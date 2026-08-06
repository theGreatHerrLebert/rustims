"""Compute per-PSM optimal CE with a frozen checkpoint and write it to a parquet.

For each prepared PSM, sweep collision energy over a grid and pick the value
that maximises spectral-angle similarity to the observed spectrum. The output
is a parquet of ``(accession, psm_id, calibrated_ce)`` that ``train.train``
can load via ``--ce-calibration`` to fine-tune with per-PSM CE instead of a
flat default.

Recipe: calibrate the *pretrained* (or other freeze-point) checkpoint once
over the full prepared corpus; then fine-tune from that same checkpoint with
the calibration applied. v4's effective recipe.

Example:
    python -m peptide_property_ng.train.calibrate_ce \\
        --checkpoint /scratch/.../small-pretrained.pt \\
        --datasets-glob '/scratch/claudius-proteomics/*' \\
        --catalog .../timstof_catalog.tsv \\
        --out /scratch/.../ce_calibration_pretrained_small.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from peptide_property_ng.data.collate import make_collate_fn
from peptide_property_ng.data.sage_dataset import (
    SagePropertyDataset,
    build_split_datasets,
    discover_sage_dirs,
)
from peptide_property_ng.losses import intensity_signal_mask, masked_spectral_angle
from peptide_property_ng.model.config import PRESETS
from peptide_property_ng.model.multitask import UnifiedPeptidePropertyModel
from peptide_property_ng.modifications.composition import CompositionTable


@torch.no_grad()
def calibrate(model, loader, device: str, ce_grid: torch.Tensor) -> tuple[list, list, list, list]:
    """Sweep the grid per batch, return per-example ``(acc, psm, ce, fixed_sa, opt_sa)``."""
    model.eval()
    ce_grid = ce_grid.to(device)
    n_ce = ce_grid.numel()

    accs: list[str] = []
    pids: list[int] = []
    opt_ces: list[float] = []
    fixed_sas: list[float] = []
    opt_sas: list[float] = []

    for batch in loader:
        meta_acc = batch.pop("_accession")
        meta_pid = batch.pop("_psm_id")
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch["intensity_target"]
        signal = intensity_signal_mask(target)
        B = target.shape[0]

        ce_orig = batch["collision_energy"].clone()
        sa_matrix = torch.full((B, n_ce), float("nan"), device=device)
        for j, ce in enumerate(ce_grid):
            batch["collision_energy"] = ce.expand(B).float()
            pred = model(batch, tasks=["intensity"])["intensity"]
            sa_matrix[:, j] = 1.0 - masked_spectral_angle(pred, target)

        # PSMs with no signal: keep their CE at the loader's default (no calibration);
        # also mark fixed_sa / opt_sa as NaN so the parquet records "uncalibrated".
        opt_sa, opt_idx = sa_matrix.max(dim=1)
        opt_ce = ce_grid[opt_idx]
        opt_ce = torch.where(signal, opt_ce, ce_orig)

        batch["collision_energy"] = ce_orig
        pred = model(batch, tasks=["intensity"])["intensity"]
        fixed_sa = 1.0 - masked_spectral_angle(pred, target)

        for i in range(B):
            accs.append(meta_acc[i])
            pids.append(int(meta_pid[i].item()))
            opt_ces.append(float(opt_ce[i].item()))
            if bool(signal[i]):
                fixed_sas.append(float(fixed_sa[i].item()))
                opt_sas.append(float(opt_sa[i].item()))
            else:
                fixed_sas.append(float("nan"))
                opt_sas.append(float("nan"))
    return accs, pids, opt_ces, fixed_sas, opt_sas


def _collate_with_meta(pad_token_id: int, max_charge: int):
    """Same as ``make_collate_fn`` but also returns per-example accession + psm_id."""
    inner = make_collate_fn(pad_token_id, max_charge)

    def _fn(examples: list[dict]) -> dict:
        out = inner(examples)
        out["_accession"] = [e["accession"] for e in examples]
        out["_psm_id"] = torch.tensor([e["psm_id"] for e in examples], dtype=torch.int64)
        return out

    return _fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint to calibrate against")
    ap.add_argument("--datasets-glob", default="/scratch/claudius-proteomics/*")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--default-ce", type=float, default=0.26)
    ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--max-datasets", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0,
                    help="MUST match what train.train uses (controls per-dataset PSM sampling)")
    ap.add_argument("--ce-min", type=float, default=0.00)
    ap.add_argument("--ce-max", type=float, default=0.60)
    ap.add_argument("--ce-step", type=float, default=0.02)
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sage_dirs = discover_sage_dirs(args.datasets_glob)
    if args.max_datasets:
        sage_dirs = sage_dirs[: args.max_datasets]
    print(f"[{time.strftime('%H:%M:%S')}] {len(sage_dirs)} datasets (cap {args.cap}/dataset)")
    splits = build_split_datasets(sage_dirs, cap=args.cap, seed=args.seed,
                                  catalog_path=args.catalog, default_ce=args.default_ce)
    examples = splits["train"].examples + splits["val"].examples + splits["test"].examples
    print(f"  total {len(examples):,} prepared examples to calibrate "
          f"({len(splits['train']):,} train + {len(splits['val']):,} val + {len(splits['test']):,} test)")

    import dataclasses
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    preset = ckpt.get("preset", "small")
    cfg = PRESETS[preset]
    intensity_head = ckpt.get("intensity_head")
    if intensity_head is None:
        # Legacy checkpoint without metadata — infer from the state-dict keys.
        keys = ckpt["model_state_dict"].keys()
        if any("heads.intensity.attention." in k for k in keys):
            intensity_head = "pooled"
        else:
            intensity_head = "site"
    cfg = dataclasses.replace(cfg, intensity_head=intensity_head)
    model = UnifiedPeptidePropertyModel(cfg, CompositionTable.load()).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  checkpoint preset={preset}, intensity_head={intensity_head}, "
          f"{model.num_parameters():,} parameters")

    n_pts = int(round((args.ce_max - args.ce_min) / args.ce_step)) + 1
    ce_grid = torch.linspace(args.ce_min, args.ce_max, n_pts)
    print(f"  CE grid ({n_pts} pts): {args.ce_min:.3f} .. {args.ce_max:.3f} step {args.ce_step:.3f}")

    loader = DataLoader(
        SagePropertyDataset(examples), batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate_with_meta(cfg.pad_token_id, cfg.max_charge),
    )
    t0 = time.time()
    accs, pids, opt_ces, fixed_sas, opt_sas = calibrate(model, loader, args.device, ce_grid)
    print(f"[{time.strftime('%H:%M:%S')}] calibrated {len(accs):,} PSMs in {time.time() - t0:.0f}s")

    table = pa.table({
        "accession": pa.array(accs, type=pa.string()),
        "psm_id": pa.array(pids, type=pa.int64()),
        "calibrated_ce": pa.array(opt_ces, type=pa.float32()),
        "fixed_sa": pa.array(fixed_sas, type=pa.float32()),
        "opt_sa": pa.array(opt_sas, type=pa.float32()),
    })
    pq.write_table(table, out_path)
    print(f"  wrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")

    # Quick summary of the calibration so the user can see the lift.
    import numpy as np
    fa = np.array(fixed_sas, dtype=np.float64)
    oa = np.array(opt_sas, dtype=np.float64)
    valid = ~np.isnan(fa) & ~np.isnan(oa)
    print(f"\nover the {int(valid.sum()):,} signal PSMs:")
    print(f"  fixed (per-PSM default/catalog CE)  mean SA  {fa[valid].mean():.4f}")
    print(f"  per-PSM optimal CE                  mean SA  {oa[valid].mean():.4f}  "
          f"(lift +{(oa[valid] - fa[valid]).mean():.4f})")
    ce_arr = np.array(opt_ces, dtype=np.float64)[valid]
    print(f"  CE: mean {ce_arr.mean():.3f}  median {np.median(ce_arr):.3f}  "
          f"q05 {np.quantile(ce_arr, 0.05):.3f}  q95 {np.quantile(ce_arr, 0.95):.3f}")


if __name__ == "__main__":
    main()
