"""Per-PSM optimal NCE calibration — v4's inference recipe applied to ppng.

For each PSM, sweep the model over a grid of collision energies and pick the CE
that maximises spectral-angle similarity to the observed spectrum. Reports the
mean optimal-CE SA, the mean fixed-CE SA at the original (per-PSM dataset/catalog)
value, and the optimal-CE distribution. No retraining — this isolates "how good
is the model when the right CE is used per spectrum" from "how good is the
flat CE we used at fine-tune time."

Example:
    python -m peptide_property_ng.eval.evaluate_optimal_nce \\
        --checkpoint runs/finetune-canonicalSA/best.pt \\
        --datasets-glob '/scratch/claudius-proteomics/*' \\
        --catalog .../timstof_catalog.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from peptide_property_ng.data.collate import make_collate_fn
from peptide_property_ng.data.sage_dataset import (
    build_split_datasets, discover_sage_dirs, load_ce_calibration,
)
from peptide_property_ng.losses import intensity_signal_mask, masked_spectral_angle
from peptide_property_ng.model.config import PRESETS
from peptide_property_ng.model.multitask import UnifiedPeptidePropertyModel
from peptide_property_ng.modifications.composition import CompositionTable


@torch.no_grad()
def evaluate_optimal_nce(
    model,
    loader,
    device: str,
    ce_grid: torch.Tensor,
):
    """Return per-PSM (opt_sa, opt_ce, fixed_sa) tensors over the signal subset.

    ``fixed_sa`` uses the loader's original (per-PSM dataset/catalog) CE — i.e.
    the same CE the model saw during fine-tune; ``opt_sa`` is the per-PSM max
    over the supplied grid.
    """
    model.eval()
    ce_grid = ce_grid.to(device)
    n_ce = ce_grid.numel()

    opt_sa_list: list[torch.Tensor] = []
    opt_ce_list: list[torch.Tensor] = []
    fixed_sa_list: list[torch.Tensor] = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch["intensity_target"]
        signal = intensity_signal_mask(target)
        if not signal.any():
            continue

        B = target.shape[0]
        ce_orig = batch["collision_energy"].clone()

        # Per-CE SA similarity (B, n_ce).
        sa_matrix = torch.zeros(B, n_ce, device=device)
        for j, ce in enumerate(ce_grid):
            batch["collision_energy"] = ce.expand(B).float()
            pred = model(batch, tasks=["intensity"])["intensity"]
            sa_matrix[:, j] = 1.0 - masked_spectral_angle(pred, target)

        opt_sa, opt_idx = sa_matrix.max(dim=1)
        opt_ce = ce_grid[opt_idx]

        # Fixed reference: model's original CE (reproduces the training test SA).
        batch["collision_energy"] = ce_orig
        pred = model(batch, tasks=["intensity"])["intensity"]
        fixed_sa = 1.0 - masked_spectral_angle(pred, target)

        opt_sa_list.append(opt_sa[signal].cpu())
        opt_ce_list.append(opt_ce[signal].cpu())
        fixed_sa_list.append(fixed_sa[signal].cpu())

    return (
        torch.cat(opt_sa_list),
        torch.cat(opt_ce_list),
        torch.cat(fixed_sa_list),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="path to a best.pt")
    ap.add_argument("--datasets-glob", default="/scratch/claudius-proteomics/*")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--default-ce", type=float, default=0.26)
    ap.add_argument("--ce-calibration", default=None,
                    help="parquet of per-PSM calibrated CEs; if supplied, the 'fixed' baseline "
                         "uses these (matches a fine-tune that was trained with the same calibration)")
    ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--max-datasets", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ce-min", type=float, default=0.10)
    ap.add_argument("--ce-max", type=float, default=0.40)
    ap.add_argument("--ce-step", type=float, default=0.02)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"checkpoint: {args.checkpoint}")
    sage_dirs = discover_sage_dirs(args.datasets_glob)
    if args.max_datasets:
        sage_dirs = sage_dirs[: args.max_datasets]
    print(f"  {len(sage_dirs)} datasets, cap {args.cap}/dataset")
    ce_cal = load_ce_calibration(args.ce_calibration) if args.ce_calibration else None
    splits = build_split_datasets(sage_dirs, cap=args.cap, seed=args.seed,
                                  catalog_path=args.catalog, default_ce=args.default_ce,
                                  ce_calibration=ce_cal)
    ds = splits[args.split]

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    preset = ckpt.get("preset", "small")
    cfg = PRESETS[preset]
    collate = make_collate_fn(cfg.pad_token_id, cfg.max_charge)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = UnifiedPeptidePropertyModel(cfg, CompositionTable.load()).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    n_pts = int(round((args.ce_max - args.ce_min) / args.ce_step)) + 1
    ce_grid = torch.linspace(args.ce_min, args.ce_max, n_pts)
    print(f"split={args.split}, |dataset|={len(ds):,}, preset={preset}")
    print(f"CE grid ({n_pts} pts): {args.ce_min:.3f} .. {args.ce_max:.3f} step {args.ce_step:.3f}")
    print()

    opt_sa, opt_ce, fixed_sa = evaluate_optimal_nce(model, loader, args.device, ce_grid)
    n = opt_sa.numel()
    lift = float(opt_sa.mean() - fixed_sa.mean())
    print(f"PSMs scored:                       {n:,}")
    print(f"fixed (original per-PSM CE)  SA:   {float(fixed_sa.mean()):.4f}")
    print(f"per-PSM optimal CE           SA:   {float(opt_sa.mean()):.4f}   (lift +{lift:.4f})")
    print()
    print("optimal CE distribution (per-PSM):")
    for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        print(f"  q{int(q*100):02d}   {float(opt_ce.quantile(q)):.3f}")
    print(f"  mean   {float(opt_ce.mean()):.3f}")
    # Mode by histogram (CE grid is discrete, so this is exact).
    vals, counts = torch.unique(opt_ce, return_counts=True)
    print(f"  mode   {float(vals[counts.argmax()]):.3f}  ({int(counts.max())} / {n} PSMs)")
    print()
    print("share of PSMs at each CE grid point:")
    for v, c in zip(vals.tolist(), counts.tolist()):
        bar = "#" * int(40 * c / n)
        print(f"  {v:.3f}  {c:>5d}  {c/n:5.1%}  {bar}")


if __name__ == "__main__":
    main()
