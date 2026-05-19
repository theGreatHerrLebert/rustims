#!/usr/bin/env python
"""Phase 1.1 CCS-only diagnostic — standalone fine-tune of the CCS head.

The multi-task fine-tune (finetune_dia_pasef_multi.py) only moved CCS
~3% on the holdout — likely because cross-task gradient interference
and a 1e-4 LR is undersized for the CCS head's SquareRootProjectionLayer
slopes/intercepts (charge-specific, big-target params). This is the
diagnostic isolation: just CCS, more epochs, higher LR, no RT loss.

If this converges to <10 Å² MAE on holdout, the multi-task hyperparameters
were the bottleneck. If it plateaus at ~30 Å² MAE (≈ pretrained + linear
projection), the architecture has a real ceiling and we need a
1/K0-native head.

Supervises in CCS Å² (head's native output scale). Reports:
  - CCS Å² MAE (head's space)
  - 1/K0 MAE (predicted CCS converted via Mason-Schamp; library use)
  - Per-charge breakdown (z=2 vs z=3 — the dominant charges)
  - Pre/post per-charge slope+intercept of the physics layer
    (sanity: did the physics-prior parameters move at all?)
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from imspy_predictors.models.unified import UnifiedPeptideModel
from imspy_predictors.utilities import ProformaTokenizer
from imspy_predictors.utility import get_model_path
from imspy_core.chemistry.mobility import (
    ccs_to_one_over_k0_par, one_over_k0_to_ccs_par,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("finetune_ccs")

PROTON = 1.007276466879


def read_pseudo_bin_apex_scans(bin_path: Path) -> np.ndarray:
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        if magic != b"PSSP":
            raise SystemExit(f"bad magic {magic!r}")
        version, n_spec, _ = struct.unpack("<III", f.read(12))
        if version != 2:
            raise SystemExit(f"version {version}")
        f.seek(4 * n_spec, 1)
        f.seek(8 * n_spec, 1)
        f.seek(4 * n_spec, 1)
        return np.frombuffer(f.read(4 * n_spec), dtype="<f4").copy()


def build_scan_to_im_lut(d_path: Path, sample_frame_id: int = 1) -> np.ndarray:
    from imspy_core.timstof import TimsDataset
    ds = TimsDataset(str(d_path))
    n_scans = ds.num_scans
    scans = np.arange(1, n_scans + 1, dtype=np.int32)
    im = np.asarray(ds.scan_to_inverse_mobility(sample_frame_id, scans),
                    dtype=np.float64)
    lut = np.zeros(n_scans + 2, dtype=np.float64)
    lut[1:n_scans + 1] = im
    return lut


def lookup_im(scan, lut):
    s = np.asarray(scan, dtype=np.float64)
    s_lo = np.floor(s).astype(np.int64)
    s_hi = s_lo + 1
    s_lo = np.clip(s_lo, 1, len(lut) - 2)
    s_hi = np.clip(s_hi, 1, len(lut) - 1)
    f = s - s_lo
    return lut[s_lo] * (1.0 - f) + lut[s_hi] * f


def load_psm_data(rescore_dir: Path, q_cutoff: float, pseudo_bin: Path, d_path: Path):
    tdc = pd.read_csv(rescore_dir / "rescored_canonical.tdc.csv",
                      usecols=["spec_idx", "match_idx", "decoy", "q_value"])
    tdc = tdc[(~tdc.decoy.astype(bool)) & (tdc.q_value <= q_cutoff)]
    log.info(f"  high-conf PSMs: {len(tdc):,}")
    psm = pd.read_csv(rescore_dir / "rescored_canonical.csv",
                      usecols=["spec_idx", "match_idx", "rt", "charge", "calcmass"])
    df = tdc.merge(psm, on=["spec_idx", "match_idx"], how="inner")
    apex_scan = read_pseudo_bin_apex_scans(pseudo_bin)
    lut = build_scan_to_im_lut(d_path)
    bin_idx = df.spec_idx.str.removeprefix("pseudo_").astype(np.int64).values
    one_over_k0 = lookup_im(apex_scan[bin_idx], lut)
    charge_arr = df.charge.values.astype(np.float64)
    mz_arr = (df.calcmass.values + charge_arr * PROTON) / charge_arr
    observed_ccs = np.asarray(
        one_over_k0_to_ccs_par(one_over_k0, mz_arr, charge_arr.astype(np.int32)),
        dtype=np.float64,
    )
    df = df.assign(
        observed_ims=one_over_k0,
        observed_ccs=observed_ccs,
    )
    df = df.rename(columns={"match_idx": "sequence"})
    # Per-(peptide, charge) aggregation — fragmentation isn't relevant for CCS
    # but charge IS, so we keep separate rows per charge state of same peptide.
    agg = df.groupby(["sequence", "charge"], as_index=False).agg(
        observed_rt=("rt", "mean"),
        observed_ims=("observed_ims", "mean"),
        observed_ccs=("observed_ccs", "mean"),
        calcmass=("calcmass", "mean"),
        n_psm=("spec_idx", "size"),
    )
    log.info(
        f"  unique (peptide, charge) pairs: {len(agg):,}  "
        f"CCS Å²: {agg.observed_ccs.min():.1f}-{agg.observed_ccs.max():.1f}  "
        f"1/K0: {agg.observed_ims.min():.4f}-{agg.observed_ims.max():.4f}  "
        f"charge counts: {dict(agg.charge.astype(int).value_counts().head())}"
    )
    return agg


def split_peptide_holdout(df: pd.DataFrame, holdout_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    seqs = df.sequence.unique()
    perm = rng.permutation(len(seqs))
    n_hold = int(round(len(seqs) * holdout_frac))
    holdout_set = set(seqs[perm[:n_hold]])
    train = df[~df.sequence.isin(holdout_set)].reset_index(drop=True)
    hold = df[df.sequence.isin(holdout_set)].reset_index(drop=True)
    log.info(f"  split (seed={seed}): train {len(train):,} / hold {len(hold):,}")
    return train, hold


def make_tensors(df: pd.DataFrame, tokenizer, device, pad_len=50):
    result = tokenizer(df.sequence.tolist(), padding=True, return_tensors="pt")
    tokens = result["input_ids"]
    if tokens.shape[1] < pad_len:
        pad = torch.zeros(tokens.shape[0], pad_len - tokens.shape[1], dtype=torch.long)
        tokens = torch.cat([tokens, pad], dim=1)
    elif tokens.shape[1] > pad_len:
        tokens = tokens[:, :pad_len]
    tokens = tokens.to(device)
    charge = torch.tensor(df.charge.values.astype(np.int64), dtype=torch.long, device=device)
    mz = (df.calcmass.values + df.charge.values * PROTON) / df.charge.values
    mz_t = torch.tensor(mz, dtype=torch.float32, device=device).unsqueeze(1)
    ccs = torch.tensor(df.observed_ccs.values, dtype=torch.float32, device=device).unsqueeze(1)
    ims = torch.tensor(df.observed_ims.values, dtype=torch.float32, device=device).unsqueeze(1)
    return {"tokens": tokens, "charge": charge, "mz": mz_t,
            "ccs": ccs, "ims": ims}


@torch.no_grad()
def predict_ccs(model, batch, batch_size=512):
    n = batch["tokens"].shape[0]
    out_chunks = []
    model.eval()
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        ccs_mean, _ = model.predict_ccs(
            batch["tokens"][sl],
            mz=batch["mz"][sl],
            charge=batch["charge"][sl],
        )
        out_chunks.append(ccs_mean.squeeze(-1).cpu().numpy())
    return np.concatenate(out_chunks, axis=0).astype(np.float64)


def report_per_charge_mae(pred_ccs, obs_ccs, charges, label):
    log.info(f"  [{label}] per-charge:")
    for z in (1, 2, 3, 4):
        m = charges == z
        if m.sum() == 0: continue
        mae = float(np.mean(np.abs(pred_ccs[m] - obs_ccs[m])))
        log.info(f"    z={z}: n={int(m.sum()):,}  MAE={mae:.2f} Å²")


def custom_finetune_loop(
    model, train_t, *, epochs, batch_size, lr, patience, val_frac, seed
):
    import torch.nn.functional as F
    n = train_t["tokens"].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx = torch.tensor(perm[:n_val], device=train_t["tokens"].device)
    train_idx = torch.tensor(perm[n_val:], device=train_t["tokens"].device)

    def _slice(b, idx):
        return {k: v[idx] for k, v in b.items()}

    tr = _slice(train_t, train_idx)
    va = _slice(train_t, val_idx)
    n_tr = tr["tokens"].shape[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5, min_lr=1e-6
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    log.info(f"CCS-only fine-tune: epochs={epochs} batch={batch_size} lr={lr:.2e}")

    for ep in range(1, epochs + 1):
        model.train()
        epoch_perm = torch.randperm(n_tr, device=tr["tokens"].device)
        t_loss = 0.0; nb = 0
        for i in range(0, n_tr, batch_size):
            idx = epoch_perm[i:i + batch_size]
            optimizer.zero_grad()
            ccs_mean, _ = model.predict_ccs(
                tr["tokens"][idx], mz=tr["mz"][idx], charge=tr["charge"][idx],
            )
            loss = F.l1_loss(ccs_mean, tr["ccs"][idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item(); nb += 1
        t_loss /= nb

        model.eval()
        v_loss = 0.0; vb = 0
        with torch.no_grad():
            for i in range(0, va["tokens"].shape[0], batch_size):
                sl = slice(i, i + batch_size)
                ccs_mean, _ = model.predict_ccs(
                    va["tokens"][sl], mz=va["mz"][sl], charge=va["charge"][sl],
                )
                v_loss += F.l1_loss(ccs_mean, va["ccs"][sl]).item()
                vb += 1
            v_loss /= max(1, vb)
        scheduler.step(v_loss)
        if ep <= 5 or ep % 5 == 0:
            log.info(
                f"  ep {ep:3d}/{epochs}  tr={t_loss:.3f} val={v_loss:.3f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        if v_loss < best_val - 1e-3:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info(f"  early stop at ep {ep} (best val L1 {best_val:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rescore-dir", type=Path, required=True)
    parser.add_argument("--pseudo-bin", type=Path, required=True)
    parser.add_argument("--d-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--q-cutoff", type=float, default=0.01)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_psm_data(args.rescore_dir, args.q_cutoff, args.pseudo_bin, args.d_path)
    train_df, hold_df = split_peptide_holdout(df, args.holdout_frac, args.seed)

    log.info("loading pretrained UnifiedPeptideModel with CCS head")
    ccs_path = get_model_path("ccs/best_model.pt")
    model = UnifiedPeptideModel.from_pretrained(str(ccs_path), tasks=["ccs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log.info(f"  device: {device}  params: {sum(p.numel() for p in model.parameters()):,}")

    # Sanity: print physics-layer slopes/intercepts before
    sqrt_proj = model.heads["ccs"].sqrt_proj
    log.info(f"  pre-finetune SquareRootProjectionLayer: "
             f"slopes={sqrt_proj.slopes.detach().cpu().numpy()}  "
             f"intercepts={sqrt_proj.intercepts.detach().cpu().numpy()}")

    pretrained_state = copy.deepcopy(model.state_dict())
    tokenizer = ProformaTokenizer.with_defaults()
    train_t = make_tensors(train_df, tokenizer, device)
    hold_t = make_tensors(hold_df, tokenizer, device)

    obs_hold_ccs = hold_df.observed_ccs.values.astype(np.float64)
    obs_hold_ims = hold_df.observed_ims.values.astype(np.float64)
    z_hold = hold_df.charge.values.astype(np.int32)
    mz_hold = (hold_df.calcmass.values + hold_df.charge.values * PROTON) / hold_df.charge.values

    # Baseline: pretrained predictions, with optional linear projection on train
    ccs_pred_pre = predict_ccs(model, hold_t)
    ccs_pred_train_pre = predict_ccs(model, train_t)
    train_obs_ccs = train_df.observed_ccs.values.astype(np.float64)
    slope, intercept = np.polyfit(ccs_pred_train_pre, train_obs_ccs, 1)
    log.info(f"baseline: pretrained CCS → observed projection: "
             f"obs ≈ {slope:.4f}*pred + {intercept:.4f} Å²")
    pre_raw_mae = float(np.mean(np.abs(ccs_pred_pre - obs_hold_ccs)))
    pre_proj_mae = float(np.mean(np.abs(slope * ccs_pred_pre + intercept - obs_hold_ccs)))
    pre_ims_proj_mae = float(np.mean(np.abs(
        np.asarray(ccs_to_one_over_k0_par(slope * ccs_pred_pre + intercept, mz_hold, z_hold),
                   dtype=np.float64) - obs_hold_ims
    )))
    log.info(f"  pretrained raw       MAE: {pre_raw_mae:.2f} Å²  (no projection)")
    log.info(f"  pretrained+proj      MAE: {pre_proj_mae:.2f} Å²")
    log.info(f"  pretrained+proj→1/K0 MAE: {pre_ims_proj_mae:.4f}")
    report_per_charge_mae(slope * ccs_pred_pre + intercept, obs_hold_ccs, z_hold,
                          "baseline+proj")

    # Fine-tune
    custom_finetune_loop(
        model, train_t,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
        val_frac=0.2, seed=args.seed,
    )

    # Post
    sqrt_proj = model.heads["ccs"].sqrt_proj
    log.info(f"  post-finetune SquareRootProjectionLayer: "
             f"slopes={sqrt_proj.slopes.detach().cpu().numpy()}  "
             f"intercepts={sqrt_proj.intercepts.detach().cpu().numpy()}")
    ccs_pred_post = predict_ccs(model, hold_t)
    post_mae = float(np.mean(np.abs(ccs_pred_post - obs_hold_ccs)))
    post_ims_mae = float(np.mean(np.abs(
        np.asarray(ccs_to_one_over_k0_par(ccs_pred_post, mz_hold, z_hold),
                   dtype=np.float64) - obs_hold_ims
    )))
    log.info(f"  finetuned       MAE: {post_mae:.2f} Å²")
    log.info(f"  finetuned→1/K0  MAE: {post_ims_mae:.4f}")
    report_per_charge_mae(ccs_pred_post, obs_hold_ccs, z_hold, "finetuned")

    # Save
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pretrained_state_dict": pretrained_state,
            "metrics": {
                "baseline_raw_mae_a2": pre_raw_mae,
                "baseline_proj_mae_a2": pre_proj_mae,
                "baseline_proj_ims_mae": pre_ims_proj_mae,
                "finetuned_mae_a2": post_mae,
                "finetuned_ims_mae": post_ims_mae,
            },
            "ccs_projection": {"slope": float(slope), "intercept": float(intercept)},
        },
        args.out_dir / "ccs_finetuned.pt",
    )
    summary = {
        "ccs_a2": {"baseline_proj": pre_proj_mae, "finetuned": post_mae,
                   "delta": pre_proj_mae - post_mae,
                   "delta_pct": 100.0 * (pre_proj_mae - post_mae) / pre_proj_mae if pre_proj_mae > 0 else 0.0},
        "ims": {"baseline_proj": pre_ims_proj_mae, "finetuned": post_ims_mae,
                "delta": pre_ims_proj_mae - post_ims_mae,
                "delta_pct": 100.0 * (pre_ims_proj_mae - post_ims_mae) / pre_ims_proj_mae if pre_ims_proj_mae > 0 else 0.0},
        "n_train": int(len(train_df)),
        "n_hold": int(len(hold_df)),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    log.info("=" * 60)
    log.info(f"CCS Å² baseline+proj MAE={pre_proj_mae:.2f}  finetuned MAE={post_mae:.2f}  "
             f"delta={summary['ccs_a2']['delta']:+.2f} ({summary['ccs_a2']['delta_pct']:+.2f}%)")
    log.info(f"1/K0   baseline+proj MAE={pre_ims_proj_mae:.4f}  finetuned MAE={post_ims_mae:.4f}  "
             f"delta={summary['ims']['delta']:+.4f} ({summary['ims']['delta_pct']:+.2f}%)")
    log.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
