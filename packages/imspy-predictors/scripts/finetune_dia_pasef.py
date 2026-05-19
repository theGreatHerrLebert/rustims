#!/usr/bin/env python
"""Phase 1.0 — RT-only smoke test for the calibrated-library pipeline.

Takes a rescored DIA-PASEF PSM table (sagepy/sage canonical schema), filters
to high-confidence target PSMs, fine-tunes the pretrained RT predictor on
this run's observations, and reports RT MAE on a peptide-level holdout
(20% of unique sequences, seeded) before vs. after fine-tuning.

If the held-out MAE drops meaningfully on the same run, the calibration
hypothesis is alive and we expand to multi-task (RT + CCS + intensity)
on UnifiedPeptideModel. If it doesn't, we go back to the drawing board.

Inputs:
  - --rescore-dir: directory containing the canonical rescore output:
      * rescored_canonical.csv      (pre-TDC, has `retention_time_projected`)
      * rescored_canonical.tdc.csv  (post-TDC, has `q_value` + `decoy`)
    Joined on (spec_idx, match_idx) — match_idx is the peptide sequence.
  - --out-dir: where to write the fine-tuned checkpoint + metrics JSON.

Holdout strategy: random 20% of UNIQUE peptide sequences (no train/test
leak across charge states). Per-PSM aggregation: mean
retention_time_projected per peptide reduces noise from multiple PSMs of
the same sequence — RT predictors are sequence-conditional.

Usage:
    python scripts/finetune_dia_pasef.py \\
        --psm-csv /path/to/rescored_canonical.csv \\
        --out-dir ./checkpoints/finetune_o240206_rt
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from imspy_predictors.rt.predictors import (
    DeepChromatographyApex,
    load_deep_retention_time_predictor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("finetune_dia_pasef")


def load_and_aggregate(
    rescore_dir: Path, q_cutoff: float
) -> pd.DataFrame:
    """Join pre-TDC features with post-TDC q-values, filter, aggregate per peptide.

    The canonical rescore output splits across two files:
      * rescored_canonical.csv      — full feature matrix incl. observed `rt` (minutes)
      * rescored_canonical.tdc.csv  — post-TDC q_value + decoy flag
    Both keyed by (spec_idx, match_idx). Inner join → high-conf target PSMs
    with their observed RT.

    Supervision target is OBSERVED `rt` (minutes from the .d file) — NOT
    `retention_time_projected`. The latter is sage's projection into ITS
    predictor's space (different scale than imspy_predictors' RT model;
    using it as a target gives nonsense MAE numbers because the imspy
    predictor's pretrained output is on yet another scale).
    """
    psm_path = rescore_dir / "rescored_canonical.csv"
    tdc_path = rescore_dir / "rescored_canonical.tdc.csv"

    log.info(f"reading {tdc_path}")
    tdc = pd.read_csv(
        tdc_path,
        usecols=["spec_idx", "match_idx", "decoy", "q_value"],
    )
    log.info(f"  tdc rows: {len(tdc):,}")
    tdc = tdc[(~tdc.decoy.astype(bool)) & (tdc.q_value <= q_cutoff)]
    log.info(f"  tdc rows q≤{q_cutoff} & target: {len(tdc):,}")
    if len(tdc) == 0:
        raise SystemExit("no high-confidence PSMs in tdc table after filter")

    log.info(f"reading {psm_path} (observed rt + keys only)")
    psm = pd.read_csv(
        psm_path,
        usecols=["spec_idx", "match_idx", "rt"],
    )
    log.info(f"  psm rows: {len(psm):,}")

    df = tdc.merge(psm, on=["spec_idx", "match_idx"], how="inner")
    log.info(f"  joined rows: {len(df):,}")
    if len(df) == 0:
        raise SystemExit("inner join produced no rows — check rescore-dir contents")

    agg = (
        df.rename(columns={"match_idx": "sequence", "rt": "observed_rt"})
        .groupby("sequence", as_index=False)
        .agg(
            observed_rt=("observed_rt", "mean"),
            n_psm=("spec_idx", "size"),
        )
    )
    log.info(
        f"  unique peptides: {len(agg):,}  "
        f"(median PSMs/peptide={int(agg.n_psm.median())}, "
        f"observed_rt min={agg.observed_rt.min():.2f} max={agg.observed_rt.max():.2f})"
    )
    return agg


def split_peptide_holdout(
    agg: pd.DataFrame, holdout_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(agg))
    n_hold = int(round(len(agg) * holdout_frac))
    hold_idx = perm[:n_hold]
    train_idx = perm[n_hold:]
    train = agg.iloc[train_idx].reset_index(drop=True)
    hold = agg.iloc[hold_idx].reset_index(drop=True)
    log.info(
        f"  holdout split (seed={seed}): "
        f"train {len(train):,} / hold {len(hold):,}"
    )
    return train, hold


def evaluate_in_observed_space(
    predictor: DeepChromatographyApex,
    hold: pd.DataFrame,
    tag: str,
    projection: tuple[float, float] | None = None,
) -> dict:
    """Predict for holdout, optionally apply a (slope, intercept) linear
    projection to land in observed-RT minutes, then MAE vs observed RT.

    Pre-fine-tune the predictor outputs in its own native scale
    (~70-115 for typical peptides on the v0.5.0 checkpoint); a fair
    "before" baseline projects via least-squares onto observed RT,
    so we measure prediction *quality* rather than scale offset.
    Post-fine-tune the predictor itself outputs in observed-RT
    minutes — pass projection=None there.
    """
    pred = predictor.simulate_separation_times(hold.sequence.tolist())
    pred = pred.astype(np.float64)
    obs = hold.observed_rt.values.astype(np.float64)
    if projection is not None:
        slope, intercept = projection
        pred = slope * pred + intercept
    abs_err = np.abs(pred - obs)
    metrics = {
        "tag": tag,
        "n": int(len(hold)),
        "mae_min": float(np.mean(abs_err)),
        "median_ae_min": float(np.median(abs_err)),
        "p90_ae_min": float(np.percentile(abs_err, 90)),
        "rmse_min": float(np.sqrt(np.mean(abs_err ** 2))),
        "projected": projection is not None,
    }
    log.info(
        f"  [{tag}] n={metrics['n']:,}  "
        f"MAE={metrics['mae_min']:.3f} min  "
        f"median={metrics['median_ae_min']:.3f}  "
        f"p90={metrics['p90_ae_min']:.3f}  "
        f"RMSE={metrics['rmse_min']:.3f}"
    )
    return metrics


def fit_linear_projection(
    predictor: DeepChromatographyApex, train: pd.DataFrame
) -> tuple[float, float]:
    """Least-squares fit predictor_output → observed_rt on training set.
    Returns (slope, intercept) such that observed ≈ slope * pred + intercept.
    """
    pred = predictor.simulate_separation_times(train.sequence.tolist())
    pred = pred.astype(np.float64)
    obs = train.observed_rt.values.astype(np.float64)
    slope, intercept = np.polyfit(pred, obs, deg=1)
    log.info(
        f"  pretrained → observed RT projection: "
        f"observed ≈ {slope:.4f} * pred + {intercept:.4f}"
    )
    return float(slope), float(intercept)


def custom_finetune_loop(
    predictor: DeepChromatographyApex,
    train: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    val_frac: float,
    seed: int,
    device: torch.device,
) -> None:
    """Bypass the legacy fine_tune_model() — it calls bare model(tokens)
    which on UnifiedPeptideModel returns a dict, not a tensor (loss
    function chokes). We call model.predict_rt(tokens) which extracts
    `outputs['rt']` and returns a tensor.

    Trains the predictor to output observed RT in minutes directly.
    Internal 80/20 split for early-stopping val (no overlap with the
    outer caller's holdout — that's a separate set).
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.default_rng(seed)
    n = len(train)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    seqs_train = train.sequence.values[train_idx].tolist()
    seqs_val = train.sequence.values[val_idx].tolist()
    rt_train = train.observed_rt.values[train_idx].astype(np.float32)
    rt_val = train.observed_rt.values[val_idx].astype(np.float32)

    tokens_train = predictor._preprocess_sequences(seqs_train)
    tokens_val = predictor._preprocess_sequences(seqs_val)
    rt_train_t = torch.tensor(rt_train, device=device).unsqueeze(1)
    rt_val_t = torch.tensor(rt_val, device=device).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(tokens_train, rt_train_t),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(tokens_val, rt_val_t),
        batch_size=batch_size, shuffle=False,
    )

    model = predictor.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, min_lr=1e-6
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for tok_b, rt_b in train_loader:
            optimizer.zero_grad()
            pred = model.predict_rt(tok_b)
            loss = F.l1_loss(pred, rt_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(1, n_batches)

        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for tok_b, rt_b in val_loader:
                pred = model.predict_rt(tok_b)
                val_loss += F.l1_loss(pred, rt_b).item()
                nv += 1
        val_loss /= max(1, nv)
        scheduler.step(val_loss)
        log.info(
            f"  epoch {ep:3d}/{epochs}  "
            f"train_L1={train_loss:.4f}  val_L1={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info(f"  early stop at epoch {ep} (best val L1 {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rescore-dir", type=Path, required=True,
                        help="dir containing rescored_canonical.csv and "
                             "rescored_canonical.tdc.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--q-cutoff", type=float, default=0.01)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Lower than fine_tune_model's default 1e-3 — "
                             "we're starting from a well-trained checkpoint, "
                             "want to nudge not retrain.")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true",
                        help="Load, filter, split, print stats; do not fine-tune.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"out_dir: {args.out_dir}")

    # 1) data
    agg = load_and_aggregate(args.rescore_dir, q_cutoff=args.q_cutoff)
    train_df, hold_df = split_peptide_holdout(
        agg, holdout_frac=args.holdout_frac, seed=args.seed
    )

    if args.dry_run:
        log.info("--dry-run: stopping after data prep")
        return 0

    # 2) load pretrained RT predictor
    log.info("loading pretrained RT predictor")
    rt_model = load_deep_retention_time_predictor()
    predictor = DeepChromatographyApex(model=rt_model)
    device = next(rt_model.parameters()).device
    log.info(f"  predictor device: {device}")

    # Snapshot pretrained weights so the post-fine-tune evaluation can
    # report a true before/after on the SAME predictor instance.
    pretrained_state = copy.deepcopy(rt_model.state_dict())

    # 3) baseline eval — projects pretrained predictions onto observed RT
    #    via least-squares on the training set first. Without projection,
    #    the pretrained checkpoint outputs in its own scale (~70-115 for
    #    typical peptides) and observed is in minutes (~1-20), so raw MAE
    #    is mostly the scale offset, not prediction quality.
    log.info("fitting pretrained → observed RT projection on training set")
    proj = fit_linear_projection(predictor, train_df)

    log.info("baseline (pretrained, linearly projected) eval on holdout")
    baseline = evaluate_in_observed_space(
        predictor, hold_df, "baseline_projected", projection=proj
    )

    # 4) fine-tune via custom loop (legacy fine_tune_model crashes on
    #    UnifiedPeptideModel — its bare forward returns a dict).
    log.info(
        f"fine-tuning RT predictor: epochs={args.epochs} "
        f"batch_size={args.batch_size} lr={args.lr}"
    )
    custom_finetune_loop(
        predictor, train_df,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
        val_frac=0.2, seed=args.seed,
        device=device,
    )

    # 5) post-finetune eval — predictor now outputs in observed-RT
    #    minutes; no projection needed.
    log.info("fine-tuned eval on holdout (no projection)")
    finetuned = evaluate_in_observed_space(
        predictor, hold_df, "finetuned", projection=None
    )

    # 6) save artifacts
    finetuned_state = copy.deepcopy(rt_model.state_dict())
    torch.save(
        {
            "model_state_dict": finetuned_state,
            "pretrained_state_dict": pretrained_state,
            "metrics": {"baseline": baseline, "finetuned": finetuned},
            "args": vars(args) | {"rescore_dir": str(args.rescore_dir),
                                  "out_dir": str(args.out_dir)},
        },
        args.out_dir / "rt_finetuned.pt",
    )
    metrics_summary = {
        "baseline_projected": baseline,
        "finetuned": finetuned,
        "delta_mae_min": baseline["mae_min"] - finetuned["mae_min"],
        "delta_mae_pct": (
            100.0 * (baseline["mae_min"] - finetuned["mae_min"]) / baseline["mae_min"]
            if baseline["mae_min"] > 0 else 0.0
        ),
        "projection": {"slope": proj[0], "intercept": proj[1]},
        "n_train_peptides": int(len(train_df)),
        "n_hold_peptides": int(len(hold_df)),
        "n_train_psms": int(train_df.n_psm.sum()),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2))
    log.info(f"wrote {metrics_path}")

    # 7) summary
    log.info("=" * 60)
    log.info(f"baseline (projected)  MAE: {baseline['mae_min']:.3f} min")
    log.info(f"fine-tuned            MAE: {finetuned['mae_min']:.3f} min")
    log.info(
        f"delta:                     {metrics_summary['delta_mae_min']:+.3f} min  "
        f"({metrics_summary['delta_mae_pct']:+.2f}%)"
    )
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
