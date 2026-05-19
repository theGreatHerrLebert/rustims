#!/usr/bin/env python
"""Phase 1.2 — fragment-intensity fine-tune for the calibrated-library pipeline.

Loads `rescored_canonical.fragments.parquet` (per-PSM fragment annotations
from sage with `annotate_matches=True`, dumped by the patched
rescore_canonical.py), filters to high-confidence target PSMs via
rescored_canonical.tdc.csv, encodes targets to the canonical Prosit
174-dim layout via intensity_target_encoder, fine-tunes the
UnifiedPeptideModel intensity head jointly with the shared encoder,
and reports spectral angle similarity on a peptide-level holdout.

The training-data unit is one row per PSM (NOT aggregated per peptide):
fragmentation depends on precursor charge AND collision energy, so a
peptide observed at z=2 vs z=3 produces different ground-truth vectors.
Holdout is by sequence — no peptide overlap between train and test.

Loss: masked_spectral_distance (canonical Prosit loss; treats zeros as
"unmatched" → ignored in the masked metric).

Usage:
    python finetune_dia_pasef_intensity.py \\
        --rescore-dir /path/to/rescore-output \\
        --out-dir ./checkpoints/finetune_o240206_intensity
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from imspy_predictors.models.unified import UnifiedPeptideModel
from imspy_predictors.utilities import ProformaTokenizer
from imspy_predictors.losses import masked_spectral_distance
from imspy_predictors.utility import get_model_path

# Encoder lives next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from intensity_target_encoder import (
    encode_psm_target_vec, PROSIT_DIM, ION_INT_TO_LABEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("finetune_intensity")


def load_psm_targets(rescore_dir: Path, q_cutoff: float):
    """Returns DataFrame keyed by PSM (one row per high-conf PSM) with
    columns sequence, charge, collision_energy, target (174-vec).
    """
    tdc_path = rescore_dir / "rescored_canonical.tdc.csv"
    psm_path = rescore_dir / "rescored_canonical.csv"
    frag_path = rescore_dir / "rescored_canonical.fragments.parquet"

    log.info(f"reading {tdc_path}")
    tdc = pd.read_csv(tdc_path,
                      usecols=["spec_idx", "match_idx", "decoy", "q_value"])
    tdc = tdc[(~tdc.decoy.astype(bool)) & (tdc.q_value <= q_cutoff)]
    log.info(f"  high-conf PSMs: {len(tdc):,}")

    log.info(f"reading {psm_path} (charge + CE only)")
    psm = pd.read_csv(psm_path,
                      usecols=["spec_idx", "match_idx", "charge",
                               "collision_energy", "calcmass"])

    log.info(f"reading {frag_path}")
    frag = pd.read_parquet(frag_path)
    log.info(f"  fragments: {len(frag):,}")

    # Keep only fragments belonging to high-conf PSMs (inner join on
    # the (spec_idx, sequence) pair the parquet uses).
    hc_keys = tdc.rename(columns={"match_idx": "sequence"})[
        ["spec_idx", "sequence"]]
    frag_hc = frag.merge(hc_keys, on=["spec_idx", "sequence"], how="inner")
    log.info(f"  fragments ∩ high-conf: {len(frag_hc):,}")

    # Encode targets per PSM.
    log.info("encoding 174-dim targets per PSM …")
    t0 = time.time()
    psm_groups = frag_hc.groupby(["spec_idx", "sequence"], sort=False)
    rows = []
    for (spec_idx, seq), grp in psm_groups:
        target = encode_psm_target_vec(grp)
        rows.append({"spec_idx": spec_idx, "sequence": seq, "target": target})
    target_df = pd.DataFrame(rows)
    log.info(f"  encoded {len(target_df):,} PSM targets in {time.time() - t0:.1f}s")

    # Join charge + CE per PSM. Use the rank-1 match per (spec_idx, sequence).
    psm_meta = psm.rename(columns={"match_idx": "sequence"}).drop_duplicates(
        subset=["spec_idx", "sequence"], keep="first"
    )
    out = target_df.merge(
        psm_meta[["spec_idx", "sequence", "charge", "collision_energy"]],
        on=["spec_idx", "sequence"], how="inner",
    )
    log.info(f"  joined → {len(out):,} training examples  "
             f"(charge mode counts: {out.charge.astype(int).value_counts().head().to_dict()})")
    return out


def split_peptide_holdout(df: pd.DataFrame, holdout_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    seqs = df.sequence.unique()
    perm = rng.permutation(len(seqs))
    n_hold = int(round(len(seqs) * holdout_frac))
    holdout_set = set(seqs[perm[:n_hold]])
    train = df[~df.sequence.isin(holdout_set)].reset_index(drop=True)
    hold = df[df.sequence.isin(holdout_set)].reset_index(drop=True)
    log.info(f"  split (seed={seed}): train {len(train):,} examples "
             f"({train.sequence.nunique():,} peptides) / hold {len(hold):,} "
             f"examples ({hold.sequence.nunique():,} peptides)")
    return train, hold


def make_tensors(df: pd.DataFrame, tokenizer, device, pad_len=50):
    """Build a dict of tensors ready for forward()."""
    result = tokenizer(df.sequence.tolist(), padding=True, return_tensors="pt")
    tokens = result["input_ids"]
    if tokens.shape[1] < pad_len:
        pad = torch.zeros(tokens.shape[0], pad_len - tokens.shape[1],
                          dtype=torch.long)
        tokens = torch.cat([tokens, pad], dim=1)
    elif tokens.shape[1] > pad_len:
        tokens = tokens[:, :pad_len]
    tokens = tokens.to(device)

    charge = torch.tensor(df.charge.values.astype(np.int64),
                          dtype=torch.long, device=device)
    ce = torch.tensor(df.collision_energy.values.astype(np.float32),
                      dtype=torch.float32, device=device).unsqueeze(1)
    targets = np.stack(df.target.values).astype(np.float32)
    target_t = torch.tensor(targets, dtype=torch.float32, device=device)
    return {"tokens": tokens, "charge": charge, "ce": ce, "target": target_t}


@torch.no_grad()
def predict_intensity(model, batch, batch_size=256):
    n = batch["tokens"].shape[0]
    out_chunks = []
    model.eval()
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        out = model.predict_intensity(
            batch["tokens"][sl],
            charge=batch["charge"][sl],
            collision_energy=batch["ce"][sl],
        )
        # Intensity head outputs may be (B, 29, 2, 3) or flat (B, 174).
        # Flatten to (B, 174) for consistent comparison.
        if out.dim() == 4:
            out = out.reshape(out.shape[0], -1)
        elif out.dim() == 2 and out.shape[1] != 174:
            out = out.reshape(out.shape[0], -1)
        out_chunks.append(out.cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def spectral_angle(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12):
    """Spectral angle similarity per sample, ignoring positions where
    target is zero (canonical Prosit metric). Returns (B,) array."""
    mask = target > 0
    sims = np.zeros(pred.shape[0], dtype=np.float64)
    for i in range(pred.shape[0]):
        m = mask[i]
        if m.sum() < 2:
            sims[i] = np.nan
            continue
        p = pred[i][m]
        t = target[i][m]
        # L2 normalize and compute cosine
        p_norm = p / max(eps, np.linalg.norm(p))
        t_norm = t / max(eps, np.linalg.norm(t))
        cos = float(np.clip(np.dot(p_norm, t_norm), -1.0, 1.0))
        sims[i] = 1.0 - 2.0 * np.arccos(cos) / np.pi  # spectral angle (1 = identical)
    return sims


def custom_finetune_loop(
    model, train_t, *, epochs, batch_size, lr, patience, val_frac, seed
):
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
        optimizer, patience=2, factor=0.5, min_lr=1e-6
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    log.info(f"intensity fine-tune: epochs={epochs} batch={batch_size} lr={lr:.2e}")

    for ep in range(1, epochs + 1):
        model.train()
        epoch_perm = torch.randperm(n_tr, device=tr["tokens"].device)
        t_loss = 0.0; nb = 0
        for i in range(0, n_tr, batch_size):
            idx = epoch_perm[i:i + batch_size]
            optimizer.zero_grad()
            pred = model.predict_intensity(
                tr["tokens"][idx],
                charge=tr["charge"][idx],
                collision_energy=tr["ce"][idx],
            )
            target = tr["target"][idx]
            # Prosit head may output (B, 29, 2, 3) — flatten target to match.
            if pred.dim() == 4:
                target = target.reshape(pred.shape)
            loss = masked_spectral_distance(target, pred)
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
                pred = model.predict_intensity(
                    va["tokens"][sl],
                    charge=va["charge"][sl],
                    collision_energy=va["ce"][sl],
                )
                target = va["target"][sl]
                if pred.dim() == 4:
                    target = target.reshape(pred.shape)
                v_loss += masked_spectral_distance(target, pred).item()
                vb += 1
            v_loss /= max(1, vb)
        scheduler.step(v_loss)
        log.info(
            f"  ep {ep:3d}/{epochs}  train_msd={t_loss:.4f}  val_msd={v_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if v_loss < best_val - 1e-4:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info(f"  early stop at ep {ep} (best val msd {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rescore-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--q-cutoff", type=float, default=0.01)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    psm_df = load_psm_targets(args.rescore_dir, q_cutoff=args.q_cutoff)
    train_df, hold_df = split_peptide_holdout(psm_df, args.holdout_frac, args.seed)

    log.info("loading pretrained UnifiedPeptideModel with intensity head")
    intensity_path = get_model_path("intensity/best_model.pt")
    model = UnifiedPeptideModel.from_pretrained(str(intensity_path),
                                                tasks=["intensity"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log.info(f"  device: {device}  params: {sum(p.numel() for p in model.parameters()):,}")
    pretrained_state = copy.deepcopy(model.state_dict())

    tokenizer = ProformaTokenizer.with_defaults()
    train_t = make_tensors(train_df, tokenizer, device)
    hold_t = make_tensors(hold_df, tokenizer, device)

    # Baseline eval on holdout.
    log.info("baseline (pretrained) eval on holdout")
    pred_pre = predict_intensity(model, hold_t)
    target_hold = hold_t["target"].cpu().numpy()
    sa_pre = spectral_angle(pred_pre, target_hold)
    log.info(f"  baseline spectral angle: mean={np.nanmean(sa_pre):.4f}  "
             f"median={np.nanmedian(sa_pre):.4f}  n_nan={int(np.isnan(sa_pre).sum())}")

    # Fine-tune.
    custom_finetune_loop(
        model, train_t,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
        val_frac=0.2, seed=args.seed,
    )

    # Post eval.
    log.info("fine-tuned eval on holdout")
    pred_post = predict_intensity(model, hold_t)
    sa_post = spectral_angle(pred_post, target_hold)
    log.info(f"  finetuned spectral angle: mean={np.nanmean(sa_post):.4f}  "
             f"median={np.nanmedian(sa_post):.4f}")

    # Save artifacts.
    finetuned_state = copy.deepcopy(model.state_dict())
    torch.save(
        {
            "model_state_dict": finetuned_state,
            "pretrained_state_dict": pretrained_state,
            "metrics": {
                "baseline_sa_mean": float(np.nanmean(sa_pre)),
                "baseline_sa_median": float(np.nanmedian(sa_pre)),
                "finetuned_sa_mean": float(np.nanmean(sa_post)),
                "finetuned_sa_median": float(np.nanmedian(sa_post)),
            },
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
        },
        args.out_dir / "intensity_finetuned.pt",
    )
    summary = {
        "baseline": {"sa_mean": float(np.nanmean(sa_pre)),
                     "sa_median": float(np.nanmedian(sa_pre))},
        "finetuned": {"sa_mean": float(np.nanmean(sa_post)),
                      "sa_median": float(np.nanmedian(sa_post))},
        "delta_sa_mean": float(np.nanmean(sa_post) - np.nanmean(sa_pre)),
        "n_train": int(len(train_df)),
        "n_hold": int(len(hold_df)),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    log.info("=" * 60)
    log.info(f"baseline    SA mean={summary['baseline']['sa_mean']:.4f}  "
             f"median={summary['baseline']['sa_median']:.4f}")
    log.info(f"fine-tuned  SA mean={summary['finetuned']['sa_mean']:.4f}  "
             f"median={summary['finetuned']['sa_median']:.4f}")
    log.info(f"delta:           {summary['delta_sa_mean']:+.4f}")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
