#!/usr/bin/env python
"""Phase B — joint RT+CCS+intensity fine-tune ablation.

Built on the Phase A finding: loading from rt/best_model.pt with
tasks=['rt','ccs','intensity'] and swapping the ccs/intensity heads from
their per-task checkpoints gives baselines within 2% of standalone for
all three tasks. So joint training IS viable — Phase B answers WHICH
fine-tune flavor wins on the per-run calibration task.

Five flavors of fine-tune (all from the same joint-init):

  B1: Heads only, encoder frozen. Safest, no encoder drift.
  B2: Heads + encoder, joint loss with static weights.
  B3: Heads + encoder with split LR groups (heads 1e-3, encoder 1e-5).
  B4: Heads + encoder with uncertainty weighting (Kendall et al.).
  B5: Sequential — intensity-only first, then RT+CCS on top.

Reports per-task held-out MAE/SA after each flavor; picks winner.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from imspy_predictors.models.unified import UnifiedPeptideModel
from imspy_predictors.utilities import ProformaTokenizer
from imspy_predictors.utility import get_model_path
from imspy_predictors.losses import masked_spectral_distance
from imspy_core.chemistry.mobility import (
    ccs_to_one_over_k0_par, one_over_k0_to_ccs_par,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intensity_target_encoder import encode_psm_target_vec, PROSIT_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("joint_finetune")

PROTON = 1.007276466879
TASK_NAMES = ("rt", "ccs", "intensity")
BASE_ENCODER = "rt"  # Phase A winner — RT-encoder + swapped CCS/intensity heads


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_pseudo_bin_apex_scans(bin_path: Path) -> np.ndarray:
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"PSSP", magic
        version, n_spec, _ = struct.unpack("<III", f.read(12))
        assert version == 2, version
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


def load_data(rescore_dir: Path, q_cutoff: float, pseudo_bin: Path, d_path: Path):
    """Build per-PSM training table with all three tasks' targets attached.

    Per-PSM unit so we can supervise intensity per fragment-set. RT and
    CCS targets are the per-(peptide, charge) means (avoids noise from
    individual PSM RT/IM jitter).
    """
    tdc = pd.read_csv(rescore_dir / "rescored_canonical.tdc.csv",
                      usecols=["spec_idx", "match_idx", "decoy", "q_value"])
    tdc = tdc[(~tdc.decoy.astype(bool)) & (tdc.q_value <= q_cutoff)]
    psm = pd.read_csv(rescore_dir / "rescored_canonical.csv",
                      usecols=["spec_idx", "match_idx", "rt", "charge",
                               "calcmass", "collision_energy"])
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
    df = df.assign(observed_ims=one_over_k0, observed_ccs=observed_ccs,
                   precursor_mz=mz_arr)
    df = df.rename(columns={"match_idx": "sequence"})

    # Per-(peptide, charge) means for RT/CCS
    pep_charge = df.groupby(["sequence", "charge"], as_index=False).agg(
        rt_target=("rt", "mean"),
        ccs_target=("observed_ccs", "mean"),
        ims_target=("observed_ims", "mean"),
    )
    df = df.merge(pep_charge, on=["sequence", "charge"], how="inner")
    log.info(f"  PSMs joined: {len(df):,}  unique (pep, z): {len(pep_charge):,}")

    # Encode intensity targets per PSM
    log.info("encoding intensity targets per PSM …")
    t0 = time.time()
    frag_df = pd.read_parquet(rescore_dir / "rescored_canonical.fragments.parquet")
    hc = df[["spec_idx", "sequence"]]
    frag_hc = frag_df.merge(hc, on=["spec_idx", "sequence"], how="inner")
    grp = frag_hc.groupby(["spec_idx", "sequence"], sort=False)
    intensity_target = {}
    for key, g in grp:
        intensity_target[key] = encode_psm_target_vec(g)
    log.info(f"  encoded {len(intensity_target):,} PSM intensity targets in {time.time()-t0:.1f}s")

    # Attach intensity target column (numpy array per row)
    df["intensity_target"] = df.apply(
        lambda r: intensity_target.get((r.spec_idx, r.sequence)), axis=1
    )
    n_before = len(df)
    df = df[df.intensity_target.notna()].reset_index(drop=True)
    log.info(f"  PSMs with intensity targets: {len(df):,} (dropped {n_before-len(df):,})")
    return df


def split_seq_holdout(df: pd.DataFrame, holdout_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    seqs = df.sequence.unique()
    perm = rng.permutation(len(seqs))
    n_hold = int(round(len(seqs) * holdout_frac))
    holdout_set = set(seqs[perm[:n_hold]])
    train = df[~df.sequence.isin(holdout_set)].reset_index(drop=True)
    hold = df[df.sequence.isin(holdout_set)].reset_index(drop=True)
    return train, hold


def make_tensors(df: pd.DataFrame, tokenizer, device, pad_len=50):
    result = tokenizer(df.sequence.tolist(), padding=True, return_tensors="pt")
    tokens = result["input_ids"]
    if tokens.shape[1] < pad_len:
        pad = torch.zeros(tokens.shape[0], pad_len - tokens.shape[1], dtype=torch.long)
        tokens = torch.cat([tokens, pad], dim=1)
    elif tokens.shape[1] > pad_len:
        tokens = tokens[:, :pad_len]
    return {
        "tokens": tokens.to(device),
        "charge": torch.tensor(df.charge.values.astype(np.int64),
                               dtype=torch.long, device=device),
        "mz": torch.tensor(df.precursor_mz.values, dtype=torch.float32, device=device).unsqueeze(1),
        "ce": torch.tensor(df.collision_energy.values.astype(np.float32),
                           dtype=torch.float32, device=device).unsqueeze(1),
        "rt_target": torch.tensor(df.rt_target.values, dtype=torch.float32, device=device).unsqueeze(1),
        "ccs_target": torch.tensor(df.ccs_target.values, dtype=torch.float32, device=device).unsqueeze(1),
        "ims_target": torch.tensor(df.ims_target.values, dtype=torch.float32, device=device).unsqueeze(1),
        "intensity_target": torch.tensor(np.stack(df.intensity_target.values).astype(np.float32),
                                          dtype=torch.float32, device=device),
    }


# ---------------------------------------------------------------------------
# Joint init + flavor configuration
# ---------------------------------------------------------------------------

def make_joint_init(device: torch.device) -> UnifiedPeptideModel:
    base_path = get_model_path(f"{BASE_ENCODER}/best_model.pt")
    log.info(f"  joint init: encoder + heads from {base_path.parent.parent}")
    model = UnifiedPeptideModel.from_pretrained(
        str(base_path), tasks=list(TASK_NAMES),
    )
    for task in TASK_NAMES:
        if task == BASE_ENCODER: continue
        task_path = get_model_path(f"{task}/best_model.pt")
        ckpt = torch.load(str(task_path), map_location="cpu", weights_only=False)
        head_state = ckpt["heads_state_dict"][task]
        model.heads[task].load_state_dict(head_state)
    return model.to(device)


def configure_optimizer(model, flavor: str, lr: float):
    """Returns (optimizer, extra_params_dict) — extra_params for B4 (uncertainty weighting)."""
    extras = {}
    if flavor == "B1":
        # Heads only, encoder frozen
        for p in model.encoder.parameters():
            p.requires_grad = False
        head_params = [p for h in model.heads.values() for p in h.parameters()]
        opt = torch.optim.AdamW(head_params, lr=lr, weight_decay=1e-4)
    elif flavor == "B2":
        # All trainable, single LR
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif flavor == "B3":
        # Split LR: heads at lr, encoder at lr/100
        head_params = [p for h in model.heads.values() for p in h.parameters()]
        enc_params = list(model.encoder.parameters())
        opt = torch.optim.AdamW(
            [
                {"params": head_params, "lr": lr},
                {"params": enc_params, "lr": lr / 100.0},
            ],
            weight_decay=1e-4,
        )
    elif flavor == "B4":
        # Uncertainty weighting: log_sigma per task, learnable
        log_sigmas = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1, device=next(model.parameters()).device))
            for task in TASK_NAMES
        })
        all_params = list(model.parameters()) + list(log_sigmas.parameters())
        opt = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
        extras["log_sigmas"] = log_sigmas
    elif flavor == "B5":
        # Phase 1 of B5: only intensity head + encoder.
        # The training loop handles the two phases separately.
        head_params = list(model.heads["intensity"].parameters())
        enc_params = list(model.encoder.parameters())
        opt = torch.optim.AdamW(
            [
                {"params": head_params, "lr": lr},
                {"params": enc_params, "lr": lr / 100.0},
            ],
            weight_decay=1e-4,
        )
    else:
        raise ValueError(f"unknown flavor {flavor}")
    return opt, extras


def compute_joint_loss(model, batch, idx, flavor, extras, weights):
    """Forward all three tasks, compute weighted loss. Returns (total_loss,
    per-task losses dict, raw per-task losses dict)."""
    out = model.forward(
        tokens=batch["tokens"][idx],
        mz=batch["mz"][idx],
        charge=batch["charge"][idx],
        collision_energy=batch["ce"][idx],
        tasks=list(TASK_NAMES),
    )
    rt_loss = F.l1_loss(out["rt"], batch["rt_target"][idx])
    ccs_mean, _ = out["ccs"]
    ccs_loss = F.l1_loss(ccs_mean, batch["ccs_target"][idx])
    int_pred = out["intensity"]
    int_target = batch["intensity_target"][idx]
    if int_pred.dim() == 4:
        int_target = int_target.reshape(int_pred.shape)
    int_loss = masked_spectral_distance(int_target, int_pred)

    raw = {"rt": rt_loss.item(), "ccs": ccs_loss.item(), "intensity": int_loss.item()}

    if flavor == "B4":
        # Kendall et al. uncertainty weighting:
        #   L = sum_t [ exp(-2*log_sigma_t) * loss_t + log_sigma_t ]
        # (simplified Laplace-likelihood for L1 / spectral distance).
        ls = extras["log_sigmas"]
        total = (
            torch.exp(-2 * ls["rt"]) * rt_loss + ls["rt"]
            + torch.exp(-2 * ls["ccs"]) * ccs_loss + ls["ccs"]
            + torch.exp(-2 * ls["intensity"]) * int_loss + ls["intensity"]
        ).sum()
    else:
        total = (
            weights["rt"] * rt_loss
            + weights["ccs"] * ccs_loss
            + weights["intensity"] * int_loss
        )

    return total, raw


def train_one_flavor(
    flavor: str, train_t, val_t, hold_t, *, epochs, batch_size, lr, patience,
    weights, device,
):
    """Build joint init, train per flavor, return (model, best_val_loss, n_epochs_run, time_s)."""
    model = make_joint_init(device)
    optimizer, extras = configure_optimizer(model, flavor, lr)

    n_tr = train_t["tokens"].shape[0]
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    t0 = time.time()

    if flavor == "B5":
        # Phase 1: intensity only. Phase 2: full joint.
        log.info(f"  [{flavor}] phase 1 — intensity-only fine-tune")
        for ep in range(1, epochs // 2 + 1):
            model.train()
            perm = torch.randperm(n_tr, device=device)
            for i in range(0, n_tr, batch_size):
                idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                out = model.forward(
                    tokens=train_t["tokens"][idx],
                    charge=train_t["charge"][idx],
                    collision_energy=train_t["ce"][idx],
                    tasks=["intensity"],
                )
                int_target = train_t["intensity_target"][idx]
                int_pred = out["intensity"]
                if int_pred.dim() == 4:
                    int_target = int_target.reshape(int_pred.shape)
                loss = masked_spectral_distance(int_target, int_pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        log.info(f"  [{flavor}] phase 2 — joint RT+CCS+intensity")
        # Reconfigure optimizer to update all heads
        head_params = [p for h in model.heads.values() for p in h.parameters()]
        enc_params = list(model.encoder.parameters())
        optimizer = torch.optim.AdamW(
            [{"params": head_params, "lr": lr},
             {"params": enc_params, "lr": lr / 100.0}],
            weight_decay=1e-4,
        )

    n_val = val_t["tokens"].shape[0]
    n_epochs_run = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        train_loss = 0.0; nb = 0
        train_raw = {"rt": 0.0, "ccs": 0.0, "intensity": 0.0}
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            total, raw = compute_joint_loss(model, train_t, idx, flavor, extras, weights)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += total.item(); nb += 1
            for k, v in raw.items(): train_raw[k] += v
        train_loss /= max(1, nb)
        for k in train_raw: train_raw[k] /= max(1, nb)

        # Val
        model.eval()
        v_loss = 0.0; vb = 0
        v_raw = {"rt": 0.0, "ccs": 0.0, "intensity": 0.0}
        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                sl = slice(i, i + batch_size)
                idx = torch.arange(i, min(i + batch_size, n_val), device=device)
                total, raw = compute_joint_loss(model, val_t, idx, flavor, extras, weights)
                v_loss += total.item(); vb += 1
                for k, v in raw.items(): v_raw[k] += v
        v_loss /= max(1, vb)
        for k in v_raw: v_raw[k] /= max(1, vb)

        n_epochs_run = ep
        if ep <= 3 or ep % 5 == 0:
            log.info(
                f"  [{flavor}] ep {ep:3d}/{epochs}  "
                f"tr[rt={train_raw['rt']:.3f} ccs={train_raw['ccs']:.2f} int={train_raw['intensity']:.3f}]  "
                f"val[rt={v_raw['rt']:.3f} ccs={v_raw['ccs']:.2f} int={v_raw['intensity']:.3f}]"
            )

        if v_loss < best_val - 1e-3:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info(f"  [{flavor}] early stop at ep {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val, n_epochs_run, time.time() - t0


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_all_tasks(model, hold_t, batch_size=256):
    """Per-task held-out metrics. RT MAE in min, CCS MAE in Å²,
    1/K0 MAE via Mason-Schamp, intensity SA mean/median."""
    model.eval()
    n = hold_t["tokens"].shape[0]
    rt_p, ccs_p, int_p = [], [], []
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        out = model.forward(
            tokens=hold_t["tokens"][sl],
            mz=hold_t["mz"][sl],
            charge=hold_t["charge"][sl],
            collision_energy=hold_t["ce"][sl],
            tasks=list(TASK_NAMES),
        )
        rt_p.append(out["rt"].squeeze(-1).cpu().numpy())
        ccs_mean, _ = out["ccs"]
        ccs_p.append(ccs_mean.squeeze(-1).cpu().numpy())
        int_pred = out["intensity"]
        if int_pred.dim() == 4:
            int_pred = int_pred.reshape(int_pred.shape[0], -1)
        int_p.append(int_pred.cpu().numpy())
    rt_pred = np.concatenate(rt_p).astype(np.float64)
    ccs_pred = np.concatenate(ccs_p).astype(np.float64)
    int_pred_arr = np.concatenate(int_p).astype(np.float64)

    rt_obs = hold_t["rt_target"].squeeze(-1).cpu().numpy().astype(np.float64)
    ccs_obs = hold_t["ccs_target"].squeeze(-1).cpu().numpy().astype(np.float64)
    ims_obs = hold_t["ims_target"].squeeze(-1).cpu().numpy().astype(np.float64)
    int_obs = hold_t["intensity_target"].cpu().numpy().astype(np.float64)
    mz_obs = hold_t["mz"].squeeze(-1).cpu().numpy().astype(np.float64)
    z_obs = hold_t["charge"].cpu().numpy().astype(np.int32)

    # CCS → 1/K0 via Mason-Schamp for library-relevant metric
    ims_pred = np.asarray(
        ccs_to_one_over_k0_par(ccs_pred, mz_obs, z_obs), dtype=np.float64
    )

    # Spectral angle (masked)
    sas = []
    for i in range(len(int_pred_arr)):
        m = int_obs[i] > 0
        if m.sum() < 2: continue
        p = int_pred_arr[i][m]; t = int_obs[i][m]
        p_n = p / max(1e-12, np.linalg.norm(p))
        t_n = t / max(1e-12, np.linalg.norm(t))
        cos = float(np.clip(np.dot(p_n, t_n), -1.0, 1.0))
        sas.append(1.0 - 2.0 * np.arccos(cos) / np.pi)

    return {
        "rt_mae_min": float(np.mean(np.abs(rt_pred - rt_obs))),
        "ccs_mae_a2": float(np.mean(np.abs(ccs_pred - ccs_obs))),
        "ims_mae": float(np.mean(np.abs(ims_pred - ims_obs))),
        "intensity_sa_mean": float(np.mean(sas)) if sas else float("nan"),
        "intensity_sa_median": float(np.median(sas)) if sas else float("nan"),
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rescore-dir", type=Path, required=True)
    parser.add_argument("--pseudo-bin", type=Path, required=True)
    parser.add_argument("--d-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--q-cutoff", type=float, default=0.01)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--flavors", type=str, default="B1,B2,B3,B4,B5")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data(args.rescore_dir, args.q_cutoff, args.pseudo_bin, args.d_path)

    train_df, hold_df = split_seq_holdout(df, args.holdout_frac, args.seed)
    # Internal val split for early stopping (within train)
    rng = np.random.default_rng(args.seed + 1)
    seqs_tr = train_df.sequence.unique()
    perm = rng.permutation(len(seqs_tr))
    n_inner_val = int(round(len(seqs_tr) * 0.2))
    inner_val_seqs = set(seqs_tr[perm[:n_inner_val]])
    real_train = train_df[~train_df.sequence.isin(inner_val_seqs)].reset_index(drop=True)
    inner_val = train_df[train_df.sequence.isin(inner_val_seqs)].reset_index(drop=True)
    log.info(f"split: real train {len(real_train):,} / inner val {len(inner_val):,} / hold {len(hold_df):,}")

    tokenizer = ProformaTokenizer.with_defaults()
    train_t = make_tensors(real_train, tokenizer, device)
    val_t = make_tensors(inner_val, tokenizer, device)
    hold_t = make_tensors(hold_df, tokenizer, device)

    weights = {"rt": 1.0, "ccs": 0.1, "intensity": 1.0}
    log.info(f"loss weights (B2/B3/B5): {weights}")

    flavors = args.flavors.split(",")
    summary = []
    for flavor in flavors:
        log.info("")
        log.info("=" * 70)
        log.info(f"FLAVOR {flavor}")
        log.info("=" * 70)
        model, best_val, n_ep, t_s = train_one_flavor(
            flavor, train_t, val_t, hold_t,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            patience=args.patience, weights=weights, device=device,
        )
        metrics = eval_all_tasks(model, hold_t)
        log.info(f"  [{flavor}] {n_ep} ep, {t_s:.1f}s wall")
        log.info(
            f"  [{flavor}] HOLDOUT: RT={metrics['rt_mae_min']:.3f} min  "
            f"CCS={metrics['ccs_mae_a2']:.2f} Å²  1/K0={metrics['ims_mae']:.4f}  "
            f"SA={metrics['intensity_sa_mean']:.4f}"
        )
        summary.append({
            "flavor": flavor, "epochs": n_ep, "wall_s": round(t_s, 1),
            **metrics,
        })

    log.info("")
    log.info("=" * 80)
    log.info("PHASE B SUMMARY (held-out 20% of unique peptides)")
    log.info("=" * 80)
    log.info(f"{'flavor':<5} {'eps':>4} {'time':>7}  {'RT min':>8}  {'CCS Å²':>8}  {'1/K0':>8}  {'SA':>7}")
    log.info("Reference (per-task standalone):  RT 0.43       CCS 6.85    1/K0 0.0139  SA 0.620")
    for s in summary:
        log.info(
            f"{s['flavor']:<5} {s['epochs']:>4} {s['wall_s']:>7.1f}s  "
            f"{s['rt_mae_min']:>8.3f}  {s['ccs_mae_a2']:>8.2f}  {s['ims_mae']:>8.4f}  "
            f"{s['intensity_sa_mean']:>7.4f}"
        )
    log.info("=" * 80)

    (args.out_dir / "phase_b_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
