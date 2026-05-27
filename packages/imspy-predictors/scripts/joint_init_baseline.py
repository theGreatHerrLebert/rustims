#!/usr/bin/env python
"""Phase A — does merging the 3 pretrained heads into one UnifiedPeptideModel
regress per-task baselines?

For each choice of base encoder (rt / ccs / intensity), load that
checkpoint as the encoder, then swap in the OTHER two heads from their
respective per-task checkpoints. Measure per-task baseline MAE on the
same M3 holdout used by the standalone fine-tune scripts.

Decision rule: if all 3 baselines are within ±10% of the standalone
references, joint training is viable; we proceed to Phase B (fine-tune
flavor ablation). If any regresses by >20%, the encoder-head pairing
matters task-specifically and joint is a regression — per-task wins.

Standalone reference baselines (from M3 anchor set, q≤0.01, peptide-level
or per-(peptide, charge) holdout, seed=0):
    RT       (linear-projected pretrained):  1.535 min MAE  on 5,371 peptides
    CCS Å²   (linear-projected pretrained):  13.24 Å²       on 5,335 (pep, z) pairs
    Intensity (raw pretrained):              SA mean 0.380  on 5,335 PSMs
"""
from __future__ import annotations

import argparse
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

# Re-use the encoder + sanity-check infra from intensity_target_encoder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from intensity_target_encoder import encode_psm_target_vec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("joint_init_baseline")

PROTON = 1.007276466879
TASK_NAMES = ("rt", "ccs", "intensity")


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


def make_merged_model(base: str, device: torch.device) -> UnifiedPeptideModel:
    """Load `base` checkpoint as encoder + base-task head; load the other
    two task heads' weights from their respective checkpoints. Returns
    a UnifiedPeptideModel with all three heads pretrained."""
    base_path = get_model_path(f"{base}/best_model.pt")
    log.info(f"  building joint init: encoder + heads from {base_path}")
    model = UnifiedPeptideModel.from_pretrained(
        str(base_path), tasks=list(TASK_NAMES),
    )
    # Load the other two heads' weights from their per-task checkpoints
    for task in TASK_NAMES:
        if task == base:
            continue
        task_path = get_model_path(f"{task}/best_model.pt")
        ckpt = torch.load(str(task_path), map_location="cpu", weights_only=False)
        head_state = ckpt.get("heads_state_dict", {}).get(task)
        if head_state is None:
            raise RuntimeError(
                f"checkpoint {task_path} has no heads_state_dict[{task}]"
            )
        model.heads[task].load_state_dict(head_state)
        log.info(f"    swapped in pretrained {task} head from {task_path}")
    model = model.to(device)
    model.eval()
    return model


def load_psm_data(rescore_dir: Path, q_cutoff: float, pseudo_bin: Path, d_path: Path):
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
    df = df.assign(observed_ims=one_over_k0, observed_ccs=observed_ccs)
    df = df.rename(columns={"match_idx": "sequence"})
    log.info(f"  high-conf PSMs: {len(df):,}")
    return df


def aggregate_per_pepcharge(df_psm: pd.DataFrame) -> pd.DataFrame:
    """For RT and CCS evaluation — per-(peptide, charge)."""
    return df_psm.groupby(["sequence", "charge"], as_index=False).agg(
        observed_rt=("rt", "mean"),
        observed_ims=("observed_ims", "mean"),
        observed_ccs=("observed_ccs", "mean"),
        calcmass=("calcmass", "mean"),
        collision_energy=("collision_energy", "first"),
        n_psm=("spec_idx", "size"),
    )


def split_seq_holdout(df: pd.DataFrame, holdout_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    seqs = df.sequence.unique()
    perm = rng.permutation(len(seqs))
    n_hold = int(round(len(seqs) * holdout_frac))
    holdout_set = set(seqs[perm[:n_hold]])
    train = df[~df.sequence.isin(holdout_set)].reset_index(drop=True)
    hold = df[df.sequence.isin(holdout_set)].reset_index(drop=True)
    return train, hold


def make_token_batch(seqs: list[str], tokenizer, device, pad_len=50):
    result = tokenizer(seqs, padding=True, return_tensors="pt")
    tokens = result["input_ids"]
    if tokens.shape[1] < pad_len:
        pad = torch.zeros(tokens.shape[0], pad_len - tokens.shape[1], dtype=torch.long)
        tokens = torch.cat([tokens, pad], dim=1)
    elif tokens.shape[1] > pad_len:
        tokens = tokens[:, :pad_len]
    return tokens.to(device)


@torch.no_grad()
def eval_rt(model, train_df, hold_df, tokenizer, device, batch_size=512):
    """RT baseline with linear projection fit on train (predicted → observed)."""
    train_tokens = make_token_batch(train_df.sequence.tolist(), tokenizer, device)
    hold_tokens = make_token_batch(hold_df.sequence.tolist(), tokenizer, device)
    def _predict(tokens):
        out = []
        for i in range(0, tokens.shape[0], batch_size):
            sl = slice(i, i + batch_size)
            out.append(model.predict_rt(tokens[sl]).squeeze(-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)
    train_pred = _predict(train_tokens)
    hold_pred = _predict(hold_tokens)
    train_obs = train_df.observed_rt.values.astype(np.float64)
    hold_obs = hold_df.observed_rt.values.astype(np.float64)
    slope, intercept = np.polyfit(train_pred, train_obs, 1)
    proj = slope * hold_pred + intercept
    return float(np.mean(np.abs(proj - hold_obs))), (float(slope), float(intercept))


@torch.no_grad()
def eval_ccs(model, train_df, hold_df, tokenizer, device, batch_size=512):
    """CCS baseline with linear projection fit on train. Reports CCS Å²."""
    def _build_inputs(df):
        tokens = make_token_batch(df.sequence.tolist(), tokenizer, device)
        charge = torch.tensor(df.charge.values.astype(np.int64),
                              dtype=torch.long, device=device)
        mz = (df.calcmass.values + df.charge.values * PROTON) / df.charge.values
        mz_t = torch.tensor(mz, dtype=torch.float32, device=device).unsqueeze(1)
        return tokens, charge, mz_t
    def _predict(tokens, charge, mz):
        out = []
        for i in range(0, tokens.shape[0], batch_size):
            sl = slice(i, i + batch_size)
            ccs_mean, _ = model.predict_ccs(
                tokens[sl], mz=mz[sl], charge=charge[sl],
            )
            out.append(ccs_mean.squeeze(-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)
    tk_t, ch_t, mz_t = _build_inputs(train_df)
    tk_h, ch_h, mz_h = _build_inputs(hold_df)
    train_pred = _predict(tk_t, ch_t, mz_t)
    hold_pred = _predict(tk_h, ch_h, mz_h)
    train_obs = train_df.observed_ccs.values.astype(np.float64)
    hold_obs = hold_df.observed_ccs.values.astype(np.float64)
    slope, intercept = np.polyfit(train_pred, train_obs, 1)
    proj = slope * hold_pred + intercept
    return float(np.mean(np.abs(proj - hold_obs))), (float(slope), float(intercept))


@torch.no_grad()
def eval_intensity(
    model, hold_psms, frag_df, tokenizer, device, batch_size=256,
):
    """Intensity baseline as spectral angle (raw, no projection — SA is
    scale-invariant). hold_psms is a per-PSM dataframe; frag_df is
    fragment rows used to encode targets per (spec_idx, sequence)."""
    # Encode targets per PSM
    psm_keys = list(zip(hold_psms.spec_idx, hold_psms.sequence))
    psm_groups = {key: g for key, g in frag_df.groupby(["spec_idx", "sequence"])}
    targets = []
    seqs_for_eval = []
    charges = []
    ces = []
    for key in psm_keys:
        if key not in psm_groups:
            continue
        target = encode_psm_target_vec(psm_groups[key])
        targets.append(target)
        seqs_for_eval.append(key[1])
        # Recover charge/CE from the PSM row
        match = hold_psms[(hold_psms.spec_idx == key[0]) & (hold_psms.sequence == key[1])].iloc[0]
        charges.append(int(match.charge))
        ces.append(float(match.collision_energy))
    if not targets:
        return float("nan"), float("nan"), 0
    target_arr = np.stack(targets).astype(np.float64)
    tokens = make_token_batch(seqs_for_eval, tokenizer, device)
    charge_t = torch.tensor(charges, dtype=torch.long, device=device)
    ce_t = torch.tensor(ces, dtype=torch.float32, device=device).unsqueeze(1)

    pred_chunks = []
    for i in range(0, tokens.shape[0], batch_size):
        sl = slice(i, i + batch_size)
        pred = model.predict_intensity(
            tokens[sl], charge=charge_t[sl], collision_energy=ce_t[sl],
        )
        if pred.dim() == 4:
            pred = pred.reshape(pred.shape[0], -1)
        pred_chunks.append(pred.cpu().numpy())
    pred_arr = np.concatenate(pred_chunks, axis=0).astype(np.float64)

    # Spectral angle (Prosit canonical, masked on target>0)
    sas = []
    for i in range(len(target_arr)):
        m = target_arr[i] > 0
        if m.sum() < 2:
            continue
        p = pred_arr[i][m]; t = target_arr[i][m]
        p_norm = p / max(1e-12, np.linalg.norm(p))
        t_norm = t / max(1e-12, np.linalg.norm(t))
        cos = float(np.clip(np.dot(p_norm, t_norm), -1.0, 1.0))
        sas.append(1.0 - 2.0 * np.arccos(cos) / np.pi)
    return float(np.mean(sas)), float(np.median(sas)), len(sas)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rescore-dir", type=Path, required=True)
    parser.add_argument("--pseudo-bin", type=Path, required=True)
    parser.add_argument("--d-path", type=Path, required=True)
    parser.add_argument("--q-cutoff", type=float, default=0.01)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")

    log.info("loading PSM data + side-loaded 1/K0 / CCS")
    df_psm = load_psm_data(args.rescore_dir, args.q_cutoff,
                           args.pseudo_bin, args.d_path)

    # For RT/CCS — per-(peptide, charge), holdout by sequence
    pep_charge = aggregate_per_pepcharge(df_psm)
    train_pc, hold_pc = split_seq_holdout(pep_charge, args.holdout_frac, args.seed)
    log.info(f"  RT/CCS eval: train {len(train_pc):,} (pep, z) / hold {len(hold_pc):,}")

    # For intensity — per-PSM, holdout by sequence
    psm_train, psm_hold = split_seq_holdout(df_psm, args.holdout_frac, args.seed)
    log.info(f"  intensity eval: train {len(psm_train):,} PSMs / hold {len(psm_hold):,}")

    log.info("loading fragments parquet")
    frag_df = pd.read_parquet(args.rescore_dir / "rescored_canonical.fragments.parquet")
    # Filter to fragments belonging to PSMs in our high-conf set (for intensity eval)
    hc_keys = df_psm[["spec_idx", "sequence"]]
    frag_hc = frag_df.merge(hc_keys, on=["spec_idx", "sequence"], how="inner")
    log.info(f"  fragments ∩ high-conf: {len(frag_hc):,}")

    tokenizer = ProformaTokenizer.with_defaults()

    # Standalone reference baselines (from prior runs in the session)
    references = {
        "rt_mae_min":     1.535,   # finetune_dia_pasef.py baseline
        "ccs_mae_a2":    13.240,   # finetune_dia_pasef_ccs.py baseline+proj
        "intensity_sa_mean": 0.380, # finetune_dia_pasef_intensity.py baseline
    }

    log.info("=" * 70)
    log.info("REFERENCE BASELINES (standalone per-task checkpoints):")
    log.info(f"  RT MAE min:       {references['rt_mae_min']:.3f}")
    log.info(f"  CCS MAE Å²:       {references['ccs_mae_a2']:.3f}")
    log.info(f"  Intensity SA mean:{references['intensity_sa_mean']:.4f}")
    log.info("=" * 70)

    results = []
    for base in ("intensity", "rt", "ccs"):
        log.info("")
        log.info("=" * 70)
        log.info(f"BASE = {base}")
        log.info("=" * 70)
        model = make_merged_model(base, device)

        rt_mae, rt_proj = eval_rt(model, train_pc, hold_pc, tokenizer, device)
        log.info(f"  RT  MAE: {rt_mae:.3f} min   "
                 f"(ref {references['rt_mae_min']:.3f}, "
                 f"delta {100*(rt_mae - references['rt_mae_min']) / references['rt_mae_min']:+.1f}%)")

        ccs_mae, ccs_proj = eval_ccs(model, train_pc, hold_pc, tokenizer, device)
        log.info(f"  CCS MAE: {ccs_mae:.3f} Å²   "
                 f"(ref {references['ccs_mae_a2']:.3f}, "
                 f"delta {100*(ccs_mae - references['ccs_mae_a2']) / references['ccs_mae_a2']:+.1f}%)")

        sa_mean, sa_median, n_sa = eval_intensity(
            model, psm_hold, frag_hc, tokenizer, device,
        )
        log.info(f"  Intensity SA mean: {sa_mean:.4f}  median: {sa_median:.4f}  "
                 f"n={n_sa:,}  (ref {references['intensity_sa_mean']:.4f}, "
                 f"delta {100*(sa_mean - references['intensity_sa_mean']) / references['intensity_sa_mean']:+.1f}%)")

        results.append({
            "base": base,
            "rt_mae_min": rt_mae,
            "ccs_mae_a2": ccs_mae,
            "intensity_sa_mean": sa_mean,
        })

    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY (positive delta = WORSE than per-task standalone)")
    log.info("=" * 70)
    log.info(f"{'base':<12} {'RT MAE':>10} {'Δ%':>7} | "
             f"{'CCS MAE':>9} {'Δ%':>7} | {'Int SA':>9} {'Δ%':>7}")
    for r in results:
        rt_d = 100*(r['rt_mae_min'] - references['rt_mae_min'])/references['rt_mae_min']
        ccs_d = 100*(r['ccs_mae_a2'] - references['ccs_mae_a2'])/references['ccs_mae_a2']
        sa_d = 100*(r['intensity_sa_mean'] - references['intensity_sa_mean'])/references['intensity_sa_mean']
        log.info(
            f"{r['base']:<12} {r['rt_mae_min']:>10.3f} {rt_d:+6.1f}% | "
            f"{r['ccs_mae_a2']:>9.3f} {ccs_d:+6.1f}% | "
            f"{r['intensity_sa_mean']:>9.4f} {sa_d:+6.1f}%"
        )
    log.info("=" * 70)
    log.info("decision: if all 3 deltas are within ±10% for at least one base, "
             "joint training is viable; pick that base for Phase B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
