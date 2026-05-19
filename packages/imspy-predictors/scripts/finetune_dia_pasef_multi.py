#!/usr/bin/env python
"""Phase 1.1 — multi-task RT+CCS fine-tune (DIAGNOSTIC ONLY, not production).

ARCHITECTURAL LIMITATION (discovered 2026-05-02): the off-the-shelf
`imspy_predictors` checkpoints (rt/best_model.pt, ccs/best_model.pt,
intensity/best_model.pt) are each pre-trained STANDALONE — each carries
its own encoder weights paired with its own task head. There is no
unified pre-trained encoder paired with all three heads.

Loading from rt/best_model.pt with tasks=['rt', 'ccs'] gets the
RT-paired encoder + RT head (good) + a FRESH RANDOMLY-INITIALIZED CCS
head (bad). Linear projection then fits the random head's output to
observed CCS, giving a ~30 Å² baseline MAE — much worse than the
standalone CCS run's 13 Å² baseline (same data, but from
ccs/best_model.pt where encoder + CCS head are correctly paired).

For the production calibration-library pipeline, use the 3 per-task
scripts: finetune_dia_pasef.py (RT), finetune_dia_pasef_ccs.py (CCS),
finetune_dia_pasef_intensity.py (intensity). Each loads its own
pre-trained checkpoint and fine-tunes cleanly. Library generation
loads all three checkpoints independently.

This multi-task script remains for future research (e.g., when a
unified pre-trained encoder lands), and to validate the per-(peptide,
charge) data-prep fix that this commit also lands.

Intensity (the third predictor in UnifiedPeptideModel) is deferred to
Phase 1.2 — the rescored CSV only carries scalar ms2_intensity, not the
per-peak vectors the intensity head expects. Fragment-level targets
live in the .pseudo.bin / upstream.

Usage:
    python scripts/finetune_dia_pasef_multi.py \\
        --rescore-dir /path/to/rescore-output \\
        --out-dir ./checkpoints/finetune_o240206_multi
"""

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
from imspy_core.chemistry.mobility import (
    ccs_to_one_over_k0_par, one_over_k0_to_ccs_par,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("finetune_dia_pasef_multi")

# Proton mass for m/z calculation: (M + z*H+) / z.
PROTON = 1.007276466879


def read_pseudo_bin_apex_scans(bin_path: Path) -> np.ndarray:
    """Read env_apex_scan (float32) from a .pseudo.bin produced by
    pipeline_spec_centric_real. Header layout (little-endian):
        char[4] magic = "PSSP" ; u32 version=2 ; u32 n_spec ; u32 total_frag
    Then per-spec arrays in order: env_charge[u32], env_mono_mz[f64],
    env_apex_rt_s[f32], env_apex_scan[f32], env_intensity[u64], ...
    We only read the apex_scan slice and skip the rest.
    """
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        if magic != b"PSSP":
            raise SystemExit(f"bad magic {magic!r} in {bin_path}")
        version, n_spec, _ = struct.unpack("<III", f.read(12))
        if version != 2:
            raise SystemExit(f"pseudo.bin version {version}, expected 2")
        # skip env_charge (4*n) + env_mono_mz (8*n) + env_apex_rt_s (4*n)
        f.seek(4 * n_spec, 1)
        f.seek(8 * n_spec, 1)
        f.seek(4 * n_spec, 1)
        apex_scan = np.frombuffer(f.read(4 * n_spec), dtype="<f4").copy()
    log.info(f"  pseudo.bin: {n_spec:,} spectra, apex_scan range "
             f"{apex_scan.min():.1f}–{apex_scan.max():.1f}")
    return apex_scan


def build_scan_to_im_lut(d_path: Path, sample_frame_id: int = 1) -> np.ndarray:
    """Sample scan→1/K0 from one mid-run frame. TimsTOF mobility is
    near-frame-invariant (drift tube settings don't change across the
    run), so a single-frame LUT is fine for the per-peptide aggregate
    1/K0 we feed the predictor. For sub-percent accuracy, sample
    multiple frames and average — out of scope here.
    """
    from imspy_core.timstof import TimsDataset
    ds = TimsDataset(str(d_path))
    n_scans = ds.num_scans
    scans = np.arange(1, n_scans + 1, dtype=np.int32)
    im = np.asarray(ds.scan_to_inverse_mobility(sample_frame_id, scans),
                    dtype=np.float64)
    log.info(f"  scan→1/K0 LUT: n_scans={n_scans}, "
             f"1/K0 range {im.min():.4f}–{im.max():.4f} "
             f"(sampled at frame_id={sample_frame_id})")
    # Build a flat lookup with index 0 unused (scan 1-indexed)
    lut = np.zeros(n_scans + 2, dtype=np.float64)
    lut[1:n_scans + 1] = im
    return lut


def lookup_im(scan: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Convert apex_scan (float32 fractional) to 1/K0 by linear
    interpolation between integer scan slots in `lut`."""
    s = np.asarray(scan, dtype=np.float64)
    s_lo = np.floor(s).astype(np.int64)
    s_hi = s_lo + 1
    s_lo = np.clip(s_lo, 1, len(lut) - 2)
    s_hi = np.clip(s_hi, 1, len(lut) - 1)
    f = s - s_lo
    return lut[s_lo] * (1.0 - f) + lut[s_hi] * f


def load_and_aggregate(
    rescore_dir: Path, q_cutoff: float, pseudo_bin: Path, d_path: Path
) -> pd.DataFrame:
    """Join pre-TDC features with post-TDC q-values, side-load 1/K0
    from the .pseudo.bin's env_apex_scan + .d's scan→1/K0 LUT, filter,
    aggregate per peptide.

    Returns a DataFrame with columns:
      sequence, observed_rt (min), observed_ims (1/K0), charge, calcmass,
      collision_energy, n_psm.

    Why side-load: the rescore_canonical.py pipeline calls build_query
    without inverse_ion_mobility, so sage stores 0.0 in the `ims` column
    of rescored_canonical.csv. The right long-term fix is upstream
    (rescore computes scan→1/K0 once and passes per-spectrum 1/K0 to
    build_query); this side-load avoids re-running rescore.
    """
    psm_path = rescore_dir / "rescored_canonical.csv"
    tdc_path = rescore_dir / "rescored_canonical.tdc.csv"

    log.info(f"reading {tdc_path}")
    tdc = pd.read_csv(
        tdc_path,
        usecols=["spec_idx", "match_idx", "decoy", "q_value"],
    )
    tdc = tdc[(~tdc.decoy.astype(bool)) & (tdc.q_value <= q_cutoff)]
    log.info(f"  tdc rows q≤{q_cutoff} & target: {len(tdc):,}")

    log.info(f"reading {psm_path} (rt/charge/mass/CE + keys only)")
    psm = pd.read_csv(
        psm_path,
        usecols=["spec_idx", "match_idx", "rt", "charge",
                 "calcmass", "collision_energy"],
    )
    log.info(f"  psm rows: {len(psm):,}")

    df = tdc.merge(psm, on=["spec_idx", "match_idx"], how="inner")
    log.info(f"  joined rows: {len(df):,}")
    if len(df) == 0:
        raise SystemExit("inner join produced no rows")

    # Side-load observed 1/K0. spec_idx like "pseudo_NNN" → bin index NNN.
    log.info(f"side-loading 1/K0 from {pseudo_bin}")
    apex_scan = read_pseudo_bin_apex_scans(pseudo_bin)
    log.info(f"building scan→1/K0 LUT from {d_path}")
    lut = build_scan_to_im_lut(d_path)

    bin_idx = df.spec_idx.str.removeprefix("pseudo_").astype(np.int64).values
    if bin_idx.max() >= len(apex_scan):
        raise SystemExit(
            f"spec_idx {bin_idx.max()} out of bin range "
            f"({len(apex_scan)}) — pseudo.bin / rescore mismatch")
    one_over_k0 = lookup_im(apex_scan[bin_idx], lut).astype(np.float64)
    # m/z = (calcmass + z*proton)/z; needed for the Mason-Schamp
    # 1/K0 ↔ CCS conversion. The pretrained CCS head outputs CCS in Å²
    # (with a physics-informed sqrt(m/z * z) projection layer); supervising
    # in 1/K0 directly fights that prior and the head can't shift its
    # output scale fast enough during fine-tune. Supervise in CCS Å²
    # instead; convert predicted CCS → 1/K0 at inference for library use.
    charge_arr = df.charge.values.astype(np.float64)
    mz_arr = (df.calcmass.values + charge_arr * PROTON) / charge_arr
    observed_ccs = one_over_k0_to_ccs_par(
        one_over_k0, mz_arr, charge_arr.astype(np.int32)
    )
    observed_ccs = np.asarray(observed_ccs, dtype=np.float64)
    df = df.assign(
        observed_ims=one_over_k0,
        observed_ccs=observed_ccs,
    )
    log.info(
        f"  joined+1/K0+CCS: rows={len(df):,}  "
        f"1/K0 {df.observed_ims.min():.4f}–{df.observed_ims.max():.4f}  "
        f"CCS {df.observed_ccs.min():.1f}–{df.observed_ccs.max():.1f} Å²"
    )

    # Aggregate per (peptide, charge). CCS depends on charge — averaging
    # observed_ccs across charge states for the same peptide mixes two
    # distinct targets and gives meaningless supervision. The CCS-only
    # diagnostic (finetune_dia_pasef_ccs.py) caught this: switching to
    # per-(peptide, charge) aggregation drops baseline+proj MAE from
    # 31 Å² to 13 Å², and fine-tuned MAE from 30 → 6.85 Å² (−48%).
    df = df.rename(columns={"match_idx": "sequence"})

    def _mode_or_first(s: pd.Series):
        v = s.mode()
        return v.iloc[0] if len(v) else s.iloc[0]

    agg = df.groupby(["sequence", "charge"], as_index=False).agg(
        observed_rt=("rt", "mean"),
        observed_ims=("observed_ims", "mean"),
        observed_ccs=("observed_ccs", "mean"),
        calcmass=("calcmass", "mean"),
        collision_energy=("collision_energy", _mode_or_first),
        n_psm=("spec_idx", "size"),
    )

    log.info(
        f"  unique peptides: {len(agg):,}  "
        f"(rt {agg.observed_rt.min():.2f}-{agg.observed_rt.max():.2f} min, "
        f"1/K0 {agg.observed_ims.min():.4f}-{agg.observed_ims.max():.4f}, "
        f"charge mode counts: "
        f"{dict(agg.charge.astype(int).value_counts().head(5))})"
    )
    return agg


def split_peptide_holdout(
    agg: pd.DataFrame, holdout_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Holdout split by UNIQUE peptide sequence (not row index). With
    per-(peptide, charge) aggregation a peptide may appear at multiple
    charges; splitting by row could leak the same sequence into both
    train and test, defeating the held-out-peptide eval."""
    rng = np.random.default_rng(seed)
    seqs = agg.sequence.unique()
    perm = rng.permutation(len(seqs))
    n_hold = int(round(len(seqs) * holdout_frac))
    holdout_set = set(seqs[perm[:n_hold]])
    train = agg[~agg.sequence.isin(holdout_set)].reset_index(drop=True)
    hold = agg[agg.sequence.isin(holdout_set)].reset_index(drop=True)
    return train, hold


def make_tensors(
    df: pd.DataFrame, tokenizer, device: torch.device, pad_len: int = 50
) -> dict:
    """Tokenise + tensorise a dataframe slice. Returns dict ready for forward().

    Mirrors DeepChromatographyApex._preprocess_sequences: pad to pad_len,
    pass tokens (zeros = padding) without an explicit padding_mask — the
    transformer encoder treats the zero token as padding by convention
    in this codebase (and forward() accepts padding_mask=None).
    """
    result = tokenizer(df.sequence.tolist(), padding=True, return_tensors="pt")
    tokens = result["input_ids"]
    if tokens.shape[1] < pad_len:
        pad = torch.zeros(tokens.shape[0], pad_len - tokens.shape[1], dtype=torch.long)
        tokens = torch.cat([tokens, pad], dim=1)
    elif tokens.shape[1] > pad_len:
        tokens = tokens[:, :pad_len]
    tokens = tokens.to(device)

    charge = torch.tensor(df.charge.values, dtype=torch.long, device=device)
    # m/z = (calcmass + z * proton) / z. calcmass is monoisotopic mass.
    mz = (df.calcmass.values + df.charge.values * PROTON) / df.charge.values
    mz_t = torch.tensor(mz, dtype=torch.float32, device=device).unsqueeze(1)
    ce = torch.tensor(df.collision_energy.values,
                      dtype=torch.float32, device=device).unsqueeze(1)

    rt = torch.tensor(df.observed_rt.values, dtype=torch.float32, device=device).unsqueeze(1)
    ims = torch.tensor(df.observed_ims.values, dtype=torch.float32, device=device).unsqueeze(1)
    ccs = torch.tensor(df.observed_ccs.values, dtype=torch.float32, device=device).unsqueeze(1)

    return {
        "tokens": tokens, "padding_mask": None,
        "charge": charge, "mz": mz_t, "ce": ce,
        "rt": rt, "ims": ims, "ccs": ccs,
    }


def fit_linear_projection(pred: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(pred, obs, deg=1)
    return float(slope), float(intercept)


@torch.no_grad()
def predict_rt_ccs(
    model: UnifiedPeptideModel, batch: dict, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Predict (rt, ccs_mean) on a tensor batch in chunks. Returns 1-D arrays."""
    n = batch["tokens"].shape[0]
    rt_out = []
    ccs_out = []
    model.eval()
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        out = model.forward(
            tokens=batch["tokens"][sl],
            padding_mask=(None if batch["padding_mask"] is None
                              else batch["padding_mask"][sl]),
            mz=batch["mz"][sl],
            charge=batch["charge"][sl],
            tasks=["rt", "ccs"],
        )
        rt_out.append(out["rt"].squeeze(-1).cpu().numpy())
        # CCS head returns (mean, std).
        ccs_mean, _ = out["ccs"]
        ccs_out.append(ccs_mean.squeeze(-1).cpu().numpy())
    return (
        np.concatenate(rt_out, axis=0).astype(np.float64),
        np.concatenate(ccs_out, axis=0).astype(np.float64),
    )


def report_mae(
    pred: np.ndarray, obs: np.ndarray, label: str, unit: str
) -> dict:
    abs_err = np.abs(pred - obs)
    metrics = {
        "n": int(len(obs)),
        "mae": float(np.mean(abs_err)),
        "median_ae": float(np.median(abs_err)),
        "p90_ae": float(np.percentile(abs_err, 90)),
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
        "unit": unit,
    }
    log.info(
        f"  [{label}] n={metrics['n']:,}  "
        f"MAE={metrics['mae']:.4f} {unit}  "
        f"median={metrics['median_ae']:.4f}  "
        f"p90={metrics['p90_ae']:.4f}  "
        f"RMSE={metrics['rmse']:.4f}"
    )
    return metrics


def custom_finetune_loop(
    model: UnifiedPeptideModel,
    train_batch: dict,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    val_frac: float,
    seed: int,
    rt_weight: float,
    ccs_weight: float,
) -> None:
    """Multi-task L1 fine-tune. Loss = w_rt * L1(rt_pred, rt_obs)
    + w_ccs * L1(ccs_mean, ccs_obs). Internal val split for early stop.
    """
    import torch.nn.functional as F

    n = train_batch["tokens"].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx = torch.tensor(perm[:n_val], device=train_batch["tokens"].device)
    train_idx = torch.tensor(perm[n_val:], device=train_batch["tokens"].device)

    def _slice(b: dict, idx: torch.Tensor) -> dict:
        return {k: (None if v is None else v[idx]) for k, v in b.items()}

    tr = _slice(train_batch, train_idx)
    va = _slice(train_batch, val_idx)
    n_tr = tr["tokens"].shape[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, min_lr=1e-6
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    log.info(
        f"multi-task fine-tune: epochs={epochs} batch={batch_size} "
        f"lr={lr:.2e} weights rt={rt_weight} ccs={ccs_weight}"
    )

    for ep in range(1, epochs + 1):
        model.train()
        epoch_perm = torch.randperm(n_tr, device=tr["tokens"].device)
        t_loss = t_rt = t_ccs = 0.0
        nb = 0
        for i in range(0, n_tr, batch_size):
            idx = epoch_perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model.forward(
                tokens=tr["tokens"][idx],
                padding_mask=(None if tr["padding_mask"] is None
                              else tr["padding_mask"][idx]),
                mz=tr["mz"][idx],
                charge=tr["charge"][idx],
                tasks=["rt", "ccs"],
            )
            rt_loss = F.l1_loss(out["rt"], tr["rt"][idx])
            ccs_mean, _ = out["ccs"]
            # Supervise CCS in Å² (the head's native output scale);
            # convert predictions back to 1/K0 at inference time for
            # library generation downstream.
            ccs_loss = F.l1_loss(ccs_mean, tr["ccs"][idx])
            loss = rt_weight * rt_loss + ccs_weight * ccs_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item(); t_rt += rt_loss.item(); t_ccs += ccs_loss.item()
            nb += 1
        t_loss /= nb; t_rt /= nb; t_ccs /= nb

        # Val
        model.eval()
        with torch.no_grad():
            v_rt = v_ccs = 0.0
            vb = 0
            for i in range(0, va["tokens"].shape[0], batch_size):
                sl = slice(i, i + batch_size)
                out = model.forward(
                    tokens=va["tokens"][sl],
                    padding_mask=(None if va["padding_mask"] is None
                                  else va["padding_mask"][sl]),
                    mz=va["mz"][sl],
                    charge=va["charge"][sl],
                    tasks=["rt", "ccs"],
                )
                v_rt += F.l1_loss(out["rt"], va["rt"][sl]).item()
                ccs_mean, _ = out["ccs"]
                v_ccs += F.l1_loss(ccs_mean, va["ccs"][sl]).item()
                vb += 1
            v_rt /= vb; v_ccs /= vb
            v_loss = rt_weight * v_rt + ccs_weight * v_ccs

        scheduler.step(v_loss)
        log.info(
            f"  ep {ep:3d}/{epochs}  "
            f"tr[rt={t_rt:.4f} ccs={t_ccs:.4f}]  "
            f"val[rt={v_rt:.4f} ccs={v_ccs:.4f}]  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if v_loss < best_val - 1e-4:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info(f"  early stop at epoch {ep} (best val combined L1 {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rescore-dir", type=Path, required=True)
    parser.add_argument("--pseudo-bin", type=Path, required=True,
                        help="path to <stem>.pseudo.bin (for env_apex_scan)")
    parser.add_argument("--d-path", type=Path, required=True,
                        help="path to .d folder (for scan→1/K0 calibration)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--q-cutoff", type=float, default=0.01)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--rt-weight", type=float, default=1.0)
    parser.add_argument("--ccs-weight", type=float, default=10.0,
                        help="ims values are ~0.7-1.5 (1/K0); RT is ~1-20 min. "
                             "Boosting CCS weight prevents the tiny-magnitude "
                             "loss from being ignored by the optimizer.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    agg = load_and_aggregate(
        args.rescore_dir,
        q_cutoff=args.q_cutoff,
        pseudo_bin=args.pseudo_bin,
        d_path=args.d_path,
    )
    train_df, hold_df = split_peptide_holdout(agg, args.holdout_frac, args.seed)
    log.info(f"  split: train {len(train_df):,} / hold {len(hold_df):,}")
    if args.dry_run:
        return 0

    # Model
    log.info("loading pretrained UnifiedPeptideModel with rt+ccs heads")
    from imspy_predictors.utility import get_model_path
    rt_path = get_model_path("rt/best_model.pt")
    model = UnifiedPeptideModel.from_pretrained(str(rt_path), tasks=["rt", "ccs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log.info(f"  device: {device}  params: {sum(p.numel() for p in model.parameters()):,}")
    pretrained_state = copy.deepcopy(model.state_dict())

    # Tokenise (one-shot for the full train + hold sets — fits in mem at this scale).
    tokenizer = ProformaTokenizer.with_defaults()
    train_t = make_tensors(train_df, tokenizer, device)
    hold_t = make_tensors(hold_df, tokenizer, device)

    # Baseline — fit linear projections of pretrained predictions onto observed
    # so MAE measures prediction quality, not scale offset.
    log.info("baseline (pretrained, linear-projected) eval")
    rt_pred_pre, ccs_pred_pre = predict_rt_ccs(model, hold_t)
    rt_obs_hold = hold_df.observed_rt.values.astype(np.float64)
    ccs_obs_hold = hold_df.observed_ccs.values.astype(np.float64)
    ims_obs_hold = hold_df.observed_ims.values.astype(np.float64)
    mz_hold = (hold_df.calcmass.values + hold_df.charge.values * PROTON) / hold_df.charge.values
    z_hold = hold_df.charge.values.astype(np.int32)

    # Fit projections on the TRAINING set (avoid leakage from holdout).
    rt_pred_train, ccs_pred_train = predict_rt_ccs(model, train_t)
    rt_proj = fit_linear_projection(
        rt_pred_train, train_df.observed_rt.values.astype(np.float64)
    )
    ccs_proj = fit_linear_projection(
        ccs_pred_train, train_df.observed_ccs.values.astype(np.float64)
    )
    log.info(
        f"  rt projection:  obs ≈ {rt_proj[0]:.4f} * pred + {rt_proj[1]:.4f}"
    )
    log.info(
        f"  ccs projection: obs ≈ {ccs_proj[0]:.4f} * pred + {ccs_proj[1]:.4f} (Å²)"
    )
    rt_baseline = report_mae(
        rt_proj[0] * rt_pred_pre + rt_proj[1], rt_obs_hold,
        "baseline_rt_proj", "min"
    )
    ccs_baseline_a2 = report_mae(
        ccs_proj[0] * ccs_pred_pre + ccs_proj[1], ccs_obs_hold,
        "baseline_ccs_proj", "Å²"
    )
    # Library-relevant baseline: convert projected CCS prediction → 1/K0 and
    # MAE vs observed 1/K0.
    ccs_baseline_pred_a2 = ccs_proj[0] * ccs_pred_pre + ccs_proj[1]
    ims_baseline_pred = np.asarray(
        ccs_to_one_over_k0_par(ccs_baseline_pred_a2, mz_hold, z_hold),
        dtype=np.float64,
    )
    ims_baseline = report_mae(
        ims_baseline_pred, ims_obs_hold,
        "baseline_1/K0_via_ccs_proj", "1/K0"
    )

    # Multi-task fine-tune
    custom_finetune_loop(
        model, train_t,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
        val_frac=0.2, seed=args.seed,
        rt_weight=args.rt_weight, ccs_weight=args.ccs_weight,
    )

    # Post-finetune eval. RT is in observed minutes; CCS is in Å² (the
    # head's native scale, what we supervised). For library use we also
    # convert predicted CCS → 1/K0 and report that MAE.
    log.info("fine-tuned eval (no projection)")
    rt_pred_post, ccs_pred_post = predict_rt_ccs(model, hold_t)
    rt_finetuned = report_mae(rt_pred_post, rt_obs_hold, "finetuned_rt", "min")
    ccs_finetuned_a2 = report_mae(
        ccs_pred_post, ccs_obs_hold, "finetuned_ccs", "Å²"
    )
    ims_finetuned_pred = np.asarray(
        ccs_to_one_over_k0_par(ccs_pred_post, mz_hold, z_hold),
        dtype=np.float64,
    )
    ims_finetuned = report_mae(
        ims_finetuned_pred, ims_obs_hold,
        "finetuned_1/K0_via_ccs", "1/K0"
    )

    # Save
    finetuned_state = copy.deepcopy(model.state_dict())
    torch.save(
        {
            "model_state_dict": finetuned_state,
            "pretrained_state_dict": pretrained_state,
            "metrics": {
                "rt_baseline_proj": rt_baseline, "rt_finetuned": rt_finetuned,
                "ccs_baseline_proj": ccs_baseline_a2,
                "ccs_finetuned": ccs_finetuned_a2,
                "ims_baseline_via_ccs_proj": ims_baseline,
                "ims_finetuned_via_ccs": ims_finetuned,
            },
            "rt_projection": {"slope": rt_proj[0], "intercept": rt_proj[1]},
            "ccs_projection": {"slope": ccs_proj[0], "intercept": ccs_proj[1]},
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
        },
        args.out_dir / "rt_ccs_multi_finetuned.pt",
    )
    def _delta(b, f):
        return {"baseline": b, "finetuned": f,
                "delta": b["mae"] - f["mae"],
                "delta_pct": 100.0 * (b["mae"] - f["mae"]) / b["mae"]
                if b["mae"] > 0 else 0.0}

    summary = {
        "rt": _delta(rt_baseline, rt_finetuned),
        "ccs_a2": _delta(ccs_baseline_a2, ccs_finetuned_a2),
        "ims_via_ccs": _delta(ims_baseline, ims_finetuned),
        "n_train_peptides": int(len(train_df)),
        "n_hold_peptides": int(len(hold_df)),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    log.info("=" * 60)
    log.info(f"RT      baseline MAE={rt_baseline['mae']:.3f} min   "
             f"finetuned MAE={rt_finetuned['mae']:.3f} min   "
             f"delta={summary['rt']['delta']:+.3f} ({summary['rt']['delta_pct']:+.2f}%)")
    log.info(f"CCS Å²  baseline MAE={ccs_baseline_a2['mae']:.2f}  "
             f"finetuned MAE={ccs_finetuned_a2['mae']:.2f}  "
             f"delta={summary['ccs_a2']['delta']:+.2f} ({summary['ccs_a2']['delta_pct']:+.2f}%)")
    log.info(f"1/K0    baseline MAE={ims_baseline['mae']:.4f}  "
             f"finetuned MAE={ims_finetuned['mae']:.4f}  "
             f"delta={summary['ims_via_ccs']['delta']:+.4f} ({summary['ims_via_ccs']['delta_pct']:+.2f}%)")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
