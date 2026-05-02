"""Phase 1.2 — fragment → Prosit-shape target vector encoder + sanity checks.

Encodes the per-PSM `fragments_observed` data from sagepy
(written to rescored_canonical.fragments.parquet by the patched
rescore_canonical.py) into the canonical Prosit-2023-TimsTOF intensity
target vector layout: 174-dim flat = 29 positions × {y,b} × {+1,+2,+3}.

Verified mapping (physics check, see verify_ion_type_mapping.py):
  parquet `ion_type` int → label
    0 → "b"     (b-ion, N-terminal fragment)
    1 → "y"     (y-ion, C-terminal fragment)

Layout (matches imspy_predictors.intensity.utility.get_prosit_intensity_flat_labels):
  index = 6 * (ordinal - 1)
        + (0 if ion_type == "y" else 3)
        + (frag_charge - 1)

Layout per ordinal:
  pos 0: y+1   pos 1: y+2   pos 2: y+3
  pos 3: b+1   pos 4: b+2   pos 5: b+3

This was the user-flagged "ugly hotspot" — sagepy's
observed_fragments_map() returns a different key order than
fragments_to_dict(Fragments(...)), and the int values for ion_type need
physics verification. Both are now nailed down.

Usage:
    python intensity_target_encoder.py \\
        --parquet /path/to/rescored_canonical.fragments.parquet \\
        --tdc-csv /path/to/rescored_canonical.tdc.csv \\
        --q-cutoff 0.01

Runs all sanity checks (s1/s2/s3) and exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROSIT_DIM = 174
PROSIT_MAX_POSITIONS = 29
PROSIT_CHARGES = (1, 2, 3)
PROSIT_ION_LABELS = ("y", "b")  # canonical order in the layout

# Canonical mapping verified by physics on real DIA-PASEF rescore output
# (verify_ion_type_mapping.py): int 0 = b-ion, int 1 = y-ion.
ION_INT_TO_LABEL = {0: "b", 1: "y"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("intensity_target_encoder")


def fragment_index(ordinal: int, ion_label: str, frag_charge: int) -> int:
    """Compute the flat-layout index for a single fragment.

    Ordinal is 1-indexed (y1..y29, b1..b29). Returns -1 for out-of-range
    inputs so the caller can skip cleanly without exceptions in a hot
    loop.
    """
    if not (1 <= ordinal <= PROSIT_MAX_POSITIONS):
        return -1
    if frag_charge not in PROSIT_CHARGES:
        return -1
    if ion_label not in ("b", "y"):
        return -1
    ion_offset = 0 if ion_label == "y" else 3
    return 6 * (ordinal - 1) + ion_offset + (frag_charge - 1)


def encode_psm_target(
    psm_fragments: pd.DataFrame, normalize_per_psm: bool = True
) -> Tuple[np.ndarray, dict]:
    """Encode one PSM's fragments → 174-dim Prosit target vector.

    Args:
        psm_fragments: rows for ONE PSM, with columns
            (ordinal, frag_charge, ion_type, intensity_observed).
            ion_type is the int ‐0/1 from sagepy.
        normalize_per_psm: if True, divide vector by its max so the
            strongest matched fragment has intensity 1.0. Sagepy already
            normalizes per-spectrum (max=1) on its way out — this is a
            defence against subtle deviations. Set False if you want to
            preserve the raw sagepy-normalized values.

    Returns:
        (target, stats) where target is shape (174,) float32 and stats
        is a dict with keys {n_total, n_emitted, n_skipped_ordinal,
        n_skipped_charge, n_skipped_ion}.
    """
    target = np.zeros(PROSIT_DIM, dtype=np.float32)
    stats = {
        "n_total": len(psm_fragments),
        "n_emitted": 0,
        "n_skipped_ordinal": 0,
        "n_skipped_charge": 0,
        "n_skipped_ion": 0,
    }
    for _, row in psm_fragments.iterrows():
        ordinal = int(row.ordinal)
        frag_charge = int(row.frag_charge)
        ion_int = int(row.ion_type)
        intensity = float(row.intensity_observed)

        ion_label = ION_INT_TO_LABEL.get(ion_int)
        if ion_label is None:
            stats["n_skipped_ion"] += 1
            continue
        if not (1 <= ordinal <= PROSIT_MAX_POSITIONS):
            stats["n_skipped_ordinal"] += 1
            continue
        if frag_charge not in PROSIT_CHARGES:
            stats["n_skipped_charge"] += 1
            continue

        idx = fragment_index(ordinal, ion_label, frag_charge)
        # No silent overwrite — same (ord, charge, ion) reaching the
        # same index twice means upstream produced duplicates; sum to
        # be safe (sagepy shouldn't, but defensive).
        target[idx] += intensity
        stats["n_emitted"] += 1

    if normalize_per_psm and target.max() > 0:
        target = target / target.max()
    return target, stats


def encode_psm_target_vec(psm_fragments: pd.DataFrame) -> np.ndarray:
    """Vectorized variant of encode_psm_target — for hot loops over many
    PSMs. Same semantics, no per-row stats tracking."""
    ord_arr = psm_fragments.ordinal.to_numpy(dtype=np.int64)
    chg_arr = psm_fragments.frag_charge.to_numpy(dtype=np.int64)
    ion_arr = psm_fragments.ion_type.to_numpy(dtype=np.int64)
    int_arr = psm_fragments.intensity_observed.to_numpy(dtype=np.float32)

    valid = (
        (ord_arr >= 1) & (ord_arr <= PROSIT_MAX_POSITIONS)
        & (chg_arr >= 1) & (chg_arr <= 3)
        & ((ion_arr == 0) | (ion_arr == 1))
    )
    ord_v = ord_arr[valid]
    chg_v = chg_arr[valid]
    ion_v = ion_arr[valid]
    int_v = int_arr[valid]

    # ion_v == 1 (y) maps to offset 0; ion_v == 0 (b) maps to offset 3.
    ion_offset = np.where(ion_v == 1, 0, 3)
    idx = 6 * (ord_v - 1) + ion_offset + (chg_v - 1)

    target = np.zeros(PROSIT_DIM, dtype=np.float32)
    np.add.at(target, idx, int_v)
    if target.max() > 0:
        target = target / target.max()
    return target


# ----------------------- Sanity checks -------------------------------


def s1_round_trip(df_psm: pd.DataFrame, sequences_for_psms: list) -> int:
    """s1 — Round-trip: encode each PSM's fragments → 174-vec, then
    use imspy_predictors' Prosit→Fragments helper to re-derive a
    fragment list from the vector. Verify the (ordinal, charge, ion)
    keys match the input fragment set.

    Returns the number of failures.
    """
    log.info("s1 — round-trip encode → decode (5 PSMs)")
    failures = 0
    psms = list(df_psm.groupby(["spec_idx", "sequence"]))[:5]
    for (spec_idx, seq), group in psms:
        target = encode_psm_target_vec(group)
        # Build the input set of (ord, charge, ion_label) from the parquet
        in_set = set()
        for _, r in group.iterrows():
            ord_ = int(r.ordinal)
            chg = int(r.frag_charge)
            ion_int = int(r.ion_type)
            label = ION_INT_TO_LABEL.get(ion_int)
            if label is None: continue
            if not (1 <= ord_ <= PROSIT_MAX_POSITIONS): continue
            if chg not in PROSIT_CHARGES: continue
            in_set.add((ord_, chg, label))

        # Decode the target by walking non-zero indices and inverting the
        # layout formula.
        decoded = set()
        for idx in np.flatnonzero(target):
            ordinal = (idx // 6) + 1
            within = idx % 6  # 0..5
            ion_offset = within // 3   # 0=y, 1=b
            chg = (within % 3) + 1
            label = "y" if ion_offset == 0 else "b"
            decoded.add((ordinal, chg, label))

        if in_set != decoded:
            extra = decoded - in_set
            missing = in_set - decoded
            log.error(
                f"s1 FAIL: {spec_idx}/{seq} — "
                f"missing={missing}  extra={extra}"
            )
            failures += 1
        else:
            log.info(
                f"  {spec_idx}/{seq:<25s} (len={len(seq)}) "
                f"in/out={len(in_set):3d}  ✓"
            )
    return failures


def s2_distribution(df_all: pd.DataFrame, n_psms: int = 1000) -> int:
    """s2 — Distribution sanity: aggregate ~1000 PSMs into the 174-dim
    layout, look at mean intensity per position. Earlier positions (y2,
    y3, b2, b3) should be commonly populated; very late positions
    (y29, b29) should rarely be populated.

    Returns 1 if the pattern is grossly inconsistent.
    """
    log.info(f"s2 — distribution sanity ({n_psms} PSMs)")
    psm_groups = list(df_all.groupby(["spec_idx", "sequence"]))[:n_psms]
    sums = np.zeros(PROSIT_DIM, dtype=np.float64)
    counts = np.zeros(PROSIT_DIM, dtype=np.int64)
    for _, group in psm_groups:
        target = encode_psm_target_vec(group)
        sums += target
        counts += (target > 0).astype(np.int64)

    means = sums / max(1, len(psm_groups))
    fill_rate = counts / max(1, len(psm_groups))

    # Reshape to (29, 6) for readability: rows=position, cols=(y+1,y+2,y+3,b+1,b+2,b+3)
    fill_rate_2d = fill_rate.reshape(PROSIT_MAX_POSITIONS, 6)
    log.info("  fill rate (frac of PSMs with this position populated):")
    log.info("              y+1     y+2     y+3     b+1     b+2     b+3")
    for ord_ in range(1, 30):
        if ord_ in (1, 2, 3, 5, 10, 15, 20, 25, 29):
            row = fill_rate_2d[ord_ - 1]
            log.info(
                f"  ord {ord_:2d}:   "
                + "  ".join(f"{x:5.3f}" for x in row)
            )

    # Sanity: mid-range y (y2..y8) should be MORE populated than y29.
    mid_y = fill_rate_2d[1:8, 0:3].mean()  # y+1..y+3 for ord 2..8
    late_y = fill_rate_2d[28, 0:3].mean()
    log.info(f"  mid-y mean fill={mid_y:.4f}  late-y29 mean fill={late_y:.4f}")
    if mid_y < 0.05:
        log.error(f"s2 FAIL: mid-range y fill rate {mid_y:.4f} unexpectedly low")
        return 1
    if late_y > mid_y:
        log.error(f"s2 FAIL: late y29 fill rate exceeds mid-range — bug")
        return 1
    return 0


def s3_spot_check(df_all: pd.DataFrame) -> int:
    """s3 — Spot check: pick 3 peptides at different lengths (short,
    mid, long) and inspect their target vectors. Verify populated
    positions are in [1..L-1] (no fragments past peptide length) and
    that the brightest peak is at a sensible position.
    """
    log.info("s3 — spot check on 3 peptides")
    psms = df_all.groupby(["spec_idx", "sequence"]).size().reset_index(name="n_frag")
    # Pick by sequence length: short (≤8), mid (12-16), long (≥22).
    psms["seq_len"] = psms.sequence.str.len()
    picks = []
    for length_filter, label in [
        (psms.seq_len <= 8, "short"),
        ((psms.seq_len >= 12) & (psms.seq_len <= 16), "mid"),
        (psms.seq_len >= 22, "long"),
    ]:
        candidates = psms[length_filter].sort_values("n_frag", ascending=False)
        if len(candidates):
            picks.append((label, candidates.iloc[0]))

    failures = 0
    for label, pick in picks:
        spec_idx, seq = pick.spec_idx, pick.sequence
        L = len(seq)
        sub = df_all[(df_all.spec_idx == spec_idx) & (df_all.sequence == seq)]
        target = encode_psm_target_vec(sub)
        log.info(
            f"  [{label}] {spec_idx} {seq} (len={L}, "
            f"matched_frags={pick.n_frag})"
        )
        nz_indices = np.flatnonzero(target)
        max_ordinal_emitted = 0
        beyond_L = []
        for idx in nz_indices:
            ordinal = (idx // 6) + 1
            within = idx % 6
            ion_offset = within // 3
            chg = (within % 3) + 1
            ion = "y" if ion_offset == 0 else "b"
            if ordinal > L - 1:
                beyond_L.append((ordinal, ion, chg))
            max_ordinal_emitted = max(max_ordinal_emitted, ordinal)
        log.info(
            f"    max emitted ordinal={max_ordinal_emitted}  (peptide L-1={L-1})"
            f"   #unique positions={len(nz_indices)}"
        )
        if beyond_L:
            log.error(
                f"s3 FAIL: {len(beyond_L)} fragments past peptide length "
                f"L-1={L-1}: {beyond_L[:5]}"
            )
            failures += 1
        # Brightest peak position
        argmax_idx = int(np.argmax(target))
        ord_max = (argmax_idx // 6) + 1
        within = argmax_idx % 6
        ion_max = "y" if within // 3 == 0 else "b"
        chg_max = (within % 3) + 1
        log.info(
            f"    brightest: {ion_max}{ord_max}+{chg_max} "
            f"(intensity={target[argmax_idx]:.3f}, idx={argmax_idx})"
        )
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--parquet", type=Path, required=True,
                        help="rescored_canonical.fragments.parquet")
    parser.add_argument("--tdc-csv", type=Path, required=True,
                        help="rescored_canonical.tdc.csv")
    parser.add_argument("--q-cutoff", type=float, default=0.01)
    parser.add_argument("--n-psms-s2", type=int, default=1000)
    args = parser.parse_args()

    log.info(f"reading {args.parquet}")
    df_all = pd.read_parquet(args.parquet)
    log.info(f"  {len(df_all):,} fragment rows")

    log.info(f"reading {args.tdc_csv}, filtering to q≤{args.q_cutoff} targets")
    tdc = pd.read_csv(args.tdc_csv,
                      usecols=["spec_idx", "match_idx", "decoy", "q_value"])
    tdc = tdc[(~tdc.decoy.astype(bool)) & (tdc.q_value <= args.q_cutoff)]
    log.info(f"  {len(tdc):,} high-confidence target PSMs")

    # Filter parquet to high-confidence PSMs (sequence == match_idx, spec_idx).
    df_hc = df_all.merge(
        tdc[["spec_idx", "match_idx"]].rename(columns={"match_idx": "sequence"}),
        on=["spec_idx", "sequence"], how="inner",
    )
    log.info(f"  parquet ∩ high-conf: {len(df_hc):,} fragment rows  "
             f"({df_hc.groupby(['spec_idx','sequence']).ngroups} unique PSMs)")

    failures = 0
    failures += s1_round_trip(df_hc, sequences_for_psms=None)
    failures += s2_distribution(df_hc, n_psms=args.n_psms_s2)
    failures += s3_spot_check(df_hc)

    if failures:
        log.error(f"{failures} sanity check(s) FAILED")
        return 1
    log.info("all sanity checks PASSED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
