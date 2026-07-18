"""Build a ground-truth frame from timsim **v2** Parquet answer keys.

The v2 pipeline emits its ground truth as first-class artifacts (``precursors``, ``peptides``,
``modforms``, ``peptide_rt``, ``precursor_ccs``, ``peptide_quantities``) rather than a sim
database. This adapter joins them into exactly the frame shape ``validate/comparison.py``'s
scorers already consume (mirroring ``load_ground_truth``), so the existing metric functions
score a v2 run unchanged — the only v2-specific code in the harness.

Two conversions must match the render bit-for-bit (``timsim-cli/src/bin/render.rs``):

* **RT** — the DIA apex-frame map ``apex = 1 + (rt_index - lo)/span * (n_frames - 1)`` then
  ``rt_seconds = apex * cycle_seconds``. ``lo``/``span`` come from the ``peptide_rt`` metadata
  (``timsim.rt.index_min`` / ``index_max``), i.e. the full peptide space, not the loaded subset.
* **1/K0** — ``ccs_to_one_over_reduced_mobility(ccs, mz, charge, gas=28.013, T=31.85, T_diff=273.15)``.

Ground truth is restricted to precursors that were actually rendered AND fragmented (an MS2
spectrum exists) — a DIA engine can only identify precursors it has fragments for, so that is the
fair denominator for recall. This is an UPPER BOUND on truly-observable precursors: it tests for an
MS2 *row*, not whether the projected peaks landed in the m/z range or survived a ``--limit`` render;
any such overcount surfaces harmlessly as extra 0%-recall bins in the abundance-quantile breakdown.
Backbone (stripped-sequence) matching only; peptidoform/PTM matching is deferred to a later step.
"""
from __future__ import annotations

import os
import pandas as pd
import pyarrow.parquet as pq

# Render constants — keep in sync with timsim-cli/src/bin/render.rs (MASS_GAS/TEMP/T_DIFF) and
# mscore::chemistry::formulas::ccs_to_one_over_reduced_mobility (summary constant).
MASS_GAS = 28.013
TEMP = 31.85
T_DIFF = 273.15
_SUMMARY_CONSTANT = 18509.8632163405


def ccs_to_inverse_mobility(ccs, mz, charge, mass_gas=MASS_GAS, temp=TEMP, t_diff=T_DIFF):
    """CCS (Å²) → 1/K0, the exact Mason-Schamp form the render uses (vectorised over pandas Series)."""
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return ((reduced_mass * (temp + t_diff)) ** 0.5 * ccs) / (_SUMMARY_CONSTANT * charge)


def _rt_index_bounds(peptide_rt_path):
    """``lo, hi`` for the apex map — from stamped metadata, falling back to the column min/max
    (which the render computes over the full peptide_rt table anyway)."""
    md = pq.read_schema(peptide_rt_path).metadata or {}
    lo, hi = md.get(b"timsim.rt.index_min"), md.get(b"timsim.rt.index_max")
    if lo is not None and hi is not None:
        return float(lo), float(hi)
    col = pq.read_table(peptide_rt_path, columns=["rt_index"])["rt_index"].to_pandas()
    return float(col.min()), float(col.max())


def build_truth_from_v2(
    gen_dir,
    n_frames,
    cycle_seconds,
    sample=None,
    precursors="precursors_inrange",
    ion_spectra="ion_spectra_ce",
    require_ms2=True,
):
    """Return a ground-truth ``DataFrame`` in ``load_ground_truth`` shape:
    ``sequence, sequence_modified, charge, rt`` (seconds), ``inverse_mobility, intensity,
    protein_id, sequence_normalized, sequence_modified_normalized``.
    """
    from .parsing import replace_I_with_L

    def path(name):
        return os.path.join(gen_dir, name + ".parquet")

    prec = pq.read_table(path(precursors)).to_pandas()
    pep = pq.read_table(path("peptides"), columns=["peptide_id", "sequence"]).to_pandas()
    rtt = pq.read_table(path("peptide_rt"), columns=["peptide_id", "rt_index"]).to_pandas()
    ccs = pq.read_table(path("precursor_ccs"), columns=["precursor_id", "ccs"]).to_pandas()
    quant = pq.read_table(path("peptide_quantities")).to_pandas()

    # rt_index is nullable; the render skips precursors whose peptide has no RT, so they must not sit in
    # the denominator. Drop them here (the inner-join below then excludes their precursors).
    rtt = rtt[rtt["rt_index"].notna()]
    # Defensive: the render's per-key maps are last-wins, so dedup each join key the same way rather than
    # letting a stray duplicate row multiply the truth via the merge.
    pep = pep.drop_duplicates(subset=["peptide_id"], keep="last")
    rtt = rtt.drop_duplicates(subset=["peptide_id"], keep="last")
    ccs = ccs.drop_duplicates(subset=["precursor_id"], keep="last")

    # Pick the rendered sample (default: first sorted id — matches the render's default).
    if sample is None:
        sample = sorted(quant["sample_id"].unique())[0]
    quant = quant.loc[quant["sample_id"] == sample, ["peptide_id", "amount_amol"]]
    quant = quant.drop_duplicates(subset=["peptide_id"], keep="last")

    lo, hi = _rt_index_bounds(path("peptide_rt"))
    span = max(hi - lo, 1e-9)

    df = (
        prec.merge(pep, on="peptide_id")
        .merge(rtt, on="peptide_id")
        .merge(ccs, on="precursor_id")
        .merge(quant, on="peptide_id", how="left")
    )
    # amount fallback to 1.0 matches the render (amounts.get(pid).unwrap_or(1.0)).
    df["amount_amol"] = df["amount_amol"].fillna(1.0)

    if require_ms2:
        sp = pq.read_table(path(ion_spectra), columns=["precursor_id", "ms_level"]).to_pandas()
        ms2 = set(sp.loc[sp["ms_level"] == 2, "precursor_id"])
        df = df[df["precursor_id"].isin(ms2)]

    # RT (seconds): DIA apex-frame convention (render.rs run_dia), frame → seconds.
    apex_frame = 1.0 + (df["rt_index"] - lo) / span * (n_frames - 1.0)
    df["rt"] = apex_frame * cycle_seconds
    # 1/K0 from CCS, same constants as the render.
    df["inverse_mobility"] = ccs_to_inverse_mobility(df["ccs"], df["mz"], df["charge"])
    # True abundance = the render's factorisation (peptide amount × propensities).
    df["intensity"] = (
        df["amount_amol"] * df["ionization_propensity"] * df["modform_fraction"] * df["charge_fraction"]
    )

    # Backbone sequence is the bare peptide; peptidoform (modforms) matching is a later step.
    df["sequence_modified"] = df["sequence"]
    df["protein_id"] = ""  # protein-inference scoring is a later step
    df["sequence_normalized"] = df["sequence"].apply(replace_I_with_L)
    df["sequence_modified_normalized"] = df["sequence_modified"].apply(replace_I_with_L)

    cols = [
        "sequence", "sequence_modified", "charge", "rt", "inverse_mobility", "intensity",
        "protein_id", "sequence_normalized", "sequence_modified_normalized",
    ]
    return df[cols].reset_index(drop=True)
