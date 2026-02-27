"""
Load search engine findings (DiaNN, FragPipe, Sage) and convert them into
TimSim's internal peptides / proteins / ions DataFrames.

This allows running a simulation from real identifications rather than
starting from a FASTA file. The original .d dataset is used as the
reference for acquisition layout and instrument settings.
"""

import logging

import numpy as np
import pandas as pd

from imspy_core.chemistry.utility import calculate_mz
from imspy_core.data.peptide import PeptideSequence

from imspy_simulation.timsim.validate.parsing import (
    parse_diann_report,
    parse_fragpipe_psm,
    parse_sage_results,
)

logger = logging.getLogger(__name__)


def load_findings(
    findings_path: str,
    findings_format: str,
    fdr_threshold: float,
    rt_lower: float,
    rt_upper: float,
    mz_lower: float,
    mz_upper: float,
    im_lower: float,
    im_upper: float,
    upscale_factor: int,
    inverse_mobility_std_mean: float,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse search engine results and build TimSim peptides/proteins/ions.

    Args:
        findings_path: Path to the search engine output file.
        findings_format: One of "diann", "diann2", "fragpipe", "sage".
        fdr_threshold: FDR threshold for filtering.
        rt_lower: Minimum RT in seconds from reference frame table.
        rt_upper: Maximum RT in seconds from reference frame table.
        mz_lower: Minimum m/z from reference dataset.
        mz_upper: Maximum m/z from reference dataset.
        im_lower: Minimum inverse mobility from reference dataset.
        im_upper: Maximum inverse mobility from reference dataset.
        upscale_factor: Maximum event count for log-scaling intensities.
        inverse_mobility_std_mean: Fixed std value for inverse mobility.
        verbose: Whether to log progress.

    Returns:
        Tuple of (peptides, proteins, ions) DataFrames in TimSim format.
    """

    # ------------------------------------------------------------------
    # 1. Parse findings with the appropriate parser
    # ------------------------------------------------------------------
    if verbose:
        logger.info(f"  Parsing {findings_format} findings from: {findings_path}")

    parsed = _parse_findings(findings_path, findings_format, fdr_threshold)

    if verbose:
        logger.info(f"  Parsed {len(parsed)} PSMs passing FDR <= {fdr_threshold}")

    # ------------------------------------------------------------------
    # 2. Convert RT from minutes to seconds
    # ------------------------------------------------------------------
    parsed["rt_seconds"] = parsed["rt"] * 60.0

    # ------------------------------------------------------------------
    # 3. Clip RT to reference gradient range
    # ------------------------------------------------------------------
    parsed["rt_seconds"] = parsed["rt_seconds"].clip(lower=rt_lower, upper=rt_upper)

    # ------------------------------------------------------------------
    # 4. Deduplicate by (sequence_modified, charge) — keep best q_value
    # ------------------------------------------------------------------
    parsed, ions_df = _deduplicate_findings(parsed, verbose)

    # ------------------------------------------------------------------
    # 5. Build ions DataFrame
    # ------------------------------------------------------------------
    ions = _build_ions(ions_df, mz_lower, mz_upper, im_lower, im_upper,
                       inverse_mobility_std_mean, verbose)

    # ------------------------------------------------------------------
    # 6. Build peptides DataFrame
    # ------------------------------------------------------------------
    peptides = _build_peptides(ions, upscale_factor, verbose)

    # ------------------------------------------------------------------
    # 7. Set relative_abundance on ions
    # ------------------------------------------------------------------
    ions = _set_relative_abundance(ions, peptides)

    # ------------------------------------------------------------------
    # 8. Build proteins DataFrame
    # ------------------------------------------------------------------
    proteins = _build_proteins(peptides)

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    if verbose:
        logger.info("")
        logger.info(f"  Proteins:  {len(proteins)}")
        logger.info(f"  Peptides:  {len(peptides)}")
        logger.info(f"  Ions:      {len(ions)}")

    return peptides, proteins, ions


# ======================================================================
# Internal helpers
# ======================================================================


def _parse_findings(
    path: str, fmt: str, fdr_threshold: float
) -> pd.DataFrame:
    """Dispatch to the correct parser."""
    fmt = fmt.lower().strip()
    if fmt == "diann":
        return parse_diann_report(path, fdr_threshold=fdr_threshold)
    elif fmt == "diann2":
        return parse_diann_report(path, diann_version_2=True, fdr_threshold=fdr_threshold)
    elif fmt == "fragpipe":
        return parse_fragpipe_psm(path, fdr_threshold=fdr_threshold)
    elif fmt == "sage":
        return parse_sage_results(path, fdr_threshold=fdr_threshold)
    else:
        raise ValueError(
            f"Unknown findings format: '{fmt}'. "
            f"Supported: diann, diann2, fragpipe, sage"
        )


def _deduplicate_findings(
    parsed: pd.DataFrame, verbose: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate by (sequence_modified, charge).

    For each unique precursor:
      - RT and IM come from the best-scoring entry (lowest q_value).
      - Intensity is summed across all observations.

    Returns the original (filtered) DataFrame and a deduplicated ions DataFrame.
    """
    total_psms = len(parsed)

    # Sort by q_value ascending so the first entry per group is the best
    parsed = parsed.sort_values("q_value", ascending=True)

    # Aggregate: best-scoring RT/IM, summed intensity
    grouped = parsed.groupby(["sequence_modified", "charge"], sort=False)

    best = grouped.first().reset_index()
    summed_intensity = grouped["intensity"].sum().reset_index()
    summed_intensity = summed_intensity.rename(columns={"intensity": "total_intensity"})

    ions_df = best.merge(summed_intensity, on=["sequence_modified", "charge"])

    n_unique = len(ions_df)
    n_dupes = total_psms - n_unique

    if verbose:
        logger.info("")
        logger.info(f"  Deduplication report:")
        logger.info(f"    Total PSMs:           {total_psms}")
        logger.info(f"    Unique precursors:    {n_unique}")
        logger.info(f"    Duplicates removed:   {n_dupes}")
        if n_unique > 0:
            counts = grouped.size()
            logger.info(f"    Mean PSMs/precursor:  {counts.mean():.1f}")
            logger.info(f"    Max  PSMs/precursor:  {counts.max()}")

    return parsed, ions_df


def _build_ions(
    ions_df: pd.DataFrame,
    mz_lower: float,
    mz_upper: float,
    im_lower: float,
    im_upper: float,
    inverse_mobility_std_mean: float,
    verbose: bool,
) -> pd.DataFrame:
    """Build ions DataFrame from deduplicated findings."""
    # Calculate mz from sequence and charge
    mz_values = []
    mono_masses = []
    valid_mask = []

    for _, row in ions_df.iterrows():
        try:
            ps = PeptideSequence(row["sequence_modified"])
            mass = ps.mono_isotopic_mass
            mz = calculate_mz(mass, row["charge"])
            mono_masses.append(mass)
            mz_values.append(mz)
            valid_mask.append(True)
        except Exception:
            mono_masses.append(np.nan)
            mz_values.append(np.nan)
            valid_mask.append(False)

    ions_df = ions_df.copy()
    ions_df["monoisotopic-mass"] = mono_masses
    ions_df["mz"] = mz_values

    n_invalid = sum(not v for v in valid_mask)
    if n_invalid > 0:
        if verbose:
            logger.warning(f"  Dropped {n_invalid} ions with unparseable sequences")
        ions_df = ions_df[valid_mask].reset_index(drop=True)

    # Drop NaN inverse mobility
    n_before = len(ions_df)
    ions_df = ions_df.dropna(subset=["inverse_mobility"]).reset_index(drop=True)
    n_dropped_im = n_before - len(ions_df)
    if n_dropped_im > 0 and verbose:
        logger.warning(f"  Dropped {n_dropped_im} ions with NaN inverse mobility")

    # Filter by peptide length (Prosit model supports max 30 amino acids)
    aa_counts = ions_df["sequence_modified"].apply(
        lambda s: PeptideSequence(s).amino_acid_count
    )
    n_too_long = (aa_counts > 30).sum()
    if n_too_long > 0 and verbose:
        logger.warning(f"  Dropped {n_too_long} ions with sequence length > 30")
    ions_df = ions_df[aa_counts <= 30].reset_index(drop=True)

    # Filter by mz and IM range
    ions_df = ions_df[
        (ions_df["mz"] >= mz_lower) & (ions_df["mz"] <= mz_upper) &
        (ions_df["inverse_mobility"] >= im_lower) & (ions_df["inverse_mobility"] <= im_upper)
    ].reset_index(drop=True)

    if verbose:
        logger.info(f"  Ions after m/z and IM filtering: {len(ions_df)}")

    # Build the final ions structure
    ions = pd.DataFrame({
        "sequence_modified": ions_df["sequence_modified"],
        "sequence": ions_df["sequence_modified"],  # TimSim uses 'sequence' for modified sequences
        "charge": ions_df["charge"],
        "mz": ions_df["mz"],
        "monoisotopic-mass": ions_df["monoisotopic-mass"],
        "observed_intensity": ions_df["total_intensity"],
        "inv_mobility_gru_predictor": ions_df["inverse_mobility"].astype(np.float32),
        "inv_mobility_gru_predictor_std": np.float32(inverse_mobility_std_mean),
        "retention_time_gru_predictor": ions_df["rt_seconds"],
        "protein_id": ions_df["protein_id"],
    })

    return ions


def _build_peptides(
    ions: pd.DataFrame,
    upscale_factor: int,
    verbose: bool,
) -> pd.DataFrame:
    """Build peptides DataFrame from ions.

    Each unique sequence_modified becomes one peptide. Events are derived
    from proportional scaling of total observed intensity across all charge
    states, preserving the original dynamic range. The median intensity is
    mapped to ``upscale_factor``, so brighter peptides exceed it and dimmer
    ones fall below — matching the multi-order-of-magnitude range that the
    normal FASTA-based pipeline produces.
    """
    grouped = ions.groupby("sequence_modified", sort=False)

    # Build protein_id mapping from unique protein names
    unique_proteins = ions["protein_id"].unique()
    protein_name_to_id = {name: idx for idx, name in enumerate(unique_proteins)}

    # Collect per-peptide total intensities first for scaling
    peptide_intensities = grouped["observed_intensity"].sum()
    median_intensity = peptide_intensities.median()
    if median_intensity <= 0:
        median_intensity = peptide_intensities[peptide_intensities > 0].median()
    if median_intensity <= 0:
        median_intensity = 1.0

    if verbose:
        logger.info(f"  Intensity scaling: median={median_intensity:.0f} -> "
                     f"upscale_factor={upscale_factor}")

    peptide_rows = []
    peptide_id = 0

    for seq_mod, group in grouped:
        # Monoisotopic mass — take from the first ion (same sequence)
        mass = group["monoisotopic-mass"].iloc[0]

        # RT from the best ion (ions are already sorted by q_value via dedup)
        rt = group["retention_time_gru_predictor"].iloc[0]

        # Total intensity across all charge states
        total_intensity = group["observed_intensity"].sum()

        # Protein from first entry
        protein = group["protein_id"].iloc[0]
        prot_id = protein_name_to_id[protein]

        # Proportional scaling: median intensity -> upscale_factor
        # This preserves the full dynamic range of observed intensities
        if total_intensity > 0:
            events = max(1, int(total_intensity / median_intensity * upscale_factor))
        else:
            events = 1

        peptide_rows.append({
            "protein_id": prot_id,
            "peptide_id": peptide_id,
            "sequence": seq_mod,  # TimSim uses 'sequence' for modified sequences
            "protein": protein,
            "decoy": 0,
            "missed_cleavages": 0,
            "n_term": None,
            "c_term": None,
            "monoisotopic-mass": mass,
            "retention_time_gru_predictor": rt,
            "events": events,
        })
        peptide_id += 1

    peptides = pd.DataFrame(peptide_rows)

    # Map peptide_id back to ions
    seq_to_pid = dict(zip(peptides["sequence"], peptides["peptide_id"]))
    ions["peptide_id"] = ions["sequence_modified"].map(seq_to_pid)

    if verbose:
        logger.info(f"  Peptides: {len(peptides)}, Events range: "
                     f"[{peptides['events'].min()}, {peptides['events'].max()}]")

    return peptides


def _set_relative_abundance(
    ions: pd.DataFrame, peptides: pd.DataFrame
) -> pd.DataFrame:
    """Set relative_abundance on ions: fraction of peptide intensity per charge state."""
    # Total intensity per peptide
    peptide_totals = ions.groupby("peptide_id")["observed_intensity"].sum()
    ions = ions.copy()
    ions["relative_abundance"] = ions.apply(
        lambda row: (
            row["observed_intensity"] / peptide_totals[row["peptide_id"]]
            if peptide_totals[row["peptide_id"]] > 0
            else 1.0
        ),
        axis=1,
    )

    # Drop helper columns not needed by downstream jobs
    ions = ions.drop(columns=["sequence_modified", "observed_intensity",
                               "retention_time_gru_predictor", "protein_id",
                               "monoisotopic-mass"],
                      errors="ignore")

    return ions.reset_index(drop=True)


def _build_proteins(peptides: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal proteins DataFrame from peptides."""
    unique_proteins = peptides[["protein_id", "protein"]].drop_duplicates(
        subset=["protein_id"]
    ).reset_index(drop=True)

    proteins = pd.DataFrame({
        "protein_id": unique_proteins["protein_id"],
        "protein": unique_proteins["protein"],
        "sequence": [""] * len(unique_proteins),
        "events": [0] * len(unique_proteins),
    })
    return proteins
