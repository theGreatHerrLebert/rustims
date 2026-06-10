"""
Load a standardized findings table and convert it into TimSim's internal
peptides / proteins / ions DataFrames.

This allows running a simulation from real identifications rather than
starting from a FASTA file.  The user is responsible for converting their
search engine output (DiaNN, FragPipe, Sage, MaxQuant, ...) into the
required format before feeding it to TimSim.

Required input columns (TSV or CSV):
    protein   - Protein identifier (e.g. "P12345")
    sequence  - Modified peptide sequence in UNIMOD notation (e.g. "PEPTC[UNIMOD:4]IDE")
    intensity - Observed intensity (float)

Optional input columns (simulated if missing):
    charge    - Precursor charge state (integer)
    rt        - Retention time in seconds (float)
    im        - Inverse ion mobility 1/K0 (float)
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from imspy_core.chemistry.utility import calculate_mz
from imspy_core.data.peptide import PeptideSequence

logger = logging.getLogger(__name__)

# Regex for sequences safe to pass to PeptideSequence (standard amino acids + UNIMOD mods).
# The Rust core aborts on truly invalid input instead of raising, so we must pre-validate.
_VALID_SEQUENCE_RE = re.compile(r'^([ACDEFGHIKLMNPQRSTVWY](\[UNIMOD:\d+\])?)+$')

REQUIRED_COLUMNS = {"protein", "sequence", "intensity"}
OPTIONAL_COLUMNS = {"charge", "rt", "im"}



@dataclass
class FindingsResult:
    """Result of loading a findings table."""
    peptides: pd.DataFrame
    proteins: pd.DataFrame
    ions: Optional[pd.DataFrame]  # None when charge column was absent
    has_rt: bool
    has_charge: bool
    has_im: bool


def load_findings(
    findings_path: str,
    rt_lower: float,
    rt_upper: float,
    mz_lower: float,
    mz_upper: float,
    im_lower: float,
    im_upper: float,
    upscale_factor: int,
    inverse_mobility_std_mean: float,
    intensity_multiplier: float,
    verbose: bool,
    reference_median: float | None = None,
) -> FindingsResult:
    """Read a standardized findings table and build TimSim DataFrames.

    Args:
        findings_path: Path to TSV/CSV with required + optional columns.
        rt_lower: Minimum RT in seconds from reference frame table.
        rt_upper: Maximum RT in seconds from reference frame table.
        mz_lower: Minimum m/z from reference dataset.
        mz_upper: Maximum m/z from reference dataset.
        im_lower: Minimum inverse mobility from reference dataset.
        im_upper: Maximum inverse mobility from reference dataset.
        upscale_factor: Base scaling anchor for intensity -> event count
            conversion.  Internally this is further multiplied by a
            per-pixel dilution factor so that the median peptide produces
            visible signal after frame/scan spreading.
        inverse_mobility_std_mean: Fixed std value for inverse mobility.
        intensity_multiplier: User-facing knob applied *after* the
            auto-scaling (e.g. 10.0 for 10x brighter, 0.1 for 10x dimmer).
        verbose: Whether to log progress.
        reference_median: Optional fixed intensity median used as the
            event-scaling denominator instead of this sample's own median.
            Default None = per-sample median (legacy behaviour). Pass the
            SAME value to every condition of a multi-sample experiment to
            preserve cross-sample (e.g. A/B) intensity ratios — otherwise the
            per-sample median differs between conditions and silently rescales
            every cross-sample ratio by median(condition_i)/median(condition_j).

    Returns:
        FindingsResult with peptides, proteins, ions (if charge provided),
        and flags indicating which optional columns were present.
    """

    # ------------------------------------------------------------------
    # 1. Read and validate the input table
    # ------------------------------------------------------------------
    if verbose:
        logger.info(f"  Reading findings from: {findings_path}")

    raw = _read_findings(findings_path)

    has_charge = "charge" in raw.columns
    has_rt = "rt" in raw.columns
    has_im = "im" in raw.columns

    if verbose:
        logger.info(f"  Read {len(raw)} rows")
        logger.info(f"  Optional columns present: "
                     f"charge={has_charge}, rt={has_rt}, im={has_im}")

    # IM without charge is meaningless (IM is per-ion)
    if has_im and not has_charge:
        logger.warning("  'im' column ignored because 'charge' is absent "
                        "(ion mobility is charge-state dependent)")
        has_im = False

    # ------------------------------------------------------------------
    # 2. Pre-filter: validate sequences and apply length constraint
    # ------------------------------------------------------------------
    raw = _filter_sequences(raw, verbose)

    if len(raw) == 0:
        raise ValueError(
            "No valid rows remain after sequence validation. "
            "Check that all sequences use valid amino acids and UNIMOD notation."
        )

    # ------------------------------------------------------------------
    # 3. Clip RT to reference gradient range (if provided)
    # ------------------------------------------------------------------
    if has_rt:
        raw["rt"] = raw["rt"].clip(lower=rt_lower, upper=rt_upper)

    # ------------------------------------------------------------------
    # 4. Deduplicate
    # ------------------------------------------------------------------
    if has_charge:
        deduped = _deduplicate_with_charge(raw, has_rt, has_im, verbose)
    else:
        deduped = _deduplicate_peptides_only(raw, has_rt, verbose)

    if len(deduped) == 0:
        raise ValueError("No valid entries remain after deduplication.")

    # ------------------------------------------------------------------
    # 5. Build tables — order depends on whether charge is present
    # ------------------------------------------------------------------
    if has_charge:
        # Build ions FIRST (applies mz/im filtering), then derive
        # peptides only from sequences that have surviving ions.
        ions = _build_ions(deduped, mz_lower, mz_upper, im_lower, im_upper,
                           inverse_mobility_std_mean, has_im, verbose)

        if len(ions) == 0:
            raise ValueError(
                "No ions survive m/z and ion mobility filtering. "
                "Check that your findings match the reference dataset ranges."
            )

        peptides = _build_peptides_from_ions(ions, deduped, upscale_factor,
                                              intensity_multiplier, has_rt, verbose,
                                              reference_median)
    else:
        # No ions to build — peptides directly from deduped rows
        peptides = _build_peptides_no_charge(deduped, upscale_factor,
                                              intensity_multiplier, has_rt, verbose,
                                              reference_median)
        ions = None

    if len(peptides) == 0:
        raise ValueError("No valid peptides could be constructed from findings.")

    # ------------------------------------------------------------------
    # 6. Build proteins DataFrame
    # ------------------------------------------------------------------
    proteins = _build_proteins(peptides)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    if verbose:
        logger.info("")
        logger.info(f"  Proteins:  {len(proteins)}")
        logger.info(f"  Peptides:  {len(peptides)}")
        if ions is not None:
            logger.info(f"  Ions:      {len(ions)}")

    return FindingsResult(
        peptides=peptides,
        proteins=proteins,
        ions=ions,
        has_rt=has_rt,
        has_charge=has_charge,
        has_im=has_im,
    )


# ======================================================================
# Reading & validation
# ======================================================================


def _read_findings(path: str) -> pd.DataFrame:
    """Read a TSV or CSV findings file and validate columns."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")

    # Normalise column names: strip whitespace, lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Findings file is missing required columns: {sorted(missing)}. "
            f"Expected columns: {sorted(REQUIRED_COLUMNS)}"
        )

    # Drop rows with missing values BEFORE type casting so that NaN in
    # optional integer columns (e.g. charge) does not blow up .astype(int).
    check_cols = [c for c in (REQUIRED_COLUMNS | OPTIONAL_COLUMNS) if c in df.columns]
    n_before = len(df)
    df = df.dropna(subset=check_cols).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"  Dropped {n_dropped} rows with missing values")

    # Enforce types for required columns
    df["intensity"] = df["intensity"].astype(float)
    df["sequence"] = df["sequence"].astype(str)
    df["protein"] = df["protein"].astype(str)

    # Enforce types for optional columns (if present)
    if "charge" in df.columns:
        df["charge"] = df["charge"].astype(int)
    if "rt" in df.columns:
        df["rt"] = df["rt"].astype(float)
    if "im" in df.columns:
        df["im"] = df["im"].astype(float)

    return df


def _filter_sequences(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Validate sequences and enforce length <= 30 amino acids.

    Uses a regex pre-check to avoid passing invalid strings to the Rust
    PeptideSequence constructor (which aborts instead of raising).
    """
    valid_mask = []
    for seq in df["sequence"]:
        if not _VALID_SEQUENCE_RE.match(seq):
            valid_mask.append(False)
            continue
        ps = PeptideSequence(seq)
        valid_mask.append(ps.amino_acid_count <= 30)

    n_dropped = sum(not v for v in valid_mask)
    if n_dropped > 0 and verbose:
        logger.warning(f"  Dropped {n_dropped} rows with unparseable or >30 aa sequences")

    return df[valid_mask].reset_index(drop=True)


# ======================================================================
# Deduplication
# ======================================================================


def _deduplicate_with_charge(
    df: pd.DataFrame, has_rt: bool, has_im: bool, verbose: bool
) -> pd.DataFrame:
    """Deduplicate by (sequence, charge) when charge column is present.

    Intensity-weighted mean for RT/IM, sum for intensity.
    """
    total_rows = len(df)
    grouped = df.groupby(["sequence", "charge"], sort=False)

    def _aggregate(g):
        weights = g["intensity"].values
        total = weights.sum()
        result = {"intensity": total, "protein": g["protein"].iloc[0]}

        if has_rt:
            result["rt"] = (np.average(g["rt"].values, weights=weights)
                            if total > 0 else g["rt"].mean())
        if has_im:
            result["im"] = (np.average(g["im"].values, weights=weights)
                            if total > 0 else g["im"].mean())

        return pd.Series(result)

    deduped = grouped.apply(_aggregate, include_groups=False).reset_index()

    n_unique = len(deduped)
    n_dupes = total_rows - n_unique

    if verbose:
        logger.info("")
        logger.info(f"  Deduplication report (by sequence + charge):")
        logger.info(f"    Total rows:           {total_rows}")
        logger.info(f"    Unique precursors:    {n_unique}")
        logger.info(f"    Duplicates merged:    {n_dupes}")

    return deduped


def _deduplicate_peptides_only(
    df: pd.DataFrame, has_rt: bool, verbose: bool
) -> pd.DataFrame:
    """Deduplicate by sequence only (no charge column).

    Intensity-weighted mean for RT, sum for intensity.
    """
    total_rows = len(df)
    grouped = df.groupby("sequence", sort=False)

    def _aggregate(g):
        weights = g["intensity"].values
        total = weights.sum()
        result = {"intensity": total, "protein": g["protein"].iloc[0]}

        if has_rt:
            result["rt"] = (np.average(g["rt"].values, weights=weights)
                            if total > 0 else g["rt"].mean())

        return pd.Series(result)

    deduped = grouped.apply(_aggregate, include_groups=False).reset_index()

    n_unique = len(deduped)
    n_dupes = total_rows - n_unique

    if verbose:
        logger.info("")
        logger.info(f"  Deduplication report (by sequence):")
        logger.info(f"    Total rows:           {total_rows}")
        logger.info(f"    Unique peptides:      {n_unique}")
        logger.info(f"    Duplicates merged:    {n_dupes}")

    return deduped


# ======================================================================
# Building DataFrames
# ======================================================================


def _build_ions(
    deduped: pd.DataFrame,
    mz_lower: float,
    mz_upper: float,
    im_lower: float,
    im_upper: float,
    inverse_mobility_std_mean: float,
    has_im: bool,
    verbose: bool,
) -> pd.DataFrame:
    """Build ions DataFrame from deduplicated findings (charge present).

    Applies m/z and IM range filtering. Sequences are already validated
    by _filter_sequences so PeptideSequence calls here should not fail.
    """
    # Compute m/z and mass for each ion
    mz_values = []
    mono_masses = []

    for _, row in deduped.iterrows():
        ps = PeptideSequence(row["sequence"])
        mass = ps.mono_isotopic_mass
        mz_values.append(calculate_mz(mass, row["charge"]))
        mono_masses.append(mass)

    ions_df = deduped.copy()
    ions_df["monoisotopic-mass"] = mono_masses
    ions_df["mz"] = mz_values

    # Filter by m/z range
    n_before = len(ions_df)
    ions_df = ions_df[
        (ions_df["mz"] >= mz_lower) & (ions_df["mz"] <= mz_upper)
    ].reset_index(drop=True)
    if verbose and len(ions_df) < n_before:
        logger.info(f"  Dropped {n_before - len(ions_df)} ions outside m/z range")

    # Filter by IM range (if IM provided)
    if has_im:
        n_before = len(ions_df)
        ions_df = ions_df.dropna(subset=["im"]).reset_index(drop=True)
        ions_df = ions_df[
            (ions_df["im"] >= im_lower) & (ions_df["im"] <= im_upper)
        ].reset_index(drop=True)
        if verbose and len(ions_df) < n_before:
            logger.info(f"  Dropped {n_before - len(ions_df)} ions outside IM range")

    if verbose:
        logger.info(f"  Ions after filtering: {len(ions_df)}")

    if len(ions_df) == 0:
        return ions_df  # caller checks and raises

    # Compute relative abundance per peptide (charge state distribution)
    peptide_totals = ions_df.groupby("sequence")["intensity"].sum()
    ions_df["relative_abundance"] = ions_df.apply(
        lambda row: (
            row["intensity"] / peptide_totals[row["sequence"]]
            if peptide_totals[row["sequence"]] > 0
            else 1.0
        ),
        axis=1,
    )

    # Build final ions structure
    result = pd.DataFrame({
        "sequence": ions_df["sequence"].values,
        "charge": ions_df["charge"].values,
        "mz": ions_df["mz"].values,
        "relative_abundance": ions_df["relative_abundance"].values,
    })

    if has_im:
        result["inv_mobility_gru_predictor"] = ions_df["im"].astype(np.float32).values
        result["inv_mobility_gru_predictor_std"] = np.float32(inverse_mobility_std_mean)

    return result


def _build_peptides_from_ions(
    ions: pd.DataFrame,
    deduped: pd.DataFrame,
    upscale_factor: int,
    intensity_multiplier: float,
    has_rt: bool,
    verbose: bool,
    reference_median: float | None = None,
) -> pd.DataFrame:
    """Build peptides from the surviving ion set (charge path).

    Only sequences present in ``ions`` get a peptide row, so there are
    no orphan peptides.
    """
    surviving_sequences = ions["sequence"].drop_duplicates().tolist()
    surviving_set = set(surviving_sequences)

    # Build a lookup for RT from deduped (intensity-weighted across charges)
    rt_lookup: dict = {}
    if has_rt:
        for seq, grp in deduped.groupby("sequence", sort=False):
            if seq not in surviving_set:
                continue
            weights = grp["intensity"].values
            total = weights.sum()
            rt_lookup[seq] = (
                np.average(grp["rt"].values, weights=weights)
                if total > 0 else grp["rt"].mean()
            )

    # Protein lookup
    protein_lookup = (
        deduped.drop_duplicates("sequence", keep="first")
        .set_index("sequence")["protein"]
        .to_dict()
    )
    unique_proteins = list(dict.fromkeys(
        protein_lookup[s] for s in surviving_sequences if s in protein_lookup
    ))  # ordered by first appearance in ions
    protein_name_to_id = {name: idx for idx, name in enumerate(unique_proteins)}

    # Per-peptide total intensity from the ions table
    peptide_intensities = ions.groupby("sequence")["relative_abundance"].first()  # placeholder
    # Recompute from deduped to include all charge-state intensities
    ion_intensity_by_seq = deduped[deduped["sequence"].isin(surviving_set)].groupby("sequence")["intensity"].sum()
    median_intensity = reference_median if reference_median else _safe_median(ion_intensity_by_seq)

    if verbose:
        src = "reference (shared)" if reference_median else "per-sample"
        logger.info(f"  Intensity scaling: median={median_intensity:.0f} ({src}) -> "
                     f"upscale_factor={upscale_factor}")

    peptide_rows = []
    peptide_id = 0

    for seq in surviving_sequences:
        total_intensity = ion_intensity_by_seq.get(seq, 0.0)
        protein = protein_lookup.get(seq, "UNKNOWN")
        prot_id = protein_name_to_id.get(protein, 0)

        mass = PeptideSequence(seq).mono_isotopic_mass
        events = max(1, int(total_intensity / median_intensity * upscale_factor * intensity_multiplier)) if total_intensity > 0 else 1

        # Column order must match what the Rust frame builder expects
        # (hardcoded column indices): RT at [9], events at [10].
        row = {
            "protein_id": prot_id,
            "peptide_id": peptide_id,
            "sequence": seq,
            "protein": protein,
            "decoy": 0,
            "missed_cleavages": 0,
            "n_term": None,
            "c_term": None,
            "monoisotopic-mass": mass,
            "retention_time_gru_predictor": rt_lookup.get(seq, 0.0) if has_rt else 0.0,
            "events": events,
        }

        peptide_rows.append(row)
        peptide_id += 1

    peptides = pd.DataFrame(peptide_rows)

    # Map peptide_id back to ions
    seq_to_pid = dict(zip(peptides["sequence"], peptides["peptide_id"]))
    ions["peptide_id"] = ions["sequence"].map(seq_to_pid)

    if verbose:
        logger.info(f"  Peptides: {len(peptides)}, Events range: "
                     f"[{peptides['events'].min()}, {peptides['events'].max()}]")

    return peptides


def _build_peptides_no_charge(
    deduped: pd.DataFrame,
    upscale_factor: int,
    intensity_multiplier: float,
    has_rt: bool,
    verbose: bool,
    reference_median: float | None = None,
) -> pd.DataFrame:
    """Build peptides when charge column is absent (each row = one peptide)."""
    unique_proteins = deduped["protein"].unique()
    protein_name_to_id = {name: idx for idx, name in enumerate(unique_proteins)}

    median_intensity = reference_median if reference_median else _safe_median(deduped["intensity"])

    if verbose:
        src = "reference (shared)" if reference_median else "per-sample"
        logger.info(f"  Intensity scaling: median={median_intensity:.0f} ({src}) -> "
                     f"upscale_factor={upscale_factor}")

    peptide_rows = []
    peptide_id = 0

    for _, row_data in deduped.iterrows():
        total_intensity = row_data["intensity"]
        protein = row_data["protein"]
        prot_id = protein_name_to_id[protein]

        events = max(1, int(total_intensity / median_intensity * upscale_factor * intensity_multiplier)) if total_intensity > 0 else 1
        mass = PeptideSequence(row_data["sequence"]).mono_isotopic_mass

        # Column order must match Rust frame builder expectations:
        # RT at [9], events at [10].
        row = {
            "protein_id": prot_id,
            "peptide_id": peptide_id,
            "sequence": row_data["sequence"],
            "protein": protein,
            "decoy": 0,
            "missed_cleavages": 0,
            "n_term": None,
            "c_term": None,
            "monoisotopic-mass": mass,
            "retention_time_gru_predictor": row_data["rt"] if has_rt else 0.0,
            "events": events,
        }

        peptide_rows.append(row)
        peptide_id += 1

    peptides = pd.DataFrame(peptide_rows)

    if verbose and len(peptides) > 0:
        logger.info(f"  Peptides: {len(peptides)}, Events range: "
                     f"[{peptides['events'].min()}, {peptides['events'].max()}]")

    return peptides


def _build_proteins(peptides: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal proteins DataFrame from peptides."""
    unique_proteins = peptides[["protein_id", "protein"]].drop_duplicates(
        subset=["protein_id"]
    ).reset_index(drop=True)

    return pd.DataFrame({
        "protein_id": unique_proteins["protein_id"],
        "protein": unique_proteins["protein"],
        "sequence": [""] * len(unique_proteins),
        "events": [0] * len(unique_proteins),
    })


# ======================================================================
# Utilities
# ======================================================================


def _safe_median(series: pd.Series) -> float:
    """Compute median, falling back to positive-only or 1.0."""
    med = series.median()
    if med <= 0:
        med = series[series > 0].median()
    if med <= 0 or np.isnan(med):
        med = 1.0
    return med
