"""
DiaNN output parsing utilities for timsim-validate.

Adapted from timsim_bench/benchmark/table_processing.py
"""

import re
import pandas as pd
from typing import Optional


def remove_unimod_annotation(sequence: str) -> str:
    """
    Remove UNIMOD annotations from a peptide sequence.

    Args:
        sequence: A peptide sequence with UNIMOD annotations.

    Returns:
        The peptide sequence without UNIMOD annotations.
    """
    pattern = r'\[UNIMOD:\d+\]'
    return re.sub(pattern, '', sequence)


def format_diann_sequence(modified_sequence: str) -> str:
    """
    Convert a DiaNN modified sequence to standard UNIMOD format.

    DiaNN uses parentheses and "UniMod", we convert to brackets and "UNIMOD".

    Args:
        modified_sequence: The modified peptide sequence from DiaNN.

    Returns:
        The formatted peptide sequence with UNIMOD annotations.
    """
    return modified_sequence.replace("(", "[").replace(")", "]").replace("UniMod", "UNIMOD")


def replace_I_with_L(sequence: str) -> str:
    """
    Replace all occurrences of 'I' with 'L' in the sequence, except inside brackets.

    This normalizes leucine/isoleucine which are indistinguishable by mass spectrometry.

    Args:
        sequence: The input peptide sequence.

    Returns:
        The modified string with 'I' replaced by 'L' outside of modification brackets.
    """
    def replace_match(match):
        if match.group(1):
            return match.group(1)  # Return text inside brackets unchanged
        return match.group(2).replace("I", "L")  # Replace outside brackets

    pattern = r'(\[.*?\])|([^\[\]]+)'
    modified_sequence = re.sub(pattern, replace_match, sequence)
    return modified_sequence.replace("UNLMOD", "UNIMOD")


def parse_diann_report(
    report_path: str,
    diann_version_2: bool = False,
    fdr_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Parse a DiaNN report.tsv file and extract relevant columns.

    Args:
        report_path: Path to the DiaNN report.tsv file.
        diann_version_2: If True, parse as DiaNN 2.0 format, otherwise 1.9.
        fdr_threshold: FDR threshold for filtering results.

    Returns:
        DataFrame with standardized columns:
        - sequence: Plain sequence without modifications
        - sequence_modified: Sequence with UNIMOD annotations
        - charge: Precursor charge state
        - mz: Precursor m/z (if available)
        - rt: Retention time in minutes
        - inverse_mobility: Ion mobility (1/K0)
        - intensity: Precursor quantity
        - q_value: Q-value for FDR filtering
        - protein_id: Protein identifier(s)
    """
    # Read the report file
    if report_path.lower().endswith(".parquet"):
        df = pd.read_parquet(report_path)
    else:
        df = pd.read_csv(report_path, sep='\t')

    # Format sequences
    df["sequence_modified"] = df["Modified.Sequence"].apply(format_diann_sequence)
    df["sequence"] = df["sequence_modified"].apply(remove_unimod_annotation)

    # Define column mapping based on DiaNN version
    column_mapping = {
        "Precursor.Charge": "charge",
        "Precursor.Quantity": "intensity",
        "RT": "rt",
        "IM": "inverse_mobility",
        "Protein.Ids": "protein_id",
        "PEP": "pep",
    }

    # Q-value column name differs between versions
    if diann_version_2:
        column_mapping["Peptidoform.Q.Value"] = "q_value"
    else:
        column_mapping["Q.Value"] = "q_value"

    # Add m/z if available
    if "Precursor.Mz" in df.columns:
        column_mapping["Precursor.Mz"] = "mz"

    # Add mass accuracy columns if available
    if "Ms1.Apex.Mz.Delta" in df.columns:
        column_mapping["Ms1.Apex.Mz.Delta"] = "mz_delta"

    # Rename columns
    df = df.rename(columns=column_mapping)

    # Filter by FDR threshold
    df = df[df["q_value"] <= fdr_threshold]

    # Filter charge states (typically 1-4 for timsTOF)
    df = df[df["charge"] <= 4]

    # Select output columns
    output_columns = [
        "sequence",
        "sequence_modified",
        "charge",
        "rt",
        "inverse_mobility",
        "intensity",
        "q_value",
        "protein_id",
    ]

    if "mz" in df.columns:
        output_columns.insert(3, "mz")

    if "mz_delta" in df.columns:
        output_columns.append("mz_delta")

    # Keep only existing columns
    output_columns = [col for col in output_columns if col in df.columns]

    return df[output_columns].reset_index(drop=True)


def normalize_sequence_for_matching(sequence: str) -> str:
    """
    Normalize a sequence for matching between DiaNN and ground truth.

    Applies I→L normalization and ensures consistent UNIMOD formatting.

    Args:
        sequence: The peptide sequence to normalize.

    Returns:
        Normalized sequence suitable for matching.
    """
    return replace_I_with_L(sequence)


def create_precursor_id(sequence: str, charge: int) -> str:
    """
    Create a unique precursor identifier from sequence and charge.

    Args:
        sequence: The peptide sequence (with or without modifications).
        charge: The precursor charge state.

    Returns:
        A unique precursor identifier string.
    """
    return f"{sequence}_{charge}"


# =============================================================================
# FragPipe parsing utilities
# =============================================================================

def mass_to_unimod(mass: float) -> str:
    """
    Convert a modification mass to UNIMOD annotation.

    Args:
        mass: The modification mass in Daltons.

    Returns:
        UNIMOD annotation string.
    """
    # Common modifications
    mass_map = {
        57: "[UNIMOD:4]",   # Carbamidomethyl (C)
        16: "[UNIMOD:35]",  # Oxidation (M)
        42: "[UNIMOD:1]",   # Acetyl (Protein N-term)
        80: "[UNIMOD:21]",  # Phospho (STY)
    }

    rounded = int(round(mass))
    return mass_map.get(rounded, f"[{mass:.4f}]")


def parse_fragpipe_modification(mod_string: str, sequence: str) -> list:
    """
    Parse FragPipe modification string into position-modification pairs.

    Args:
        mod_string: The modification string from FragPipe (e.g., "2M(15.9949), 6C(57.0214)").
        sequence: The peptide sequence.

    Returns:
        List of (position, modified_aa) tuples.
    """
    import numpy as np

    modifications = []
    if pd.isna(mod_string) or not mod_string:
        return modifications

    try:
        mods_list = mod_string.split(", ")
        for mod in mods_list:
            # Handle N-terminal acetylation
            if mod == 'N-term(42.0106)':
                modifications.append((0, f"[UNIMOD:1]{sequence[0]}"))
            else:
                # Pattern: position + amino acid + mass in parentheses
                # e.g., "2M(15.9949)" or "6C(57.0214)"
                pattern = r"^(\d+)([A-Za-z])\(([\d.]+)\)$"
                match = re.match(pattern, mod)
                if match:
                    index, aa, mass = match.groups()
                    unimod = mass_to_unimod(float(mass))
                    modifications.append((int(index) - 1, aa + unimod))
    except Exception:
        pass  # Handle malformed modification strings

    return modifications


def fragpipe_mods_to_unimod(sequence: str, mods: str) -> str:
    """
    Convert FragPipe modifications to UNIMOD-style sequence.

    Args:
        sequence: The plain peptide sequence.
        mods: The FragPipe modifications string.

    Returns:
        Sequence with UNIMOD annotations.
    """
    r_dict = {index: aa for index, aa in enumerate(sequence)}
    modifications = parse_fragpipe_modification(mods, sequence)
    for index, mod in modifications:
        r_dict[index] = mod
    return "".join(r_dict.values())


def parse_fragpipe_psm(
    psm_path: str,
    fdr_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Parse FragPipe PSM.tsv file and extract relevant columns.

    Args:
        psm_path: Path to the FragPipe psm.tsv file.
        fdr_threshold: Probability threshold for filtering (default: 0.01 = 99% probability).

    Returns:
        DataFrame with standardized columns matching DiaNN output format.
    """
    df = pd.read_csv(psm_path, sep='\t')

    # Convert modifications to UNIMOD format (handle NaN values)
    def safe_mods_to_unimod(row):
        mods = row.get("Assigned Modifications", "")
        if pd.isna(mods):
            mods = ""
        return fragpipe_mods_to_unimod(row["Peptide"], str(mods))

    df["sequence_modified"] = df.apply(safe_mods_to_unimod, axis=1)
    df["sequence"] = df["Peptide"]

    # Rename columns to match standard format
    column_mapping = {
        "Charge": "charge",
        "Retention": "rt",
        "Ion Mobility": "inverse_mobility",
        "Intensity": "intensity",
        "Probability": "probability",
        "Protein ID": "protein_id",
        "Hyperscore": "hyperscore",
    }

    df = df.rename(columns=column_mapping)

    # Convert retention time from seconds to minutes
    if "rt" in df.columns:
        df["rt"] = df["rt"] / 60.0

    # Calculate q-value proxy from probability (1 - probability)
    df["q_value"] = 1.0 - df["probability"]

    # Filter by probability (equivalent to FDR filtering)
    df = df[df["probability"] >= (1.0 - fdr_threshold)]

    # Filter charge states
    df = df[df["charge"] <= 4]

    # Select output columns
    output_columns = [
        "sequence",
        "sequence_modified",
        "charge",
        "rt",
        "inverse_mobility",
        "intensity",
        "q_value",
        "protein_id",
    ]

    # Keep only existing columns
    output_columns = [col for col in output_columns if col in df.columns]

    return df[output_columns].reset_index(drop=True)


def parse_fragpipe_combined(
    psm_path: str,
    peptide_path: str = None,
    protein_path: str = None,
    ion_path: str = None,
    fdr_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Parse FragPipe output files and combine into a unified format.

    This combines PSM, peptide, and protein level information similar to
    the process_fragpipe_psm_table function from timsim_bench.

    Args:
        psm_path: Path to FragPipe psm.tsv file.
        peptide_path: Path to FragPipe peptide.tsv file (optional).
        protein_path: Path to FragPipe protein.tsv file (optional).
        ion_path: Path to FragPipe ion.tsv file for intensities (optional).
        fdr_threshold: FDR threshold for filtering.

    Returns:
        DataFrame with standardized columns and combined quality scores.
    """
    # Parse PSM table
    psm_df = parse_fragpipe_psm(psm_path, fdr_threshold)

    # Add peptide-level quality if available
    if peptide_path and os.path.exists(peptide_path):
        peptide_df = pd.read_csv(peptide_path, sep='\t')
        # Handle both individual (Peptide) and combined (Peptide Sequence) column names
        peptide_col = "Peptide" if "Peptide" in peptide_df.columns else "Peptide Sequence"
        if peptide_col in peptide_df.columns:
            # Handle presence/absence of Probability column
            if "Probability" in peptide_df.columns:
                peptide_quality = dict(zip(peptide_df[peptide_col], peptide_df["Probability"]))
            else:
                # Default to 0.99 if probability column not present
                peptide_quality = dict(zip(peptide_df[peptide_col], [0.99] * len(peptide_df)))
            psm_df["peptide_probability"] = psm_df["sequence"].map(peptide_quality).fillna(0.99)

    # Add protein-level quality if available
    if protein_path and os.path.exists(protein_path):
        protein_df = pd.read_csv(protein_path, sep='\t')
        if "Protein ID" in protein_df.columns:
            # Handle presence/absence of Protein Probability column
            if "Protein Probability" in protein_df.columns:
                protein_quality = dict(zip(protein_df["Protein ID"], protein_df["Protein Probability"]))
            else:
                # Default to 0.99 if probability column not present
                protein_quality = dict(zip(protein_df["Protein ID"], [0.99] * len(protein_df)))
            psm_df["protein_probability"] = psm_df["protein_id"].map(protein_quality).fillna(0.99)

    # Add ion intensities if available
    if ion_path and os.path.exists(ion_path):
        ion_df = pd.read_csv(ion_path, sep='\t')
        # Ion table has more detailed intensity information
        # Merge on sequence and charge
        if "Modified Sequence" in ion_df.columns:
            def safe_format_sequence(s):
                if pd.isna(s):
                    return ""
                return str(s).replace("(", "[").replace(")", "]").replace("UniMod", "UNIMOD")

            # Find intensity column (may be "Intensity" or "{experiment} Intensity")
            intensity_col = None
            for col in ion_df.columns:
                if col == "Intensity" or col.endswith(" Intensity"):
                    intensity_col = col
                    break

            if intensity_col:
                ion_df["sequence_modified"] = ion_df["Modified Sequence"].apply(safe_format_sequence)
                ion_intensity = ion_df.groupby(["sequence_modified", "Charge"])[intensity_col].max().reset_index()
                ion_intensity = ion_intensity.rename(columns={"Charge": "charge", intensity_col: "ion_intensity"})
                psm_df = psm_df.merge(
                    ion_intensity,
                    on=["sequence_modified", "charge"],
                    how="left"
                )
                # Use ion intensity if available and PSM intensity is missing/zero
                if "ion_intensity" in psm_df.columns:
                    psm_df["intensity"] = psm_df["intensity"].fillna(psm_df["ion_intensity"])
                    psm_df = psm_df.drop(columns=["ion_intensity"])

    return psm_df


# Import os for file existence checks
import os
