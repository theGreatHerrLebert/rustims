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
