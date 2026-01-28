"""Input filtering for Koina models based on model-specific requirements."""

import logging
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Eligible models and their restrictions:
# Intensity: Prosit_2023_intensity_timsTOF, AlphaPeptDeep_ms2_generic, ms2pip_timsTOF2023, ms2pip_timsTOF2024
# RT: AlphaPeptDeep_rt_generic, Deeplc_hela_hf, Chronologer_RT, Prosit_2019_irt
# CCS: AlphaPeptDeep_ccs_generic, IM2Deep
# Flyability: pfly_2024_fine_tuned

# Model filter specifications
# None = no restrictions, list = list of filter dictionaries
MODEL_FILTERS: Dict[str, Optional[List[Dict[str, Any]]]] = {
    "prosit": [
        {"length": [0, 30]},
        {"modifications": ["C[UNIMOD:4]", "M[UNIMOD:35]"]},
        {"precursor_charges": [1, 2, 3, 4, 5, 6]},
    ],
    "alphapeptdeep": None,  # Supports all modifications
    "ms2pip": [{"length": [0, 30]}],
    "deeplc": [{"length": [0, 60]}],
    "chronologer": None,
    "im2deep": [{"length": [0, 60]}],
    "pfly": None,
}

# Human-readable model descriptions for error messages
MODEL_DESCRIPTIONS = {
    "prosit": "Prosit (max 30 AA, only C[UNIMOD:4] and M[UNIMOD:35] mods, charges 1-6)",
    "alphapeptdeep": "AlphaPeptDeep (supports all modifications)",
    "ms2pip": "MS2PIP (max 30 AA)",
    "deeplc": "DeepLC (max 60 AA)",
    "chronologer": "Chronologer (no restrictions)",
    "im2deep": "IM2Deep (max 60 AA)",
    "pfly": "pFly (no restrictions)",
}


def get_model_type(model_name: str) -> str:
    """Extract model type from full model name."""
    return model_name.split("_")[0].lower()


def get_supported_models() -> List[str]:
    """Get list of supported model types."""
    return list(MODEL_FILTERS.keys())


def get_model_restrictions(model_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Get filter restrictions for a model.

    Args:
        model_name: Full model name (e.g., "Prosit_2023_intensity_timsTOF")

    Returns:
        List of filter dictionaries or None if no restrictions
    """
    model_type = get_model_type(model_name)
    if model_type not in MODEL_FILTERS:
        return None
    return MODEL_FILTERS[model_type]


def validate_model_compatibility(
    model_name: str,
    peptide_sequences: List[str],
    charges: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Validate peptides against model requirements without filtering.

    Args:
        model_name: Full model name
        peptide_sequences: List of peptide sequences
        charges: Optional list of charge states

    Returns:
        Dictionary with validation results:
        - 'compatible': bool - whether all peptides are compatible
        - 'incompatible_count': int - number of incompatible peptides
        - 'reasons': list - reasons for incompatibility
    """
    model_type = get_model_type(model_name)
    filters = MODEL_FILTERS.get(model_type)

    if filters is None:
        return {"compatible": True, "incompatible_count": 0, "reasons": []}

    reasons = []
    incompatible_count = 0

    for f in filters:
        if "length" in f:
            min_len, max_len = f["length"]
            for seq in peptide_sequences:
                # Remove modification annotations to get actual length
                clean_seq = "".join(c for c in seq if c.isalpha())
                if len(clean_seq) > max_len:
                    incompatible_count += 1
                    if f"Length > {max_len}" not in reasons:
                        reasons.append(f"Length > {max_len}")

        if "modifications" in f:
            allowed_mods = f["modifications"]
            import re
            for seq in peptide_sequences:
                mods = re.findall(r".\[[^\]]*\]", seq)
                for mod in mods:
                    if mod not in allowed_mods:
                        incompatible_count += 1
                        if f"Unsupported modification: {mod}" not in reasons:
                            reasons.append(f"Unsupported modification: {mod}")

        if "precursor_charges" in f and charges:
            allowed_charges = f["precursor_charges"]
            for charge in charges:
                if charge not in allowed_charges:
                    incompatible_count += 1
                    if f"Charge {charge} not in {allowed_charges}" not in reasons:
                        reasons.append(f"Charge {charge} not in {allowed_charges}")

    return {
        "compatible": incompatible_count == 0,
        "incompatible_count": incompatible_count,
        "reasons": reasons,
    }


def filter_input_by_model(model_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter input DataFrame based on the model requirements.

    Args:
        model_name: Name of the model.
        df: Input DataFrame with peptide sequences.

    Returns:
        Filtered DataFrame.

    Raises:
        ValueError: If model is not supported.
    """
    model_type = get_model_type(model_name)

    if model_type not in MODEL_FILTERS:
        supported = ", ".join(get_supported_models())
        raise ValueError(
            f"Model type '{model_type}' (from '{model_name}') not supported. "
            f"Supported model types: {supported}"
        )

    filters = MODEL_FILTERS[model_type]
    if filters is None:
        logger.debug(f"No filtering required for model {model_name}")
        return df

    original_count = len(df)

    for filter_dict in filters:
        if "length" in filter_dict:
            min_len, max_len = filter_dict["length"]
            df = filter_peptide_length(df, min_len, max_len)
        if "modifications" in filter_dict:
            allowed_mods = filter_dict["modifications"]
            df = filter_peptide_modifications(df, allowed_mods)
        if "precursor_charges" in filter_dict:
            allowed_charges = filter_dict["precursor_charges"]
            df = filter_precursor_charges(df, allowed_charges)
        if "instrument_types" in filter_dict:
            allowed_instruments = filter_dict["instrument_types"]
            df = filter_instrument_types(df, allowed_instruments)

    filtered_count = len(df)
    if filtered_count < original_count:
        logger.info(
            f"Model {model_name}: Filtered {original_count - filtered_count}/{original_count} "
            f"peptides based on model requirements"
        )

    return df


def filter_peptide_length(
    df: pd.DataFrame, min_len: int = 5, max_len: int = 30
) -> pd.DataFrame:
    """
    Filter peptides by length.

    Args:
        df: DataFrame with peptide sequences.
        min_len: Minimum length of peptide sequence.
        max_len: Maximum length of peptide sequence.

    Returns:
        DataFrame with filtered peptide sequences.
    """
    if "peptide_sequences" not in df.columns:
        raise ValueError("DataFrame must contain 'peptide_sequences' column.")

    if max_len is None and min_len is None:
        return df

    if max_len is None:
        max_len = np.inf
    if min_len is None:
        min_len = 0

    # Calculate actual peptide length (excluding modification annotations)
    df = df.copy()
    df["_peptide_length"] = (
        df["peptide_sequences"].str.replace(r"\[.*?\]", "", regex=True).str.len()
    )

    filtered_df = df[
        (df["_peptide_length"] >= min_len) & (df["_peptide_length"] <= max_len)
    ]

    removed = len(df) - len(filtered_df)
    if removed > 0:
        logger.debug(
            f"Removed {removed} peptides based on length ({min_len}-{max_len} AA)"
        )

    return filtered_df.drop(columns=["_peptide_length"])


def filter_peptide_modifications(
    df: pd.DataFrame, allowed_mods: List[str] = None
) -> pd.DataFrame:
    """
    Filter peptides by modifications.

    Args:
        df: DataFrame with peptide sequences.
        allowed_mods: List of allowed modifications (e.g., ["C[UNIMOD:4]", "M[UNIMOD:35]"]).
                     If None, no filtering is done.

    Returns:
        DataFrame with filtered peptide sequences.
    """
    if "peptide_sequences" not in df.columns:
        raise ValueError("DataFrame must contain 'peptide_sequences' column.")

    if allowed_mods is None:
        return df

    df = df.copy()
    # Extract modifications including the preceding amino acid
    df["_modifications"] = df["peptide_sequences"].str.findall(r".\[[^\]]*\]")

    def check_mods(mods):
        if not isinstance(mods, list) or len(mods) == 0:
            return True  # No modifications is always OK
        return all(mod in allowed_mods for mod in mods)

    filtered_df = df[df["_modifications"].apply(check_mods)]

    removed = len(df) - len(filtered_df)
    if removed > 0:
        logger.debug(
            f"Removed {removed} peptides with unsupported modifications "
            f"(allowed: {allowed_mods})"
        )

    return filtered_df.drop(columns=["_modifications"])


def filter_precursor_charges(
    df: pd.DataFrame, allowed_charges: List[int] = None
) -> pd.DataFrame:
    """
    Filter peptides by precursor charges.

    Args:
        df: DataFrame with peptide sequences.
        allowed_charges: List of allowed precursor charges. If None, no filtering is done.

    Returns:
        DataFrame with filtered peptide sequences.
    """
    if "precursor_charges" not in df.columns:
        raise ValueError("DataFrame must contain 'precursor_charges' column.")

    if allowed_charges is None:
        return df

    filtered_df = df[df["precursor_charges"].isin(allowed_charges)]

    removed = len(df) - len(filtered_df)
    if removed > 0:
        logger.debug(
            f"Removed {removed} peptides with unsupported charges "
            f"(allowed: {allowed_charges})"
        )

    return filtered_df


def filter_instrument_types(
    df: pd.DataFrame, allowed_instruments: List[str] = None
) -> pd.DataFrame:
    """
    Filter peptides by instrument types.

    Args:
        df: DataFrame with peptide sequences.
        allowed_instruments: List of allowed instrument types. If None, no filtering is done.

    Returns:
        DataFrame with filtered peptide sequences.
    """
    if "instrument_types" not in df.columns:
        raise ValueError("DataFrame must contain 'instrument_types' column.")

    if allowed_instruments is None:
        return df

    filtered_df = df[df["instrument_types"].isin(allowed_instruments)]

    removed = len(df) - len(filtered_df)
    if removed > 0:
        logger.debug(
            f"Removed {removed} peptides with unsupported instruments "
            f"(allowed: {allowed_instruments})"
        )

    return filtered_df
