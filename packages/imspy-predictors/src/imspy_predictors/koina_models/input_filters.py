import pandas as pd
import numpy as np


# eligible models:
## Intensity: Prosit_2023_intensity_timsTOF, AlphaPeptDeep_ms2_generic, ms2pip_timsTOF2023, ms2pip_timsTOF2024
## RT: AlphaPeptDeep_rt_generic, Deeplc_hela_hf, Chronologer_RT, Prosit_2019_irt
## CCS: AlphaPeptDeep_ccs_generic, IM2Deep
## flyabilit: pfly_2024_fine_tuned

model_filters = {
    "prosit": [
        {"length": [0, 30]},
        {"modifications": ["C[UNIMOD:4]", "M[UNIMOD:35]"]},
        {"precursor_charges": [1, 2, 3, 4, 5, 6]},
    ],
    "alphapeptdeep": None,
    "ms2pip": [{"length": [0, 30]}],
    "deeplc": [{"length": [0, 60]}],
    "chronologer": None,
    "im2deep": [{"length": [0, 60]}],
    "pfly": None,
}


def filter_input_by_model(model_name, df) -> pd.DataFrame:
    """
    Filter input DataFrame based on the model name.
    Args:
        model_name: Name of the model.
        df: Input DataFrame with peptide sequences.
    Returns:
        Filtered DataFrame.
    """
    model_type = model_name.split("_")[0].lower()
    if model_type in model_filters:
        filters = model_filters[model_type]
        if filters is not None:
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
    else:
        raise ValueError(f"Model {model_name} not supported.")
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
    if (max_len is not None) or (min_len is not None):
        if max_len is None:
            max_len = np.inf
        if min_len is None:
            min_len = 0

        df["peptide_length"] = (
            df["peptide_sequences"].str.replace(r"\[.*?\]", "", regex=True).str.len()
        )
        filtered_df = df[
            (df["peptide_length"] >= min_len)
            & (df["peptide_sequences"].str.len() <= max_len)
        ]
        print(
            f"Removed {len(df) - len(filtered_df)} peptides based on length "
            f"({min_len}-{max_len} amino acids)."
        )
        filtered_df = filtered_df.drop(columns=["peptide_length"])
    else:
        filtered_df = df
        print("No filtering based on peptide length.")
    return filtered_df


def filter_peptide_modifications(
    df: pd.DataFrame, allowed_mods: list = ["C[UNIMOD:4]", "M[UNIMOD:35]"]
) -> pd.DataFrame:
    """
    Filter peptides by modifications.
    Args:
        df: DataFrame with peptide sequences.
        allowed_mods: List of allowed modifications, including [] and no space. If None, no filtering is done.
    Returns:
        DataFrame with filtered peptide sequences.
    """
    if "peptide_sequences" not in df.columns:
        raise ValueError("DataFrame must contain 'peptide_sequences' column.")

    if allowed_mods is not None:
        df["modifications"] = df["peptide_sequences"].str.findall(r".\[[^\]]*\]")
        # the match also include the preceding amino acid before the unimod charater
        filtered_df = df[
            df["modifications"].apply(
                lambda x: (
                    all(mod in allowed_mods for mod in x)
                    if isinstance(x, list)
                    else True
                )
            )
        ]

        print(
            f"Removed {len(df) - len(filtered_df)} peptides based on modifications "
            f"{allowed_mods}."
        )
        filtered_df = filtered_df.drop(columns=["modifications"])
    else:
        filtered_df = df
        print("No filtering based on modifications.")

    return filtered_df


def filter_precursor_charges(
    df: pd.DataFrame, allowed_charges: list[int] = [1, 2, 3, 4, 5, 6]
) -> pd.DataFrame:
    """
    Filter peptides by precursor charges.
    Args:
        df: DataFrame with peptide sequences.
        allowed_charges: List of allowed precursor charges (int). If None, no filtering is done.
    Returns:
        DataFrame with filtered peptide sequences.
    """
    if "precursor_charges" not in df.columns:
        raise ValueError("DataFrame must contain 'precursor_charges' column.")

    if allowed_charges is not None:
        filtered_df = df[df["precursor_charges"].isin(allowed_charges)]
        print(
            f"Removed {len(df) - len(filtered_df)} peptides based on precursor charges "
            f"{allowed_charges}."
        )
    else:
        filtered_df = df
        print("No filtering based on precursor charges.")

    return filtered_df


def filter_instrument_types(
    df: pd.DataFrame, allowed_instruments: list = ["QE", "LUMOS", "TIMSTOF", "SCIEXTOF"]
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

    if allowed_instruments is not None:
        filtered_df = df[df["instrument_types"].isin(allowed_instruments)]
        print(
            f"Removed {len(df) - len(filtered_df)} peptides based on instrument types "
            f"{allowed_instruments}."
        )
    else:
        filtered_df = df
        print("No filtering based on instrument types.")

    return filtered_df
