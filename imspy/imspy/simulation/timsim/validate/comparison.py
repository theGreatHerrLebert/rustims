"""
Ground truth loading and comparison utilities for timsim-validate.

Adapted from timsim_bench/benchmark/table_concat.py and utility.py
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple, Set, Optional
from scipy import stats

from .parsing import remove_unimod_annotation, replace_I_with_L, create_precursor_id


def load_ground_truth(
    database_path: str,
    experiment_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load ground truth peptide/ion data from a timsim simulation database.

    Args:
        database_path: Path to the synthetic_data.db SQLite database.
        experiment_name: Optional experiment name for tagging.

    Returns:
        DataFrame with ground truth data including:
        - sequence: Plain sequence
        - sequence_modified: Sequence with modifications
        - charge: Charge state
        - rt: Retention time (seconds)
        - inverse_mobility: Ion mobility (1/K0)
        - intensity: Simulated intensity (events * relative_abundance)
        - precursor_id: Unique identifier (sequence_charge)
    """
    from imspy.simulation.experiment import SyntheticExperimentDataHandle

    # Initialize handle
    handle = SyntheticExperimentDataHandle(database_path, verbose=False)

    # Load tables
    peptides = handle.get_table("peptides")
    ions = handle.get_table("ions")

    # Check if fragment_ions table exists (for transmitted ions)
    tables = handle.list_tables()
    if "fragment_ions" in tables:
        fragment_ions = handle.get_table("fragment_ions")
        # Merge to get only transmitted ions
        ions = pd.merge(
            ions,
            fragment_ions[["peptide_id", "ion_id"]].drop_duplicates(),
            on=["peptide_id", "ion_id"],
        )

    # Merge peptides and ions
    ground_truth = pd.merge(
        peptides[["peptide_id", "sequence", "retention_time_gru_predictor", "events", "protein"]],
        ions[["peptide_id", "charge", "inv_mobility_gru_predictor", "relative_abundance"]],
        on="peptide_id",
    )

    # Calculate intensity
    ground_truth["intensity"] = ground_truth["events"] * ground_truth["relative_abundance"]

    # Rename columns
    ground_truth = ground_truth.rename(columns={
        "retention_time_gru_predictor": "rt",
        "inv_mobility_gru_predictor": "inverse_mobility",
    })

    # Store modified sequence and create plain sequence
    ground_truth["sequence_modified"] = ground_truth["sequence"]
    ground_truth["sequence"] = ground_truth["sequence_modified"].apply(remove_unimod_annotation)

    # Create precursor ID for matching
    ground_truth["precursor_id"] = ground_truth.apply(
        lambda r: create_precursor_id(r["sequence_modified"], r["charge"]),
        axis=1
    )

    # Normalize sequences for matching
    ground_truth["sequence_normalized"] = ground_truth["sequence"].apply(replace_I_with_L)
    ground_truth["sequence_modified_normalized"] = ground_truth["sequence_modified"].apply(replace_I_with_L)

    # Add experiment name if provided
    if experiment_name:
        ground_truth["experiment"] = experiment_name

    return ground_truth


def create_peptide_sets(
    df: pd.DataFrame,
    use_modifications: bool = True,
    normalize: bool = True,
) -> Tuple[Set[str], Set[Tuple[str, int]]]:
    """
    Create sets of peptides and precursors (peptide + charge) from a DataFrame.

    Args:
        df: DataFrame with 'sequence', 'sequence_modified', and 'charge' columns.
        use_modifications: If True, use modified sequences; otherwise plain sequences.
        normalize: If True, apply I→L normalization.

    Returns:
        Tuple of (peptide_set, precursor_set) where precursor_set contains
        (sequence, charge) tuples.
    """
    if use_modifications:
        seq_col = "sequence_modified_normalized" if normalize else "sequence_modified"
    else:
        seq_col = "sequence_normalized" if normalize else "sequence"

    # Ensure normalized columns exist
    if normalize and seq_col not in df.columns:
        if use_modifications:
            df = df.copy()
            df["sequence_modified_normalized"] = df["sequence_modified"].apply(replace_I_with_L)
            seq_col = "sequence_modified_normalized"
        else:
            df = df.copy()
            df["sequence_normalized"] = df["sequence"].apply(replace_I_with_L)
            seq_col = "sequence_normalized"

    peptide_set = set(df[seq_col])
    precursor_set = set(zip(df[seq_col], df["charge"]))

    return peptide_set, precursor_set


def calculate_identification_metrics(
    ground_truth_set: Set,
    identified_set: Set,
) -> dict:
    """
    Calculate identification metrics from ground truth and identified sets.

    Args:
        ground_truth_set: Set of ground truth identifiers.
        identified_set: Set of identified identifiers.

    Returns:
        Dictionary with metrics:
        - true_positives: Count of correctly identified
        - false_positives: Count of incorrectly identified (not in ground truth)
        - false_negatives: Count of missed (in ground truth but not identified)
        - identification_rate: TP / (TP + FN)
        - precision: TP / (TP + FP)
        - fdr: FP / (TP + FP)
    """
    true_positives = ground_truth_set & identified_set
    false_positives = identified_set - ground_truth_set
    false_negatives = ground_truth_set - identified_set

    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)

    # Identification rate (sensitivity/recall)
    total_ground_truth = tp_count + fn_count
    id_rate = tp_count / total_ground_truth if total_ground_truth > 0 else 0.0

    # Precision
    total_identified = tp_count + fp_count
    precision = tp_count / total_identified if total_identified > 0 else 0.0

    # FDR
    fdr = fp_count / total_identified if total_identified > 0 else 0.0

    return {
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "identification_rate": id_rate,
        "precision": precision,
        "fdr": fdr,
    }


def match_results(
    ground_truth: pd.DataFrame,
    diann_results: pd.DataFrame,
    match_on: str = "sequence_modified",
    normalize: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Match DiaNN results to ground truth and return matched, unidentified, and false positive sets.

    Args:
        ground_truth: DataFrame with ground truth data.
        diann_results: DataFrame with DiaNN identification results.
        match_on: Column to use for matching ('sequence' or 'sequence_modified').
        normalize: If True, apply I→L normalization before matching.

    Returns:
        Tuple of (matched, unidentified, false_positives) DataFrames.
        - matched: Ground truth entries that were identified, with DiaNN values merged
        - unidentified: Ground truth entries not found by DiaNN
        - false_positives: DiaNN entries not in ground truth
    """
    gt = ground_truth.copy()
    diann = diann_results.copy()

    # Create match keys
    if normalize:
        gt["match_key"] = gt[match_on].apply(replace_I_with_L) + "_" + gt["charge"].astype(str)
        diann["match_key"] = diann[match_on].apply(replace_I_with_L) + "_" + diann["charge"].astype(str)
    else:
        gt["match_key"] = gt[match_on] + "_" + gt["charge"].astype(str)
        diann["match_key"] = diann[match_on] + "_" + diann["charge"].astype(str)

    # Find sets
    gt_keys = set(gt["match_key"])
    diann_keys = set(diann["match_key"])

    matched_keys = gt_keys & diann_keys
    unidentified_keys = gt_keys - diann_keys
    fp_keys = diann_keys - gt_keys

    # Create result DataFrames
    matched_gt = gt[gt["match_key"].isin(matched_keys)].copy()
    matched_diann = diann[diann["match_key"].isin(matched_keys)].copy()

    # Merge matched data (keep ground truth as base, add DiaNN observed values)
    # Rename DiaNN columns to distinguish from ground truth
    diann_cols_to_merge = ["match_key", "rt", "inverse_mobility", "intensity"]
    diann_cols_to_merge = [c for c in diann_cols_to_merge if c in matched_diann.columns]

    matched_diann_subset = matched_diann[diann_cols_to_merge].drop_duplicates(subset=["match_key"])
    matched_diann_subset = matched_diann_subset.rename(columns={
        "rt": "rt_observed",
        "inverse_mobility": "im_observed",
        "intensity": "intensity_observed",
    })

    matched = matched_gt.merge(matched_diann_subset, on="match_key", how="left")

    # Rename ground truth columns for clarity
    matched = matched.rename(columns={
        "rt": "rt_true",
        "inverse_mobility": "im_true",
        "intensity": "intensity_true",
    })

    unidentified = gt[gt["match_key"].isin(unidentified_keys)].copy()
    false_positives = diann[diann["match_key"].isin(fp_keys)].copy()

    return matched, unidentified, false_positives


def calculate_correlation_metrics(
    matched: pd.DataFrame,
) -> dict:
    """
    Calculate correlation metrics between ground truth and observed values.

    Args:
        matched: DataFrame with matched results containing both true and observed values.

    Returns:
        Dictionary with correlation metrics for RT, IM, and intensity.
    """
    metrics = {}

    # RT correlation
    if "rt_true" in matched.columns and "rt_observed" in matched.columns:
        rt_true = matched["rt_true"].dropna()
        rt_obs = matched.loc[rt_true.index, "rt_observed"].dropna()
        common_idx = rt_true.index.intersection(rt_obs.index)

        if len(common_idx) >= 3:
            rt_true = rt_true.loc[common_idx]
            rt_obs = rt_obs.loc[common_idx]

            # DiaNN reports RT in minutes, ground truth is in seconds
            # Convert ground truth to minutes for comparison
            rt_true_min = rt_true / 60.0

            r, p = stats.pearsonr(rt_true_min, rt_obs)
            mae = np.mean(np.abs(rt_true_min - rt_obs))
            median_error = np.median(np.abs(rt_true_min - rt_obs))

            metrics["rt_pearson_r"] = r
            metrics["rt_pearson_p"] = p
            metrics["rt_mae_minutes"] = mae
            metrics["rt_median_error_minutes"] = median_error
        else:
            metrics["rt_pearson_r"] = np.nan
            metrics["rt_pearson_p"] = np.nan
            metrics["rt_mae_minutes"] = np.nan
            metrics["rt_median_error_minutes"] = np.nan

    # IM correlation
    if "im_true" in matched.columns and "im_observed" in matched.columns:
        im_true = matched["im_true"].dropna()
        im_obs = matched.loc[im_true.index, "im_observed"].dropna()
        common_idx = im_true.index.intersection(im_obs.index)

        if len(common_idx) >= 3:
            im_true = im_true.loc[common_idx]
            im_obs = im_obs.loc[common_idx]

            r, p = stats.pearsonr(im_true, im_obs)
            mae = np.mean(np.abs(im_true - im_obs))
            median_error = np.median(np.abs(im_true - im_obs))

            metrics["im_pearson_r"] = r
            metrics["im_pearson_p"] = p
            metrics["im_mae"] = mae
            metrics["im_median_error"] = median_error
        else:
            metrics["im_pearson_r"] = np.nan
            metrics["im_pearson_p"] = np.nan
            metrics["im_mae"] = np.nan
            metrics["im_median_error"] = np.nan

    # Intensity/quantification correlation (log-transformed)
    if "intensity_true" in matched.columns and "intensity_observed" in matched.columns:
        int_true = matched["intensity_true"].dropna()
        int_obs = matched.loc[int_true.index, "intensity_observed"].dropna()
        common_idx = int_true.index.intersection(int_obs.index)

        # Filter positive values for log transform
        valid_idx = common_idx[
            (int_true.loc[common_idx] > 0) & (int_obs.loc[common_idx] > 0)
        ]

        if len(valid_idx) >= 3:
            log_true = np.log1p(int_true.loc[valid_idx])
            log_obs = np.log1p(int_obs.loc[valid_idx])

            r, p = stats.pearsonr(log_true, log_obs)
            spearman_r, spearman_p = stats.spearmanr(
                int_true.loc[valid_idx], int_obs.loc[valid_idx]
            )

            metrics["quant_pearson_r"] = r
            metrics["quant_pearson_p"] = p
            metrics["quant_spearman_r"] = spearman_r
            metrics["quant_spearman_p"] = spearman_p
        else:
            metrics["quant_pearson_r"] = np.nan
            metrics["quant_pearson_p"] = np.nan
            metrics["quant_spearman_r"] = np.nan
            metrics["quant_spearman_p"] = np.nan

    return metrics
