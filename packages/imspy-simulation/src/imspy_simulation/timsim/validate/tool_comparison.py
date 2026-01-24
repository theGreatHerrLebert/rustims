"""
Multi-tool comparison module for timsim-validate.

Compares simulation ground truth with results from multiple analysis tools
(DIA-NN, FragPipe) to evaluate relative performance.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple, Any, Union
import pandas as pd
import numpy as np

from .parsing import (
    parse_diann_report,
    parse_fragpipe_combined,
    parse_sage_results,
    normalize_sequence_for_matching,
    create_precursor_id,
)
from .comparison import load_ground_truth, create_peptide_sets

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Results from a single analysis tool."""
    name: str
    df: pd.DataFrame
    peptides: Set[str]
    precursors: Set[str]
    num_psms: int
    num_peptides: int
    num_precursors: int


@dataclass
class OverlapStats:
    """Statistics about overlap between tools."""
    # Ground truth comparisons
    gt_only: int  # Only in ground truth
    tool_only: int  # Only in tool (false positives)
    both: int  # In both (true positives)

    # Derived metrics
    identification_rate: float
    precision: float
    recall: float


@dataclass
class PairwiseComparison:
    """Comparison between two tools."""
    tool1_name: str
    tool2_name: str
    tool1_only: int
    tool2_only: int
    both: int
    jaccard_index: float


@dataclass
class IntensityBinMetrics:
    """Metrics for an intensity bin."""
    bin_index: int
    intensity_range: str
    log_range: Tuple[float, float]
    ground_truth_count: int
    metrics_per_tool: Dict[str, Dict[str, Any]]  # tool name -> {identified, id_rate}


@dataclass
class ChargeStateMetrics:
    """Metrics per charge state."""
    charge: int
    ground_truth_count: int
    metrics_per_tool: Dict[str, Dict[str, Any]]  # tool name -> {identified, id_rate}


@dataclass
class SpeciesRatioMetrics:
    """Species ratio metrics for HYE-type experiments."""
    # Expected ratios from dilution factors (ground truth)
    expected_ratios: Dict[str, float]  # species name -> expected fraction

    # Observed ratios per tool
    observed_ratios_per_tool: Dict[str, Dict[str, float]]  # tool name -> {species: fraction}

    # Per-species counts
    ground_truth_counts: Dict[str, int]  # species -> peptide count in GT
    identified_counts_per_tool: Dict[str, Dict[str, int]]  # tool -> {species -> count}

    # Ratio errors per tool (absolute difference from expected)
    ratio_errors_per_tool: Dict[str, Dict[str, float]]  # tool -> {species -> abs error}

    # Max ratio error per tool (used for pass/fail)
    max_ratio_error_per_tool: Dict[str, float]  # tool -> max error across species


@dataclass
class PTMLocalizationMetrics:
    """PTM site localization metrics for phosphoproteomics experiments."""
    # Total phosphopeptides in ground truth
    ground_truth_phosphopeptides: int

    # Per-tool metrics
    identified_phosphopeptides_per_tool: Dict[str, int]  # tool -> count identified

    # Correctly localized sites (where tool's reported site matches ground truth)
    correctly_localized_per_tool: Dict[str, int]  # tool -> count with correct site

    # Site localization accuracy per tool (correctly_localized / identified)
    site_accuracy_per_tool: Dict[str, float]  # tool -> accuracy (0.0 - 1.0)


@dataclass
class DDAMetrics:
    """DDA-specific acquisition and identification metrics."""
    # MS2 acquisition metrics
    total_ms2_events: int  # Total MS2 scans/events
    unique_precursors_selected: int  # Unique precursors targeted for MS2
    ms2_frames: int  # Number of MS2 frames
    total_precursors_available: int  # Total precursors in ground truth

    # Derived metrics
    precursor_selection_rate: float  # fraction of available precursors selected
    avg_precursors_per_frame: float  # average precursors per MS2 frame
    precursor_redundancy: float  # avg times each precursor was selected (>1 means resampling)

    # Per-tool MS2 identification efficiency
    ms2_id_efficiency_per_tool: Dict[str, float]  # tool -> (IDs / MS2 events)

    # Per-tool metrics
    identified_per_tool: Dict[str, int]  # tool -> precursors identified


@dataclass
class ComparisonResult:
    """Complete comparison results."""
    ground_truth_precursors: int
    ground_truth_peptides: int

    tool_results: Dict[str, ToolResult]
    gt_overlaps: Dict[str, OverlapStats]  # Tool name -> overlap with GT
    pairwise: Dict[str, PairwiseComparison]  # "tool1_vs_tool2" -> comparison

    # Common across all tools
    common_to_all: int
    unique_to_gt: int  # Not found by any tool

    # Correlation metrics per tool
    rt_correlations: Dict[str, float]
    im_correlations: Dict[str, float]

    # Extended breakdown metrics (optional)
    intensity_breakdown: Optional[List[IntensityBinMetrics]] = None
    charge_breakdown: Optional[List[ChargeStateMetrics]] = None

    # Species ratio metrics (for HYE experiments)
    species_breakdown: Optional[SpeciesRatioMetrics] = None

    # PTM localization metrics (for phosphoproteomics experiments)
    ptm_metrics: Optional[PTMLocalizationMetrics] = None

    # DDA-specific metrics
    dda_metrics: Optional[DDAMetrics] = None

    # Tool version information
    tool_versions: Optional[Dict[str, str]] = None  # Tool name -> version string


def load_tool_result(
    name: str,
    df: pd.DataFrame,
    normalize: bool = True,
) -> ToolResult:
    """
    Create a ToolResult from a parsed DataFrame.

    Args:
        name: Name of the tool (e.g., "DIA-NN", "FragPipe").
        df: Parsed results DataFrame.
        normalize: Whether to normalize sequences (I->L).

    Returns:
        ToolResult with sets of peptides and precursors.
    """
    peptides, precursors = create_peptide_sets(
        df, use_modifications=False, normalize=normalize
    )

    return ToolResult(
        name=name,
        df=df,
        peptides=peptides,
        precursors=precursors,
        num_psms=len(df),
        num_peptides=len(peptides),
        num_precursors=len(precursors),
    )


def calculate_overlap_with_gt(
    gt_precursors: Set[str],
    tool_precursors: Set[str],
) -> OverlapStats:
    """
    Calculate overlap statistics between ground truth and tool results.

    Args:
        gt_precursors: Set of ground truth precursor IDs.
        tool_precursors: Set of tool precursor IDs.

    Returns:
        OverlapStats with identification metrics.
    """
    both = gt_precursors & tool_precursors
    gt_only = gt_precursors - tool_precursors
    tool_only = tool_precursors - gt_precursors

    identification_rate = len(both) / len(gt_precursors) if gt_precursors else 0
    precision = len(both) / len(tool_precursors) if tool_precursors else 0
    recall = identification_rate  # Same as ID rate for ground truth comparison

    return OverlapStats(
        gt_only=len(gt_only),
        tool_only=len(tool_only),
        both=len(both),
        identification_rate=identification_rate,
        precision=precision,
        recall=recall,
    )


def calculate_pairwise_comparison(
    tool1: ToolResult,
    tool2: ToolResult,
) -> PairwiseComparison:
    """
    Calculate pairwise comparison between two tools.

    Args:
        tool1: First tool result.
        tool2: Second tool result.

    Returns:
        PairwiseComparison with overlap statistics.
    """
    both = tool1.precursors & tool2.precursors
    tool1_only = tool1.precursors - tool2.precursors
    tool2_only = tool2.precursors - tool1.precursors

    union = tool1.precursors | tool2.precursors
    jaccard = len(both) / len(union) if union else 0

    return PairwiseComparison(
        tool1_name=tool1.name,
        tool2_name=tool2.name,
        tool1_only=len(tool1_only),
        tool2_only=len(tool2_only),
        both=len(both),
        jaccard_index=jaccard,
    )


def calculate_rt_correlation(
    ground_truth: pd.DataFrame,
    tool_df: pd.DataFrame,
    normalize: bool = True,
) -> Optional[float]:
    """
    Calculate RT correlation between ground truth and tool results.

    Args:
        ground_truth: Ground truth DataFrame with 'sequence', 'charge', 'rt' columns.
        tool_df: Tool results DataFrame.
        normalize: Whether to normalize sequences.

    Returns:
        Pearson correlation coefficient or None if insufficient data.
    """
    # Handle empty DataFrames
    if tool_df.empty or ground_truth.empty:
        return None

    # Create matching keys
    gt = ground_truth.copy()
    tool = tool_df.copy()

    if normalize:
        gt["match_key"] = gt.apply(
            lambda r: create_precursor_id(
                normalize_sequence_for_matching(r["sequence"]),
                int(r["charge"])
            ), axis=1
        )
        tool["match_key"] = tool.apply(
            lambda r: create_precursor_id(
                normalize_sequence_for_matching(r["sequence"]),
                int(r["charge"])
            ), axis=1
        )
    else:
        gt["match_key"] = gt.apply(
            lambda r: create_precursor_id(r["sequence"], int(r["charge"])), axis=1
        )
        tool["match_key"] = tool.apply(
            lambda r: create_precursor_id(r["sequence"], int(r["charge"])), axis=1
        )

    # Merge on key
    merged = gt.merge(
        tool[["match_key", "rt"]],
        on="match_key",
        suffixes=("_gt", "_tool"),
    )

    if len(merged) < 2:
        return None

    # Handle column naming
    rt_gt = merged["rt_gt"] if "rt_gt" in merged.columns else merged["rt"]
    rt_tool = merged["rt_tool"] if "rt_tool" in merged.columns else merged["rt"]

    return rt_gt.corr(rt_tool)


def calculate_im_correlation(
    ground_truth: pd.DataFrame,
    tool_df: pd.DataFrame,
    normalize: bool = True,
) -> Optional[float]:
    """
    Calculate IM correlation between ground truth and tool results.

    Args:
        ground_truth: Ground truth DataFrame.
        tool_df: Tool results DataFrame.
        normalize: Whether to normalize sequences.

    Returns:
        Pearson correlation coefficient or None if insufficient data.
    """
    # Handle empty DataFrames
    if tool_df.empty or ground_truth.empty:
        return None

    gt = ground_truth.copy()
    tool = tool_df.copy()

    # Check for IM column
    gt_im_col = "inverse_mobility" if "inverse_mobility" in gt.columns else "im"
    tool_im_col = "inverse_mobility" if "inverse_mobility" in tool.columns else "im"

    if gt_im_col not in gt.columns or tool_im_col not in tool.columns:
        return None

    if normalize:
        gt["match_key"] = gt.apply(
            lambda r: create_precursor_id(
                normalize_sequence_for_matching(r["sequence"]),
                int(r["charge"])
            ), axis=1
        )
        tool["match_key"] = tool.apply(
            lambda r: create_precursor_id(
                normalize_sequence_for_matching(r["sequence"]),
                int(r["charge"])
            ), axis=1
        )
    else:
        gt["match_key"] = gt.apply(
            lambda r: create_precursor_id(r["sequence"], int(r["charge"])), axis=1
        )
        tool["match_key"] = tool.apply(
            lambda r: create_precursor_id(r["sequence"], int(r["charge"])), axis=1
        )

    # Rename IM columns for merge
    gt = gt.rename(columns={gt_im_col: "im_gt"})
    tool = tool.rename(columns={tool_im_col: "im_tool"})

    merged = gt.merge(
        tool[["match_key", "im_tool"]],
        on="match_key",
    )

    if len(merged) < 2:
        return None

    return merged["im_gt"].corr(merged["im_tool"])


def calculate_intensity_breakdown(
    ground_truth_df: pd.DataFrame,
    tool_results: Dict[str, ToolResult],
    normalize: bool = True,
    num_bins: int = 5,
) -> List[IntensityBinMetrics]:
    """
    Calculate identification metrics per intensity bin for all tools.

    Args:
        ground_truth_df: Ground truth DataFrame with 'intensity' column.
        tool_results: Dictionary of ToolResult objects.
        normalize: Whether to normalize sequences.
        num_bins: Number of intensity bins.

    Returns:
        List of IntensityBinMetrics for each bin.
    """
    if "intensity" not in ground_truth_df.columns:
        return []

    gt = ground_truth_df.copy()
    gt["log_intensity"] = np.log10(gt["intensity"].clip(lower=1))

    # Determine sequence column based on normalization
    if normalize:
        if "sequence_normalized" not in gt.columns:
            gt["sequence_normalized"] = gt["sequence"].apply(normalize_sequence_for_matching)
        seq_col = "sequence_normalized"
    else:
        seq_col = "sequence"

    # Create bins
    try:
        gt["intensity_bin"] = pd.qcut(gt["log_intensity"], num_bins, labels=False, duplicates="drop")
    except ValueError:
        return []

    results = []
    for bin_idx in sorted(gt["intensity_bin"].unique()):
        bin_data = gt[gt["intensity_bin"] == bin_idx]
        log_range = (float(bin_data["log_intensity"].min()), float(bin_data["log_intensity"].max()))
        intensity_range = f"{10**log_range[0]:.0e}-{10**log_range[1]:.0e}"

        # Create ground truth precursor set for this bin as tuples (seq, charge)
        gt_bin_precursors = set(zip(bin_data[seq_col], bin_data["charge"]))

        # Calculate per-tool metrics for this bin
        tool_metrics = {}
        for name, tool in tool_results.items():
            identified = len(gt_bin_precursors & tool.precursors)
            id_rate = identified / len(gt_bin_precursors) if gt_bin_precursors else 0
            tool_metrics[name] = {"identified": identified, "id_rate": id_rate}

        results.append(IntensityBinMetrics(
            bin_index=int(bin_idx),
            intensity_range=intensity_range,
            log_range=log_range,
            ground_truth_count=len(gt_bin_precursors),
            metrics_per_tool=tool_metrics,
        ))

    return results


def calculate_charge_breakdown(
    ground_truth_df: pd.DataFrame,
    tool_results: Dict[str, ToolResult],
    normalize: bool = True,
) -> List[ChargeStateMetrics]:
    """
    Calculate identification metrics per charge state for all tools.

    Args:
        ground_truth_df: Ground truth DataFrame with 'charge' column.
        tool_results: Dictionary of ToolResult objects.
        normalize: Whether to normalize sequences.

    Returns:
        List of ChargeStateMetrics for each charge state.
    """
    if "charge" not in ground_truth_df.columns:
        return []

    gt = ground_truth_df.copy()

    # Determine sequence column based on normalization
    if normalize:
        if "sequence_normalized" not in gt.columns:
            gt["sequence_normalized"] = gt["sequence"].apply(normalize_sequence_for_matching)
        seq_col = "sequence_normalized"
    else:
        seq_col = "sequence"

    results = []
    for charge in sorted(gt["charge"].unique()):
        charge_data = gt[gt["charge"] == charge]

        # Create ground truth precursor set for this charge as tuples (seq, charge)
        gt_charge_precursors = set(zip(charge_data[seq_col], charge_data["charge"]))

        # Calculate per-tool metrics for this charge
        tool_metrics = {}
        for name, tool in tool_results.items():
            identified = len(gt_charge_precursors & tool.precursors)
            id_rate = identified / len(gt_charge_precursors) if gt_charge_precursors else 0
            tool_metrics[name] = {"identified": identified, "id_rate": id_rate}

        results.append(ChargeStateMetrics(
            charge=int(charge),
            ground_truth_count=len(gt_charge_precursors),
            metrics_per_tool=tool_metrics,
        ))

    return results


def calculate_species_breakdown(
    ground_truth_df: pd.DataFrame,
    tool_results: Dict[str, ToolResult],
    normalize: bool = True,
    dilution_factors: Optional[Dict[str, float]] = None,
) -> Optional[SpeciesRatioMetrics]:
    """
    Calculate species ratio metrics for HYE-type experiments.

    This function computes how well each tool identifies peptides from different
    species (Human, Yeast, E.coli) compared to expected ratios from dilution factors.

    Args:
        ground_truth_df: Ground truth DataFrame with 'fasta' column indicating species.
        tool_results: Dictionary of ToolResult objects.
        normalize: Whether to normalize sequences.
        dilution_factors: Optional expected dilution factors. If None, calculated from GT.

    Returns:
        SpeciesRatioMetrics if 'fasta' column exists, None otherwise.
    """
    # Check if fasta column exists (indicates proteome_mix mode)
    if "fasta" not in ground_truth_df.columns:
        return None

    gt = ground_truth_df.copy()

    # Determine sequence column based on normalization
    if normalize:
        if "sequence_normalized" not in gt.columns:
            gt["sequence_normalized"] = gt["sequence"].apply(normalize_sequence_for_matching)
        seq_col = "sequence_normalized"
    else:
        seq_col = "sequence"

    # Get unique species from fasta column
    species_list = gt["fasta"].dropna().unique().tolist()
    if len(species_list) < 2:
        return None  # Need at least 2 species for ratio comparison

    # Count ground truth peptides per species
    gt_counts: Dict[str, int] = {}
    gt_precursors_by_species: Dict[str, Set[Tuple[str, int]]] = {}

    for species in species_list:
        species_data = gt[gt["fasta"] == species]
        gt_counts[species] = len(species_data)
        gt_precursors_by_species[species] = set(zip(species_data[seq_col], species_data["charge"]))

    # Calculate expected ratios (from dilution factors or from ground truth counts)
    total_gt = sum(gt_counts.values())
    if dilution_factors:
        expected_ratios = dilution_factors.copy()
    else:
        # Calculate from ground truth distribution
        expected_ratios = {s: c / total_gt for s, c in gt_counts.items()}

    # Normalize expected ratios to sum to 1
    ratio_sum = sum(expected_ratios.values())
    if ratio_sum > 0:
        expected_ratios = {s: r / ratio_sum for s, r in expected_ratios.items()}

    # Create a mapping from precursor (seq, charge) to species
    precursor_to_species: Dict[Tuple[str, int], str] = {}
    for species in species_list:
        for precursor in gt_precursors_by_species[species]:
            precursor_to_species[precursor] = species

    # Calculate identified counts per species per tool
    identified_counts_per_tool: Dict[str, Dict[str, int]] = {}
    observed_ratios_per_tool: Dict[str, Dict[str, float]] = {}
    ratio_errors_per_tool: Dict[str, Dict[str, float]] = {}
    max_ratio_error_per_tool: Dict[str, float] = {}

    for tool_name, tool in tool_results.items():
        # Count identified precursors by species
        species_counts: Dict[str, int] = {s: 0 for s in species_list}

        for precursor in tool.precursors:
            if precursor in precursor_to_species:
                species = precursor_to_species[precursor]
                species_counts[species] += 1

        identified_counts_per_tool[tool_name] = species_counts

        # Calculate observed ratios
        total_identified = sum(species_counts.values())
        if total_identified > 0:
            observed_ratios = {s: c / total_identified for s, c in species_counts.items()}
        else:
            observed_ratios = {s: 0.0 for s in species_list}

        observed_ratios_per_tool[tool_name] = observed_ratios

        # Calculate ratio errors (absolute difference from expected)
        ratio_errors = {}
        for species in species_list:
            expected = expected_ratios.get(species, 0.0)
            observed = observed_ratios.get(species, 0.0)
            ratio_errors[species] = abs(observed - expected)

        ratio_errors_per_tool[tool_name] = ratio_errors
        max_ratio_error_per_tool[tool_name] = max(ratio_errors.values()) if ratio_errors else 0.0

    return SpeciesRatioMetrics(
        expected_ratios=expected_ratios,
        observed_ratios_per_tool=observed_ratios_per_tool,
        ground_truth_counts=gt_counts,
        identified_counts_per_tool=identified_counts_per_tool,
        ratio_errors_per_tool=ratio_errors_per_tool,
        max_ratio_error_per_tool=max_ratio_error_per_tool,
    )


def calculate_ptm_localization(
    ground_truth_df: pd.DataFrame,
    tool_results: Dict[str, ToolResult],
    normalize: bool = True,
) -> Optional[PTMLocalizationMetrics]:
    """
    Calculate PTM site localization metrics for phosphoproteomics experiments.

    This function computes how accurately each tool localizes the phosphorylation
    site compared to the ground truth phospho_site_a.

    Args:
        ground_truth_df: Ground truth DataFrame with phospho columns.
        tool_results: Dictionary of ToolResult objects.
        normalize: Whether to normalize sequences.

    Returns:
        PTMLocalizationMetrics if phospho columns exist, None otherwise.
    """
    # Check if phospho columns exist (indicates phospho_mode)
    if "phospho_site_a" not in ground_truth_df.columns:
        return None

    gt = ground_truth_df.copy()

    # Create a mapping from (sequence_modified, charge) to phospho_site_a
    # The sequence_modified already contains the UNIMOD:21 at phospho_site_a
    if normalize:
        if "sequence_modified_normalized" not in gt.columns:
            gt["sequence_modified_normalized"] = gt["sequence_modified"].apply(
                normalize_sequence_for_matching
            )
        seq_col = "sequence_modified_normalized"
    else:
        seq_col = "sequence_modified"

    # Build lookup: (normalized_seq_with_mod, charge) -> phospho_site_a
    gt_phospho_lookup: Dict[Tuple[str, int], int] = {}
    for _, row in gt.iterrows():
        key = (row[seq_col], int(row["charge"]))
        gt_phospho_lookup[key] = int(row["phospho_site_a"])

    # Count ground truth phosphopeptides (unique modified sequences)
    gt_phosphopeptides = len(set(gt[seq_col]))

    # Calculate per-tool metrics
    identified_per_tool: Dict[str, int] = {}
    correctly_localized_per_tool: Dict[str, int] = {}
    site_accuracy_per_tool: Dict[str, float] = {}

    for tool_name, tool in tool_results.items():
        # Build precursor set from tool.df using sequence_modified column
        # This is necessary because tool.precursors uses bare sequences (without mods)
        tool_df = tool.df.copy()

        # Ensure sequence_modified column exists
        if "sequence_modified" not in tool_df.columns:
            logger.warning(f"{tool_name} results missing sequence_modified column")
            identified_per_tool[tool_name] = 0
            correctly_localized_per_tool[tool_name] = 0
            site_accuracy_per_tool[tool_name] = 0.0
            continue

        # Normalize tool sequences if needed
        if normalize:
            tool_df["sequence_modified_normalized"] = tool_df["sequence_modified"].apply(
                normalize_sequence_for_matching
            )
            tool_seq_col = "sequence_modified_normalized"
        else:
            tool_seq_col = "sequence_modified"

        # Build set of (modified_sequence, charge) tuples from tool results
        tool_modified_precursors: Set[Tuple[str, int]] = set()
        for _, row in tool_df.iterrows():
            key = (row[tool_seq_col], int(row["charge"]))
            tool_modified_precursors.add(key)

        # Get tool's identified phosphopeptides that match ground truth
        identified_count = 0
        correct_count = 0

        for precursor in tool_modified_precursors:
            if precursor in gt_phospho_lookup:
                identified_count += 1
                # Since we matched on sequence_modified, the modification position
                # is encoded in the sequence string - if it matches, site is correct
                correct_count += 1

        identified_per_tool[tool_name] = identified_count
        correctly_localized_per_tool[tool_name] = correct_count

        # Calculate accuracy
        if identified_count > 0:
            site_accuracy_per_tool[tool_name] = correct_count / identified_count
        else:
            site_accuracy_per_tool[tool_name] = 0.0

    return PTMLocalizationMetrics(
        ground_truth_phosphopeptides=gt_phosphopeptides,
        identified_phosphopeptides_per_tool=identified_per_tool,
        correctly_localized_per_tool=correctly_localized_per_tool,
        site_accuracy_per_tool=site_accuracy_per_tool,
    )


def calculate_dda_metrics(
    database_path: str,
    tool_results: Dict[str, ToolResult],
    ground_truth_precursors: int,
) -> Optional[DDAMetrics]:
    """
    Calculate DDA-specific acquisition and identification metrics.

    Args:
        database_path: Path to simulation database.
        tool_results: Dictionary of ToolResult objects.
        ground_truth_precursors: Total precursors in ground truth.

    Returns:
        DDAMetrics if DDA tables exist in database, None otherwise.
    """
    import sqlite3

    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Check if pasef_meta table exists (DDA-specific)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pasef_meta'"
        )
        if not cursor.fetchone():
            conn.close()
            return None

        # Get MS2 acquisition metrics
        cursor.execute("SELECT COUNT(*) FROM pasef_meta")
        total_ms2_events = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT precursor) FROM pasef_meta")
        unique_precursors_selected = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT frame) FROM pasef_meta")
        ms2_frames = cursor.fetchone()[0]

        conn.close()

        # If no MS2 events, this is likely not DDA
        if total_ms2_events == 0:
            return None

        # Calculate derived metrics
        precursor_selection_rate = (
            unique_precursors_selected / ground_truth_precursors
            if ground_truth_precursors > 0
            else 0.0
        )

        avg_precursors_per_frame = (
            total_ms2_events / ms2_frames if ms2_frames > 0 else 0.0
        )

        precursor_redundancy = (
            total_ms2_events / unique_precursors_selected
            if unique_precursors_selected > 0
            else 0.0
        )

        # Calculate per-tool MS2 identification efficiency
        ms2_id_efficiency_per_tool: Dict[str, float] = {}
        identified_per_tool: Dict[str, int] = {}

        for tool_name, tool in tool_results.items():
            identified = tool.num_precursors
            identified_per_tool[tool_name] = identified

            # MS2 ID efficiency = identified precursors / MS2 events
            ms2_id_efficiency_per_tool[tool_name] = (
                identified / total_ms2_events if total_ms2_events > 0 else 0.0
            )

        return DDAMetrics(
            total_ms2_events=total_ms2_events,
            unique_precursors_selected=unique_precursors_selected,
            ms2_frames=ms2_frames,
            total_precursors_available=ground_truth_precursors,
            precursor_selection_rate=precursor_selection_rate,
            avg_precursors_per_frame=avg_precursors_per_frame,
            precursor_redundancy=precursor_redundancy,
            ms2_id_efficiency_per_tool=ms2_id_efficiency_per_tool,
            identified_per_tool=identified_per_tool,
        )

    except Exception as e:
        logger.warning(f"Error calculating DDA metrics: {e}")
        return None


def compare_tools(
    ground_truth_df: pd.DataFrame,
    tool_results: Dict[str, pd.DataFrame],
    normalize: bool = True,
    database_path: Optional[str] = None,
) -> ComparisonResult:
    """
    Compare multiple analysis tools against ground truth.

    Args:
        ground_truth_df: Ground truth DataFrame from simulation.
        database_path: Optional path to simulation database (for DDA metrics).
        tool_results: Dictionary mapping tool name to parsed results DataFrame.
        normalize: Whether to normalize sequences for matching.

    Returns:
        ComparisonResult with all comparison statistics.
    """
    # Load ground truth
    gt_peptides, gt_precursors = create_peptide_sets(
        ground_truth_df, use_modifications=False, normalize=normalize
    )

    # Load tool results
    tools: Dict[str, ToolResult] = {}
    for name, df in tool_results.items():
        tools[name] = load_tool_result(name, df, normalize=normalize)

    # Calculate overlaps with ground truth
    gt_overlaps: Dict[str, OverlapStats] = {}
    for name, tool in tools.items():
        gt_overlaps[name] = calculate_overlap_with_gt(gt_precursors, tool.precursors)

    # Calculate pairwise comparisons
    pairwise: Dict[str, PairwiseComparison] = {}
    tool_names = list(tools.keys())
    for i, name1 in enumerate(tool_names):
        for name2 in tool_names[i+1:]:
            key = f"{name1}_vs_{name2}"
            pairwise[key] = calculate_pairwise_comparison(tools[name1], tools[name2])

    # Calculate common to all tools
    if tools:
        all_tool_precursors = [t.precursors for t in tools.values()]
        common_to_all_tools = set.intersection(*all_tool_precursors) if all_tool_precursors else set()
        common_to_all = len(common_to_all_tools & gt_precursors)

        any_tool_precursors = set.union(*all_tool_precursors) if all_tool_precursors else set()
        unique_to_gt = len(gt_precursors - any_tool_precursors)
    else:
        common_to_all = 0
        unique_to_gt = len(gt_precursors)

    # Calculate correlations
    rt_correlations: Dict[str, float] = {}
    im_correlations: Dict[str, float] = {}

    for name, tool in tools.items():
        rt_corr = calculate_rt_correlation(ground_truth_df, tool.df, normalize)
        if rt_corr is not None:
            rt_correlations[name] = rt_corr

        im_corr = calculate_im_correlation(ground_truth_df, tool.df, normalize)
        if im_corr is not None:
            im_correlations[name] = im_corr

    # Calculate intensity and charge breakdown metrics
    intensity_breakdown = calculate_intensity_breakdown(ground_truth_df, tools, normalize)
    charge_breakdown = calculate_charge_breakdown(ground_truth_df, tools, normalize)

    # Calculate species breakdown metrics (for HYE experiments)
    species_breakdown = calculate_species_breakdown(ground_truth_df, tools, normalize)

    # Calculate PTM localization metrics (for phosphoproteomics experiments)
    ptm_metrics = calculate_ptm_localization(ground_truth_df, tools, normalize)

    # Calculate DDA-specific metrics (if database_path provided)
    dda_metrics = None
    if database_path:
        dda_metrics = calculate_dda_metrics(database_path, tools, len(gt_precursors))

    return ComparisonResult(
        ground_truth_precursors=len(gt_precursors),
        ground_truth_peptides=len(gt_peptides),
        tool_results=tools,
        gt_overlaps=gt_overlaps,
        pairwise=pairwise,
        common_to_all=common_to_all,
        unique_to_gt=unique_to_gt,
        rt_correlations=rt_correlations,
        im_correlations=im_correlations,
        intensity_breakdown=intensity_breakdown,
        charge_breakdown=charge_breakdown,
        species_breakdown=species_breakdown,
        ptm_metrics=ptm_metrics,
        dda_metrics=dda_metrics,
    )


def generate_comparison_text_report(result: ComparisonResult) -> str:
    """
    Generate a text report comparing all tools.

    Args:
        result: ComparisonResult from compare_tools().

    Returns:
        Formatted text report string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("              TIMSIM MULTI-TOOL COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Ground truth summary
    lines.append("GROUND TRUTH (SIMULATION)")
    lines.append("-" * 40)
    lines.append(f"  Total Precursors:    {result.ground_truth_precursors:,}")
    lines.append(f"  Total Peptides:      {result.ground_truth_peptides:,}")
    lines.append("")

    # Per-tool results
    lines.append("TOOL RESULTS")
    lines.append("-" * 40)

    header = f"  {'Tool':<15} {'PSMs':>10} {'Peptides':>10} {'Precursors':>12} {'ID Rate':>10}"
    lines.append(header)
    lines.append("  " + "-" * 59)

    for name, tool in result.tool_results.items():
        overlap = result.gt_overlaps[name]
        lines.append(
            f"  {name:<15} {tool.num_psms:>10,} {tool.num_peptides:>10,} "
            f"{tool.num_precursors:>12,} {overlap.identification_rate:>9.1%}"
        )
    lines.append("")

    # Overlap with ground truth
    lines.append("IDENTIFICATION OVERLAP WITH GROUND TRUTH")
    lines.append("-" * 40)

    header = f"  {'Tool':<15} {'True Pos':>10} {'False Pos':>10} {'False Neg':>10} {'Precision':>10}"
    lines.append(header)
    lines.append("  " + "-" * 56)

    for name, overlap in result.gt_overlaps.items():
        lines.append(
            f"  {name:<15} {overlap.both:>10,} {overlap.tool_only:>10,} "
            f"{overlap.gt_only:>10,} {overlap.precision:>9.1%}"
        )
    lines.append("")

    # Pairwise comparisons
    if result.pairwise:
        lines.append("PAIRWISE TOOL COMPARISONS (Precursor Level)")
        lines.append("-" * 40)

        for key, comp in result.pairwise.items():
            lines.append(f"  {comp.tool1_name} vs {comp.tool2_name}:")
            lines.append(f"    Common:           {comp.both:>10,}")
            lines.append(f"    Only {comp.tool1_name}:{' '*(10-len(comp.tool1_name))}{comp.tool1_only:>10,}")
            lines.append(f"    Only {comp.tool2_name}:{' '*(10-len(comp.tool2_name))}{comp.tool2_only:>10,}")
            lines.append(f"    Jaccard Index:    {comp.jaccard_index:>10.3f}")
            lines.append("")

    # Common and unique
    lines.append("OVERALL COVERAGE")
    lines.append("-" * 40)
    lines.append(f"  Found by all tools:      {result.common_to_all:>10,}")
    lines.append(f"  Not found by any tool:   {result.unique_to_gt:>10,}")
    lines.append("")

    # Correlation metrics
    lines.append("RETENTION TIME CORRELATIONS (vs Ground Truth)")
    lines.append("-" * 40)
    for name, corr in result.rt_correlations.items():
        lines.append(f"  {name:<15} R = {corr:.4f}")
    lines.append("")

    lines.append("ION MOBILITY CORRELATIONS (vs Ground Truth)")
    lines.append("-" * 40)
    for name, corr in result.im_correlations.items():
        lines.append(f"  {name:<15} R = {corr:.4f}")
    lines.append("")

    # Intensity breakdown
    if result.intensity_breakdown:
        lines.append("INTENSITY-DEPENDENT IDENTIFICATION RATES")
        lines.append("-" * 80)
        tool_names = list(result.tool_results.keys())
        header = f"  {'Bin':<5} {'Intensity Range':<18} {'GT Count':>10}"
        for name in tool_names:
            header += f" {name[:8]:>10}"
        lines.append(header)
        lines.append("  " + "-" * (45 + 12 * len(tool_names)))

        for bin_metrics in result.intensity_breakdown:
            line = f"  {bin_metrics.bin_index:<5} {bin_metrics.intensity_range:<18} {bin_metrics.ground_truth_count:>10}"
            for name in tool_names:
                id_rate = bin_metrics.metrics_per_tool.get(name, {}).get("id_rate", 0)
                line += f" {id_rate:>9.1%}"
            lines.append(line)
        lines.append("")

    # Charge state breakdown
    if result.charge_breakdown:
        lines.append("CHARGE STATE BREAKDOWN")
        lines.append("-" * 80)
        tool_names = list(result.tool_results.keys())
        header = f"  {'Charge':<8} {'GT Count':>10}"
        for name in tool_names:
            header += f" {name[:8]:>10}"
        lines.append(header)
        lines.append("  " + "-" * (20 + 12 * len(tool_names)))

        for charge_metrics in result.charge_breakdown:
            line = f"  {charge_metrics.charge}+{'':<5} {charge_metrics.ground_truth_count:>10}"
            for name in tool_names:
                id_rate = charge_metrics.metrics_per_tool.get(name, {}).get("id_rate", 0)
                line += f" {id_rate:>9.1%}"
            lines.append(line)
        lines.append("")

    # Species breakdown (for HYE experiments)
    if result.species_breakdown:
        lines.append("SPECIES RATIO ANALYSIS (HYE)")
        lines.append("-" * 80)
        tool_names = list(result.tool_results.keys())
        species_list = list(result.species_breakdown.expected_ratios.keys())

        # Header
        lines.append("  Expected Ratios (Ground Truth):")
        for species in species_list:
            expected = result.species_breakdown.expected_ratios[species]
            gt_count = result.species_breakdown.ground_truth_counts.get(species, 0)
            lines.append(f"    {species:<20} {expected:>6.1%} ({gt_count:,} precursors)")
        lines.append("")

        # Per-tool observed ratios
        lines.append("  Observed Ratios by Tool:")
        header = f"    {'Species':<20}"
        for name in tool_names:
            header += f" {name[:10]:>12}"
        header += f" {'Expected':>12}"
        lines.append(header)
        lines.append("    " + "-" * (22 + 14 * (len(tool_names) + 1)))

        for species in species_list:
            line = f"    {species:<20}"
            for name in tool_names:
                observed = result.species_breakdown.observed_ratios_per_tool[name].get(species, 0)
                line += f" {observed:>11.1%}"
            expected = result.species_breakdown.expected_ratios[species]
            line += f" {expected:>11.1%}"
            lines.append(line)
        lines.append("")

        # Ratio errors
        lines.append("  Ratio Errors by Tool:")
        header = f"    {'Species':<20}"
        for name in tool_names:
            header += f" {name[:10]:>12}"
        lines.append(header)
        lines.append("    " + "-" * (22 + 14 * len(tool_names)))

        for species in species_list:
            line = f"    {species:<20}"
            for name in tool_names:
                error = result.species_breakdown.ratio_errors_per_tool[name].get(species, 0)
                line += f" {error:>11.1%}"
            lines.append(line)
        lines.append("")

        # Max errors summary
        lines.append("  Max Ratio Error per Tool:")
        for name in tool_names:
            max_error = result.species_breakdown.max_ratio_error_per_tool[name]
            status = "PASS" if max_error < 0.20 else "FAIL"
            lines.append(f"    {name:<15} {max_error:>6.1%}  [{status}]")
        lines.append("")

    # PTM localization metrics (for phosphoproteomics experiments)
    if result.ptm_metrics:
        lines.append("PTM SITE LOCALIZATION (PHOSPHOPROTEOMICS)")
        lines.append("-" * 80)
        tool_names = list(result.tool_results.keys())

        lines.append(f"  Ground Truth Phosphopeptides: {result.ptm_metrics.ground_truth_phosphopeptides:,}")
        lines.append("")

        # Per-tool metrics
        header = f"  {'Tool':<15} {'Identified':>12} {'Correct Site':>14} {'Accuracy':>12} {'Status':>10}"
        lines.append(header)
        lines.append("  " + "-" * 65)

        for name in tool_names:
            identified = result.ptm_metrics.identified_phosphopeptides_per_tool.get(name, 0)
            correct = result.ptm_metrics.correctly_localized_per_tool.get(name, 0)
            accuracy = result.ptm_metrics.site_accuracy_per_tool.get(name, 0.0)
            status = "PASS" if accuracy >= 0.80 else "FAIL"
            lines.append(f"  {name:<15} {identified:>12,} {correct:>14,} {accuracy:>11.1%} {status:>10}")
        lines.append("")

    # DDA-specific metrics
    if result.dda_metrics:
        lines.append("DDA ACQUISITION METRICS")
        lines.append("-" * 80)
        dda = result.dda_metrics
        tool_names = list(result.tool_results.keys())

        lines.append(f"  MS2 Events (scans):          {dda.total_ms2_events:,}")
        lines.append(f"  Unique Precursors Selected:  {dda.unique_precursors_selected:,}")
        lines.append(f"  MS2 Frames:                  {dda.ms2_frames:,}")
        lines.append(f"  Total Precursors Available:  {dda.total_precursors_available:,}")
        lines.append("")
        lines.append(f"  Precursor Selection Rate:    {dda.precursor_selection_rate:.1%}")
        lines.append(f"  Avg Precursors per Frame:    {dda.avg_precursors_per_frame:.1f}")
        lines.append(f"  Precursor Redundancy:        {dda.precursor_redundancy:.2f}x")
        lines.append("")

        # Per-tool MS2 identification efficiency
        lines.append("  MS2 Identification Efficiency (IDs / MS2 Events):")
        for name in tool_names:
            efficiency = dda.ms2_id_efficiency_per_tool.get(name, 0.0)
            identified = dda.identified_per_tool.get(name, 0)
            lines.append(f"    {name:<15} {efficiency:>6.1%}  ({identified:,} IDs from {dda.total_ms2_events:,} MS2)")
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def generate_comparison_plots(
    result: ComparisonResult,
    ground_truth_df: pd.DataFrame,
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate comparison plots for multi-tool analysis.

    Args:
        result: ComparisonResult from compare_tools().
        ground_truth_df: Ground truth DataFrame.
        output_dir: Directory to save plots.

    Returns:
        Dictionary mapping plot name to file path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Try to import matplotlib_venn, use None as fallback marker
    try:
        from matplotlib_venn import venn2, venn3
        has_venn = True
    except ImportError:
        has_venn = False
        logger.warning("matplotlib_venn not available, using bar chart fallback for overlap plots")

    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    tool_names = list(result.tool_results.keys())
    colors = {'DIA-NN': '#3498db', 'FragPipe': '#e74c3c', 'Ground Truth': '#2ecc71'}

    # 1. Bar chart: Identification counts
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(tool_names) + 1)
    width = 0.6

    counts = [result.ground_truth_precursors] + [result.tool_results[n].num_precursors for n in tool_names]
    bar_colors = [colors.get('Ground Truth', '#95a5a6')] + [colors.get(n, '#95a5a6') for n in tool_names]

    bars = ax.bar(x, counts, width, color=bar_colors, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Number of Precursors', fontsize=12)
    ax.set_title('Precursor Identification Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Simulation\n(Ground Truth)'] + tool_names, fontsize=11)

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.annotate(f'{count:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(output_dir, "identification_counts.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['identification_counts'] = path

    # 2. Bar chart: Identification rates
    fig, ax = plt.subplots(figsize=(8, 6))

    rates = [result.gt_overlaps[n].identification_rate * 100 for n in tool_names]
    bar_colors = [colors.get(n, '#95a5a6') for n in tool_names]

    bars = ax.bar(tool_names, rates, color=bar_colors, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Identification Rate (%)', fontsize=12)
    ax.set_title('Identification Rate vs Ground Truth', fontsize=14, fontweight='bold')
    # Scale y-axis to highest bar + 15% padding for labels
    max_rate = max(rates) if rates else 100
    ax.set_ylim(0, min(100, max_rate * 1.15))

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add threshold line
    ax.axhline(y=30, color='red', linestyle='--', linewidth=1, label='Threshold (30%)')
    ax.legend(loc='upper right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(output_dir, "identification_rates.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['identification_rates'] = path

    # 3. Venn diagram: Tool overlap with ground truth
    # Helper function to convert tuple precursor sets to string format
    def precursor_set_to_strings(precursor_set: set) -> set:
        """Convert (sequence, charge) tuples to 'sequence_charge' strings."""
        return {f"{seq}_{charge}" for seq, charge in precursor_set}

    if len(tool_names) >= 2:
        # Get precursor sets - all as string format for consistency
        gt_set = set()
        for _, row in ground_truth_df.iterrows():
            key = create_precursor_id(
                normalize_sequence_for_matching(row["sequence"]),
                int(row["charge"])
            )
            gt_set.add(key)

        # Convert tool precursor sets from tuples to strings
        tool_sets = {
            name: precursor_set_to_strings(result.tool_results[name].precursors)
            for name in tool_names
        }
        tool1_set = tool_sets[tool_names[0]]
        tool2_set = tool_sets[tool_names[1]]
        tool3_set = tool_sets[tool_names[2]] if len(tool_names) >= 3 else None

        if has_venn:
            # Use Venn diagrams
            if len(tool_names) == 2:
                # 2-tool comparison: GT vs each tool + tools vs each other
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Venn: GT vs Tool 1
                ax = axes[0]
                venn2([gt_set, tool1_set], set_labels=('Ground Truth', tool_names[0]), ax=ax)
                ax.set_title(f'Ground Truth vs {tool_names[0]}', fontsize=12, fontweight='bold')

                # Venn: GT vs Tool 2
                ax = axes[1]
                venn2([gt_set, tool2_set], set_labels=('Ground Truth', tool_names[1]), ax=ax)
                ax.set_title(f'Ground Truth vs {tool_names[1]}', fontsize=12, fontweight='bold')

                # Venn: Tool 1 vs Tool 2
                ax = axes[2]
                venn2([tool1_set, tool2_set], set_labels=(tool_names[0], tool_names[1]), ax=ax)
                ax.set_title(f'{tool_names[0]} vs {tool_names[1]}', fontsize=12, fontweight='bold')

                plt.suptitle('Precursor Overlap Analysis', fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()

                path = os.path.join(output_dir, "venn_overlap.png")
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.close()
                plot_paths['venn_overlap'] = path

                # 3-way Venn: GT vs both tools
                fig, ax = plt.subplots(figsize=(8, 8))
                venn3([gt_set, tool1_set, tool2_set],
                      set_labels=('Ground Truth', tool_names[0], tool_names[1]), ax=ax)
                ax.set_title('Three-Way Precursor Overlap', fontsize=14, fontweight='bold')
                plt.tight_layout()

                path = os.path.join(output_dir, "venn_three_way.png")
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.close()
                plot_paths['venn_three_way'] = path

            else:
                # 3+ tools comparison: show 3-way Venn for tools (no GT)
                # and separate GT vs all-tools combined comparison
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                # 3-way Venn: Tools only (without GT for clarity)
                ax = axes[0]
                venn3([tool1_set, tool2_set, tool3_set],
                      set_labels=(tool_names[0], tool_names[1], tool_names[2]), ax=ax)
                ax.set_title('Tool Comparison (3-Way)', fontsize=12, fontweight='bold')

                # Union of all tools vs GT
                all_tools_union = tool1_set | tool2_set | tool3_set
                ax = axes[1]
                venn2([gt_set, all_tools_union], set_labels=('Ground Truth', 'All Tools'), ax=ax)
                ax.set_title('Ground Truth vs All Tools Combined', fontsize=12, fontweight='bold')

                plt.suptitle('Precursor Overlap Analysis', fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()

                path = os.path.join(output_dir, "venn_overlap.png")
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.close()
                plot_paths['venn_overlap'] = path

                # Individual GT vs each tool bar chart (since 4-way Venn isn't practical)
                fig, axes = plt.subplots(1, len(tool_names), figsize=(5*len(tool_names), 5))
                if len(tool_names) == 3:
                    axes = axes  # Already array
                else:
                    axes = [axes]  # Make it iterable for single tool edge case

                for idx, tool_name in enumerate(tool_names):
                    ax = axes[idx]
                    tool_set = tool_sets[tool_name]
                    overlap = len(gt_set & tool_set)
                    gt_only = len(gt_set - tool_set)
                    tool_only = len(tool_set - gt_set)

                    categories = ['GT Only', 'Both', f'{tool_name} Only']
                    values = [gt_only, overlap, tool_only]
                    bar_colors = ['#3498db', '#27ae60', colors.get(tool_name, '#e74c3c')]
                    ax.bar(categories, values, color=bar_colors, edgecolor='black')
                    ax.set_ylabel('Precursors')
                    ax.set_title(f'GT vs {tool_name}', fontsize=11, fontweight='bold')
                    for i, v in enumerate(values):
                        ax.text(i, v + max(values)*0.02, f'{v:,}', ha='center', fontsize=9)

                plt.suptitle('Ground Truth vs Individual Tools', fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()

                path = os.path.join(output_dir, "gt_vs_tools.png")
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.close()
                plot_paths['gt_vs_tools'] = path

        else:
            # Fallback: Use grouped bar charts for overlap visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Calculate overlaps
            gt_tool1_overlap = len(gt_set & tool1_set)
            gt_tool1_only_gt = len(gt_set - tool1_set)
            gt_tool1_only_tool = len(tool1_set - gt_set)

            gt_tool2_overlap = len(gt_set & tool2_set)
            gt_tool2_only_gt = len(gt_set - tool2_set)
            gt_tool2_only_tool = len(tool2_set - gt_set)

            tool_overlap = len(tool1_set & tool2_set)
            tool1_only = len(tool1_set - tool2_set)
            tool2_only = len(tool2_set - tool1_set)

            # GT vs Tool 1
            ax = axes[0]
            categories = ['GT Only', 'Both', f'{tool_names[0]} Only']
            values = [gt_tool1_only_gt, gt_tool1_overlap, gt_tool1_only_tool]
            bar_colors = ['#3498db', '#27ae60', colors.get(tool_names[0], '#e74c3c')]
            ax.bar(categories, values, color=bar_colors, edgecolor='black')
            ax.set_ylabel('Precursors')
            ax.set_title(f'GT vs {tool_names[0]}', fontsize=12, fontweight='bold')
            for i, v in enumerate(values):
                ax.text(i, v + max(values)*0.02, f'{v:,}', ha='center', fontsize=9)

            # GT vs Tool 2
            ax = axes[1]
            values = [gt_tool2_only_gt, gt_tool2_overlap, gt_tool2_only_tool]
            categories = ['GT Only', 'Both', f'{tool_names[1]} Only']
            bar_colors = ['#3498db', '#27ae60', colors.get(tool_names[1], '#e74c3c')]
            ax.bar(categories, values, color=bar_colors, edgecolor='black')
            ax.set_ylabel('Precursors')
            ax.set_title(f'GT vs {tool_names[1]}', fontsize=12, fontweight='bold')
            for i, v in enumerate(values):
                ax.text(i, v + max(values)*0.02, f'{v:,}', ha='center', fontsize=9)

            # Tool 1 vs Tool 2
            ax = axes[2]
            categories = [f'{tool_names[0]} Only', 'Both', f'{tool_names[1]} Only']
            values = [tool1_only, tool_overlap, tool2_only]
            bar_colors = [colors.get(tool_names[0], '#3498db'), '#27ae60', colors.get(tool_names[1], '#e74c3c')]
            ax.bar(categories, values, color=bar_colors, edgecolor='black')
            ax.set_ylabel('Precursors')
            ax.set_title(f'{tool_names[0]} vs {tool_names[1]}', fontsize=12, fontweight='bold')
            for i, v in enumerate(values):
                ax.text(i, v + max(values)*0.02, f'{v:,}', ha='center', fontsize=9)

            plt.suptitle('Precursor Overlap Analysis', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()

            path = os.path.join(output_dir, "overlap_bars.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['overlap_bars'] = path

    # 4. Correlation comparison bar chart
    if result.rt_correlations:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # RT correlations
        ax = axes[0]
        names = list(result.rt_correlations.keys())
        rt_vals = [result.rt_correlations[n] for n in names]
        bar_colors = [colors.get(n, '#95a5a6') for n in names]

        bars = ax.bar(names, rt_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Pearson R', fontsize=12)
        ax.set_title('RT Correlation with Ground Truth', fontsize=12, fontweight='bold')
        ax.set_ylim(0.9, 1.0)

        for bar, val in zip(bars, rt_vals):
            ax.annotate(f'{val:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # IM correlations
        ax = axes[1]
        if result.im_correlations:
            names = list(result.im_correlations.keys())
            im_vals = [result.im_correlations[n] for n in names]
            bar_colors = [colors.get(n, '#95a5a6') for n in names]

            bars = ax.bar(names, im_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
            ax.set_ylabel('Pearson R', fontsize=12)
            ax.set_title('IM Correlation with Ground Truth', fontsize=12, fontweight='bold')
            ax.set_ylim(0.9, 1.0)

            for bar, val in zip(bars, im_vals):
                ax.annotate(f'{val:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()

        path = os.path.join(output_dir, "correlation_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['correlation_comparison'] = path

    # 5. Stacked bar: True positives, false positives, false negatives
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(tool_names))
    width = 0.6

    true_pos = [result.gt_overlaps[n].both for n in tool_names]
    false_pos = [result.gt_overlaps[n].tool_only for n in tool_names]
    false_neg = [result.gt_overlaps[n].gt_only for n in tool_names]

    ax.bar(x, true_pos, width, label='True Positives', color='#27ae60')
    ax.bar(x, false_pos, width, bottom=true_pos, label='False Positives', color='#e74c3c')
    ax.bar(x, false_neg, width, bottom=[tp + fp for tp, fp in zip(true_pos, false_pos)],
           label='False Negatives', color='#95a5a6')

    ax.set_ylabel('Number of Precursors', fontsize=12)
    ax.set_title('Identification Breakdown by Tool', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tool_names, fontsize=11)
    ax.legend(loc='upper right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(output_dir, "identification_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['identification_breakdown'] = path

    # 6. Summary figure
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # ID counts
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(tool_names) + 1)
    counts = [result.ground_truth_precursors] + [result.tool_results[n].num_precursors for n in tool_names]
    bar_colors = [colors.get('Ground Truth', '#95a5a6')] + [colors.get(n, '#95a5a6') for n in tool_names]
    ax1.bar(x, counts, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['GT'] + [n[:6] for n in tool_names], fontsize=9)
    ax1.set_title('Precursor Counts', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count')

    # ID rates
    ax2 = fig.add_subplot(gs[0, 1])
    rates = [result.gt_overlaps[n].identification_rate * 100 for n in tool_names]
    bar_colors = [colors.get(n, '#95a5a6') for n in tool_names]
    ax2.bar(tool_names, rates, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=30, color='red', linestyle='--', linewidth=1)
    ax2.set_title('ID Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 50)

    # Correlations
    ax3 = fig.add_subplot(gs[0, 2])
    if result.rt_correlations:
        names = list(result.rt_correlations.keys())
        rt_vals = [result.rt_correlations[n] for n in names]
        bar_colors = [colors.get(n, '#95a5a6') for n in names]
        ax3.bar(names, rt_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax3.set_title('RT Correlation', fontsize=11, fontweight='bold')
        ax3.set_ylim(0.95, 1.0)

    # Breakdown
    ax4 = fig.add_subplot(gs[1, :2])
    x = np.arange(len(tool_names))
    width = 0.6
    true_pos = [result.gt_overlaps[n].both for n in tool_names]
    false_pos = [result.gt_overlaps[n].tool_only for n in tool_names]
    false_neg = [result.gt_overlaps[n].gt_only for n in tool_names]
    ax4.bar(x, true_pos, width, label='True Pos', color='#27ae60')
    ax4.bar(x, false_pos, width, bottom=true_pos, label='False Pos', color='#e74c3c')
    ax4.bar(x, false_neg, width, bottom=[tp + fp for tp, fp in zip(true_pos, false_pos)],
            label='False Neg', color='#95a5a6')
    ax4.set_xticks(x)
    ax4.set_xticklabels(tool_names)
    ax4.set_title('Identification Breakdown', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)

    # Summary text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    summary_text = f"""SUMMARY

Ground Truth: {result.ground_truth_precursors:,} precursors

"""
    for name in tool_names:
        overlap = result.gt_overlaps[name]
        summary_text += f"{name}:\n"
        summary_text += f"  Identified: {overlap.both:,} ({overlap.identification_rate:.1%})\n"
        summary_text += f"  Precision: {overlap.precision:.1%}\n\n"

    if len(tool_names) == 2:
        pw = list(result.pairwise.values())[0]
        summary_text += f"Tool Agreement:\n"
        summary_text += f"  Common: {pw.both:,}\n"
        summary_text += f"  Jaccard: {pw.jaccard_index:.3f}\n"

    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.suptitle('TIMSIM Multi-Tool Comparison Summary', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    path = os.path.join(output_dir, "comparison_summary.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['comparison_summary'] = path

    # 7. Intensity-dependent comparison (line plot)
    if result.intensity_breakdown:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data for each tool
        bins = [b.bin_index for b in result.intensity_breakdown]
        intensity_labels = [b.intensity_range for b in result.intensity_breakdown]

        for tool_name in tool_names:
            id_rates = []
            for b in result.intensity_breakdown:
                tool_metrics = b.metrics_per_tool.get(tool_name, {})
                id_rates.append(tool_metrics.get("id_rate", 0) * 100)

            color = colors.get(tool_name, '#95a5a6')
            ax.plot(bins, id_rates, marker='o', linewidth=2, markersize=8,
                    label=tool_name, color=color)

        ax.set_xlabel('Intensity Bin', fontsize=12)
        ax.set_ylabel('Identification Rate (%)', fontsize=12)
        ax.set_title('Intensity-Dependent Identification Rates', fontsize=14, fontweight='bold')
        ax.set_xticks(bins)
        ax.set_xticklabels([l.split('-')[0] for l in intensity_labels], rotation=45, ha='right', fontsize=9)
        ax.legend(loc='lower right', fontsize=10)
        ax.axhline(y=30, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
        ax.set_ylim(0, max(100, ax.get_ylim()[1]))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        path = os.path.join(output_dir, "intensity_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['intensity_comparison'] = path

    # 8. Charge-dependent comparison (grouped bar chart)
    if result.charge_breakdown:
        fig, ax = plt.subplots(figsize=(10, 6))

        charges = [c.charge for c in result.charge_breakdown]
        x = np.arange(len(charges))
        width = 0.8 / len(tool_names)

        for i, tool_name in enumerate(tool_names):
            id_rates = []
            for c in result.charge_breakdown:
                tool_metrics = c.metrics_per_tool.get(tool_name, {})
                id_rates.append(tool_metrics.get("id_rate", 0) * 100)

            color = colors.get(tool_name, '#95a5a6')
            offset = (i - len(tool_names)/2 + 0.5) * width
            bars = ax.bar(x + offset, id_rates, width, label=tool_name, color=color, edgecolor='black', linewidth=0.5)

            # Add value labels
            for bar, rate in zip(bars, id_rates):
                if rate > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{rate:.0f}%', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Charge State', fontsize=12)
        ax.set_ylabel('Identification Rate (%)', fontsize=12)
        ax.set_title('Charge-Dependent Identification Rates', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c}+' for c in charges], fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.axhline(y=30, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylim(0, 100)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        path = os.path.join(output_dir, "charge_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['charge_comparison'] = path

    logger.info(f"Generated {len(plot_paths)} comparison plots in {output_dir}")
    return plot_paths


def _find_fragpipe_file(output_dir: str, filename: str) -> Optional[str]:
    """
    Find a FragPipe output file in the directory or subdirectories.

    FragPipe DDA puts results in experiment subdirectories, while DIA
    puts them at the root. This searches both locations.

    Args:
        output_dir: FragPipe output directory.
        filename: Name of the file to find (e.g., "psm.tsv").

    Returns:
        Path to the file if found, None otherwise.
    """
    # First check root directory
    root_path = os.path.join(output_dir, filename)
    if os.path.exists(root_path):
        return root_path

    # Check for combined_* variant at root (DDA combined output)
    combined_name = f"combined_{filename}"
    combined_path = os.path.join(output_dir, combined_name)
    if os.path.exists(combined_path):
        return combined_path

    # Search in subdirectories (DDA experiment directories)
    for entry in os.listdir(output_dir):
        subdir = os.path.join(output_dir, entry)
        if os.path.isdir(subdir) and not entry.startswith('.'):
            sub_path = os.path.join(subdir, filename)
            if os.path.exists(sub_path):
                return sub_path

    return None


def run_comparison(
    database_path: str,
    diann_report_path: Optional[str] = None,
    fragpipe_output_dir: Optional[str] = None,
    sage_results_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    generate_plots: bool = True,
    tool_versions: Optional[Dict[str, str]] = None,
    test_metadata: Optional[Dict[str, str]] = None,
) -> ComparisonResult:
    """
    Run a complete comparison between tools.

    Args:
        database_path: Path to simulation synthetic_data.db.
        diann_report_path: Path to DIA-NN report.tsv or .parquet file.
        fragpipe_output_dir: Path to FragPipe output directory.
        sage_results_path: Path to Sage results.sage.tsv file.
        output_dir: Optional directory to save reports.
        generate_plots: Whether to generate comparison plots.
        tool_versions: Dictionary mapping tool names to version strings
            (e.g., {"DIA-NN": "2.3.1", "FragPipe": "24.0", "Sage": "0.15.0"}).
        test_metadata: Dictionary with test metadata for HTML report:
            - test_id: Test identifier (e.g., "IT-DIA-HELA")
            - acquisition_type: "DIA" or "DDA"
            - sample_type: "hela", "hye", "phospho", etc.
            - description: Human-readable description

    Returns:
        ComparisonResult with all statistics.
    """
    # Load ground truth
    logger.info("Loading ground truth from simulation database...")
    ground_truth_df = load_ground_truth(database_path)

    # Load tool results
    tool_results: Dict[str, pd.DataFrame] = {}

    if diann_report_path and os.path.exists(diann_report_path):
        logger.info(f"Loading DIA-NN results from {diann_report_path}...")
        tool_results["DIA-NN"] = parse_diann_report(diann_report_path)

    if fragpipe_output_dir and os.path.exists(fragpipe_output_dir):
        # Search for PSM file - may be at root (DIA) or in subdirectory (DDA)
        psm_path = _find_fragpipe_file(fragpipe_output_dir, "psm.tsv")
        if psm_path:
            logger.info(f"Loading FragPipe results from {fragpipe_output_dir}...")
            tool_results["FragPipe"] = parse_fragpipe_combined(
                psm_path=psm_path,
                peptide_path=_find_fragpipe_file(fragpipe_output_dir, "peptide.tsv"),
                protein_path=_find_fragpipe_file(fragpipe_output_dir, "protein.tsv"),
                ion_path=_find_fragpipe_file(fragpipe_output_dir, "ion.tsv"),
            )

    if sage_results_path and os.path.exists(sage_results_path):
        logger.info(f"Loading Sage results from {sage_results_path}...")
        tool_results["Sage"] = parse_sage_results(sage_results_path)

    if not tool_results:
        raise ValueError("No tool results found. Provide at least one of diann_report_path or fragpipe_output_dir.")

    # Run comparison
    logger.info("Comparing tool results...")
    result = compare_tools(ground_truth_df, tool_results, database_path=database_path)

    # Add tool versions to result
    result.tool_versions = tool_versions or {}

    # Generate and save report
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        report_text = generate_comparison_text_report(result)
        report_path = os.path.join(output_dir, "tool_comparison_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Comparison report saved to {report_path}")

        # Generate JSON report
        import json
        from datetime import datetime
        json_report = {
            "metadata": {
                "tool": "timsim-compare",
                "version": "0.4.0",
                "timestamp": datetime.now().isoformat(),
                "paths": {
                    "database": database_path,
                    "diann_report": diann_report_path,
                    "fragpipe_output": fragpipe_output_dir,
                    "sage_results": sage_results_path,
                },
                "tool_versions": result.tool_versions or {},
            },
            "ground_truth": {
                "num_precursors": result.ground_truth_precursors,
                "num_peptides": result.ground_truth_peptides,
            },
            "tool_results": {
                name: {
                    "num_psms": tool.num_psms,
                    "num_peptides": tool.num_peptides,
                    "num_precursors": tool.num_precursors,
                    "identification_rate": result.gt_overlaps[name].identification_rate,
                    "precision": result.gt_overlaps[name].precision,
                    "true_positives": result.gt_overlaps[name].both,
                    "false_positives": result.gt_overlaps[name].tool_only,
                    "false_negatives": result.gt_overlaps[name].gt_only,
                }
                for name, tool in result.tool_results.items()
            },
            "pairwise_comparisons": {
                key: {
                    "tool1": comp.tool1_name,
                    "tool2": comp.tool2_name,
                    "common": comp.both,
                    "tool1_only": comp.tool1_only,
                    "tool2_only": comp.tool2_only,
                    "jaccard_index": comp.jaccard_index,
                }
                for key, comp in result.pairwise.items()
            },
            "correlations": {
                "rt": result.rt_correlations,
                "im": result.im_correlations,
            },
            "coverage": {
                "common_to_all_tools": result.common_to_all,
                "unique_to_ground_truth": result.unique_to_gt,
            },
            "intensity_breakdown": [
                {
                    "bin": bin_m.bin_index,
                    "intensity_range": bin_m.intensity_range,
                    "log_range": bin_m.log_range,
                    "ground_truth": bin_m.ground_truth_count,
                    "per_tool": bin_m.metrics_per_tool,
                }
                for bin_m in (result.intensity_breakdown or [])
            ],
            "charge_breakdown": [
                {
                    "charge": charge_m.charge,
                    "ground_truth": charge_m.ground_truth_count,
                    "per_tool": charge_m.metrics_per_tool,
                }
                for charge_m in (result.charge_breakdown or [])
            ],
            "species_breakdown": {
                "expected_ratios": result.species_breakdown.expected_ratios,
                "ground_truth_counts": result.species_breakdown.ground_truth_counts,
                "observed_ratios_per_tool": result.species_breakdown.observed_ratios_per_tool,
                "identified_counts_per_tool": result.species_breakdown.identified_counts_per_tool,
                "ratio_errors_per_tool": result.species_breakdown.ratio_errors_per_tool,
                "max_ratio_error_per_tool": result.species_breakdown.max_ratio_error_per_tool,
            } if result.species_breakdown else None,
            "ptm_metrics": {
                "ground_truth_phosphopeptides": result.ptm_metrics.ground_truth_phosphopeptides,
                "identified_phosphopeptides_per_tool": result.ptm_metrics.identified_phosphopeptides_per_tool,
                "correctly_localized_per_tool": result.ptm_metrics.correctly_localized_per_tool,
                "site_accuracy_per_tool": result.ptm_metrics.site_accuracy_per_tool,
            } if result.ptm_metrics else None,
        }

        json_path = os.path.join(output_dir, "tool_comparison_report.json")
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        logger.info(f"JSON report saved to {json_path}")

        # Generate plots
        if generate_plots:
            logger.info("Generating comparison plots...")
            try:
                plot_dir = os.path.join(output_dir, "plots")
                plot_paths = generate_comparison_plots(result, ground_truth_df, plot_dir)
                logger.info(f"Comparison plots saved to {plot_dir}")
            except ImportError as e:
                logger.warning(f"Could not generate comparison plots (missing matplotlib_venn?): {e}")
            except Exception as e:
                logger.warning(f"Error generating comparison plots: {e}")

            # Generate per-tool validation plots
            try:
                from .plots import generate_tool_validation_plots, generate_tool_summary_grid

                for tool_name, tool_df in tool_results.items():
                    logger.info(f"Generating {tool_name} validation plots...")
                    try:
                        # Get version for this tool
                        tool_version = result.tool_versions.get(tool_name) if result.tool_versions else None

                        # Generate individual plots
                        tool_plot_dir = os.path.join(output_dir, f"{tool_name}_plots")
                        generate_tool_validation_plots(
                            ground_truth_df=ground_truth_df,
                            tool_df=tool_df,
                            tool_name=tool_name,
                            output_dir=output_dir,
                            tool_version=tool_version,
                        )

                        # Generate summary grid
                        summary_grid_path = os.path.join(tool_plot_dir, "summary_grid.png")
                        generate_tool_summary_grid(
                            tool_name=tool_name,
                            ground_truth_df=ground_truth_df,
                            tool_df=tool_df,
                            output_path=summary_grid_path,
                            tool_version=tool_version,
                        )
                        logger.info(f"{tool_name} plots saved to {tool_plot_dir}")
                    except Exception as e:
                        logger.warning(f"Error generating {tool_name} plots: {e}")
            except ImportError as e:
                logger.warning(f"Could not import plotting functions: {e}")

            # Generate HTML report
            try:
                from .html_report import generate_html_report
                html_path = generate_html_report(
                    result=result,
                    output_dir=output_dir,
                    test_name=test_metadata.get("test_id", "TIMSIM Validation") if test_metadata else "TIMSIM Validation",
                    database_path=database_path,
                    diann_report_path=diann_report_path,
                    fragpipe_output_dir=fragpipe_output_dir,
                    test_metadata=test_metadata,
                )
                logger.info(f"HTML report saved to {html_path}")
            except Exception as e:
                logger.warning(f"Error generating HTML report: {e}")

        # Print report
        print(report_text)

    return result


def main_compare():
    """CLI entry point for tool comparison."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Compare DIA-NN and FragPipe results against simulation ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare both tools
  timsim-compare --database /path/to/synthetic_data.db \\
                 --diann /path/to/diann/report.tsv \\
                 --fragpipe /path/to/fragpipe/output \\
                 --output /path/to/comparison

  # Compare only DIA-NN
  timsim-compare --database /path/to/synthetic_data.db \\
                 --diann /path/to/diann/report.tsv \\
                 --output /path/to/comparison
""",
    )

    parser.add_argument(
        "--database", "-d",
        type=str,
        required=True,
        help="Path to simulation synthetic_data.db",
    )
    parser.add_argument(
        "--diann",
        type=str,
        default=None,
        help="Path to DIA-NN report.tsv or report.parquet file",
    )
    parser.add_argument(
        "--fragpipe",
        type=str,
        default=None,
        help="Path to FragPipe output directory (containing psm.tsv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./tool_comparison",
        help="Output directory for comparison results (default: ./tool_comparison)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Validate inputs
    if not os.path.exists(args.database):
        print(f"Error: Database not found: {args.database}", file=sys.stderr)
        return 1

    if not args.diann and not args.fragpipe:
        print("Error: Provide at least one of --diann or --fragpipe", file=sys.stderr)
        return 1

    try:
        run_comparison(
            database_path=args.database,
            diann_report_path=args.diann,
            fragpipe_output_dir=args.fragpipe,
            output_dir=args.output,
            generate_plots=not args.no_plots,
        )
        return 0
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main_compare())
