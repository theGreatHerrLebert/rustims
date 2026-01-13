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


def compare_tools(
    ground_truth_df: pd.DataFrame,
    tool_results: Dict[str, pd.DataFrame],
    normalize: bool = True,
) -> ComparisonResult:
    """
    Compare multiple analysis tools against ground truth.

    Args:
        ground_truth_df: Ground truth DataFrame from simulation.
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
    from matplotlib_venn import venn2, venn3

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
    ax.set_ylim(0, 100)

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
    if len(tool_names) == 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Get precursor sets
        gt_set = set()
        for _, row in ground_truth_df.iterrows():
            key = create_precursor_id(
                normalize_sequence_for_matching(row["sequence"]),
                int(row["charge"])
            )
            gt_set.add(key)

        tool1_set = result.tool_results[tool_names[0]].precursors
        tool2_set = result.tool_results[tool_names[1]].precursors

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

    logger.info(f"Generated {len(plot_paths)} comparison plots in {output_dir}")
    return plot_paths


def run_comparison(
    database_path: str,
    diann_report_path: Optional[str] = None,
    fragpipe_output_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    generate_plots: bool = True,
) -> ComparisonResult:
    """
    Run a complete comparison between tools.

    Args:
        database_path: Path to simulation synthetic_data.db.
        diann_report_path: Path to DIA-NN report.tsv or .parquet file.
        fragpipe_output_dir: Path to FragPipe output directory.
        output_dir: Optional directory to save reports.
        generate_plots: Whether to generate comparison plots.

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
        psm_path = os.path.join(fragpipe_output_dir, "psm.tsv")
        if os.path.exists(psm_path):
            logger.info(f"Loading FragPipe results from {fragpipe_output_dir}...")
            tool_results["FragPipe"] = parse_fragpipe_combined(
                psm_path=psm_path,
                peptide_path=os.path.join(fragpipe_output_dir, "peptide.tsv"),
                protein_path=os.path.join(fragpipe_output_dir, "protein.tsv"),
                ion_path=os.path.join(fragpipe_output_dir, "ion.tsv"),
            )

    if not tool_results:
        raise ValueError("No tool results found. Provide at least one of diann_report_path or fragpipe_output_dir.")

    # Run comparison
    logger.info("Comparing tool results...")
    result = compare_tools(ground_truth_df, tool_results)

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
                },
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
                logger.info(f"Plots saved to {plot_dir}")
            except ImportError as e:
                logger.warning(f"Could not generate plots (missing matplotlib_venn?): {e}")
            except Exception as e:
                logger.warning(f"Error generating plots: {e}")

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
