"""
Visualization module for timsim-validate.

Generates plots for validation reports comparing simulated ground truth
with DiaNN identification results.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

# Color scheme
COLOR_SIMULATED = "#1f77b4"  # Blue
COLOR_IDENTIFIED = "#2ca02c"  # Green
COLOR_UNIDENTIFIED = "#d62728"  # Red
COLOR_FALSE_POSITIVE = "#ff7f0e"  # Orange


@dataclass
class PlotPaths:
    """Paths to generated plot files."""
    summary_plot: Optional[str] = None
    rt_correlation: Optional[str] = None
    im_correlation: Optional[str] = None
    intensity_histogram: Optional[str] = None
    venn_diagram: Optional[str] = None
    quant_correlation: Optional[str] = None


def _setup_style():
    """Setup matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 150
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.labelsize'] = 10


def plot_rt_correlation(
    ax: plt.Axes,
    matched_df: pd.DataFrame,
    rt_sim_col: str = "rt_sim",
    rt_obs_col: str = "rt_obs",
    title: str = "Retention Time Correlation",
    sim_in_seconds: bool = True,
    obs_in_minutes: bool = True,
) -> Dict[str, float]:
    """
    Plot retention time correlation between simulated and observed values.

    Args:
        ax: Matplotlib axes to plot on.
        matched_df: DataFrame with matched peptides containing RT columns.
        rt_sim_col: Column name for simulated RT.
        rt_obs_col: Column name for observed RT.
        title: Plot title.
        sim_in_seconds: If True, simulated RT is in seconds (will convert to minutes).
        obs_in_minutes: If True, observed RT is already in minutes (no conversion).

    Returns:
        Dictionary with correlation statistics.
    """
    if matched_df.empty or rt_sim_col not in matched_df.columns or rt_obs_col not in matched_df.columns:
        ax.text(0.5, 0.5, "No RT data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return {"r": np.nan, "r2": np.nan, "mae": np.nan, "n": 0}

    rt_sim = matched_df[rt_sim_col].values
    rt_obs = matched_df[rt_obs_col].values

    # Remove NaN values
    mask = ~(np.isnan(rt_sim) | np.isnan(rt_obs))
    rt_sim = rt_sim[mask]
    rt_obs = rt_obs[mask]

    if len(rt_sim) < 2:
        ax.text(0.5, 0.5, "Insufficient RT data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return {"r": np.nan, "r2": np.nan, "mae": np.nan, "n": len(rt_sim)}

    # Convert to common unit (minutes)
    # Ground truth (simulated) is in seconds, DiaNN observed is in minutes
    if sim_in_seconds:
        rt_sim = rt_sim / 60.0
    if not obs_in_minutes:
        rt_obs = rt_obs / 60.0
    unit = "min"

    # Calculate statistics
    r = np.corrcoef(rt_sim, rt_obs)[0, 1]
    r2 = r ** 2
    mae = np.mean(np.abs(rt_sim - rt_obs))
    n = len(rt_sim)

    # Plot hexbin
    hb = ax.hexbin(
        rt_sim, rt_obs,
        gridsize=50,
        cmap="viridis",
        norm=mpl.colors.LogNorm(),
        mincnt=1,
        linewidths=0,
    )

    # Add diagonal line
    lims = [
        min(rt_sim.min(), rt_obs.min()),
        max(rt_sim.max(), rt_obs.max()),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=1, label='y=x')

    # Labels and title
    ax.set_xlabel(f"True RT ({unit})")
    ax.set_ylabel(f"Observed RT ({unit})")
    ax.set_title(title)

    # Stats annotation
    ax.text(
        0.05, 0.95,
        f"R = {r:.3f}\nR² = {r2:.3f}\nMAE = {mae:.2f} {unit}\nN = {n}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    return {"r": r, "r2": r2, "mae": mae, "n": n}


def plot_im_correlation(
    ax: plt.Axes,
    matched_df: pd.DataFrame,
    im_sim_col: str = "im_sim",
    im_obs_col: str = "im_obs",
    title: str = "Ion Mobility Correlation",
) -> Dict[str, float]:
    """
    Plot ion mobility correlation between simulated and observed values.

    Args:
        ax: Matplotlib axes to plot on.
        matched_df: DataFrame with matched peptides containing IM columns.
        im_sim_col: Column name for simulated IM (1/K0).
        im_obs_col: Column name for observed IM (1/K0).
        title: Plot title.

    Returns:
        Dictionary with correlation statistics.
    """
    if matched_df.empty or im_sim_col not in matched_df.columns or im_obs_col not in matched_df.columns:
        ax.text(0.5, 0.5, "No IM data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return {"r": np.nan, "r2": np.nan, "mae": np.nan, "n": 0}

    im_sim = matched_df[im_sim_col].values
    im_obs = matched_df[im_obs_col].values

    # Remove NaN values
    mask = ~(np.isnan(im_sim) | np.isnan(im_obs))
    im_sim = im_sim[mask]
    im_obs = im_obs[mask]

    if len(im_sim) < 2:
        ax.text(0.5, 0.5, "Insufficient IM data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return {"r": np.nan, "r2": np.nan, "mae": np.nan, "n": len(im_sim)}

    # Calculate statistics
    r = np.corrcoef(im_sim, im_obs)[0, 1]
    r2 = r ** 2
    mae = np.mean(np.abs(im_sim - im_obs))
    n = len(im_sim)

    # Plot hexbin
    hb = ax.hexbin(
        im_sim, im_obs,
        gridsize=50,
        cmap="viridis",
        norm=mpl.colors.LogNorm(),
        mincnt=1,
        linewidths=0,
    )

    # Add diagonal line
    lims = [
        min(im_sim.min(), im_obs.min()),
        max(im_sim.max(), im_obs.max()),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=1, label='y=x')

    # Labels and title
    ax.set_xlabel("True 1/K0 (Vs/cm²)")
    ax.set_ylabel("Observed 1/K0 (Vs/cm²)")
    ax.set_title(title)

    # Stats annotation
    ax.text(
        0.05, 0.95,
        f"R = {r:.3f}\nR² = {r2:.3f}\nMAE = {mae:.4f}\nN = {n}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    return {"r": r, "r2": r2, "mae": mae, "n": n}


def plot_quant_correlation(
    ax: plt.Axes,
    matched_df: pd.DataFrame,
    intensity_sim_col: str = "intensity_sim",
    intensity_obs_col: str = "intensity_obs",
    title: str = "Quantification Correlation",
) -> Dict[str, float]:
    """
    Plot intensity/quantification correlation in log space.

    Args:
        ax: Matplotlib axes to plot on.
        matched_df: DataFrame with matched peptides containing intensity columns.
        intensity_sim_col: Column name for simulated intensity.
        intensity_obs_col: Column name for observed intensity.
        title: Plot title.

    Returns:
        Dictionary with correlation statistics.
    """
    if matched_df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return {"r": np.nan, "r2": np.nan, "n": 0}

    # Get intensity columns (try different naming conventions)
    int_sim = None
    int_obs = None

    if intensity_sim_col in matched_df.columns:
        int_sim = matched_df[intensity_sim_col].values
    elif "intensity_true" in matched_df.columns:
        int_sim = matched_df["intensity_true"].values
    elif "intensity_sim" in matched_df.columns:
        int_sim = matched_df["intensity_sim"].values
    elif "intensity" in matched_df.columns:
        int_sim = matched_df["intensity"].values

    if intensity_obs_col in matched_df.columns:
        int_obs = matched_df[intensity_obs_col].values
    elif "intensity_observed" in matched_df.columns:
        int_obs = matched_df["intensity_observed"].values
    elif "intensity_obs" in matched_df.columns:
        int_obs = matched_df["intensity_obs"].values
    elif "Precursor.Quantity" in matched_df.columns:
        int_obs = matched_df["Precursor.Quantity"].values

    if int_sim is None or int_obs is None:
        ax.text(0.5, 0.5, "No intensity data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return {"r": np.nan, "r2": np.nan, "n": 0}

    # Remove NaN and zero values
    mask = ~(np.isnan(int_sim) | np.isnan(int_obs) | (int_sim <= 0) | (int_obs <= 0))
    int_sim = int_sim[mask]
    int_obs = int_obs[mask]

    if len(int_sim) < 2:
        ax.text(0.5, 0.5, "Insufficient intensity data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return {"r": np.nan, "r2": np.nan, "n": len(int_sim)}

    # Log transform
    log_sim = np.log10(int_sim + 1)
    log_obs = np.log10(int_obs + 1)

    # Calculate statistics
    r = np.corrcoef(log_sim, log_obs)[0, 1]
    r2 = r ** 2
    n = len(int_sim)

    # Plot hexbin
    hb = ax.hexbin(
        log_sim, log_obs,
        gridsize=50,
        cmap="inferno",
        norm=mpl.colors.LogNorm(),
        mincnt=1,
        linewidths=0,
    )

    # Add diagonal line
    lims = [
        min(log_sim.min(), log_obs.min()),
        max(log_sim.max(), log_obs.max()),
    ]
    ax.plot(lims, lims, 'w--', alpha=0.75, linewidth=1, label='y=x')

    # Labels and title
    ax.set_xlabel("log10(True Intensity)")
    ax.set_ylabel("log10(Observed Intensity)")
    ax.set_title(title)

    # Stats annotation
    ax.text(
        0.95, 0.05,
        f"R² = {r2:.3f}\nN = {n}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        color="white",
        fontweight="bold",
    )

    return {"r": r, "r2": r2, "n": n}


def plot_venn_identification(
    ax: plt.Axes,
    n_ground_truth: int,
    n_identified: int,
    n_overlap: int,
    n_false_positive: int = 0,
    title: str = "Identification Overlap",
) -> None:
    """
    Plot a simplified Venn-style diagram showing identification overlap.

    Uses bar chart representation since matplotlib-venn may not be available.

    Args:
        ax: Matplotlib axes to plot on.
        n_ground_truth: Total number of ground truth peptides.
        n_identified: Total number of identified peptides.
        n_overlap: Number of true positives (correctly identified).
        n_false_positive: Number of false positives.
        title: Plot title.
    """
    # Calculate metrics
    n_missed = n_ground_truth - n_overlap  # False negatives
    id_rate = n_overlap / n_ground_truth if n_ground_truth > 0 else 0
    fdr = n_false_positive / n_identified if n_identified > 0 else 0

    # Create stacked bar representation
    categories = ['Ground Truth', 'Identified']

    # Ground truth bar: overlap (green) + missed (red)
    # Identified bar: overlap (green) + false positive (orange)

    bar_width = 0.6
    x = np.arange(len(categories))

    # Ground truth bar
    ax.bar(x[0], n_overlap, bar_width, color=COLOR_IDENTIFIED, label=f'True Positives ({n_overlap})')
    ax.bar(x[0], n_missed, bar_width, bottom=n_overlap, color=COLOR_UNIDENTIFIED, label=f'Missed ({n_missed})')

    # Identified bar
    ax.bar(x[1], n_overlap, bar_width, color=COLOR_IDENTIFIED)
    ax.bar(x[1], n_false_positive, bar_width, bottom=n_overlap, color=COLOR_FALSE_POSITIVE, label=f'False Positives ({n_false_positive})')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

    # Add text annotations
    ax.text(
        0.5, 0.02,
        f"ID Rate: {id_rate:.1%} | FDR: {fdr:.1%}",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def plot_intensity_histogram(
    ax: plt.Axes,
    ground_truth_df: pd.DataFrame,
    identified_sequences: set,
    intensity_col: str = "intensity",
    sequence_col: str = "sequence",
    title: str = "Intensity Distribution",
    bins: int = 50,
) -> Dict[str, Any]:
    """
    Plot histogram of intensity distribution for identified vs unidentified peptides.

    Args:
        ax: Matplotlib axes to plot on.
        ground_truth_df: DataFrame with ground truth peptides.
        identified_sequences: Set of identified peptide sequences.
        intensity_col: Column name for intensity values.
        sequence_col: Column name for peptide sequences.
        title: Plot title.
        bins: Number of histogram bins.

    Returns:
        Dictionary with distribution statistics.
    """
    if ground_truth_df.empty or intensity_col not in ground_truth_df.columns:
        ax.text(0.5, 0.5, "No intensity data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return {"median_identified": np.nan, "median_unidentified": np.nan}

    # Aggregate by sequence if needed
    if sequence_col in ground_truth_df.columns:
        agg_df = ground_truth_df.groupby(sequence_col, as_index=False).agg({intensity_col: "sum"})
    else:
        agg_df = ground_truth_df.copy()

    # Split into identified and unidentified
    if sequence_col in agg_df.columns:
        identified_mask = agg_df[sequence_col].isin(identified_sequences)
    else:
        identified_mask = pd.Series([False] * len(agg_df))

    identified_intensity = agg_df.loc[identified_mask, intensity_col].values
    unidentified_intensity = agg_df.loc[~identified_mask, intensity_col].values

    # Log transform
    eps = 1.0
    identified_log = np.log10(identified_intensity + eps) if len(identified_intensity) > 0 else np.array([])
    unidentified_log = np.log10(unidentified_intensity + eps) if len(unidentified_intensity) > 0 else np.array([])

    # Determine bin edges
    all_log = np.concatenate([identified_log, unidentified_log]) if len(identified_log) > 0 or len(unidentified_log) > 0 else np.array([0])
    bin_edges = np.linspace(all_log.min(), all_log.max(), bins + 1)

    # Plot histograms
    if len(unidentified_log) > 0:
        ax.hist(unidentified_log, bins=bin_edges, alpha=0.7, color=COLOR_UNIDENTIFIED,
                label=f'Unidentified (N={len(unidentified_log)})')
    if len(identified_log) > 0:
        ax.hist(identified_log, bins=bin_edges, alpha=0.7, color=COLOR_IDENTIFIED,
                label=f'Identified (N={len(identified_log)})')

    ax.set_xlabel("log10(Intensity)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

    # Calculate medians
    median_id = np.median(identified_log) if len(identified_log) > 0 else np.nan
    median_unid = np.median(unidentified_log) if len(unidentified_log) > 0 else np.nan

    # Add median lines
    if not np.isnan(median_id):
        ax.axvline(median_id, color=COLOR_IDENTIFIED, linestyle='--', linewidth=1.5, alpha=0.8)
    if not np.isnan(median_unid):
        ax.axvline(median_unid, color=COLOR_UNIDENTIFIED, linestyle='--', linewidth=1.5, alpha=0.8)

    return {
        "median_identified": median_id,
        "median_unidentified": median_unid,
        "n_identified": len(identified_log),
        "n_unidentified": len(unidentified_log),
    }


def generate_summary_figure(
    ground_truth_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    metrics: Any,
    output_path: str,
    title: str = "TimSim Validation Summary",
) -> str:
    """
    Generate a multi-panel summary figure for validation results.

    Args:
        ground_truth_df: DataFrame with ground truth peptides.
        matched_df: DataFrame with matched peptides (ground truth joined with DiaNN results).
        metrics: ValidationMetrics object with computed metrics.
        output_path: Path to save the figure.
        title: Figure title.

    Returns:
        Path to the saved figure.
    """
    _setup_style()

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Identification overlap (top-left)
    ax_venn = fig.add_subplot(gs[0, 0])
    if hasattr(metrics, 'identification_rate') and hasattr(metrics, 'num_ground_truth'):
        n_gt = metrics.num_ground_truth
        n_tp = int(metrics.identification_rate * n_gt) if not np.isnan(metrics.identification_rate) else 0
        n_fp = getattr(metrics, 'num_false_positives', 0) or 0
        n_identified = n_tp + n_fp
        plot_venn_identification(
            ax_venn,
            n_ground_truth=n_gt,
            n_identified=n_identified,
            n_overlap=n_tp,
            n_false_positive=n_fp,
            title="A) Identification Overlap"
        )
    else:
        ax_venn.text(0.5, 0.5, "No identification data", ha="center", va="center", transform=ax_venn.transAxes)
        ax_venn.set_title("A) Identification Overlap")

    # Panel B: RT Correlation (top-middle)
    ax_rt = fig.add_subplot(gs[0, 1])
    if not matched_df.empty:
        # Try different column naming conventions
        rt_sim_col = None
        rt_obs_col = None
        for col in ['rt_true', 'rt_sim', 'retention_time', 'rt', 'RT']:
            if col in matched_df.columns:
                rt_sim_col = col
                break
        for col in ['rt_observed', 'rt_obs', 'RT', 'iRT', 'retention_time_obs']:
            if col in matched_df.columns and col != rt_sim_col:
                rt_obs_col = col
                break

        if rt_sim_col and rt_obs_col:
            plot_rt_correlation(ax_rt, matched_df, rt_sim_col, rt_obs_col, title="B) RT Correlation")
        else:
            ax_rt.text(0.5, 0.5, "No RT columns found", ha="center", va="center", transform=ax_rt.transAxes)
            ax_rt.set_title("B) RT Correlation")
    else:
        ax_rt.text(0.5, 0.5, "No matched data", ha="center", va="center", transform=ax_rt.transAxes)
        ax_rt.set_title("B) RT Correlation")

    # Panel C: IM Correlation (top-right)
    ax_im = fig.add_subplot(gs[0, 2])
    if not matched_df.empty:
        im_sim_col = None
        im_obs_col = None
        for col in ['im_true', 'im_sim', 'inv_mobility', 'mobility', 'im', 'IM']:
            if col in matched_df.columns:
                im_sim_col = col
                break
        for col in ['im_observed', 'im_obs', 'IM', 'inv_mobility_obs', 'CCS']:
            if col in matched_df.columns and col != im_sim_col:
                im_obs_col = col
                break

        if im_sim_col and im_obs_col:
            plot_im_correlation(ax_im, matched_df, im_sim_col, im_obs_col, title="C) Ion Mobility Correlation")
        else:
            ax_im.text(0.5, 0.5, "No IM columns found", ha="center", va="center", transform=ax_im.transAxes)
            ax_im.set_title("C) Ion Mobility Correlation")
    else:
        ax_im.text(0.5, 0.5, "No matched data", ha="center", va="center", transform=ax_im.transAxes)
        ax_im.set_title("C) Ion Mobility Correlation")

    # Panel D: Intensity histogram (bottom-left)
    ax_hist = fig.add_subplot(gs[1, 0])
    if not ground_truth_df.empty and not matched_df.empty:
        # Get identified sequences from matched_df
        seq_col = 'sequence' if 'sequence' in matched_df.columns else matched_df.columns[0]
        identified_seqs = set(matched_df[seq_col].unique()) if seq_col in matched_df.columns else set()

        int_col = 'intensity' if 'intensity' in ground_truth_df.columns else None
        seq_col_gt = 'sequence' if 'sequence' in ground_truth_df.columns else None

        if int_col and seq_col_gt:
            plot_intensity_histogram(
                ax_hist, ground_truth_df, identified_seqs,
                intensity_col=int_col, sequence_col=seq_col_gt,
                title="D) Intensity Distribution"
            )
        else:
            ax_hist.text(0.5, 0.5, "No intensity data", ha="center", va="center", transform=ax_hist.transAxes)
            ax_hist.set_title("D) Intensity Distribution")
    else:
        ax_hist.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax_hist.transAxes)
        ax_hist.set_title("D) Intensity Distribution")

    # Panel E: Quantification correlation (bottom-middle)
    ax_quant = fig.add_subplot(gs[1, 1])
    if not matched_df.empty:
        plot_quant_correlation(ax_quant, matched_df, title="E) Quantification Correlation")
    else:
        ax_quant.text(0.5, 0.5, "No matched data", ha="center", va="center", transform=ax_quant.transAxes)
        ax_quant.set_title("E) Quantification Correlation")

    # Panel F: Metrics summary (bottom-right)
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.axis('off')

    # Build metrics summary text
    summary_lines = ["Validation Metrics Summary", "=" * 30, ""]

    if hasattr(metrics, 'identification_rate'):
        id_rate = metrics.identification_rate
        summary_lines.append(f"Identification Rate: {id_rate:.1%}" if not np.isnan(id_rate) else "Identification Rate: N/A")

    if hasattr(metrics, 'rt_correlation'):
        rt_corr = metrics.rt_correlation
        summary_lines.append(f"RT Correlation (R): {rt_corr:.3f}" if not np.isnan(rt_corr) else "RT Correlation: N/A")

    if hasattr(metrics, 'rt_mae_minutes'):
        rt_mae = metrics.rt_mae_minutes
        summary_lines.append(f"RT MAE: {rt_mae:.2f} min" if not np.isnan(rt_mae) else "RT MAE: N/A")

    if hasattr(metrics, 'im_correlation'):
        im_corr = metrics.im_correlation
        summary_lines.append(f"IM Correlation (R): {im_corr:.3f}" if not np.isnan(im_corr) else "IM Correlation: N/A")

    if hasattr(metrics, 'im_mae'):
        im_mae = metrics.im_mae
        summary_lines.append(f"IM MAE: {im_mae:.4f} 1/K0" if not np.isnan(im_mae) else "IM MAE: N/A")

    summary_lines.append("")
    summary_lines.append("=" * 30)

    if hasattr(metrics, 'overall_pass'):
        status = "PASSED" if metrics.overall_pass else "FAILED"
        color = "green" if metrics.overall_pass else "red"
        summary_lines.append(f"Overall: {status}")

    ax_summary.text(
        0.5, 0.5,
        "\n".join(summary_lines),
        transform=ax_summary.transAxes,
        ha="center", va="center",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )
    ax_summary.set_title("F) Summary")

    # Main title
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    logger.info(f"Saved summary figure to {output_path}")
    return output_path


def generate_all_plots(
    ground_truth_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    metrics: Any,
    output_dir: str,
) -> PlotPaths:
    """
    Generate all validation plots and save to output directory.

    Args:
        ground_truth_df: DataFrame with ground truth peptides.
        matched_df: DataFrame with matched peptides.
        metrics: ValidationMetrics object.
        output_dir: Directory to save plots.

    Returns:
        PlotPaths with paths to all generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    _setup_style()
    paths = PlotPaths()

    # Generate summary figure
    summary_path = os.path.join(plots_dir, "validation_summary.png")
    try:
        generate_summary_figure(
            ground_truth_df=ground_truth_df,
            matched_df=matched_df,
            metrics=metrics,
            output_path=summary_path,
        )
        paths.summary_plot = summary_path
    except Exception as e:
        logger.warning(f"Failed to generate summary plot: {e}")

    # Generate individual plots
    # RT Correlation
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        if not matched_df.empty:
            for col in ['rt_true', 'rt_sim', 'retention_time', 'rt']:
                if col in matched_df.columns:
                    rt_sim_col = col
                    break
            else:
                rt_sim_col = None
            for col in ['rt_observed', 'rt_obs', 'RT', 'retention_time_obs']:
                if col in matched_df.columns and col != rt_sim_col:
                    rt_obs_col = col
                    break
            else:
                rt_obs_col = None

            if rt_sim_col and rt_obs_col:
                plot_rt_correlation(ax, matched_df, rt_sim_col, rt_obs_col)
        rt_path = os.path.join(plots_dir, "rt_correlation.png")
        plt.savefig(rt_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        paths.rt_correlation = rt_path
    except Exception as e:
        logger.warning(f"Failed to generate RT correlation plot: {e}")

    # IM Correlation
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        if not matched_df.empty:
            for col in ['im_true', 'im_sim', 'inv_mobility', 'mobility']:
                if col in matched_df.columns:
                    im_sim_col = col
                    break
            else:
                im_sim_col = None
            for col in ['im_observed', 'im_obs', 'IM', 'inv_mobility_obs']:
                if col in matched_df.columns and col != im_sim_col:
                    im_obs_col = col
                    break
            else:
                im_obs_col = None

            if im_sim_col and im_obs_col:
                plot_im_correlation(ax, matched_df, im_sim_col, im_obs_col)
        im_path = os.path.join(plots_dir, "im_correlation.png")
        plt.savefig(im_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        paths.im_correlation = im_path
    except Exception as e:
        logger.warning(f"Failed to generate IM correlation plot: {e}")

    # Intensity histogram
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        if not ground_truth_df.empty and not matched_df.empty:
            seq_col = 'sequence' if 'sequence' in matched_df.columns else None
            if seq_col:
                identified_seqs = set(matched_df[seq_col].unique())
                int_col = 'intensity' if 'intensity' in ground_truth_df.columns else None
                seq_col_gt = 'sequence' if 'sequence' in ground_truth_df.columns else None
                if int_col and seq_col_gt:
                    plot_intensity_histogram(ax, ground_truth_df, identified_seqs, int_col, seq_col_gt)
        hist_path = os.path.join(plots_dir, "intensity_histogram.png")
        plt.savefig(hist_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        paths.intensity_histogram = hist_path
    except Exception as e:
        logger.warning(f"Failed to generate intensity histogram: {e}")

    # Quantification correlation
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        if not matched_df.empty:
            plot_quant_correlation(ax, matched_df)
        quant_path = os.path.join(plots_dir, "quant_correlation.png")
        plt.savefig(quant_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        paths.quant_correlation = quant_path
    except Exception as e:
        logger.warning(f"Failed to generate quantification plot: {e}")

    logger.info(f"Generated validation plots in {plots_dir}")
    return paths
