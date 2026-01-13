"""
HTML report generation for timsim-validate.

Generates self-contained HTML reports with embedded images and metrics.
"""

import base64
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def encode_image_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_status_badge(passed: bool) -> str:
    """Generate HTML for a pass/fail status badge."""
    if passed:
        return '<span class="badge pass">PASS</span>'
    return '<span class="badge fail">FAIL</span>'


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal as percentage string."""
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_correlation(value: float) -> str:
    """Format a correlation coefficient."""
    if value is None or (isinstance(value, float) and value != value):
        return "N/A"
    return f"{value:.4f}"


CSS_STYLES = """
:root {
    --pass-color: #27ae60;
    --fail-color: #e74c3c;
    --diann-color: #3498db;
    --fragpipe-color: #e74c3c;
    --bg-light: #f8f9fa;
    --border-color: #dee2e6;
}

* {
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
    background: #fff;
    color: #333;
}

h1 {
    color: #2c3e50;
    border-bottom: 3px solid var(--diann-color);
    padding-bottom: 10px;
    margin-bottom: 20px;
}

h2 {
    color: #34495e;
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 8px;
    margin-top: 30px;
}

h3 {
    color: #495057;
    margin-top: 20px;
}

.header-info {
    background: var(--bg-light);
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
}

.header-info div {
    padding: 5px;
}

.header-info .label {
    font-weight: 600;
    color: #666;
    font-size: 0.85em;
}

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9em;
    text-transform: uppercase;
}

.badge.pass {
    background: var(--pass-color);
    color: white;
}

.badge.fail {
    background: var(--fail-color);
    color: white;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.tool-card {
    background: var(--bg-light);
    border-radius: 10px;
    padding: 20px;
    border-left: 4px solid #95a5a6;
}

.tool-card.diann {
    border-left-color: var(--diann-color);
}

.tool-card.fragpipe {
    border-left-color: var(--fragpipe-color);
}

.tool-card h3 {
    margin-top: 0;
    margin-bottom: 15px;
}

.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-color);
}

.metric-row:last-child {
    border-bottom: none;
}

.metric-label {
    font-weight: 500;
    color: #555;
}

.metric-value {
    font-weight: 600;
    font-family: 'Monaco', 'Menlo', monospace;
}

.metric-value.good {
    color: var(--pass-color);
}

.metric-value.bad {
    color: var(--fail-color);
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 0.95em;
}

th, td {
    padding: 12px 15px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

th {
    background: var(--bg-light);
    font-weight: 600;
    color: #495057;
}

tr:hover {
    background: rgba(52, 152, 219, 0.05);
}

.plot-section {
    margin: 30px 0;
}

.plot-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.plot-container {
    background: var(--bg-light);
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}

.plot-container img {
    max-width: 100%;
    height: auto;
    border-radius: 5px;
    cursor: pointer;
    transition: transform 0.2s;
}

.plot-container img:hover {
    transform: scale(1.02);
}

.plot-container .caption {
    margin-top: 10px;
    font-size: 0.9em;
    color: #666;
    font-weight: 500;
}

.full-width-plot {
    text-align: center;
    margin: 20px 0;
}

.full-width-plot img {
    max-width: 100%;
    height: auto;
    border-radius: 10px;
}

.collapsible {
    background: var(--bg-light);
    border: none;
    padding: 15px 20px;
    width: 100%;
    text-align: left;
    font-size: 1.1em;
    font-weight: 600;
    cursor: pointer;
    border-radius: 8px;
    margin-top: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.collapsible:hover {
    background: #e9ecef;
}

.collapsible:after {
    content: '+';
    font-size: 1.3em;
    font-weight: bold;
}

.collapsible.active:after {
    content: '-';
}

.collapsible-content {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease-out;
    background: white;
    border-radius: 0 0 8px 8px;
}

.collapsible-content.show {
    max-height: none;
    padding: 20px;
    border: 1px solid var(--border-color);
    border-top: none;
}

.footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 2px solid var(--border-color);
    text-align: center;
    color: #666;
    font-size: 0.9em;
}

/* Lightbox styles */
.lightbox {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.9);
    cursor: pointer;
}

.lightbox img {
    max-width: 95%;
    max-height: 95%;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}

.lightbox-close {
    position: absolute;
    top: 20px;
    right: 30px;
    color: white;
    font-size: 40px;
    font-weight: bold;
    cursor: pointer;
}

@media print {
    .collapsible-content {
        max-height: none !important;
        padding: 20px !important;
    }
    .collapsible:after {
        display: none;
    }
}
"""

JS_SCRIPT = """
function toggleCollapsible(element) {
    element.classList.toggle('active');
    var content = element.nextElementSibling;
    content.classList.toggle('show');
}

function openLightbox(src) {
    var lightbox = document.getElementById('lightbox');
    var lightboxImg = document.getElementById('lightbox-img');
    lightboxImg.src = src;
    lightbox.style.display = 'block';
}

function closeLightbox() {
    document.getElementById('lightbox').style.display = 'none';
}

// Close lightbox on escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeLightbox();
    }
});
"""


def generate_html_report(
    result: Any,  # ComparisonResult
    output_dir: str,
    test_name: str = "TIMSIM Validation",
    database_path: Optional[str] = None,
    diann_report_path: Optional[str] = None,
    fragpipe_output_dir: Optional[str] = None,
    thresholds: Optional[Dict[str, float]] = None,
    test_metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate a self-contained HTML validation report.

    Args:
        result: ComparisonResult from compare_tools().
        output_dir: Directory containing plots and where to save HTML.
        test_name: Name of the test for the report title.
        database_path: Path to simulation database.
        diann_report_path: Path to DIA-NN report.
        fragpipe_output_dir: Path to FragPipe output.
        thresholds: Dictionary of pass/fail thresholds.
        test_metadata: Dictionary with test metadata:
            - test_id: Test identifier (e.g., "IT-DIA-HELA")
            - acquisition_type: "DIA" or "DDA"
            - sample_type: "hela", "hye", "phospho", etc.
            - description: Human-readable description

    Returns:
        Path to the generated HTML file.
    """
    if thresholds is None:
        thresholds = {
            "min_id_rate": 0.30,
            "min_rt_correlation": 0.90,
            "min_im_correlation": 0.90,
        }

    if test_metadata is None:
        test_metadata = {}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tool_names = list(result.tool_results.keys())
    tool_versions = result.tool_versions or {}

    # Extract benchmark metadata
    test_id = test_metadata.get("test_id", test_name)
    acquisition_type = test_metadata.get("acquisition_type", "DIA")
    sample_type = test_metadata.get("sample_type", "unknown")
    description = test_metadata.get("description", "")

    # Create human-readable benchmark type string
    sample_type_labels = {
        "hela": "HeLa Proteome",
        "hye": "HYE Mixed Species (Human/Yeast/E.coli)",
        "phospho": "Phosphoproteomics (PTM Localization)",
    }
    sample_type_label = sample_type_labels.get(sample_type.lower(), sample_type.title())
    benchmark_type_str = f"{acquisition_type}-PASEF {sample_type_label}"

    # Collect all plot paths
    plots_dir = os.path.join(output_dir, "plots")
    comparison_plots = {}
    if os.path.exists(plots_dir):
        for filename in os.listdir(plots_dir):
            if filename.endswith(".png"):
                name = filename.replace(".png", "")
                comparison_plots[name] = os.path.join(plots_dir, filename)

    tool_plots = {}
    for tool_name in tool_names:
        tool_plot_dir = os.path.join(output_dir, f"{tool_name}_plots")
        tool_plots[tool_name] = {}
        if os.path.exists(tool_plot_dir):
            for filename in os.listdir(tool_plot_dir):
                if filename.endswith(".png"):
                    name = filename.replace(".png", "")
                    tool_plots[tool_name][name] = os.path.join(tool_plot_dir, filename)

    # Build HTML
    html_parts = []

    # Document start
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{test_name} - Validation Report</title>
    <style>{CSS_STYLES}</style>
</head>
<body>
""")

    # Lightbox for image zoom
    html_parts.append("""
<div id="lightbox" class="lightbox" onclick="closeLightbox()">
    <span class="lightbox-close">&times;</span>
    <img id="lightbox-img" src="">
</div>
""")

    # Header
    html_parts.append(f"""
<h1>{test_id} - Validation Report</h1>
<p style="font-size: 1.1em; color: #666; margin-top: -10px; margin-bottom: 20px;">
    <strong>{benchmark_type_str}</strong>
    {f' - {description}' if description else ''}
</p>

<div class="header-info">
    <div>
        <div class="label">Test ID</div>
        <div>{test_id}</div>
    </div>
    <div>
        <div class="label">Benchmark Type</div>
        <div>{benchmark_type_str}</div>
    </div>
    <div>
        <div class="label">Generated</div>
        <div>{timestamp}</div>
    </div>
    <div>
        <div class="label">Ground Truth Precursors</div>
        <div>{result.ground_truth_precursors:,}</div>
    </div>
    <div>
        <div class="label">Ground Truth Peptides</div>
        <div>{result.ground_truth_peptides:,}</div>
    </div>
    <div>
        <div class="label">Tools Compared</div>
        <div>{', '.join(f"{t} v{tool_versions.get(t, '?')}" if tool_versions.get(t) else t for t in tool_names)}</div>
    </div>
</div>
""")

    # Summary cards for each tool
    html_parts.append("<h2>Tool Summary</h2>")
    html_parts.append('<div class="summary-grid">')

    for tool_name in tool_names:
        tool_result = result.tool_results[tool_name]
        overlap = result.gt_overlaps[tool_name]
        rt_corr = result.rt_correlations.get(tool_name, float('nan'))
        im_corr = result.im_correlations.get(tool_name, float('nan'))

        # Check thresholds
        id_passed = overlap.identification_rate >= thresholds["min_id_rate"]
        rt_passed = rt_corr >= thresholds["min_rt_correlation"]
        im_passed = im_corr >= thresholds["min_im_correlation"]
        overall_passed = id_passed and rt_passed and im_passed

        card_class = "diann" if "DIA" in tool_name.upper() else "fragpipe"
        display_name = f"{tool_name} v{tool_versions.get(tool_name)}" if tool_versions.get(tool_name) else tool_name

        html_parts.append(f"""
<div class="tool-card {card_class}">
    <h3>{display_name} {get_status_badge(overall_passed)}</h3>
    <div class="metric-row">
        <span class="metric-label">Precursors Identified</span>
        <span class="metric-value">{tool_result.num_precursors:,}</span>
    </div>
    <div class="metric-row">
        <span class="metric-label">ID Rate</span>
        <span class="metric-value {'good' if id_passed else 'bad'}">{format_percentage(overlap.identification_rate)}</span>
    </div>
    <div class="metric-row">
        <span class="metric-label">Precision</span>
        <span class="metric-value good">{format_percentage(overlap.precision)}</span>
    </div>
    <div class="metric-row">
        <span class="metric-label">RT Correlation</span>
        <span class="metric-value {'good' if rt_passed else 'bad'}">{format_correlation(rt_corr)}</span>
    </div>
    <div class="metric-row">
        <span class="metric-label">IM Correlation</span>
        <span class="metric-value {'good' if im_passed else 'bad'}">{format_correlation(im_corr)}</span>
    </div>
    <div class="metric-row">
        <span class="metric-label">True Positives</span>
        <span class="metric-value">{overlap.both:,}</span>
    </div>
    <div class="metric-row">
        <span class="metric-label">False Positives</span>
        <span class="metric-value">{overlap.tool_only:,}</span>
    </div>
</div>
""")

    html_parts.append("</div>")

    # Comparison Summary Plot (full width)
    if "comparison_summary" in comparison_plots:
        img_data = encode_image_base64(comparison_plots["comparison_summary"])
        if img_data:
            html_parts.append("""
<h2>Comparison Overview</h2>
<div class="full-width-plot">
""")
            html_parts.append(f'<img src="data:image/png;base64,{img_data}" alt="Comparison Summary" onclick="openLightbox(this.src)">')
            html_parts.append("</div>")

    # Comparison Plots Section
    html_parts.append("""
<button class="collapsible" onclick="toggleCollapsible(this)">Comparison Plots</button>
<div class="collapsible-content show">
<div class="plot-grid">
""")

    plot_labels = {
        "identification_counts": "Precursor Counts",
        "identification_rates": "Identification Rates",
        "identification_breakdown": "ID Breakdown (TP/FP/FN)",
        "correlation_comparison": "Correlation Comparison",
        "intensity_comparison": "Intensity-Dependent ID",
        "charge_comparison": "Charge-Dependent ID",
        "venn_overlap": "Tool Overlap (Venn)",
        "venn_three_way": "Three-Way Overlap",
        "overlap_bars": "Tool Overlap (Bars)",
    }

    for plot_name, plot_path in comparison_plots.items():
        if plot_name == "comparison_summary":
            continue
        img_data = encode_image_base64(plot_path)
        if img_data:
            label = plot_labels.get(plot_name, plot_name.replace("_", " ").title())
            html_parts.append(f"""
<div class="plot-container">
    <img src="data:image/png;base64,{img_data}" alt="{label}" onclick="openLightbox(this.src)">
    <div class="caption">{label}</div>
</div>
""")

    html_parts.append("</div></div>")

    # Per-tool sections
    for tool_name in tool_names:
        card_class = "diann" if "DIA" in tool_name.upper() else "fragpipe"
        html_parts.append(f"""
<button class="collapsible" onclick="toggleCollapsible(this)">{tool_name} Detailed Plots</button>
<div class="collapsible-content">
""")

        # Summary grid (full width)
        if "summary_grid" in tool_plots.get(tool_name, {}):
            img_data = encode_image_base64(tool_plots[tool_name]["summary_grid"])
            if img_data:
                html_parts.append(f"""
<div class="full-width-plot">
    <img src="data:image/png;base64,{img_data}" alt="{tool_name} Summary Grid" onclick="openLightbox(this.src)">
</div>
""")

        # Other plots in grid
        html_parts.append('<div class="plot-grid">')

        tool_plot_labels = {
            "rt_correlation": "RT Correlation",
            "im_correlation": "Ion Mobility Correlation",
            "intensity_histogram": "Intensity Distribution",
            "quant_correlation": "Quantification Correlation",
        }

        for plot_name, plot_path in tool_plots.get(tool_name, {}).items():
            if plot_name == "summary_grid":
                continue
            img_data = encode_image_base64(plot_path)
            if img_data:
                label = tool_plot_labels.get(plot_name, plot_name.replace("_", " ").title())
                html_parts.append(f"""
<div class="plot-container">
    <img src="data:image/png;base64,{img_data}" alt="{label}" onclick="openLightbox(this.src)">
    <div class="caption">{label}</div>
</div>
""")

        html_parts.append("</div></div>")

    # Detailed metrics tables
    html_parts.append("""
<button class="collapsible" onclick="toggleCollapsible(this)">Detailed Metrics Tables</button>
<div class="collapsible-content">
""")

    # Intensity breakdown table
    if result.intensity_breakdown:
        html_parts.append("<h3>Intensity-Dependent Identification</h3>")
        html_parts.append("<table>")
        header_row = "<tr><th>Bin</th><th>Intensity Range</th><th>Ground Truth</th>"
        for tool_name in tool_names:
            header_row += f"<th>{tool_name}</th>"
        header_row += "</tr>"
        html_parts.append(header_row)

        for bin_data in result.intensity_breakdown:
            row = f"<tr><td>{bin_data.bin_index}</td><td>{bin_data.intensity_range}</td><td>{bin_data.ground_truth_count:,}</td>"
            for tool_name in tool_names:
                tool_metrics = bin_data.metrics_per_tool.get(tool_name, {})
                id_rate = tool_metrics.get("id_rate", 0) * 100
                row += f"<td>{id_rate:.1f}%</td>"
            row += "</tr>"
            html_parts.append(row)

        html_parts.append("</table>")

    # Charge state table
    if result.charge_breakdown:
        html_parts.append("<h3>Charge State Breakdown</h3>")
        html_parts.append("<table>")
        header_row = "<tr><th>Charge</th><th>Ground Truth</th>"
        for tool_name in tool_names:
            header_row += f"<th>{tool_name}</th>"
        header_row += "</tr>"
        html_parts.append(header_row)

        for charge_data in result.charge_breakdown:
            row = f"<tr><td>{charge_data.charge}+</td><td>{charge_data.ground_truth_count:,}</td>"
            for tool_name in tool_names:
                tool_metrics = charge_data.metrics_per_tool.get(tool_name, {})
                id_rate = tool_metrics.get("id_rate", 0) * 100
                row += f"<td>{id_rate:.1f}%</td>"
            row += "</tr>"
            html_parts.append(row)

        html_parts.append("</table>")

    # Pairwise comparisons
    if result.pairwise:
        html_parts.append("<h3>Pairwise Tool Comparisons</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Comparison</th><th>Common</th><th>Tool 1 Only</th><th>Tool 2 Only</th><th>Jaccard Index</th></tr>")

        for key, pw in result.pairwise.items():
            html_parts.append(f"""
<tr>
    <td>{pw.tool1_name} vs {pw.tool2_name}</td>
    <td>{pw.both:,}</td>
    <td>{pw.tool1_only:,}</td>
    <td>{pw.tool2_only:,}</td>
    <td>{pw.jaccard_index:.3f}</td>
</tr>
""")

        html_parts.append("</table>")

    # Species breakdown (HYE benchmarks)
    if result.species_breakdown:
        html_parts.append("<h3>Species Ratio Analysis (HYE)</h3>")
        html_parts.append("<p>Mixed species benchmark comparing observed vs expected species ratios.</p>")

        species_list = list(result.species_breakdown.expected_ratios.keys())

        # Expected vs Observed table
        html_parts.append("<table>")
        header_row = "<tr><th>Species</th><th>Expected</th><th>GT Count</th>"
        for tool_name in tool_names:
            header_row += f"<th>{tool_name} Observed</th><th>{tool_name} Error</th>"
        header_row += "</tr>"
        html_parts.append(header_row)

        for species in species_list:
            expected = result.species_breakdown.expected_ratios.get(species, 0)
            gt_count = result.species_breakdown.ground_truth_counts.get(species, 0)
            row = f"<tr><td>{species}</td><td>{expected:.1%}</td><td>{gt_count:,}</td>"

            for tool_name in tool_names:
                observed = result.species_breakdown.observed_ratios_per_tool.get(tool_name, {}).get(species, 0)
                error = result.species_breakdown.ratio_errors_per_tool.get(tool_name, {}).get(species, 0)
                error_class = "good" if error < 0.10 else "bad" if error > 0.20 else ""
                row += f"<td>{observed:.1%}</td><td class='{error_class}'>{error:.1%}</td>"
            row += "</tr>"
            html_parts.append(row)

        html_parts.append("</table>")

        # Max error summary
        html_parts.append("<h4>Species Ratio Accuracy Summary</h4>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Tool</th><th>Max Ratio Error</th><th>Status</th></tr>")
        for tool_name in tool_names:
            max_error = result.species_breakdown.max_ratio_error_per_tool.get(tool_name, 0)
            status = "PASS" if max_error < 0.20 else "FAIL"
            status_class = "good" if status == "PASS" else "bad"
            html_parts.append(f"<tr><td>{tool_name}</td><td>{max_error:.1%}</td><td class='{status_class}'><strong>{status}</strong></td></tr>")
        html_parts.append("</table>")

    # PTM localization (Phospho benchmarks)
    if result.ptm_metrics:
        html_parts.append("<h3>PTM Site Localization (Phosphoproteomics)</h3>")
        html_parts.append("<p>Phosphorylation site localization accuracy compared to ground truth.</p>")

        html_parts.append("<table>")
        html_parts.append("<tr><th>Tool</th><th>GT Phosphopeptides</th><th>Identified</th><th>Correct Site</th><th>Accuracy</th><th>Status</th></tr>")

        for tool_name in tool_names:
            gt_count = result.ptm_metrics.ground_truth_phosphopeptides
            identified = result.ptm_metrics.identified_phosphopeptides_per_tool.get(tool_name, 0)
            correct = result.ptm_metrics.correctly_localized_per_tool.get(tool_name, 0)
            accuracy = result.ptm_metrics.site_accuracy_per_tool.get(tool_name, 0)
            status = "PASS" if accuracy >= 0.80 else "FAIL"
            status_class = "good" if status == "PASS" else "bad"
            html_parts.append(f"""
<tr>
    <td>{tool_name}</td>
    <td>{gt_count:,}</td>
    <td>{identified:,}</td>
    <td>{correct:,}</td>
    <td class='{status_class}'>{accuracy:.1%}</td>
    <td class='{status_class}'><strong>{status}</strong></td>
</tr>
""")
        html_parts.append("</table>")

    # DDA-specific metrics
    if result.dda_metrics:
        dda = result.dda_metrics
        html_parts.append("<h3>DDA Acquisition Metrics</h3>")
        html_parts.append("<p>Data-Dependent Acquisition (DDA) MS2 selection and identification efficiency.</p>")

        # Acquisition overview
        html_parts.append("<h4>MS2 Acquisition Summary</h4>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Metric</th><th>Value</th><th>Description</th></tr>")
        html_parts.append(f"<tr><td>Total MS2 Events</td><td>{dda.total_ms2_events:,}</td><td>Number of MS2 scans acquired</td></tr>")
        html_parts.append(f"<tr><td>Unique Precursors Selected</td><td>{dda.unique_precursors_selected:,}</td><td>Distinct precursors targeted for fragmentation</td></tr>")
        html_parts.append(f"<tr><td>MS2 Frames</td><td>{dda.ms2_frames:,}</td><td>Number of PASEF frames with MS2 events</td></tr>")
        html_parts.append(f"<tr><td>Precursor Selection Rate</td><td>{dda.precursor_selection_rate:.1%}</td><td>Fraction of available precursors selected for MS2</td></tr>")
        html_parts.append(f"<tr><td>Avg Precursors/Frame</td><td>{dda.avg_precursors_per_frame:.1f}</td><td>Average precursors per MS2 frame (TopN efficiency)</td></tr>")
        html_parts.append(f"<tr><td>Precursor Redundancy</td><td>{dda.precursor_redundancy:.2f}x</td><td>Average times each precursor was selected (&gt;1 = resampling)</td></tr>")
        html_parts.append("</table>")

        # Per-tool identification efficiency
        html_parts.append("<h4>MS2 Identification Efficiency</h4>")
        html_parts.append("<p>How efficiently each tool converts MS2 events into identifications.</p>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Tool</th><th>Precursors ID'd</th><th>MS2 Events</th><th>ID Efficiency</th><th>Description</th></tr>")

        for tool_name in tool_names:
            identified = dda.identified_per_tool.get(tool_name, 0)
            efficiency = dda.ms2_id_efficiency_per_tool.get(tool_name, 0.0)
            eff_class = "good" if efficiency >= 0.30 else ""
            html_parts.append(f"""
<tr>
    <td>{tool_name}</td>
    <td>{identified:,}</td>
    <td>{dda.total_ms2_events:,}</td>
    <td class='{eff_class}'>{efficiency:.1%}</td>
    <td>IDs per MS2 event</td>
</tr>
""")
        html_parts.append("</table>")

    html_parts.append("</div>")

    # Footer
    html_parts.append(f"""
<div class="footer">
    <p>Generated by <strong>timsim-validate</strong> | {timestamp}</p>
    <p>Ground truth from simulation database | Thresholds: ID Rate >= {thresholds['min_id_rate']:.0%},
       RT R >= {thresholds['min_rt_correlation']:.2f}, IM R >= {thresholds['min_im_correlation']:.2f}</p>
</div>

<script>{JS_SCRIPT}</script>
</body>
</html>
""")

    # Write HTML file
    html_content = "\n".join(html_parts)
    html_path = os.path.join(output_dir, "validation_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Generated HTML report: {html_path}")
    return html_path


# CSS for meta report (extends base CSS)
META_CSS_ADDITIONS = """
.dashboard-header {
    text-align: center;
    margin-bottom: 30px;
}

.stats-row {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin: 20px 0;
}

.stat-box {
    background: var(--bg-light);
    border-radius: 10px;
    padding: 20px 40px;
    text-align: center;
    min-width: 150px;
}

.stat-box.pass {
    border-left: 4px solid var(--pass-color);
}

.stat-box.fail {
    border-left: 4px solid var(--fail-color);
}

.stat-box .number {
    font-size: 2.5em;
    font-weight: bold;
    color: #333;
}

.stat-box.pass .number {
    color: var(--pass-color);
}

.stat-box.fail .number {
    color: var(--fail-color);
}

.stat-box .label {
    font-size: 0.9em;
    color: #666;
    margin-top: 5px;
}

.benchmark-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.benchmark-card {
    background: var(--bg-light);
    border-radius: 10px;
    padding: 20px;
    border-left: 4px solid #95a5a6;
    transition: transform 0.2s, box-shadow 0.2s;
}

.benchmark-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.benchmark-card.pass {
    border-left-color: var(--pass-color);
}

.benchmark-card.fail {
    border-left-color: var(--fail-color);
}

.benchmark-card h3 {
    margin: 0 0 10px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.benchmark-card .benchmark-type {
    font-size: 0.85em;
    color: #666;
    margin-bottom: 15px;
}

.benchmark-card .tools-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 15px;
}

.tool-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 15px;
    font-size: 0.85em;
    background: white;
}

.tool-chip.pass {
    border: 1px solid var(--pass-color);
    color: var(--pass-color);
}

.tool-chip.fail {
    border: 1px solid var(--fail-color);
    color: var(--fail-color);
}

.benchmark-card .view-link {
    display: inline-block;
    margin-top: 10px;
    padding: 8px 16px;
    background: var(--diann-color);
    color: white;
    text-decoration: none;
    border-radius: 5px;
    font-size: 0.9em;
    transition: background 0.2s;
}

.benchmark-card .view-link:hover {
    background: #2980b9;
}

.nav-tabs {
    display: flex;
    border-bottom: 2px solid var(--border-color);
    margin-bottom: 20px;
}

.nav-tab {
    padding: 10px 20px;
    cursor: pointer;
    border: none;
    background: none;
    font-size: 1em;
    color: #666;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: color 0.2s, border-color 0.2s;
}

.nav-tab:hover {
    color: var(--diann-color);
}

.nav-tab.active {
    color: var(--diann-color);
    border-bottom-color: var(--diann-color);
    font-weight: 600;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.summary-table {
    width: 100%;
    margin: 20px 0;
}

.summary-table th {
    text-align: left;
    padding: 12px;
    background: var(--bg-light);
}

.summary-table td {
    padding: 12px;
    border-bottom: 1px solid var(--border-color);
}

.summary-table tr:hover {
    background: rgba(52, 152, 219, 0.05);
}
"""

META_JS_ADDITIONS = """
function switchTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));

    // Show selected tab
    document.getElementById(tabId).classList.add('active');
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
}

function filterBenchmarks(status) {
    const cards = document.querySelectorAll('.benchmark-card');
    cards.forEach(card => {
        if (status === 'all') {
            card.style.display = 'block';
        } else if (status === 'pass' && card.classList.contains('pass')) {
            card.style.display = 'block';
        } else if (status === 'fail' && card.classList.contains('fail')) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}
"""


def generate_meta_report(
    benchmark_results: List[Dict[str, Any]],
    output_path: str,
    title: str = "TIMSIM Integration Test Dashboard",
) -> str:
    """
    Generate a meta HTML report that provides an overview of all benchmarks.

    Args:
        benchmark_results: List of dictionaries containing:
            - test_id: Test identifier
            - passed: Overall pass/fail status
            - benchmark_type: Human-readable benchmark type
            - acquisition_type: DIA or DDA
            - sample_type: hela, hye, phospho, etc.
            - report_path: Path to individual HTML report (relative)
            - tool_results: Dict of tool name -> {passed, id_rate, precision, ...}
        output_path: Path to write the meta HTML report.
        title: Title for the dashboard.

    Returns:
        Path to the generated HTML file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate summary statistics
    total = len(benchmark_results)
    passed = sum(1 for r in benchmark_results if r.get("passed", False))
    failed = total - passed

    # Group by sample type
    by_sample_type = {}
    for result in benchmark_results:
        sample_type = result.get("sample_type", "unknown")
        if sample_type not in by_sample_type:
            by_sample_type[sample_type] = []
        by_sample_type[sample_type].append(result)

    # Build HTML
    html_parts = []

    # Document start
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{CSS_STYLES}
{META_CSS_ADDITIONS}
    </style>
</head>
<body>
""")

    # Header
    html_parts.append(f"""
<div class="dashboard-header">
    <h1>{title}</h1>
    <p style="color: #666;">Generated: {timestamp}</p>
</div>

<div class="stats-row">
    <div class="stat-box pass">
        <div class="number">{passed}</div>
        <div class="label">Passed</div>
    </div>
    <div class="stat-box fail">
        <div class="number">{failed}</div>
        <div class="label">Failed</div>
    </div>
    <div class="stat-box">
        <div class="number">{total}</div>
        <div class="label">Total Tests</div>
    </div>
</div>
""")

    # Navigation tabs
    html_parts.append("""
<div class="nav-tabs">
    <button class="nav-tab active" data-tab="cards-view" onclick="switchTab('cards-view')">Card View</button>
    <button class="nav-tab" data-tab="table-view" onclick="switchTab('table-view')">Table View</button>
</div>
""")

    # Card View
    html_parts.append("""
<div id="cards-view" class="tab-content active">
    <div style="margin-bottom: 15px;">
        <strong>Filter:</strong>
        <button onclick="filterBenchmarks('all')" style="margin-left: 10px;">All</button>
        <button onclick="filterBenchmarks('pass')" style="margin-left: 5px;">Passed</button>
        <button onclick="filterBenchmarks('fail')" style="margin-left: 5px;">Failed</button>
    </div>
    <div class="benchmark-grid">
""")

    # Sample type labels
    sample_type_labels = {
        "hela": "HeLa Proteome",
        "hye": "HYE Mixed Species",
        "phospho": "Phosphoproteomics",
    }

    for result in benchmark_results:
        test_id = result.get("test_id", "Unknown")
        is_passed = result.get("passed", False)
        benchmark_type = result.get("benchmark_type", "")
        report_path = result.get("report_path", "")
        tool_results = result.get("tool_results", {})

        card_class = "pass" if is_passed else "fail"

        # Build tool chips
        tool_chips = []
        for tool_name, tool_data in tool_results.items():
            tool_passed = tool_data.get("passed", False)
            chip_class = "pass" if tool_passed else "fail"
            id_rate = tool_data.get("id_rate", 0) * 100
            precision = tool_data.get("precision", 0) * 100
            tool_chips.append(
                f'<span class="tool-chip {chip_class}">{tool_name}: {id_rate:.0f}% ID, {precision:.0f}% Prec</span>'
            )

        html_parts.append(f"""
        <div class="benchmark-card {card_class}">
            <h3>
                {test_id}
                {get_status_badge(is_passed)}
            </h3>
            <div class="benchmark-type">{benchmark_type}</div>
            <div class="tools-summary">
                {''.join(tool_chips)}
            </div>
            <a href="{report_path}" class="view-link">View Full Report →</a>
        </div>
""")

    html_parts.append("""
    </div>
</div>
""")

    # Table View
    html_parts.append("""
<div id="table-view" class="tab-content">
    <table class="summary-table">
        <thead>
            <tr>
                <th>Test ID</th>
                <th>Type</th>
                <th>Status</th>
                <th>Tools</th>
                <th>Best ID Rate</th>
                <th>Best Precision</th>
                <th>Report</th>
            </tr>
        </thead>
        <tbody>
""")

    for result in benchmark_results:
        test_id = result.get("test_id", "Unknown")
        is_passed = result.get("passed", False)
        acquisition_type = result.get("acquisition_type", "DIA")
        sample_type = result.get("sample_type", "unknown")
        report_path = result.get("report_path", "")
        tool_results = result.get("tool_results", {})

        # Get best metrics
        best_id_rate = max((t.get("id_rate", 0) for t in tool_results.values()), default=0)
        best_precision = max((t.get("precision", 0) for t in tool_results.values()), default=0)

        # Tool status summary
        tool_summary = ", ".join(
            f"{name} ({'✓' if data.get('passed') else '✗'})"
            for name, data in tool_results.items()
        )

        sample_label = sample_type_labels.get(sample_type, sample_type.title())

        html_parts.append(f"""
            <tr>
                <td><strong>{test_id}</strong></td>
                <td>{acquisition_type} {sample_label}</td>
                <td>{get_status_badge(is_passed)}</td>
                <td>{tool_summary}</td>
                <td>{best_id_rate * 100:.1f}%</td>
                <td>{best_precision * 100:.1f}%</td>
                <td><a href="{report_path}">View →</a></td>
            </tr>
""")

    html_parts.append("""
        </tbody>
    </table>
</div>
""")

    # Footer
    html_parts.append(f"""
<div class="footer">
    <p>Generated by <strong>timsim-validate</strong> | {timestamp}</p>
    <p>{passed} passed, {failed} failed out of {total} benchmarks</p>
</div>

<script>
{JS_SCRIPT}
{META_JS_ADDITIONS}
</script>
</body>
</html>
""")

    # Write HTML file
    html_content = "\n".join(html_parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Generated meta report: {output_path}")
    return output_path
