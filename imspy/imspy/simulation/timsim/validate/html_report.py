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

    Returns:
        Path to the generated HTML file.
    """
    if thresholds is None:
        thresholds = {
            "min_id_rate": 0.30,
            "min_rt_correlation": 0.90,
            "min_im_correlation": 0.90,
        }

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tool_names = list(result.tool_results.keys())
    tool_versions = result.tool_versions or {}

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
<h1>{test_name} - Validation Report</h1>

<div class="header-info">
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
