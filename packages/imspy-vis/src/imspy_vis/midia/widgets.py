"""Interactive ipywidgets/Plotly dashboard for MIDIA precursor & fragment clustering.

This is the modern, imspy-backed reconstruction of the proteolizard-midia Voila tool that
let a non-coding scientist tune clustering parameters and decide a peak-picking strategy by
eye. A :class:`DataPanel` loads a retention-time slice of a run; the cluster panels expose
the HDBSCAN parameters as live controls, recompute on a button press, and render the labelled
3D point cloud together with a summary table and per-axis cluster-size histograms.

Usage (in a notebook served by Voila)::

    from imspy_vis.midia.widgets import MidiaVis
    MidiaVis(default_path="/path/to/run.d").display()
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets
from IPython.display import display

from .data import MidiaExperiment, filter_fragments_by_scan_ranges
from .clustering import (
    cluster_precursors_hdbscan,
    cluster_midia_hdbscan,
    calculate_statistics,
)
from .transforms import calculate_mz_tick_spacing

_PALETTES = ["Alphabet", "Light24", "Dark24", "Set3", "Bold", "Vivid"]

# The original exposed every key of ``hdbscan.dist_metrics.METRIC_MAPPING``, but most of those
# (cosine, mahalanobis, haversine, seuclidean, …) need extra parameters or 2D data and raise at
# fit time. We offer the broad-but-safe subset that fits an n-D point cloud unaided.
_HDBSCAN_METRICS = ["euclidean", "manhattan", "chebyshev", "l1", "l2", "cityblock",
                    "canberra", "braycurtis", "infinity"]

# Per-knob explanations shown via a hover ⓘ icon next to each clustering control. Phrased for a
# scientist tuning the result by eye: what the knob does and which way to push it.
_HELP = {
    "min_cluster_size": "Smallest group of points HDBSCAN will call a cluster. Higher → fewer, "
        "bigger clusters and small peaks get dropped as noise; lower → more, smaller clusters.",
    "min_samples": "How conservative the clustering is. Higher → more points left as noise and "
        "tighter, denser cluster cores; lower → more points pulled into clusters. Often ≈ min "
        "cluster size.",
    "metric": "Distance used to compare points in the scaled (cycle, scan, m/z[, MIDIA]) space. "
        "euclidean is the default; manhattan / cityblock is more robust to a single odd axis.",
    "cycle_scaling": "Weights the retention-time (cycle) axis as value → x / 2^scaling. More "
        "negative compresses RT (points count as closer in time); positive expands it.",
    "scan_scaling": "Same x / 2^scaling weighting for the scan (ion-mobility) axis. Raise to make "
        "mobility differences split clusters apart; lower to merge across mobility.",
    "mz_scaling": "Same x / 2^scaling weighting for the m/z axis. Raise to separate clusters that "
        "differ in m/z; lower to merge them.",
    "resolution": "Resolving power used by the peak-width-preserving m/z transform, so m/z spacing "
        "scales like real peak widths. Higher → finer m/z separation (more clusters).",
    "alpha": "HDBSCAN robust-linkage distance scaling. Above 1 is more conservative (fewer merges); "
        "1.0 is standard and rarely needs changing.",
    "leaf_size": "Internal KD/Ball-tree leaf size — a speed/memory knob for nearest-neighbour "
        "queries. Does not change the resulting labels.",
    "approx_min_span_tree": "Use a faster approximate minimum spanning tree. On → quicker on big "
        "slices, with a tiny chance of a slightly different tree; off → exact but slower.",
    "gen_min_span_tree": "Also compute and keep the minimum spanning tree (for tree diagnostics). "
        "No effect on the labels; small extra cost.",
    "use_probability": "Display only: size points by cluster-membership probability, so core points "
        "look bigger than fringe points. Does not change which points cluster.",
    "filter_noise": "Display only: hide the points HDBSCAN labelled as noise (label −1) from the "
        "3D plot.",
    "colors": "Display only: qualitative colour palette used to tint the clusters.",
    # MIDIA-only knobs
    "cluster_selection_method": "How clusters are cut from the hierarchy. 'eom' (excess of mass) "
        "favours clusters of varying density; 'leaf' takes the finest leaves → more, smaller clusters.",
    "cluster_selection_epsilon": "Merge neighbouring clusters closer than this distance. Raise to "
        "rejoin fragments of one feature that got split; low → keep the raw HDBSCAN split.",
    "use_midia": "Include the MIDIA extraction-window (precursor isolation m/z) as a 4th clustering "
        "dimension. On → fragments are also separated by which precursor window isolated them.",
    "extraction_scaling": "Divides the MIDIA-dimension (mc) axis onto a comparable scale (quad-window "
        "units). Higher → that axis matters less in the distance.",
}


def _help_html(key: str) -> str:
    """The formatted explanation block shown when a knob's ⓘ is clicked."""
    return (f"<div style='background:#eef6fb; border-left:3px solid #2b8cbe;"
            f" padding:6px 10px; margin:4px 0; border-radius:3px;'>"
            f"<b>{key.replace('_', ' ')}:</b> {_HELP.get(key, '')}</div>")


def _initial_histograms() -> go.FigureWidget:
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Cycle", "Scan", "m/z", "Size"))
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.add_trace(go.Histogram(x=[]), row=r, col=c)
    fig.update_layout(title_text="Cluster distributions", showlegend=False,
                      template="plotly_white", width=480, height=380,
                      margin=dict(l=10, r=10, b=10, t=60))
    return go.FigureWidget(fig)


class DataPanel:
    """Loads a retention-time slice of a run and applies a coarse range filter."""

    def __init__(self, default_path: str = "", browse_dir: str | None = None):
        self.experiment: MidiaExperiment | None = None
        self._loaded_path: str | None = None
        self.precursor_df = pd.DataFrame()
        self.fragment_df = pd.DataFrame()

        self.path = widgets.Text(value=default_path, description="run .d:",
                                 layout=widgets.Layout(width="60%"))

        # Browse-to-pick a run instead of typing the whole path. Bruker .d runs are
        # directories, so this navigates folders and lets you click a .d to select it
        # (rather than a generic file chooser that would descend *into* the .d). The browser
        # opens at ``browse_dir`` when given (e.g. a folder of runs), else the run's parent.
        self._suppress_browse = False
        self.browse_dir = (browse_dir if browse_dir and os.path.isdir(browse_dir)
                           else self._initial_browse_dir(default_path))
        self.browse_label = widgets.HTML()
        self.browser = widgets.Select(options=[], rows=8, description="",
                                      layout=widgets.Layout(width="60%"))
        self.browser.observe(self._on_browse_select, names="value")
        self.refresh_button = widgets.Button(description="↻", tooltip="refresh listing",
                                             layout=widgets.Layout(width="40px"))
        self.refresh_button.on_click(lambda _c: self._refresh_browser())
        self._refresh_browser()

        # RT range as a span over the whole-run TIC, so you can see which part of the
        # chromatogram you're slicing (faithful to the old timsVIS slice selector). Bounds are
        # minutes; ``max`` widens to the real run length once a run is opened (_draw_tic).
        self.rt_range = widgets.FloatRangeSlider(
            value=[5.0, 7.0], min=0.0, max=46.0, step=0.1, readout_format=".1f",
            description="RT (min):", continuous_update=False,
            layout=widgets.Layout(width="60%"))
        self.rt_range.observe(self._on_rt_range_change, names="value")
        self.tic = self._make_tic_widget()
        self.mz_min = widgets.FloatText(value=700.0, description="m/z min:")
        self.mz_max = widgets.FloatText(value=730.0, description="m/z max:")
        self.scan_min = widgets.IntText(value=300, description="scan min:")
        self.scan_max = widgets.IntText(value=800, description="scan max:")
        self.intensity_min = widgets.FloatText(value=20.0, description="min intensity:")
        self.load_button = widgets.Button(description="Load slice", button_style="primary")
        self.status = widgets.HTML(value="<i>no slice loaded</i>")
        self.load_button.on_click(self._on_load)

        self.controls = widgets.VBox([
            widgets.HTML("<b>Pick a run</b> — click folders to navigate, click a <code>.d</code> to select:"),
            widgets.HBox([self.browse_label, self.refresh_button]),
            self.browser,
            self.path,
            self.rt_range,
            self.tic,
            widgets.HBox([self.mz_min, self.mz_max]),
            widgets.HBox([self.scan_min, self.scan_max, self.intensity_min]),
            widgets.HBox([self.load_button, self.status]),
        ])

    # -- run browser --------------------------------------------------------------------
    @staticmethod
    def _initial_browse_dir(default_path: str) -> str:
        """Start browsing from the run's parent folder if known, else the home directory."""
        parent = os.path.dirname(default_path.rstrip("/")) if default_path else ""
        return parent if os.path.isdir(parent) else os.path.expanduser("~")

    def _clear_browser_selection(self) -> None:
        """Reset the Select to no-selection (under the guard) so the same item fires again."""
        self._suppress_browse = True
        try:
            self.browser.value = None
        finally:
            self._suppress_browse = False

    def _refresh_browser(self) -> None:
        """List the current directory: an optional '..' entry, sub-folders, and selectable .d runs."""
        self._suppress_browse = True
        try:
            self.browse_label.value = f"📂 <code>{self.browse_dir}</code>"
            # Only offer '..' when there is a real parent (not at the filesystem root).
            stripped = self.browse_dir.rstrip(os.sep)
            parent = os.path.dirname(stripped)
            entries = [("⬆  ..", "..")] if parent and parent != stripped else []
            try:
                for name in sorted(os.listdir(self.browse_dir), key=str.lower):
                    full = os.path.join(self.browse_dir, name)
                    if not os.path.isdir(full):
                        continue
                    entries.append((f"🎯 {name}" if name.endswith(".d") else f"📁 {name}/", full))
            except OSError as e:
                self.browse_label.value = f"<span style='color:red'>cannot read {self.browse_dir}: {e}</span>"
            self.browser.options = entries
            self.browser.value = None
        finally:
            self._suppress_browse = False

    def _on_browse_select(self, change: object) -> None:
        val = change["new"]
        if self._suppress_browse or val is None:
            return
        if val == "..":
            parent = os.path.dirname(self.browse_dir.rstrip(os.sep))
            self.browse_dir = parent or os.path.sep
            self._refresh_browser()
        elif val.endswith(".d"):
            self.path.value = val  # pick this run; the heavy point load happens on "Load slice"
            self._clear_browser_selection()  # so re-clicking the same .d works
            self._show_tic()  # open the run (cheap metadata) and draw its TIC to pick against
        else:
            try:  # validate readability before navigating, so a failed open doesn't strand us
                os.listdir(val)
            except OSError as e:
                self.status.value = f"<span style='color:red'>cannot open {os.path.basename(val)}: {e}</span>"
                self._clear_browser_selection()
                return
            self.browse_dir = val
            self._refresh_browser()

    def _ensure_experiment(self) -> None:
        """Open the run (a cheap metadata read) if not already open for the current path."""
        if self.experiment is None or self._loaded_path != self.path.value:
            self.experiment = MidiaExperiment(self.path.value)
            self._loaded_path = self.path.value

    # -- TIC slice selector (faithful to the old timsVIS DDADataLoader) -------------------
    @staticmethod
    def _make_tic_widget() -> go.FigureWidget:
        """An empty 2D TIC line. 2D FigureWidgets render fine under Voila (unlike Scatter3d)."""
        fig = go.FigureWidget(data=go.Scatter(x=[], y=[], mode="lines",
                                              line=dict(color="#444", width=1)))
        fig.update_layout(title="Total Intensity Count (MS1)", template="plotly_white",
                          xaxis_title="Time [min]", yaxis_title="Normalized intensity",
                          yaxis_range=[0, 1], width=620, height=240,
                          margin=dict(l=10, r=10, b=30, t=40))
        return fig

    def _show_tic(self) -> None:
        """Open the run and draw its whole-run TIC, so the RT span is picked against it."""
        try:
            self.status.value = "<i>reading frame metadata…</i>"
            self._ensure_experiment()
            self._draw_tic()
            self.status.value = (f"run open — <b>{self.experiment.n_cycles:,}</b> cycles; "
                                 f"select an RT span, then press Load slice")
        except Exception as e:
            self.status.value = f"<span style='color:red'>{type(e).__name__}: {e}</span>"

    def _draw_tic(self) -> None:
        """Fill the TIC line from the run's MS1 SummedIntensities and fit the RT slider to it."""
        tic = self.experiment.precursor_tic().sort_values("rt_min")
        x = tic.rt_min.to_numpy()
        y = tic.intensity.to_numpy()
        ymax = float(y.max()) if len(y) and y.max() > 0 else 1.0
        with self.tic.batch_update():
            self.tic.data[0].x = x
            self.tic.data[0].y = y / ymax
        # Widen the RT span slider to the real run length and clamp the current span into it.
        run_max = round(max(self.experiment.rt_max_min, 0.1), 1)
        lo, hi = self.rt_range.value
        self.rt_range.max = max(run_max, hi)  # never below the current high (ipywidgets invariant)
        self.rt_range.value = [min(lo, run_max), min(hi, run_max)]
        self.rt_range.max = run_max
        self._draw_span()

    def _draw_span(self) -> None:
        """Redraw the green selection span on the TIC at the current RT range."""
        lo, hi = self.rt_range.value
        self.tic.layout["shapes"] = ()
        self.tic.add_vrect(x0=lo, x1=hi, line_width=1, fillcolor="green", opacity=0.3)

    def _on_rt_range_change(self, _change: object) -> None:
        # The span only matters once a TIC is drawn; add_vrect on an empty plot is harmless.
        self._draw_span()

    def _on_load(self, _change: object) -> None:
        try:
            self.status.value = "<i>loading…</i>"
            newly_opened = self.experiment is None or self._loaded_path != self.path.value
            self._ensure_experiment()
            if newly_opened:
                self._draw_tic()  # a typed-path first load gets its TIC too (not just .d picks)
            rt0, rt1 = self.rt_range.value
            sl = self.experiment.get_slice_retention_time(
                rt0 * 60.0, rt1 * 60.0
            ).filtered(mz_min=self.mz_min.value, mz_max=self.mz_max.value,
                       scan_min=self.scan_min.value, scan_max=self.scan_max.value,
                       intensity_min=self.intensity_min.value)
            self.precursor_df = sl.precursor_coords3D()
            self.fragment_df = sl.fragment_coords4D()
            self.status.value = (f"<b>{len(self.precursor_df):,}</b> precursor pts &nbsp; "
                                 f"<b>{len(self.fragment_df):,}</b> fragment pts")
        except Exception as e:  # surface errors in the UI instead of the kernel log
            self.status.value = f"<span style='color:red'>{type(e).__name__}: {e}</span>"

    def filter_settings(self) -> dict:
        """Current data-load + range-filter settings, for parameter capture (CSV save)."""
        parts = [p for p in self.path.value.rstrip("/").split("/") if p]
        return {
            "id": parts[-1] if parts else "",
            "rt_min_min": self.rt_range.value[0], "rt_max_min": self.rt_range.value[1],
            "mz_min": self.mz_min.value, "mz_max": self.mz_max.value,
            "scan_min": self.scan_min.value, "scan_max": self.scan_max.value,
            "intensity_min": self.intensity_min.value,
        }


class PointCloudPanel:
    """A raw inferno 3D point cloud of a coordinate view — no clustering.

    Renders ``cycle x scan x m/z`` coloured by log-intensity. Useful both as a quick look at
    the loaded slice and as a render sanity check for the 3D pipeline. Drawn as a fresh
    go.Figure into an Output (the Voila-robust path), with a downsample cap so the browser
    stays responsive.
    """

    def __init__(self, data_panel: DataPanel, kind: str, title: str):
        self.data_panel = data_panel
        self.kind = kind  # "precursor" or "fragment"

        self.opacity = widgets.FloatSlider(value=0.3, min=0.1, max=1.0, step=0.1,
                                            description="opacity:", continuous_update=False)
        self.point_size = widgets.FloatSlider(value=1.0, min=0.5, max=6.0, step=0.5,
                                              description="point size:", continuous_update=False)
        self.colorscale = widgets.Dropdown(options=sorted(px.colors.named_colorscales()),
                                           value="inferno", description="colors:")
        self.max_points = widgets.IntSlider(value=60_000, min=5_000, max=200_000, step=5_000,
                                            description="max points:", continuous_update=False)
        self.render_button = widgets.Button(description="Render", button_style="info")
        self.status = widgets.HTML()
        self.render_button.on_click(self._on_render)
        self.out = widgets.Output()
        with self.out:
            display(self._figure(pd.DataFrame()))

        self.box = widgets.VBox([
            widgets.HTML(f"<b>{title}</b>"),
            widgets.HBox([self.opacity, self.point_size, self.colorscale, self.max_points]),
            widgets.HBox([self.render_button, self.status]),
            self.out,
        ])

    def _source(self) -> pd.DataFrame:
        return (self.data_panel.precursor_df if self.kind == "precursor"
                else self.data_panel.fragment_df)

    def _on_render(self, _change: object) -> None:
        try:
            df = self._source()
            n = len(df)
            if n > self.max_points.value:
                df = df.sample(self.max_points.value, random_state=0)
            self.status.value = f"showing <b>{len(df):,}</b> / {n:,} points"
            fig = self._figure(df)
            with self.out:
                self.out.clear_output(wait=True)
                display(fig)
        except Exception as e:
            self.status.value = f"<span style='color:red'>{type(e).__name__}: {e}</span>"

    def _figure(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if len(df):
            fig.add_trace(go.Scatter3d(
                x=df.cycle, y=df.scan, z=df.mz, mode="markers",
                marker=dict(size=self.point_size.value,
                            color=np.log(df.intensity.to_numpy() + 1e-4),
                            colorscale=self.colorscale.value, opacity=self.opacity.value,
                            line=dict(width=0))))
            tick = calculate_mz_tick_spacing(float(df.mz.min()), float(df.mz.max()))
        else:
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers"))
            tick = 1
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), width=820, height=600,
                          template="plotly_white",
                          scene={"xaxis": {"title": "Cycle (RT)"}, "yaxis": {"title": "Scan (mobility)"},
                                 "zaxis": {"title": "m/z", "dtick": tick}})
        return fig


class FragmentPointCloudPanel:
    """The 4D MIDIA fragment cloud: pick which dimension to drop and what to colour by.

    Faithful to the original ``MidiaFragmentPointCloudVis``. A fragment point has four
    dimensions — ``cycle`` (RT), the **extraction window** (the precursor isolation m/z
    recovered from its quadrupole step+scan, the "MIDIA dimension"), ``scan`` (mobility) and
    ``m/z``. You exclude one; the remaining three become the x/y/z axes. ``colour by`` chooses
    the marker colour (any dimension or log-intensity). ``quad scaling`` divides the
    extraction-window axis so it sits on a comparable scale to the others.
    """

    _DIMS = ["cycle", "extraction window", "scan", "m/z"]  # dim index 0..3
    _AXIS_TITLE = {0: "Cycle (RT)", 1: "Extraction window m/z", 2: "Scan (mobility)", 3: "m/z"}

    def __init__(self, data_panel: DataPanel, title: str = "Fragment point cloud"):
        self.data_panel = data_panel

        self.opacity = widgets.FloatSlider(value=0.3, min=0.1, max=1.0, step=0.1,
                                            description="opacity:", continuous_update=False)
        self.point_size = widgets.FloatSlider(value=1.0, min=0.5, max=6.0, step=0.5,
                                              description="point size:", continuous_update=False)
        self.colorscale = widgets.Dropdown(options=sorted(px.colors.named_colorscales()),
                                           value="inferno", description="colors:")
        self.max_points = widgets.IntSlider(value=60_000, min=5_000, max=200_000, step=5_000,
                                            description="max points:", continuous_update=False)
        self.exclude_dim = widgets.Dropdown(options=self._DIMS, value="extraction window",
                                            description="exclude:")
        self.color_by = widgets.Dropdown(options=self._DIMS + ["log-intensity"],
                                         value="log-intensity", description="color by:")
        self.quad_scaling = widgets.IntSlider(value=36, min=1, max=120, step=1,
                                              description="quad scaling:", continuous_update=False)
        self.render_button = widgets.Button(description="Render", button_style="info")
        self.status = widgets.HTML()
        self.render_button.on_click(self._on_render)
        self.out = widgets.Output()
        with self.out:
            display(self._figure([np.array([])] * 3, [0, 2, 3]))

        self.box = widgets.VBox([
            widgets.HTML(f"<b>{title}</b>"),
            widgets.HBox([self.opacity, self.point_size, self.colorscale, self.max_points]),
            widgets.HBox([self.exclude_dim, self.color_by, self.quad_scaling]),
            widgets.HBox([self.render_button, self.status]),
            self.out,
        ])

    def _on_render(self, _change: object) -> None:
        try:
            df = self.data_panel.fragment_df
            n = len(df)
            if n == 0 or self.data_panel.experiment is None:
                self.status.value = "<i>load a slice first</i>"
                return

            # Recover the MIDIA (extraction-window) coordinate per point and scale it.
            mc = self.data_panel.experiment.midia_dimension(
                df.step.to_numpy(), df.scan.to_numpy()) / self.quad_scaling.value

            exclude = self._DIMS.index(self.exclude_dim.value)
            remaining = [d for d in range(4) if d != exclude]

            # Always restrict to the in-window fragment population, so the displayed set does
            # not change with the axis/colour choice (and the extraction-window coordinate is
            # always defined). Window-less fragments lie outside the MIDIA isolation scheme;
            # this mirrors the 4D clustering's window filter.
            view = pd.DataFrame({"cycle": df.cycle.to_numpy(), "mc": mc,
                                 "scan": df.scan.to_numpy(), "mz": df.mz.to_numpy(),
                                 "intensity": df.intensity.to_numpy()})
            view = view[np.isfinite(view.mc)]
            n_window = len(view)
            dropped = n - n_window
            if n_window > self.max_points.value:
                view = view.sample(self.max_points.value, random_state=0)

            cols = {0: view.cycle.to_numpy(), 1: view.mc.to_numpy(),
                    2: view.scan.to_numpy(), 3: view.mz.to_numpy()}
            xyz = [cols[d] for d in remaining]
            color = self._color_values(view)

            note = f" ({dropped:,} window-less excluded)" if dropped else ""
            self.status.value = f"showing <b>{len(view):,}</b> / {n_window:,} in-window points{note}"
            fig = self._figure(xyz, remaining, color)
            with self.out:
                self.out.clear_output(wait=True)
                display(fig)
        except Exception as e:
            self.status.value = f"<span style='color:red'>{type(e).__name__}: {e}</span>"

    def _axis_title(self, dim: int) -> str:
        # The extraction-window axis plots the window left bound / quad scaling, not raw m/z.
        if dim == 1:
            return f"Extraction window m/z /{self.quad_scaling.value}"
        return self._AXIS_TITLE[dim]

    def _color_values(self, view: pd.DataFrame) -> np.ndarray:
        by = self.color_by.value
        if by == "log-intensity":
            return np.log(view.intensity.to_numpy() + 1e-4)
        return view[{"cycle": "cycle", "extraction window": "mc",
                     "scan": "scan", "m/z": "mz"}[by]].to_numpy()

    def _figure(self, xyz: list, remaining: list, color: np.ndarray | None = None) -> go.Figure:
        fig = go.Figure()
        if len(xyz[0]):
            fig.add_trace(go.Scatter3d(
                x=xyz[0], y=xyz[1], z=xyz[2], mode="markers",
                marker=dict(size=self.point_size.value, color=color,
                            colorscale=self.colorscale.value, opacity=self.opacity.value,
                            line=dict(width=0))))
        else:
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers"))

        axis_keys = ["xaxis", "yaxis", "zaxis"]
        scene = {axis_keys[i]: {"title": self._axis_title(d)} for i, d in enumerate(remaining)}
        # Put nice m/z ticks on whichever axis carries the m/z dimension.
        if 3 in remaining:
            mz_vals = xyz[remaining.index(3)]
            if len(mz_vals):
                scene[axis_keys[remaining.index(3)]]["dtick"] = calculate_mz_tick_spacing(
                    float(np.min(mz_vals)), float(np.max(mz_vals)))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), width=820, height=600,
                          template="plotly_white", scene=scene)
        return fig


class SurfacePanel:
    """A 2D intensity heatmap over two raw axes — the third is summed (folded) out.

    Recovers the old timsVIS ``TimsSurfaceVisualizer``: voxelize the precursor (MS1) slice and
    sum one axis to get an intensity surface over the other two (RT×scan, RT×m/z or scan×m/z),
    with identity/sqrt/log normalization and a coarse→fine resolution control. Rebuilt on
    ``numpy.histogram2d`` + a 2D ``go.Heatmap`` — no TensorFlow and no heavy 3D ``go.Surface``,
    so it stays light and renders under Voila. Like the original, this is an MS1-only view (the
    old tool had no MIDIA surface) and folds the already range-filtered slice.
    """

    _DIMS = ["retention time", "scan", "m/z"]              # data order: (cycle, scan, mz)
    _COL = {"retention time": "cycle", "scan": "scan", "m/z": "mz"}
    _LABEL = {"retention time": "Cycle (RT)", "scan": "Scan (mobility)", "m/z": "m/z"}
    _NBINS = {0: 60, 1: 120, 2: 240, 3: 480}              # resolution 0..3 -> bins per axis

    def __init__(self, data_panel: DataPanel, title: str = "Intensity surface (MS1)"):
        self.data_panel = data_panel

        self.exclude_dim = widgets.Dropdown(options=self._DIMS, value="m/z", description="exclude:")
        self.normalization = widgets.Dropdown(options=["identity", "sqrt", "log"],
                                              value="identity", description="normalize:")
        self.resolution = widgets.IntSlider(value=1, min=0, max=3, description="resolution:",
                                            continuous_update=False)
        self.colorscale = widgets.Dropdown(options=sorted(px.colors.named_colorscales()),
                                           value="viridis", description="colors:")
        self.render_button = widgets.Button(description="Display", button_style="info")
        self.status = widgets.HTML()
        self.render_button.on_click(self._on_render)
        self.out = widgets.Output()
        with self.out:
            display(self._figure(None, "retention time", "scan"))

        self.box = widgets.VBox([
            widgets.HTML(f"<b>{title}</b>"),
            widgets.HBox([self.exclude_dim, self.normalization, self.resolution, self.colorscale]),
            widgets.HBox([self.render_button, self.status]),
            self.out,
        ])

    def _on_render(self, _change: object) -> None:
        try:
            df = self.data_panel.precursor_df
            if len(df) == 0:
                self.status.value = "<i>load a slice first</i>"
                return
            exclude = self.exclude_dim.value
            # Remaining dims keep the (cycle, scan, mz) order; first -> rows (y), second -> cols (x).
            y_dim, x_dim = [d for d in self._DIMS if d != exclude]
            yv = df[self._COL[y_dim]].to_numpy(dtype=float)
            xv = df[self._COL[x_dim]].to_numpy(dtype=float)
            w = df.intensity.to_numpy(dtype=float)
            nb = self._NBINS[self.resolution.value]
            # histogram2d(first, second) -> H[i,j] over (first=y, second=x); edges follow that order.
            H, yedges, xedges = np.histogram2d(yv, xv, bins=nb, weights=w)
            H = self._normalize(H)
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            self.status.value = (f"<b>{len(df):,}</b> pts → {H.shape[0]}×{H.shape[1]} grid, "
                                 f"folded over {exclude}")
            fig = self._figure((H, xc, yc), y_dim, x_dim)
            with self.out:
                self.out.clear_output(wait=True)
                display(fig)
        except Exception as e:
            self.status.value = f"<span style='color:red'>{type(e).__name__}: {e}</span>"

    def _normalize(self, H: np.ndarray) -> np.ndarray:
        if self.normalization.value == "sqrt":
            return np.sqrt(H)
        if self.normalization.value == "log":
            return np.log(H + 1.0)
        return H

    def _figure(self, grid: tuple | None, y_dim: str, x_dim: str) -> go.Figure:
        fig = go.Figure()
        if grid is not None:
            H, xc, yc = grid
            fig.add_trace(go.Heatmap(z=H, x=xc, y=yc, colorscale=self.colorscale.value,
                                     colorbar=dict(title="Intensity")))
        else:
            fig.add_trace(go.Heatmap(z=[[0.0]]))
        fig.update_layout(margin=dict(l=10, r=10, b=40, t=10), width=720, height=520,
                          template="plotly_white",
                          xaxis_title=self._LABEL[x_dim], yaxis_title=self._LABEL[y_dim])
        return fig


class _ClusterPanel:
    """Shared UI: HDBSCAN sliders, a Cluster button, a 3D scatter, stats + histograms."""

    def __init__(self, data_panel: DataPanel, title: str):
        self.data_panel = data_panel
        self.title = title

        self.min_cluster_size = widgets.IntSlider(value=13, min=2, max=100, step=1,
                                                  description="min cluster size:", continuous_update=False)
        self.min_samples = widgets.IntSlider(value=5, min=1, max=50, step=1,
                                             description="min samples:", continuous_update=False)
        self.metric = widgets.Dropdown(options=_HDBSCAN_METRICS,
                                       value="euclidean", description="metric:")
        self.cycle_scaling = widgets.FloatSlider(value=-0.2, min=-2, max=2, step=0.1,
                                                 description="cycle scaling:", continuous_update=False)
        self.scan_scaling = widgets.FloatSlider(value=0.4, min=-2, max=2, step=0.1,
                                                description="scan scaling:", continuous_update=False)
        self.mz_scaling = widgets.FloatSlider(value=0.0, min=-5, max=5, step=0.1,
                                              description="mz scaling:", continuous_update=False)
        self.resolution = widgets.IntSlider(value=46_900, min=5_000, max=150_000, step=100,
                                            description="resolution:", continuous_update=False)

        # Advanced HDBSCAN knobs (faithful to the original tool; fixed defaults until now).
        self.alpha = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1,
                                         description="alpha:", continuous_update=False)
        self.leaf_size = widgets.IntSlider(value=40, min=1, max=100, step=1,
                                           description="leaf size:", continuous_update=False)
        self.approx_min_span_tree = widgets.Checkbox(value=True, description="approx tree", indent=False)
        self.gen_min_span_tree = widgets.Checkbox(value=True, description="generate tree", indent=False)
        self.use_probability = widgets.Checkbox(value=True, description="size by probability", indent=False)

        self.filter_noise = widgets.Checkbox(value=False, description="remove noise", indent=False)
        self.colors = widgets.Dropdown(options=_PALETTES, value="Alphabet", description="colors:")
        self.cluster_button = widgets.Button(description="Cluster", button_style="success")
        self.cluster_button.on_click(self._on_cluster)

        # Export a tuned strategy: a config name + every filter/cluster parameter, written to a
        # timestamped CSV (the original tool's core workflow for non-coding scientists).
        self._save_prefix = "HDBSCAN"
        self.config_name = widgets.Text(placeholder="e.g. hela-tight", description="config name:",
                                        layout=widgets.Layout(width="auto"))
        self.save_button = widgets.Button(description="Export config")
        self.save_button.on_click(self._on_save)
        self.save_status = widgets.HTML()

        # The 3D cloud is rendered as a fresh go.Figure into an Output widget rather than a
        # live FigureWidget: a Scatter3d FigureWidget does not render under Voila/anywidget
        # (2D widgets do), so we redraw the whole figure on each Cluster press. This keeps a
        # fully interactive (rotatable) plot at the cost of resetting the camera per run.
        self.scatter_out = widgets.Output()
        with self.scatter_out:
            display(self._scatter_figure(pd.DataFrame()))
        self.summary = widgets.Output()
        self.histograms = _initial_histograms()

        # Click an ⓘ to pin its explanation here (hovering shows the same text as a tooltip).
        self.help_box = widgets.HTML()

        self.box = widgets.VBox([
            widgets.HTML(f"<b>{title}</b> &nbsp;<span style='color:#888'>— hover or click the "
                         "<span style='color:#2b8cbe'>&#9432;</span> icons to learn what each knob does</span>"),
            self.help_box,
            self._extra_controls(),
            widgets.HBox([self._wrap_help(self.min_cluster_size, "min_cluster_size"),
                          self._wrap_help(self.min_samples, "min_samples"),
                          self._wrap_help(self.metric, "metric")]),
            widgets.HBox([self._wrap_help(self.cycle_scaling, "cycle_scaling"),
                          self._wrap_help(self.scan_scaling, "scan_scaling"),
                          self._wrap_help(self.mz_scaling, "mz_scaling"),
                          self._wrap_help(self.resolution, "resolution")]),
            widgets.HBox([self._wrap_help(self.alpha, "alpha"),
                          self._wrap_help(self.leaf_size, "leaf_size"),
                          self._wrap_help(self.approx_min_span_tree, "approx_min_span_tree"),
                          self._wrap_help(self.gen_min_span_tree, "gen_min_span_tree"),
                          self._wrap_help(self.use_probability, "use_probability")]),
            widgets.HBox([self.cluster_button,
                          self._wrap_help(self.filter_noise, "filter_noise"),
                          self._wrap_help(self.colors, "colors")]),
            widgets.HBox([self.config_name, self.save_button, self.save_status]),
            widgets.HBox([self.scatter_out, widgets.VBox([self.summary, self.histograms])]),
        ])

    def _extra_controls(self) -> widgets.Widget:
        return widgets.HBox([])

    # -- per-knob help (hover tooltip + click-to-pin explanation) ------------------------
    def _wrap_help(self, widget: widgets.Widget, key: str) -> widgets.HBox:
        """Pair a control with a small ⓘ button: hover shows the tooltip, click pins the text.

        A Button (not styled HTML) is used because ipywidgets' HTML sanitizes the ``title``
        attribute away; a Button's ``tooltip`` trait surfaces on hover, and ``on_click`` writes
        the explanation into the panel's shared ``help_box`` so a click does something visible.
        """
        b = widgets.Button(icon="info-circle", tooltip=_HELP.get(key, ""),
                           layout=widgets.Layout(width="30px", height="28px"))
        b.style.button_color = "transparent"
        b.on_click(lambda _b, k=key: self._show_help(k))
        return widgets.HBox([widget, b], layout=widgets.Layout(align_items="center"))

    def _show_help(self, key: str) -> None:
        self.help_box.value = _help_html(key)

    def _on_cluster(self, _change: object) -> None:
        try:
            clustered = self._run_clustering()
        except Exception as e:
            with self.summary:
                self.summary.clear_output()
                print(f"{type(e).__name__}: {e}")
            return
        noise = clustered[clustered.label == -1]
        signal = clustered[clustered.label != -1]
        table, _ = calculate_statistics(signal, noise)
        with self.summary:
            self.summary.clear_output()
            from IPython.display import display
            display(table)
        self._update_histograms(signal)
        self._render(clustered)

    def _render(self, clustered: pd.DataFrame) -> None:
        data = clustered[clustered.label != -1] if self.filter_noise.value else clustered
        fig = self._scatter_figure(data)
        with self.scatter_out:
            self.scatter_out.clear_output(wait=True)
            display(fig)

    def _scatter_figure(self, data: pd.DataFrame) -> go.Figure:
        """Build the labelled 3D cloud as a plain go.Figure (rendered via the Output)."""
        fig = go.Figure()
        if len(data):
            cmap = dict(enumerate(getattr(px.colors.qualitative, self.colors.value)))
            labels = data.label.to_numpy().astype(int)
            prob = data.probability.to_numpy() if "probability" in data else np.ones(len(data))
            # Noise stays small; signal points scale with membership probability when the
            # toggle is on, else use a flat size (original: bps=3.5).
            bps, up = 3.5, self.use_probability.value
            fig.add_trace(go.Scatter3d(
                x=data.cycle / np.power(2, self.cycle_scaling.value),
                y=data.scan / np.power(2, self.scan_scaling.value),
                z=data.mz, mode="markers",
                marker=dict(
                    size=[2 if l == -1 else (bps * p if up else bps) for l, p in zip(labels, prob)],
                    color=["grey" if l == -1 else cmap[l % len(cmap)] for l in labels],
                    opacity=0.8, line=dict(width=0))))
            tick = calculate_mz_tick_spacing(float(data.mz.min()), float(data.mz.max()))
        else:
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers"))
            tick = 1
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), width=720, height=560,
                          template="plotly_white",
                          scene={"xaxis": {"title": "Cycle"}, "yaxis": {"title": "Scan"},
                                 "zaxis": {"title": "m/z", "dtick": tick}})
        return fig

    def _update_histograms(self, signal: pd.DataFrame) -> None:
        if signal.empty:
            return
        g = signal.groupby("label")
        with self.histograms.batch_update():
            self.histograms.data[0].x = g["cycle"].nunique().to_numpy()
            self.histograms.data[1].x = g["scan"].nunique().to_numpy()
            self.histograms.data[2].x = g["mz"].nunique().to_numpy()
            self.histograms.data[3].x = g.size().to_numpy()

    def nudge_resize(self) -> None:
        """Force the 2D histogram FigureWidget to redraw after its tab is revealed."""
        fig = self.histograms
        w = fig.layout.width
        with fig.batch_update():
            fig.layout.width = (w or 480) + 1
        with fig.batch_update():
            fig.layout.width = w

    # -- parameter capture --------------------------------------------------------------
    def _cluster_settings(self) -> dict:
        """The HDBSCAN parameters the clustering actually runs with (for the saved CSV)."""
        return {
            "algorithm": "best",
            "alpha": self.alpha.value,
            "approx_min_span_tree": self.approx_min_span_tree.value,
            "gen_min_span_tree": self.gen_min_span_tree.value,
            "leaf_size": self.leaf_size.value,
            "use_probability": self.use_probability.value,
            "min_cluster_size": self.min_cluster_size.value,
            "min_samples": self.min_samples.value,
            "metric": self.metric.value,
            "cycle_scaling": self.cycle_scaling.value,
            "scan_scaling": self.scan_scaling.value,
            "mz_scaling": self.mz_scaling.value,
            "resolution": self.resolution.value,
        }

    def _on_save(self, _change: object) -> None:
        """Write a one-row CSV capturing the config name + filter + cluster settings."""
        try:
            row = {"config_name": self.config_name.value,
                   **self.data_panel.filter_settings(),
                   **self._cluster_settings()}
            stamp = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
            fname = f"{self._save_prefix}-{stamp}.csv"
            pd.DataFrame(row, index=[0]).to_csv(fname, index=False)
            self.save_status.value = f"exported <b>{fname}</b>"
        except Exception as e:
            self.save_status.value = f"<span style='color:red'>{type(e).__name__}: {e}</span>"

    # subclasses implement the actual clustering call
    def _run_clustering(self) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


class PrecursorHDBSCANPanel(_ClusterPanel):
    def __init__(self, data_panel: DataPanel):
        super().__init__(data_panel, "Precursor (MS1) HDBSCAN")
        self._save_prefix = "PRECURSOR-HDBSCAN"
        self.min_samples.value = 1  # the original precursor-HDBSCAN default

    def _run_clustering(self) -> pd.DataFrame:
        return cluster_precursors_hdbscan(
            self.data_panel.precursor_df,
            algorithm="best",
            alpha=self.alpha.value,
            approx_min_span_tree=self.approx_min_span_tree.value,
            gen_min_span_tree=self.gen_min_span_tree.value,
            leaf_size=self.leaf_size.value,
            min_cluster_size=self.min_cluster_size.value,
            min_samples=self.min_samples.value,
            metric=self.metric.value,
            cycle_scaling=self.cycle_scaling.value,
            scan_scaling=self.scan_scaling.value,
            resolution=self.resolution.value,
            mz_scaling=self.mz_scaling.value)


class MidiaHDBSCANPanel(_ClusterPanel):
    def __init__(self, data_panel: DataPanel):
        super().__init__(data_panel, "MIDIA fragment HDBSCAN (4D)")
        self._save_prefix = "MIDIA-HDBSCAN"
        self.metric.value = "manhattan"  # the original's MIDIA-fragment default
        self.min_cluster_size.value = 7  # original MIDIA-HDBSCAN defaults
        self.min_samples.value = 7

    def _extra_controls(self) -> widgets.Widget:
        self.cluster_selection_method = widgets.Dropdown(options=["eom", "leaf"], value="eom",
                                                         description="selection:")
        self.cluster_selection_epsilon = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1,
                                                             description="sel. epsilon:",
                                                             continuous_update=False)
        self.use_midia = widgets.Checkbox(value=True, description="include MIDIA dim", indent=False)
        self.extraction_scaling = widgets.IntSlider(value=36, min=1, max=120, step=1,
                                                    description="quad scaling:", continuous_update=False)
        return widgets.HBox([self._wrap_help(self.cluster_selection_method, "cluster_selection_method"),
                             self._wrap_help(self.cluster_selection_epsilon, "cluster_selection_epsilon"),
                             self._wrap_help(self.use_midia, "use_midia"),
                             self._wrap_help(self.extraction_scaling, "extraction_scaling")])

    def _cluster_settings(self) -> dict:
        return {**super()._cluster_settings(),
                "cluster_selection_method": self.cluster_selection_method.value,
                "cluster_selection_epsilon": self.cluster_selection_epsilon.value,
                "use_midia_dimension": self.use_midia.value,
                "extraction_scaling": self.extraction_scaling.value}

    def _run_clustering(self) -> pd.DataFrame:
        return cluster_midia_hdbscan(
            self.data_panel.fragment_df,
            self.data_panel.experiment,
            algorithm="best",
            cluster_selection_method=self.cluster_selection_method.value,
            cluster_selection_epsilon=self.cluster_selection_epsilon.value,
            alpha=self.alpha.value,
            approx_min_span_tree=self.approx_min_span_tree.value,
            gen_min_span_tree=self.gen_min_span_tree.value,
            leaf_size=self.leaf_size.value,
            min_cluster_size=self.min_cluster_size.value,
            min_samples=self.min_samples.value,
            metric=self.metric.value,
            cycle_scaling=self.cycle_scaling.value,
            scan_scaling=self.scan_scaling.value,
            resolution=self.resolution.value,
            mz_scaling=self.mz_scaling.value,
            extraction_scaling=self.extraction_scaling.value,
            use_midia_dimension=self.use_midia.value)


class MidiaFilterPanel:
    """The MIDIA fragment filter: isolate the fragments co-isolated with a precursor query.

    Faithful to the original ``MidiaFragmentFilter``. A precursor *query* (``query start`` +
    ``query length``), a ``quad width`` and a ``% overlap`` select, per quadrupole step, the
    scan range whose isolation window overlaps the query
    (:meth:`MidiaExperiment.fragment_scan_ranges`); an optional ``scan min/max`` clamps further.

    It also acts as the **fragment data source** for the MIDIA cloud and clustering tabs: it
    exposes the same ``experiment`` / ``fragment_df`` / ``filter_settings`` surface as
    :class:`DataPanel`, so those panels consume the filtered fragments transparently. Until
    ``Filter`` is pressed (or after ``Reset``) it passes the full loaded fragment set through.
    """

    def __init__(self, data_panel: DataPanel):
        self.data_panel = data_panel
        self._filtered: pd.DataFrame | None = None  # None => pass the full fragment_df through
        self._filtered_from = None  # the fragment_df object the filter was computed from
        self._snapshot: dict | None = None  # query params that produced `_filtered`

        self.query_start = widgets.IntSlider(value=709, min=350, max=1500, step=1,
                                             description="query start:", continuous_update=False)
        self.query_length = widgets.IntSlider(value=12, min=6, max=150, step=6,
                                              description="query length:", continuous_update=False)
        self.target_length = widgets.IntSlider(value=36, min=12, max=60, step=12,
                                               description="quad width:", continuous_update=False)
        self.percent_overlap = widgets.IntSlider(value=50, min=1, max=100, step=1,
                                                 description="% overlap:", continuous_update=False)
        self.scan_min = widgets.IntSlider(value=0, min=0, max=1000, step=10,
                                          description="scan min:", continuous_update=False)
        self.scan_max = widgets.IntSlider(value=1000, min=0, max=1000, step=10,
                                          description="scan max:", continuous_update=False)
        self.filter_button = widgets.Button(description="Filter", button_style="primary")
        self.reset_button = widgets.Button(description="Reset")
        self.status = widgets.HTML(value="<i>passing all fragments through</i>")
        self.filter_button.on_click(self._on_filter)
        self.reset_button.on_click(self._on_reset)

        self.controls = widgets.VBox([
            widgets.HTML("<b>MIDIA fragment filter</b>"),
            widgets.HBox([self.query_start, self.query_length]),
            widgets.HBox([self.target_length, self.percent_overlap]),
            widgets.HBox([self.scan_min, self.scan_max]),
            widgets.HBox([self.filter_button, self.reset_button, self.status]),
        ])

    # -- fragment-source interface (mirrors DataPanel for the downstream MIDIA panels) ----
    @property
    def experiment(self) -> "MidiaExperiment | None":
        return self.data_panel.experiment

    def _active(self) -> bool:
        """True only while the cached filter still belongs to the currently-loaded slice.

        A reload in ``DataPanel`` replaces ``fragment_df`` with a fresh object; comparing by
        identity invalidates the stale filter so downstream tabs never see old fragments.
        """
        return self._filtered is not None and self._filtered_from is self.data_panel.fragment_df

    @property
    def fragment_df(self) -> pd.DataFrame:
        return self._filtered if self._active() else self.data_panel.fragment_df

    def filter_settings(self) -> dict:
        """Data filter settings, extended with the MIDIA query params that produced the filter."""
        settings = dict(self.data_panel.filter_settings())
        if self._active():
            settings.update(self._snapshot)  # the params at Filter time, not the live widgets
        return settings

    def _on_filter(self, _change: object) -> None:
        try:
            exp = self.data_panel.experiment
            source = self.data_panel.fragment_df
            if exp is None or source.empty:
                self.status.value = "<i>load a slice first</i>"
                return
            ranges = exp.fragment_scan_ranges(self.query_start.value, self.query_length.value,
                                              self.target_length.value, self.percent_overlap.value)
            self._filtered = filter_fragments_by_scan_ranges(
                source, ranges, self.scan_min.value, self.scan_max.value)
            self._filtered_from = source
            # Snapshot the params now, so a later slider nudge can't desync data from the saved CSV.
            self._snapshot = {
                "query_start": self.query_start.value, "query_length": self.query_length.value,
                "quad_width": self.target_length.value, "percent_overlap": self.percent_overlap.value,
                "midia_scan_min": self.scan_min.value, "midia_scan_max": self.scan_max.value,
            }
            self.status.value = (f"<b>{len(self._filtered):,}</b> / {len(source):,} fragments in "
                                 f"{len(ranges)} window group(s)")
        except Exception as e:
            self.status.value = f"<span style='color:red'>{type(e).__name__}: {e}</span>"

    def _on_reset(self, _change: object) -> None:
        self._filtered = None
        self._filtered_from = None
        self._snapshot = None
        self.status.value = "<i>passing all fragments through</i>"


class MidiaVis:
    """Tabbed dashboard: data loading + precursor and MIDIA clustering."""

    def __init__(self, default_path: str = "", browse_dir: str | None = None):
        self.data_panel = DataPanel(default_path, browse_dir=browse_dir)
        self.precursor_cloud = PointCloudPanel(self.data_panel, "precursor", "Precursor point cloud")
        # Folded-intensity heatmap of the MS1 slice (recovers the old timsVIS Surface tab).
        self.precursor_surface = SurfacePanel(self.data_panel)
        self.precursor_panel = PrecursorHDBSCANPanel(self.data_panel)
        # The MIDIA filter is the fragment source for the MIDIA cloud + clustering tabs.
        self.midia_filter = MidiaFilterPanel(self.data_panel)
        self.fragment_cloud = FragmentPointCloudPanel(self.midia_filter, "Fragment point cloud")
        self.midia_panel = MidiaHDBSCANPanel(self.midia_filter)
        self.tab = widgets.Tab(children=[self.data_panel.controls,
                                         self.precursor_cloud.box,
                                         self.precursor_surface.box,
                                         self.precursor_panel.box,
                                         self.midia_filter.controls,
                                         self.fragment_cloud.box,
                                         self.midia_panel.box])
        for i, t in enumerate(["Data", "Precursor cloud", "Precursor surface", "Precursor HDBSCAN",
                               "MIDIA filter", "Fragment cloud", "MIDIA HDBSCAN"]):
            self.tab.set_title(i, t)
        self.tab.observe(self._on_tab_change, names="selected_index")

    def _on_tab_change(self, change: object) -> None:
        # Redraw the clustering histogram FigureWidgets when their (hidden) tab is shown.
        panel = {3: self.precursor_panel, 6: self.midia_panel}.get(change["new"])
        if panel is not None:
            panel.nudge_resize()

    def display(self) -> None:
        from IPython.display import display
        display(self.tab)
