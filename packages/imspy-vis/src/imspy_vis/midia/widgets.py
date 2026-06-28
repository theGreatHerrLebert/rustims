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

    def __init__(self, default_path: str = ""):
        self.experiment: MidiaExperiment | None = None
        self._loaded_path: str | None = None
        self.precursor_df = pd.DataFrame()
        self.fragment_df = pd.DataFrame()

        self.path = widgets.Text(value=default_path, description="run .d:",
                                 layout=widgets.Layout(width="60%"))

        # Browse-to-pick a run instead of typing the whole path. Bruker .d runs are
        # directories, so this navigates folders and lets you click a .d to select it
        # (rather than a generic file chooser that would descend *into* the .d).
        self._suppress_browse = False
        self.browse_dir = self._initial_browse_dir(default_path)
        self.browse_label = widgets.HTML()
        self.browser = widgets.Select(options=[], rows=8, description="",
                                      layout=widgets.Layout(width="60%"))
        self.browser.observe(self._on_browse_select, names="value")
        self.refresh_button = widgets.Button(description="↻", tooltip="refresh listing",
                                             layout=widgets.Layout(width="40px"))
        self.refresh_button.on_click(lambda _c: self._refresh_browser())
        self._refresh_browser()

        self.rt_start = widgets.FloatText(value=5.0, description="RT min (min):")
        self.rt_stop = widgets.FloatText(value=7.0, description="RT max (min):")
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
            widgets.HBox([self.rt_start, self.rt_stop]),
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
            self.path.value = val  # pick this run; load happens on "Load slice"
            self.status.value = f"selected <b>{os.path.basename(val)}</b> — press Load slice"
            self._clear_browser_selection()  # so re-clicking the same .d works
        else:
            try:  # validate readability before navigating, so a failed open doesn't strand us
                os.listdir(val)
            except OSError as e:
                self.status.value = f"<span style='color:red'>cannot open {os.path.basename(val)}: {e}</span>"
                self._clear_browser_selection()
                return
            self.browse_dir = val
            self._refresh_browser()

    def _on_load(self, _change: object) -> None:
        try:
            self.status.value = "<i>loading…</i>"
            if self.experiment is None or self._loaded_path != self.path.value:
                self.experiment = MidiaExperiment(self.path.value)
                self._loaded_path = self.path.value
            sl = self.experiment.get_slice_retention_time(
                self.rt_start.value * 60.0, self.rt_stop.value * 60.0
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
            "rt_min_min": self.rt_start.value, "rt_max_min": self.rt_stop.value,
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
        self.point_size = widgets.FloatSlider(value=2.0, min=0.5, max=6.0, step=0.5,
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
        self.point_size = widgets.FloatSlider(value=2.0, min=0.5, max=6.0, step=0.5,
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

        self.box = widgets.VBox([
            widgets.HTML(f"<b>{title}</b>"),
            self._extra_controls(),
            widgets.HBox([self.min_cluster_size, self.min_samples, self.metric]),
            widgets.HBox([self.cycle_scaling, self.scan_scaling, self.mz_scaling, self.resolution]),
            widgets.HBox([self.alpha, self.leaf_size,
                          self.approx_min_span_tree, self.gen_min_span_tree, self.use_probability]),
            widgets.HBox([self.cluster_button, self.filter_noise, self.colors]),
            widgets.HBox([self.config_name, self.save_button, self.save_status]),
            widgets.HBox([self.scatter_out, widgets.VBox([self.summary, self.histograms])]),
        ])

    def _extra_controls(self) -> widgets.Widget:
        return widgets.HBox([])

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
        return widgets.HBox([self.cluster_selection_method, self.cluster_selection_epsilon,
                             self.use_midia, self.extraction_scaling])

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

    def __init__(self, default_path: str = ""):
        self.data_panel = DataPanel(default_path)
        self.precursor_cloud = PointCloudPanel(self.data_panel, "precursor", "Precursor point cloud")
        self.precursor_panel = PrecursorHDBSCANPanel(self.data_panel)
        # The MIDIA filter is the fragment source for the MIDIA cloud + clustering tabs.
        self.midia_filter = MidiaFilterPanel(self.data_panel)
        self.fragment_cloud = FragmentPointCloudPanel(self.midia_filter, "Fragment point cloud")
        self.midia_panel = MidiaHDBSCANPanel(self.midia_filter)
        self.tab = widgets.Tab(children=[self.data_panel.controls,
                                         self.precursor_cloud.box,
                                         self.precursor_panel.box,
                                         self.midia_filter.controls,
                                         self.fragment_cloud.box,
                                         self.midia_panel.box])
        for i, t in enumerate(["Data", "Precursor cloud", "Precursor HDBSCAN",
                               "MIDIA filter", "Fragment cloud", "MIDIA HDBSCAN"]):
            self.tab.set_title(i, t)
        self.tab.observe(self._on_tab_change, names="selected_index")

    def _on_tab_change(self, change: object) -> None:
        # Redraw the clustering histogram FigureWidgets when their (hidden) tab is shown.
        panel = {2: self.precursor_panel, 5: self.midia_panel}.get(change["new"])
        if panel is not None:
            panel.nudge_resize()

    def display(self) -> None:
        from IPython.display import display
        display(self.tab)
