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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets
from IPython.display import display

from .data import MidiaExperiment
from .clustering import (
    cluster_precursors_hdbscan,
    cluster_midia_hdbscan,
    calculate_statistics,
)
from .transforms import calculate_mz_tick_spacing

_PALETTES = ["Alphabet", "Light24", "Dark24", "Set3", "Bold", "Vivid"]


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
            self.path,
            widgets.HBox([self.rt_start, self.rt_stop]),
            widgets.HBox([self.mz_min, self.mz_max]),
            widgets.HBox([self.scan_min, self.scan_max, self.intensity_min]),
            widgets.HBox([self.load_button, self.status]),
        ])

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


class _ClusterPanel:
    """Shared UI: HDBSCAN sliders, a Cluster button, a 3D scatter, stats + histograms."""

    def __init__(self, data_panel: DataPanel, title: str):
        self.data_panel = data_panel
        self.title = title

        self.min_cluster_size = widgets.IntSlider(value=13, min=2, max=100, step=1,
                                                  description="min cluster size:", continuous_update=False)
        self.min_samples = widgets.IntSlider(value=5, min=1, max=50, step=1,
                                             description="min samples:", continuous_update=False)
        self.metric = widgets.Dropdown(options=["euclidean", "manhattan", "chebyshev"],
                                       value="euclidean", description="metric:")
        self.cycle_scaling = widgets.FloatSlider(value=-0.2, min=-2, max=2, step=0.1,
                                                 description="cycle scaling:", continuous_update=False)
        self.scan_scaling = widgets.FloatSlider(value=0.4, min=-2, max=2, step=0.1,
                                                description="scan scaling:", continuous_update=False)
        self.resolution = widgets.IntSlider(value=46_900, min=5_000, max=150_000, step=100,
                                            description="resolution:", continuous_update=False)
        self.filter_noise = widgets.Checkbox(value=False, description="remove noise", indent=False)
        self.colors = widgets.Dropdown(options=_PALETTES, value="Alphabet", description="colors:")
        self.cluster_button = widgets.Button(description="Cluster", button_style="success")
        self.cluster_button.on_click(self._on_cluster)

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
            widgets.HBox([self.cycle_scaling, self.scan_scaling, self.resolution]),
            widgets.HBox([self.cluster_button, self.filter_noise, self.colors]),
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
            fig.add_trace(go.Scatter3d(
                x=data.cycle / np.power(2, self.cycle_scaling.value),
                y=data.scan / np.power(2, self.scan_scaling.value),
                z=data.mz, mode="markers",
                marker=dict(
                    size=[2 if l == -1 else 2 + 3 * p for l, p in zip(labels, prob)],
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

    # subclasses implement the actual clustering call
    def _run_clustering(self) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


class PrecursorHDBSCANPanel(_ClusterPanel):
    def __init__(self, data_panel: DataPanel):
        super().__init__(data_panel, "Precursor (MS1) HDBSCAN")

    def _run_clustering(self) -> pd.DataFrame:
        return cluster_precursors_hdbscan(
            self.data_panel.precursor_df,
            min_cluster_size=self.min_cluster_size.value,
            min_samples=self.min_samples.value,
            metric=self.metric.value,
            cycle_scaling=self.cycle_scaling.value,
            scan_scaling=self.scan_scaling.value,
            resolution=self.resolution.value)


class MidiaHDBSCANPanel(_ClusterPanel):
    def __init__(self, data_panel: DataPanel):
        super().__init__(data_panel, "MIDIA fragment HDBSCAN (4D)")

    def _extra_controls(self) -> widgets.Widget:
        self.use_midia = widgets.Checkbox(value=True, description="include MIDIA dim", indent=False)
        self.extraction_scaling = widgets.IntSlider(value=36, min=1, max=120, step=1,
                                                    description="quad scaling:", continuous_update=False)
        return widgets.HBox([self.use_midia, self.extraction_scaling])

    def _run_clustering(self) -> pd.DataFrame:
        return cluster_midia_hdbscan(
            self.data_panel.fragment_df,
            self.data_panel.experiment,
            min_cluster_size=self.min_cluster_size.value,
            min_samples=self.min_samples.value,
            metric=self.metric.value,
            cycle_scaling=self.cycle_scaling.value,
            scan_scaling=self.scan_scaling.value,
            resolution=self.resolution.value,
            extraction_scaling=self.extraction_scaling.value,
            use_midia_dimension=self.use_midia.value)


class MidiaVis:
    """Tabbed dashboard: data loading + precursor and MIDIA clustering."""

    def __init__(self, default_path: str = ""):
        self.data_panel = DataPanel(default_path)
        self.precursor_cloud = PointCloudPanel(self.data_panel, "precursor", "Precursor point cloud")
        self.fragment_cloud = PointCloudPanel(self.data_panel, "fragment", "Fragment point cloud")
        self.precursor_panel = PrecursorHDBSCANPanel(self.data_panel)
        self.midia_panel = MidiaHDBSCANPanel(self.data_panel)
        self.tab = widgets.Tab(children=[self.data_panel.controls,
                                         self.precursor_cloud.box,
                                         self.fragment_cloud.box,
                                         self.precursor_panel.box,
                                         self.midia_panel.box])
        for i, t in enumerate(["Data", "Precursor cloud", "Fragment cloud",
                               "Precursor HDBSCAN", "MIDIA HDBSCAN"]):
            self.tab.set_title(i, t)
        self.tab.observe(self._on_tab_change, names="selected_index")

    def _on_tab_change(self, change: object) -> None:
        # Redraw the clustering histogram FigureWidgets when their (hidden) tab is shown.
        panel = {3: self.precursor_panel, 4: self.midia_panel}.get(change["new"])
        if panel is not None:
            panel.nudge_resize()

    def display(self) -> None:
        from IPython.display import display
        display(self.tab)
