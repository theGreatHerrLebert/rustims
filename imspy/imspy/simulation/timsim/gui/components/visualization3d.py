"""3D Point Cloud Visualization for TimSim GUI."""

from typing import Optional, Callable
from pathlib import Path
import numpy as np
from nicegui import ui, run
import plotly.graph_objects as go


def prepare_point_cloud(
    data_path: str,
    max_points: int = 30000,
    num_frames: int = 20,
    frame_type: str = "precursor"
) -> Optional[np.ndarray]:
    """Prepare subsampled point cloud from .d dataset.

    Args:
        data_path: Path to the .d dataset folder
        max_points: Maximum number of points to return
        num_frames: Number of frames to sample from
        frame_type: Type of frames to load - "precursor" or "fragment"

    Returns:
        Array of shape (N, 4) with columns [retention_time, mobility, mz, intensity]
        or None if loading fails
    """
    try:
        from imspy.timstof.dia import TimsDatasetDIA

        dataset = TimsDatasetDIA(data_path, in_memory=False, use_bruker_sdk=True)

        # Get frame IDs based on type
        if frame_type == "fragment":
            frames = dataset.fragment_frames
            frame_label = "fragment"
        else:
            frames = dataset.precursor_frames
            frame_label = "precursor"

        if len(frames) == 0:
            print(f"No {frame_label} frames found")
            return None

        # Sample frames evenly across the run
        step = max(1, len(frames) // num_frames)
        frame_ids = frames[::step][:num_frames]

        all_points = []
        for fid in frame_ids:
            try:
                frame = dataset.get_tims_frame(int(fid))
                if frame is None:
                    continue
                df = frame.df
                if len(df) > 0:
                    all_points.append(df[['retention_time', 'mobility', 'mz', 'intensity']].values)
            except Exception as e:
                print(f"Error loading frame {fid}: {e}")
                continue

        if not all_points:
            print("No data points found")
            return None

        points = np.vstack(all_points)

        # Subsample if too many points
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]

        return points

    except Exception as e:
        print(f"Error preparing point cloud: {e}")
        return None


def create_plotly_figure(points: np.ndarray) -> go.Figure:
    """Create Plotly 3D scatter figure from point cloud data.

    Args:
        points: Array of shape (N, 4) with columns [retention_time, mobility, mz, intensity]

    Returns:
        Plotly Figure object
    """
    # Log-transform intensity for better visualization
    intensity_log = np.log1p(points[:, 3])

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],  # retention_time
        y=points[:, 1],  # mobility (1/K0)
        z=points[:, 2],  # m/z
        mode='markers',
        marker=dict(
            size=1.5,
            color=intensity_log,
            colorscale='inferno',
            opacity=0.5,
            line=dict(width=0),
            colorbar=dict(
                title=dict(text='log(Intensity)', side='right', font=dict(color='white')),
                tickfont=dict(color='white'),
            )
        ),
        hovertemplate=(
            'RT: %{x:.1f}s<br>'
            '1/K₀: %{y:.3f}<br>'
            'm/z: %{z:.2f}<br>'
            '<extra></extra>'
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text='Retention Time (s)', font=dict(color='white')),
                tickfont=dict(color='white'),
                gridcolor='#444',
                showbackground=False
            ),
            yaxis=dict(
                title=dict(text='Inverse Mobility (1/K₀)', font=dict(color='white')),
                tickfont=dict(color='white'),
                gridcolor='#444',
                showbackground=False
            ),
            zaxis=dict(
                title=dict(text='m/z', font=dict(color='white')),
                tickfont=dict(color='white'),
                gridcolor='#444',
                showbackground=False
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='#0d1117',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500
    )

    return fig


def create_3d_visualization(
    initial_data_path: str = None,
    max_points: int = 30000,
    on_path_change: Callable[[str], None] = None,
) -> dict:
    """Create a lazy-loaded 3D point cloud visualization component.

    The visualization is not loaded until the user clicks "Load 3D View".
    This prevents performance issues from having a heavy WebGL context
    always running in the background.

    Args:
        initial_data_path: Initial path to .d dataset (optional)
        max_points: Default maximum number of points
        on_path_change: Callback when data path changes

    Returns:
        Dictionary with control functions:
        - set_data_path(path): Update the data path
        - refresh(): Reload the visualization
        - clear(): Clear the current visualization
        - is_loaded(): Check if visualization is currently loaded
    """
    # State
    current_path = [initial_data_path]
    is_loaded = [False]
    plot_element = [None]

    with ui.card().classes("w-full"):
        ui.label("3D Point Cloud Preview").classes("font-bold text-lg")
        ui.label("Visualize m/z vs Ion Mobility vs Retention Time from reference data").classes("text-sm text-gray-400 mb-4")

        # Controls row
        with ui.row().classes("w-full items-center gap-4 mb-4 flex-wrap"):
            ui.label("Frame type:").classes("text-sm")
            frame_type_toggle = ui.toggle(
                ["Precursor", "Fragment"],
                value="Precursor"
            )

            max_points_input = ui.number(
                "Max points",
                value=max_points,
                min=1000,
                max=100000,
                step=5000
            ).classes("w-36")

            num_frames_input = ui.number(
                "Frames to sample",
                value=20,
                min=5,
                max=100,
                step=5
            ).classes("w-36")

            ui.space()

            load_btn = ui.button("Load 3D View", icon="view_in_ar").classes("bg-blue-600")
            clear_btn = ui.button("Clear", icon="clear").classes("bg-gray-600")

        # Status label
        status_label = ui.label("").classes("text-sm text-gray-400")

        # Plotly container (initially empty)
        plot_container = ui.column().classes("w-full")

    async def load_data():
        """Load and display the 3D point cloud."""
        if not current_path[0]:
            status_label.text = "No reference dataset path set. Please set a reference .d folder in the Files tab."
            status_label.classes(remove="text-gray-400", add="text-yellow-500")
            return

        # Check path exists
        path = Path(current_path[0])
        if not path.exists() or path.suffix != ".d":
            status_label.text = f"Invalid path: {current_path[0]}"
            status_label.classes(remove="text-gray-400", add="text-red-500")
            return

        status_label.text = "Loading data..."
        status_label.classes(remove="text-red-500 text-yellow-500", add="text-gray-400")
        load_btn.disable()

        try:
            # Get frame type from toggle
            frame_type = "fragment" if frame_type_toggle.value == "Fragment" else "precursor"

            # Run data preparation in background thread
            points = await run.io_bound(
                prepare_point_cloud,
                current_path[0],
                int(max_points_input.value),
                int(num_frames_input.value),
                frame_type
            )

            if points is None or len(points) == 0:
                status_label.text = "Failed to load data or no points found"
                status_label.classes(remove="text-gray-400", add="text-red-500")
                load_btn.enable()
                return

            # Create figure
            fig = create_plotly_figure(points)

            # Display plot
            plot_container.clear()
            with plot_container:
                plot_element[0] = ui.plotly(fig).classes("w-full")

            is_loaded[0] = True
            status_label.text = f"Loaded {len(points):,} points from {int(num_frames_input.value)} {frame_type} frames"
            status_label.classes(remove="text-red-500 text-yellow-500", add="text-green-500")

        except Exception as e:
            status_label.text = f"Error: {str(e)}"
            status_label.classes(remove="text-gray-400", add="text-red-500")

        finally:
            load_btn.enable()

    def clear_plot():
        """Clear the visualization to free resources."""
        plot_container.clear()
        plot_element[0] = None
        is_loaded[0] = False
        status_label.text = "Cleared"
        status_label.classes(remove="text-green-500 text-red-500 text-yellow-500", add="text-gray-400")

    def set_data_path(path: str):
        """Update the data path."""
        current_path[0] = path
        if on_path_change:
            on_path_change(path)

    # Connect button handlers
    load_btn.on_click(load_data)
    clear_btn.on_click(clear_plot)

    return {
        "set_data_path": set_data_path,
        "refresh": load_data,
        "clear": clear_plot,
        "is_loaded": lambda: is_loaded[0],
    }
