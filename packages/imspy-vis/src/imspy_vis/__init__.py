"""
imspy_vis - Lightweight visualization tools for timsTOF proteomics data.

This package provides visualization tools using Plotly, Matplotlib, and ipywidgets
for interactive exploration of timsTOF mass spectrometry data.

Features:
    - Frame rendering with DDA/DIA annotation overlays
    - Interactive 3D point cloud visualization
    - Video generation from timsTOF datasets
    - Jupyter notebook integration

Example:
    >>> from imspy_vis import generate_preview_video, DDAFrameRenderer
    >>> generate_preview_video('/path/to/data.d', '/path/to/output.mp4', mode='dda')
"""

__version__ = "0.4.0"

# Point cloud visualization (Plotly/ipywidgets)
from imspy_vis.pointcloud import (
    ImsPointCloudVisualizer,
    DDAPrecursorPointCloudVis,
    calculate_mz_tick_spacing,
)

# Frame rendering and video generation
from imspy_vis.frame_rendering import (
    BaseFrameRenderer,
    DDAFrameRenderer,
    DIAFrameRenderer,
    generate_preview_video,
    get_frame_matrix,
    configure_gpu_memory,
    overlay_precursor,
    overlay_fragment,
    overlay_windows,
)

__all__ = [
    # Version
    '__version__',
    # Point cloud visualization
    'ImsPointCloudVisualizer',
    'DDAPrecursorPointCloudVis',
    'calculate_mz_tick_spacing',
    # Frame rendering
    'BaseFrameRenderer',
    'DDAFrameRenderer',
    'DIAFrameRenderer',
    'generate_preview_video',
    'get_frame_matrix',
    'configure_gpu_memory',
    'overlay_precursor',
    'overlay_fragment',
    'overlay_windows',
]
