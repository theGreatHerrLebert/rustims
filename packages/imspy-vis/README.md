# imspy-vis

Lightweight visualization tools for timsTOF proteomics data.

## Installation

```bash
pip install imspy-vis
```

For Jupyter notebook support:

```bash
pip install imspy-vis[notebook]
```

## Features

- **Frame Rendering**: DDA and DIA frame visualization with annotation overlays
- **Point Cloud Visualization**: Interactive 3D visualization using Plotly
- **Video Generation**: Generate preview videos from timsTOF datasets
- **Jupyter Integration**: Interactive widgets for notebook-based exploration

## Quick Start

### Frame Rendering

```python
from imspy_vis import DDAFrameRenderer, DIAFrameRenderer, generate_preview_video

# Generate a quick preview video
generate_preview_video(
    '/path/to/data.d',
    '/path/to/output.mp4',
    mode='dda',
    max_frames=100,
    fps=10
)

# Or use the renderer directly
renderer = DDAFrameRenderer('/path/to/data.d')
renderer.render_to_video('/path/to/output.mp4', max_frames=50)
```

### Point Cloud Visualization (Jupyter)

```python
from imspy_vis import DDAPrecursorPointCloudVis

# In a Jupyter notebook
visualizer = DDAPrecursorPointCloudVis(precursor_data)
visualizer.display_widgets()
```

## Modules

- **pointcloud**: Interactive 3D point cloud visualization using Plotly
- **frame_rendering**: Frame-by-frame rendering and video generation

## Dependencies

- **imspy-core**: Core data structures (required)
- **plotly**: Interactive plotting
- **matplotlib**: Static plotting and frame rendering
- **imageio**: Video generation
- **ipywidgets**: Jupyter notebook widgets

## Related Packages

- **imspy-core**: Core data structures and timsTOF readers
- **imspy-predictors**: ML-based predictors
- **imspy-simulation**: TimSim simulation tools
- **imspy-search**: Database search functionality

## License

MIT License - see LICENSE file for details.
