"""Coordinate and intensity transforms shared by the MIDIA visualization widgets.

Ported from the (now retired) ``proteolizardalgo`` / ``proteolizardvis`` packages so
that ``imspy-vis`` has no dependency on the old proteolizard stack.
"""

import numpy as np


def peak_width_preserving_mz_transform(
        mz: np.ndarray,
        M0: float = 500.0,
        resolution: float = 50_000.0) -> np.ndarray:
    """Transform m/z into an index in which a TOF peak keeps a constant width.

    On a TOF instrument the peak width scales with m/z, so clustering in raw m/z gives
    distance-distorted neighbourhoods. Mapping ``mz`` through a logarithm normalised by the
    instrument resolution makes one "peak width" a constant step on the axis, which is what
    a density clusterer should see.

    Args:
        mz: Array of mass-to-charge ratios to transform.
        M0: The m/z at which the TOF resolution is reported.
        resolution: The resolution of the TOF instrument.
    """
    return (np.log(mz) - np.log(M0)) / np.log1p(1.0 / resolution)


def calculate_mz_tick_spacing(mz_min: float, mz_max: float, num_ticks: int = 10) -> float:
    """Round tick spacing for the m/z axis of a 3D scatter. Never 0 — plotly rejects ``dtick=0``,
    which a very narrow visible m/z span (< ~0.5) would otherwise produce and fail the render."""
    spacing = float(np.round((mz_max - mz_min) / max(num_ticks, 1), 1))
    if spacing > 0:
        return spacing
    raw = (mz_max - mz_min) / max(num_ticks, 1)  # narrow span rounded to 0 — use the finer raw step
    return float(raw) if raw > 0 else 1.0
