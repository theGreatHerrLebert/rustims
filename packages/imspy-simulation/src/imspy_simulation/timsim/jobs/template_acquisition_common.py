"""Instrument-neutral primitives shared by every build-from-template acquisition
builder (Thermo ``.raw`` Astral/Orbitrap and SCIEX ``.wiff`` ZenoTOF SWATH).

A build-from-template run does NOT read a Bruker reference ``.d``: it turns a
vendor template's (or synthesized) per-scan schedule into the four TimSim
acquisition tables, and feeds the distribution jobs only the m/z + ion-mobility
ranges they need. The transform and the two duck-typed stand-ins below are the
same regardless of vendor, so they live here once and are imported by each
builder (``astral_acquisition``, ``sciex_acquisition``).

A schedule row is ``(scan, rt_s, ms_level, center_mz|None, width_mz|None,
ce|None)`` — isolation/CE are ``None`` for MS1. This module is a pure data
transform (no connector / no DB), so it is unit-testable on its own.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# A schedule row: (scan, rt_s, ms_level, center_mz|None, width_mz|None, ce|None)
ScheduleRow = Tuple[int, float, int, Optional[float], Optional[float], Optional[float]]


def build_synthetic_scan_table(
    num_scans: int = 451,
    im_lower: float = 0.6,
    im_upper: float = 1.6,
) -> pd.DataFrame:
    """A synthetic mobility (scan) grid for a build-from-template run — NO Bruker reference.

    Astral/Orbitrap/ZenoTOF have no ion mobility, and the render marginalises the
    mobility axis away, so the actual 1/K0 values do not affect the output — but the
    simulation pipeline still projects ion-mobility distributions onto a scan grid,
    so it needs SOME grid. We provide a plausible descending 1/K0 grid (the Bruker
    ``scans`` table is descending scan index with ascending mobility) without reading
    a reference ``.d``. This is the key decoupling primitive for the lean
    build-from-template path: it removes the ``TDFWriter.helper_handle`` dependency.
    """
    if num_scans < 1:
        raise ValueError("num_scans must be >= 1")
    if not (im_lower < im_upper):
        raise ValueError("im_lower must be < im_upper")
    # Descending scan index (Bruker convention), ascending inverse mobility.
    scans = np.arange(num_scans, dtype=np.int64)[::-1]
    mobilities = np.linspace(im_lower, im_upper, num_scans, dtype=np.float64)
    return pd.DataFrame({"scan": scans, "mobility": mobilities})


def build_frame_tables_from_schedule(
    schedule: Sequence[ScheduleRow],
    *,
    num_scans: int = 1,
    ce_decimals: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[int]]:
    """Build the DIA acquisition tables from a per-scan template schedule.

    Each schedule scan becomes one frame (``frame_id`` 1..N in scan order), timed
    at its retention time. MS1 scans are precursor frames (``ms_type`` 0); MS2
    scans are fragment frames (``ms_type`` 9) carrying the isolation window +
    collision energy. Distinct (center, width, CE) windows are assigned
    window-group ids; because these instruments have no ion mobility, each window
    spans the FULL scan range ``[0, num_scans-1]`` so the (mobility-indexed)
    transmission degenerates to pure m/z gating.

    Returns ``(frame_table, dia_ms_ms_windows, frames_to_window_groups,
    frame_to_template_scan)`` where the last is a list mapping ``frame_id-1`` ->
    the template scan number (so the writer can author each frame into its slot).
    """
    if not schedule:
        raise ValueError("empty template schedule")

    frames = []
    fragment_rows = []  # (frame_id, window_key)
    frame_to_template_scan: List[int] = []
    # Deterministic window-group assignment: first occurrence order.
    window_key_to_group: dict[Tuple[float, float, float], int] = {}
    window_group_meta: dict[int, Tuple[float, float, float]] = {}

    n_ms1 = 0
    for idx, row in enumerate(schedule):
        scan, rt, ms_level, center, width, ce = row
        frame_id = idx + 1
        frame_to_template_scan.append(int(scan))
        is_ms1 = ms_level <= 1
        ms_type = 0 if is_ms1 else 9
        frames.append({"frame_id": frame_id, "time": float(rt), "ms_type": ms_type})
        if is_ms1:
            n_ms1 += 1
            continue
        # MS2: resolve / assign its window group.
        if center is None or width is None:
            raise ValueError(f"MS2 scan {scan} (frame {frame_id}) has no isolation window")
        ce_val = round(float(ce), ce_decimals) if ce is not None else 0.0
        key = (round(float(center), 4), round(float(width), 4), ce_val)
        group = window_key_to_group.get(key)
        if group is None:
            group = len(window_key_to_group) + 1  # 1-based window-group ids
            window_key_to_group[key] = group
            window_group_meta[group] = key
        fragment_rows.append({"frame": frame_id, "window_group": group})

    if n_ms1 == 0:
        raise ValueError("template schedule has no MS1 scans")
    if not fragment_rows:
        raise ValueError("template schedule has no MS2 scans")

    frame_table = pd.DataFrame(frames)
    frames_to_window_groups = pd.DataFrame(fragment_rows)

    # One dia_ms_ms_windows row per window group. No mobility, so the window spans
    # the whole scan range (mobility transmission -> pure m/z gating).
    win_rows = []
    for group, (center, width, ce_val) in sorted(window_group_meta.items()):
        win_rows.append(
            {
                "window_group": group,
                "scan_start": 0,
                "scan_end": int(max(num_scans - 1, 0)),
                "isolation_mz": center,
                "isolation_width": width,
                "collision_energy": ce_val,
            }
        )
    dia_ms_ms_windows = pd.DataFrame(win_rows)

    return frame_table, dia_ms_ms_windows, frames_to_window_groups, frame_to_template_scan


class TemplateHelperHandle:
    """Minimal stand-in for a Bruker reference's ``helper_handle`` — exposes only
    the m/z + ion-mobility ranges the distribution jobs read. These instruments
    have no IMS; the mobility range is synthetic (marginalised away at render)."""

    def __init__(self, mz_lower, mz_upper, im_lower, im_upper, num_scans):
        self.mz_lower = mz_lower
        self.mz_upper = mz_upper
        self.im_lower = im_lower
        self.im_upper = im_upper
        self.num_scans = num_scans


class TemplateTdfWriterStub:
    """Stand-in for the Bruker ``TDFWriter``: a build-from-template path writes a
    Thermo ``.raw`` or open mzML (not a ``.d``), so only the reference ranges are
    needed during simulation, never the binary writer."""

    def __init__(self, helper_handle):
        self.helper_handle = helper_handle
