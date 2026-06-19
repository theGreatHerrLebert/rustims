"""Waters SONAR build-from-parameters acquisition (the Waters analogue of
``SciexAcquisitionBuilder``).

SONAR is Waters' scanning-quadrupole DIA: the quadrupole continuously scans a
narrow (~20 Da) transmission window across a fixed precursor m/z range, synced
with the TOF pushes, so the acquisition is fully described by a few parameters
(scan range, window width/step, cycle time) — there is no per-scan timing or
windowing to read from a vendor file. We therefore SYNTHESIZE the SONAR schedule
directly in Python (no template needed, unlike the Thermo/SCIEX build-from-file
paths) and feed it to the SAME ``build_frame_tables_from_schedule`` transform the
Astral/SCIEX paths use. Output is open mzML (``render_dia_mzml``).

Measured scheme of a real SONAR run (PXD028735, Synapt XS): quad scans 400-900
m/z, 20 Da window, ~200 bins/cycle (2.5 Da step, ~8x overlap), 0.5 s/cycle. The
default here lays CONTIGUOUS 20 Da windows (a clean, light, directly-searchable
"normal DIA" scheme); set ``window_step`` < ``window_width`` to reproduce the
faithful overlapping bin scan. No ion mobility is modelled (the render
marginalises mobility away; Waters TWIMS drift-time/CCS is out of scope for v0).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .template_acquisition_common import (
    TemplateHelperHandle,
    TemplateTdfWriterStub,
    build_frame_tables_from_schedule,
    build_synthetic_scan_table,
)

ScheduleRow = Tuple[int, float, int, Optional[float], Optional[float], Optional[float]]


def build_sonar_schedule(
    *,
    mz_start: float,
    mz_end: float,
    window_width: float,
    window_step: float,
    cycle_time_s: float,
    gradient_length_s: float,
    ce_intercept: float,
    ce_slope_per_mz: float,
) -> Tuple[List[ScheduleRow], List[float], int]:
    """Synthesize a SONAR DIA schedule: 1 MS1 + N quad windows per cycle.

    Windows are laid from ``mz_start``: center ``i`` = ``mz_start + window_width/2
    + i*window_step``. Endpoint policy: the window count is chosen with ``ceil`` so
    the windows FULLY COVER ``[mz_start, mz_end]`` (no gap); the last window may
    extend slightly past ``mz_end`` rather than leave the top of the range
    uncovered. ``window_step == window_width`` gives contiguous (non-overlapping)
    windows; a smaller step gives the overlapping SONAR scan (``window_step >
    window_width`` would leave gaps and is rejected). Each cycle is 1 MS1 survey
    followed by the windows, evenly spaced across ``cycle_time_s``, repeated over
    the gradient. Collision energy is the rolling-CE linear model
    ``ce_intercept + ce_slope_per_mz * center`` (modelled as NCE for the HCD-like
    beam-type CID, as for SCIEX).

    Returns ``(schedule_rows, window_centers, n_cycles)``.
    """
    if not (mz_end > mz_start):
        raise ValueError(f"mz_end must be > mz_start (got {mz_start}..{mz_end})")
    if window_width <= 0.0 or window_step <= 0.0:
        raise ValueError("window_width and window_step must be > 0")
    if cycle_time_s <= 0.0:
        raise ValueError("cycle_time_s must be > 0")
    if not math.isfinite(gradient_length_s) or gradient_length_s <= 0.0:
        raise ValueError(f"gradient_length_s must be finite and > 0, got {gradient_length_s!r}")

    span = mz_end - mz_start
    if window_width > span:
        raise ValueError(
            f"window_width ({window_width}) must be <= the m/z span ({span}); a window "
            f"wider than the scanned range is not a valid SONAR geometry"
        )
    if window_step > window_width:
        raise ValueError(
            f"window_step ({window_step}) must be <= window_width ({window_width}); a "
            f"larger step would leave uncovered gaps between windows"
        )

    # ceil (with a float-noise epsilon) so windows fully cover [mz_start, mz_end];
    # the last window may overshoot mz_end slightly rather than leave a gap.
    n_steps = max(0, math.ceil((span - window_width) / window_step - 1e-9))
    n_windows = n_steps + 1
    centers = [round(mz_start + window_width / 2.0 + i * window_step, 4) for i in range(n_windows)]

    # Rolling CE must stay strictly positive across all windows (a non-positive NCE
    # would condition fragment-intensity prediction to empty/unphysical MS2).
    min_ce = min(ce_intercept + ce_slope_per_mz * c for c in centers)
    if min_ce <= 0.0:
        raise ValueError(
            f"rolling-CE model yields non-positive collision energy (min {min_ce:.4f}) over "
            f"windows {centers[0]}..{centers[-1]}; supply a CE model positive across the range"
        )

    n_events = 1 + n_windows  # MS1 + windows
    n_cycles = max(1, int(gradient_length_s / cycle_time_s))
    dt = cycle_time_s / n_events

    schedule: List[ScheduleRow] = []
    scan = 0
    for c in range(n_cycles):
        cycle_start = c * cycle_time_s
        scan += 1
        schedule.append((scan, cycle_start, 1, None, None, None))  # MS1 survey
        for i, center in enumerate(centers):
            scan += 1
            rt = cycle_start + (i + 1) * dt
            ce = ce_intercept + ce_slope_per_mz * center
            schedule.append((scan, rt, 2, center, window_width, ce))
    return schedule, centers, n_cycles


class WatersSonarAcquisitionBuilder:
    """Lean Waters SONAR acquisition — NO Bruker reference / TDF, NO vendor template.

    Synthesizes the frame table + DIA windows from SONAR scan parameters and a
    rolling-CE model (``build_sonar_schedule``), then writes the four acquisition
    tables to ``synthetic_data.db``. Duck-types the slice of the Bruker
    acquisition-builder interface the simulator reads (``synthetics_handle``,
    ``frame_table``, ``scan_table``, ``tdf_writer.helper_handle`` ranges,
    ``gradient_length``, ``rt_cycle_length``, ``path``). The render output is mzML.
    """

    def __init__(
        self,
        path: str,
        *,
        mz_start: float = 400.0,
        mz_end: float = 900.0,
        window_width: float = 20.0,
        window_step: Optional[float] = None,
        cycle_time_s: float = 0.5,
        gradient_length_s: float = 1800.0,
        ce_intercept: Optional[float] = 5.0,
        ce_slope_per_mz: Optional[float] = 0.04,
        num_scans: int = 451,
        im_lower: float = 0.6,
        im_upper: float = 1.6,
        mz_lower: Optional[float] = None,
        mz_upper: Optional[float] = None,
        round_collision_energy: bool = True,
        collision_energy_decimals: int = 2,
        verbose: bool = True,
    ) -> None:
        from imspy_simulation.experiment import SyntheticExperimentDataHandle

        # SONAR CE is a voltage ramp not stored anywhere here; without a model the
        # schedule's CE would be 0, conditioning fragment-intensity prediction to
        # empty/unrealistic MS2. Require an explicit linear rolling-CE model.
        if ce_intercept is None or ce_slope_per_mz is None:
            raise ValueError(
                "Waters SONAR requires a rolling-CE model: pass both ce_intercept and "
                "ce_slope_per_mz (CE = intercept + slope_per_mz * window_center_mz)"
            )
        # Default to contiguous (non-overlapping) windows; window_step < window_width
        # reproduces the faithful overlapping SONAR quad scan.
        if window_step is None:
            window_step = window_width

        self.path = path
        self.acquisition_name = "dia"
        self.synthetics_handle = SyntheticExperimentDataHandle(database_path=path)

        schedule, centers, n_cycles = build_sonar_schedule(
            mz_start=mz_start,
            mz_end=mz_end,
            window_width=window_width,
            window_step=window_step,
            cycle_time_s=cycle_time_s,
            gradient_length_s=gradient_length_s,
            ce_intercept=ce_intercept,
            ce_slope_per_mz=ce_slope_per_mz,
        )
        if verbose:
            # window_step > window_width is rejected in build_sonar_schedule, so
            # step == width is contiguous and step < width is overlapping.
            overlap = "contiguous" if window_step >= window_width else f"overlapping step {window_step}"
            print(
                f"Waters SONAR build-from-parameters: {len(centers)} windows "
                f"({mz_start:.0f}-{mz_end:.0f} m/z, {window_width} Da, {overlap}), "
                f"{n_cycles} cycles x {cycle_time_s}s ({gradient_length_s / 60.0:.1f} min), "
                f"{len(schedule)} frames"
            )

        ce_decimals = collision_energy_decimals if round_collision_energy else 6
        ft, win, f2g, f2scan = build_frame_tables_from_schedule(
            schedule, num_scans=num_scans, ce_decimals=ce_decimals
        )

        self.frame_table = ft
        self.scan_table = build_synthetic_scan_table(num_scans, im_lower, im_upper)
        self.dia_ms_ms_windows = win
        self.frames_to_window_groups = f2g
        self.frame_to_template_scan = f2scan  # synthetic (1..N); no real vendor slots
        self.num_frames = len(ft)
        self.gradient_length = float(ft["time"].max())
        # rt_cycle_length is the per-CYCLE interval (one full SONAR quad sweep), NOT the
        # per-scan frame spacing. These are per-scan frames (many scanning-quad windows per
        # cycle), so median(diff(all frame times)) is the scan spacing — using it under-scales
        # the EMG frame-abundance per cycle by ~windows-per-cycle and collapses rendered peak
        # intensities. Use the authoritative SONAR cycle time.
        self.rt_cycle_length = float(cycle_time_s)

        if mz_lower is None:
            mz_lower = float(win["isolation_mz"].min() - win["isolation_width"].max())
        if mz_upper is None:
            mz_upper = float(win["isolation_mz"].max() + win["isolation_width"].max())
        self.tdf_writer = _waters_writer_stub(mz_lower, mz_upper, im_lower, im_upper, num_scans)

        for name, tbl in (
            ("frames", ft),
            ("scans", self.scan_table),
            ("dia_ms_ms_info", f2g),
            ("dia_ms_ms_windows", win),
        ):
            self.synthetics_handle.create_table(table_name=name, table=tbl)

    def __repr__(self) -> str:
        return (
            f"WatersSonarAcquisitionBuilder(path={self.path}, frames={self.num_frames}, "
            f"windows={len(self.dia_ms_ms_windows)}, gradient={self.gradient_length:.1f}s)"
        )


def _waters_writer_stub(mz_lower, mz_upper, im_lower, im_upper, num_scans):
    return TemplateTdfWriterStub(
        TemplateHelperHandle(mz_lower, mz_upper, im_lower, im_upper, num_scans)
    )
