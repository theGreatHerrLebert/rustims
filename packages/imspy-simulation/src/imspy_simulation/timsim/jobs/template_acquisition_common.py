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

import logging
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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


def validate_template_schedule(
    schedule: Sequence[ScheduleRow],
    *,
    strict_fixed_dia: bool = False,
) -> List[str]:
    """Method-aware sanity gate for a build-from-template scan schedule.

    A build-from-template run reads a vendor ``.raw`` / ``.wiff`` and turns its
    per-scan schedule into acquisition tables. When the reader mis-decodes a file's
    scan-event layout it returns plausible-but-wrong scan levels (e.g. a run read as
    all-MS1), and the error otherwise surfaces cryptically deep in the render. This
    gate refuses such a schedule at ingestion with an actionable message.

    Hard failures (raise ``ValueError`` — unambiguous corruption / mis-parsed levels):
      - empty schedule;
      - any non-finite / negative RT, or non-monotonic (decreasing) RT;
      - an MS2 row whose isolation center/width is missing, non-finite, or non-positive
        (the scan level / isolation was mis-parsed);
      - fewer than 2 MS1 surveys, or zero MS2 scans.

    Soft (returned + logged as warnings; the irregular-cycle one raises only under
    ``strict_fixed_dia``):
      - MS1 rows carrying an isolation window — real MS1 events can carry a non-zero
        center, so this warns rather than failing (hard-failing over-rejects valid files);
      - irregular windows-per-cycle — legitimate for variable-window / scheduled /
        GPF / CE-ramp DIA.

    Returns the list of warning strings (empty when clean).
    """
    if not schedule:
        raise ValueError("empty template schedule")

    warnings: List[str] = []
    n_ms1 = 0
    n_ms2 = 0
    prev_rt: Optional[float] = None
    windows_per_cycle: List[int] = []
    cur_cycle_ms2 = 0
    seen_first_ms1 = False
    n_ms1_with_iso = 0

    for row in schedule:
        scan, rt, ms_level, center, width, _ce = row
        if rt is None or not math.isfinite(float(rt)) or rt < 0:
            raise ValueError(f"scan {scan}: invalid retention time {rt!r}")
        if prev_rt is not None and rt < prev_rt:
            raise ValueError(
                f"scan {scan}: retention time {rt} < previous {prev_rt} — the schedule "
                f"is not monotonic; scan order or the RT field looks mis-parsed"
            )
        prev_rt = rt

        if ms_level <= 1:  # MS1 survey
            # Real MS1 events can carry a non-zero isolation center; a normalized
            # schedule usually nulls it. Treat a surviving MS1 window as a WARNING, not
            # a hard failure — hard-failing here would over-reject legitimate files.
            if center is not None or width is not None:
                n_ms1_with_iso += 1
            n_ms1 += 1
            if seen_first_ms1:
                windows_per_cycle.append(cur_cycle_ms2)
            seen_first_ms1 = True
            cur_cycle_ms2 = 0
        else:  # MS2 fragment
            if center is None or width is None:
                raise ValueError(
                    f"scan {scan}: parsed as MS2 (ms_level={ms_level}) but isolation is "
                    f"missing (center={center}, width={width}) — the scan-event levels "
                    f"look mis-parsed for this file's layout"
                )
            cw, ww = float(center), float(width)
            if not (math.isfinite(cw) and cw > 0.0 and math.isfinite(ww) and ww > 0.0):
                raise ValueError(
                    f"scan {scan}: MS2 isolation center/width is non-finite or non-positive "
                    f"(center={center}, width={width}) — the scan-event fields look mis-parsed"
                )
            n_ms2 += 1
            cur_cycle_ms2 += 1

    if n_ms1 < 2:
        raise ValueError(
            f"template schedule has {n_ms1} MS1 survey(s); a valid DIA acquisition has "
            f">= 2 — the reader may not support this file's scan-event layout"
        )
    if n_ms2 == 0:
        raise ValueError(
            "template schedule has no MS2 scans — the reader may not support this "
            "file's scan-event layout"
        )

    if n_ms1_with_iso > 0:
        msg = (
            f"{n_ms1_with_iso} MS1 survey(s) carry an isolation window — normally a "
            f"normalized schedule nulls these; non-fatal, but worth checking the extractor"
        )
        warnings.append(msg)
        logger.warning("template schedule: %s", msg)

    # Windows-per-cycle regularity over INTERIOR cycles (the trailing partial cycle in
    # `cur_cycle_ms2` is excluded). Drop zero-MS2 "cycles" (consecutive MS1 surveys:
    # lock-mass / system scans / method transitions) — they are not real DIA cycles and
    # would otherwise spuriously read as irregular. Varies legitimately for non-fixed
    # methods, so it warns rather than failing unless strict_fixed_dia.
    interior = [w for w in windows_per_cycle if w > 0]
    if len(interior) >= 2:
        lo, hi = min(interior), max(interior)
        if lo != hi:
            msg = (
                f"irregular DIA cycle: windows-per-cycle varies in [{lo}, {hi}] across "
                f"{len(interior)} interior cycles (legitimate for variable-window / "
                f"scheduled / GPF methods; unusual for fixed-window DIA)"
            )
            if strict_fixed_dia:
                raise ValueError(msg + " — rejected under strict_fixed_dia")
            warnings.append(msg)
            logger.warning("template schedule: %s", msg)

    return warnings


def build_frame_tables_from_schedule(
    schedule: Sequence[ScheduleRow],
    *,
    num_scans: int = 1,
    ce_decimals: int = 2,
    strict_fixed_dia: bool = False,
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
    # Reject a mis-parsed / degenerate schedule at ingestion with an actionable
    # message, instead of failing cryptically deep in the render.
    validate_template_schedule(schedule, strict_fixed_dia=strict_fixed_dia)

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


def rt_cycle_length_from_ms1(frame_table: pd.DataFrame) -> float:
    """Per-cycle (MS1->MS1) interval from a per-scan-frame table.

    Build-from-template/parameters acquisitions use per-SCAN frames (one MS1 + many
    MS2/SWATH/SONAR windows per cycle), so ``median(diff(all frame times))`` is the per-scan
    spacing, NOT the cycle length. Feeding the scan spacing to the EMG frame-abundance
    integration spreads a peptide's elution weight across every window's frame instead of
    carrying it per cycle, under-scaling rendered peak intensities by ~windows-per-cycle.
    Derive the cycle length from the MS1 (``ms_type == 0``) survey spacing instead.

    Raises ``ValueError`` if there are <2 MS1 frames: a valid DIA acquisition has multiple MS1
    surveys, and silently falling back to the scan spacing would re-introduce the exact bug
    this fixes.
    """
    if "ms_type" in frame_table.columns:
        times = np.sort(frame_table["time"].values[frame_table["ms_type"].values == 0])
    else:
        times = np.sort(frame_table["time"].values)
    diffs = np.diff(times)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        raise ValueError(
            "cannot determine rt_cycle_length: <2 MS1 (ms_type==0) frames, so the per-cycle "
            "(MS1->MS1) interval is undefined; a valid DIA acquisition has multiple MS1 surveys."
        )
    return float(np.median(positive))
