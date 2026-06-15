"""P6e build-from-template: turn an Astral template's per-scan schedule into the
TimSim acquisition tables, so the trunk is simulated on the template's REAL
(non-uniform) scan retention times and m/z windows — not a recomputed fixed cycle.

The template schedule comes from
``imspy_connector.py_acquisition.PyAcquisitionScheme.thermo_frame_schedule(path)``
as rows ``(scan, retention_time_s, ms_level, center_mz, width_mz, collision_energy)``
(the isolation/CE are ``None`` for MS1).

This module is the pure data transform (no connector / no DB), so it is unit
testable on its own; the acquisition builder consumes its output.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# The schedule->tables transform and the duck-typed Bruker-reference stand-ins are
# instrument-neutral (the SCIEX build-from-.wiff path reuses them verbatim), so they
# live in the shared module. Imported here under both the canonical neutral names and
# the original ``*astral*`` / ``_Astral*`` names so existing call sites and tests keep
# working.
from .template_acquisition_common import (
    ScheduleRow,
    TemplateHelperHandle,
    TemplateTdfWriterStub,
    build_frame_tables_from_schedule,
    build_synthetic_scan_table,
)

# Backward-compatible aliases (the function/classes moved to template_acquisition_common).
build_astral_frame_tables = build_frame_tables_from_schedule
_AstralHelperHandle = TemplateHelperHandle
_AstralTdfWriterStub = TemplateTdfWriterStub


class AstralAcquisitionBuilder:
    """Lean Orbitrap Astral acquisition (P6e option b) — NO Bruker reference / TDF.

    Sources the frame table + DIA windows from a Thermo ``.raw`` template's real
    per-scan schedule (``thermo_frame_schedule``), so the synthetic trunk is
    simulated on the template's TRUE (non-uniform) scan retention times and m/z
    windows, with per-window NCE. It duck-types the small slice of the Bruker
    acquisition-builder interface the simulator reads (``synthetics_handle``,
    ``frame_table``, ``scan_table``, ``tdf_writer.helper_handle`` ranges,
    ``gradient_length``, ``rt_cycle_length``, ``path``), and writes the four
    acquisition tables (``frames``, ``scans``, ``dia_ms_ms_info``,
    ``dia_ms_ms_windows``) to ``synthetic_data.db`` — without a Bruker `.d`.

    ``frame_to_template_scan`` maps each frame to its template scan, so the writer
    can author each rendered frame into its slot at dispatch time.
    """

    def __init__(
        self,
        path: str,
        template_path: str,
        *,
        num_scans: int = 451,
        im_lower: float = 0.6,
        im_upper: float = 1.6,
        mz_lower: Optional[float] = None,
        mz_upper: Optional[float] = None,
        round_collision_energy: bool = True,
        collision_energy_decimals: int = 2,
        collision_energy_nce: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        import imspy_connector
        from imspy_simulation.experiment import SyntheticExperimentDataHandle

        self.path = path
        self.template_path = template_path
        self.acquisition_name = "dia"
        self.synthetics_handle = SyntheticExperimentDataHandle(database_path=path)

        # `thermo_frame_schedule` returns retention times already in SECONDS (it
        # converts the Thermo `.raw`'s native minutes at the boundary). The entire
        # timsim trunk — the RT model, the EMG sigma defaults (`calculate_rt_defaults`:
        # sigma ~= gradient_s/3600*0.75 + 1.125), and the frame-distribution sampling —
        # works in seconds, so no further scaling is needed here. (Historically this
        # boundary multiplied by 60; that now lives in the Rust extractor so the
        # `retention_time_s` field is honest for every caller.) The `.raw` writer
        # authors into the template's own slots and preserves the template's native
        # (minute) scan times, so the output file is unaffected.
        schedule = list(
            imspy_connector.py_acquisition.PyAcquisitionScheme.thermo_frame_schedule(
                template_path
            )
        )
        if verbose:
            grad_min = (max(r[1] for r in schedule) / 60.0) if schedule else 0.0
            print(
                f"Astral build-from-template: {len(schedule)} template scans "
                f"({grad_min:.1f} min gradient)"
            )

        # Round (and window-group) collision energies at the configured precision —
        # do it ONCE inside build_frame_tables_from_schedule so grouping and the stored
        # CE use the same values (no premature truncation). round_collision_energy=False
        # keeps full precision (grouped at 6 dp to avoid float over-splitting).
        ce_decimals = collision_energy_decimals if round_collision_energy else 6
        ft, win, f2g, f2scan = build_frame_tables_from_schedule(
            schedule, num_scans=num_scans, ce_decimals=ce_decimals
        )
        # Optional manual override: force a single NCE across all windows. Off by
        # default — the template's genuine per-window NCE is used as-is.
        if collision_energy_nce is not None:
            win["collision_energy"] = float(collision_energy_nce)

        self.frame_table = ft
        self.scan_table = build_synthetic_scan_table(num_scans, im_lower, im_upper)
        self.dia_ms_ms_windows = win
        self.frames_to_window_groups = f2g
        self.frame_to_template_scan = f2scan
        self.num_frames = len(ft)
        self.gradient_length = float(ft["time"].max())
        diffs = np.diff(ft["time"].values)
        positive = diffs[diffs > 0]
        self.rt_cycle_length = float(np.median(positive)) if positive.size else 0.0

        if mz_lower is None:
            mz_lower = float(win["isolation_mz"].min() - win["isolation_width"].max())
        if mz_upper is None:
            mz_upper = float(win["isolation_mz"].max() + win["isolation_width"].max())
        self.tdf_writer = TemplateTdfWriterStub(
            TemplateHelperHandle(mz_lower, mz_upper, im_lower, im_upper, num_scans)
        )

        for name, tbl in (
            ("frames", ft),
            ("scans", self.scan_table),
            ("dia_ms_ms_info", f2g),
            ("dia_ms_ms_windows", win),
        ):
            self.synthetics_handle.create_table(table_name=name, table=tbl)

    def __repr__(self) -> str:
        return (
            f"AstralAcquisitionBuilder(path={self.path}, frames={self.num_frames}, "
            f"windows={len(self.dia_ms_ms_windows)}, gradient={self.gradient_length:.1f}s)"
        )
