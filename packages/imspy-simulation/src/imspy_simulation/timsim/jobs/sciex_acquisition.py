"""SCIEX ZenoTOF SWATH build-from-.wiff acquisition (the SCIEX analogue of
``AstralAcquisitionBuilder``).

A SCIEX ``.wiff`` method carries the SWATH isolation windows + TOF calibration but NO
per-scan timing, so — unlike the Thermo build-from-template path which reads the
template's real schedule — we SYNTHESIZE a SWATH frame schedule from the windows plus a
caller-supplied cycle time + gradient (and a rolling-CE model). The Rust scheme layer does
the expansion (``PyAcquisitionScheme.sciex_frame_schedule``); from there the acquisition
tables are built by the SAME ``build_frame_tables_from_schedule`` transform the Astral path uses,
so the trunk renders identically. Output is open mzML (``render_dia_mzml``), since the
proprietary ``.wiff.scan`` spectra are not authored.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .template_acquisition_common import (
    TemplateHelperHandle,
    TemplateTdfWriterStub,
    build_frame_tables_from_schedule,
    build_synthetic_scan_table,
)


class SciexAcquisitionBuilder:
    """Lean SCIEX ZenoTOF SWATH acquisition — NO Bruker reference / TDF.

    Synthesizes the frame table + DIA windows from a SCIEX ``.wiff`` SWATH method
    (``sciex_frame_schedule``) and a supplied cycle time / gradient / rolling-CE model,
    then writes the four acquisition tables to ``synthetic_data.db``. Duck-types the
    slice of the Bruker acquisition-builder interface the simulator reads
    (``synthetics_handle``, ``frame_table``, ``scan_table``, ``tdf_writer.helper_handle``
    ranges, ``gradient_length``, ``rt_cycle_length``, ``path``). The render output is mzML
    (the proprietary ``.wiff.scan`` is not authored).
    """

    def __init__(
        self,
        path: str,
        wiff_path: str,
        *,
        cycle_time_s: float = 3.5,
        gradient_length_s: float = 1800.0,
        ce_intercept: Optional[float] = 5.0,
        ce_slope_per_mz: Optional[float] = 0.045,
        num_scans: int = 451,
        im_lower: float = 0.6,
        im_upper: float = 1.6,
        mz_lower: Optional[float] = None,
        mz_upper: Optional[float] = None,
        round_collision_energy: bool = True,
        collision_energy_decimals: int = 2,
        verbose: bool = True,
    ) -> None:
        import imspy_connector
        from imspy_simulation.experiment import SyntheticExperimentDataHandle

        # SCIEX rolling CE is not stored in the .wiff; without a model the schedule's CE
        # is Unknown -> 0, which conditions fragment-intensity prediction (empty/unrealistic
        # MS2). Require an explicit linear rolling-CE model for a SCIEX run.
        if ce_intercept is None or ce_slope_per_mz is None:
            raise ValueError(
                "SCIEX requires a rolling-CE model: pass both ce_intercept and "
                "ce_slope_per_mz (CE = intercept + slope_per_mz * precursor_mz)"
            )

        self.path = path
        self.wiff_path = wiff_path
        self.acquisition_name = "dia"
        self.synthetics_handle = SyntheticExperimentDataHandle(database_path=path)

        # Synthesize the SWATH schedule (Rust): 1 MS1 + N windows per cycle, expanded
        # over the gradient at `cycle_time_s`. CE is the rolling-CE linear model (vs
        # m/z); supply both coefficients or neither (left Unknown -> CE 0).
        sched = imspy_connector.py_acquisition.PyAcquisitionScheme.sciex_frame_schedule(
            wiff_path,
            cycle_time_s,
            gradient_length_s,
            ce_intercept=ce_intercept,
            ce_slope_per_mz=ce_slope_per_mz,
        )
        schedule = list(sched)

        # Rolling CE must stay strictly positive (and known) across the synthesized
        # windows: a None CE silently becomes 0, and a non-positive NCE conditions
        # fragment-intensity prediction to empty/unphysical MS2. The .wiff windows +
        # rolling-CE model are resolved in Rust; verify the result here (parity with
        # the Waters SONAR builder). MS2 rows are ms_level != 1.
        ms2_ce = [r[5] for r in schedule if r[2] != 1]
        if ms2_ce:
            if any(c is None for c in ms2_ce):
                raise ValueError(
                    "SCIEX schedule has windows with unknown collision energy (some .wiff "
                    "window centers fell outside the rolling-CE model); supply a CE model "
                    "covering the full SWATH precursor m/z range"
                )
            if min(ms2_ce) <= 0.0:
                raise ValueError(
                    f"SCIEX rolling-CE model (intercept={ce_intercept}, slope={ce_slope_per_mz}) "
                    f"yields non-positive collision energy (min {min(ms2_ce):.4f}) over the SWATH "
                    "windows; supply a CE model positive across the precursor m/z range"
                )

        if verbose:
            n_ms1 = sum(1 for r in schedule if r[2] == 1)
            print(
                f"SCIEX SWATH build-from-.wiff: {len(schedule)} frames "
                f"({gradient_length_s / 60.0:.1f} min, cycle {cycle_time_s}s, "
                f"{n_ms1} cycles)"
            )

        ce_decimals = collision_energy_decimals if round_collision_energy else 6
        ft, win, f2g, f2scan = build_frame_tables_from_schedule(
            schedule, num_scans=num_scans, ce_decimals=ce_decimals
        )

        self.frame_table = ft
        self.scan_table = build_synthetic_scan_table(num_scans, im_lower, im_upper)
        self.dia_ms_ms_windows = win
        self.frames_to_window_groups = f2g
        self.frame_to_template_scan = f2scan  # synthetic (1..N); no real .wiff slots
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
            f"SciexAcquisitionBuilder(path={self.path}, frames={self.num_frames}, "
            f"windows={len(self.dia_ms_ms_windows)}, gradient={self.gradient_length:.1f}s)"
        )
