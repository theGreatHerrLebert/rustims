"""P6e: build-from-template Astral acquisition-table construction.

The unconditional tests use a tiny synthetic schedule. The live test (gated on
TIMSIM_ASTRAL_TEMPLATE + a thermo-enabled connector) runs the real extractor.
"""
import os

import pytest

from imspy_simulation.timsim.jobs.astral_acquisition import build_astral_frame_tables


def _sched():
    # 2 cycles of: MS1, MS2@500, MS2@520 — with non-uniform RTs.
    return [
        (1, 0.000, 1, None, None, None),
        (2, 0.001, 2, 500.0, 20.0, 25.0),
        (3, 0.0017, 2, 520.0, 20.0, 27.0),
        (4, 0.010, 1, None, None, None),
        (5, 0.011, 2, 500.0, 20.0, 25.0),
        (6, 0.0123, 2, 520.0, 20.0, 27.0),
    ]


def test_frame_table_mirrors_schedule_with_real_rts():
    ft, win, f2g, f2scan = build_astral_frame_tables(_sched(), num_scans=10)

    # One frame per scan, in scan order, with the REAL (non-uniform) retention times.
    assert list(ft["frame_id"]) == [1, 2, 3, 4, 5, 6]
    assert list(ft["time"]) == [0.0, 0.001, 0.0017, 0.010, 0.011, 0.0123]
    # MS1 -> ms_type 0, MS2 -> 9.
    assert list(ft["ms_type"]) == [0, 9, 9, 0, 9, 9]
    # frame -> template scan mapping is identity here (scans were 1..6).
    assert f2scan == [1, 2, 3, 4, 5, 6]


def test_distinct_windows_grouped_and_full_scan_range():
    _, win, f2g, _ = build_astral_frame_tables(_sched(), num_scans=10)

    # Two distinct windows (500/25, 520/27) -> groups 1 and 2.
    assert len(win) == 2
    assert set(win["window_group"]) == {1, 2}
    # Astral has no mobility -> windows span the whole scan range (m/z-only gating).
    assert list(win["scan_start"]) == [0, 0]
    assert list(win["scan_end"]) == [9, 9]
    # CE comes from the template per window (not a single config NCE).
    ce_by_group = dict(zip(win["window_group"], win["collision_energy"]))
    assert ce_by_group[1] == 25.0 and ce_by_group[2] == 27.0

    # The same window across cycles reuses its group; one row per MS2 frame.
    assert list(f2g["frame"]) == [2, 3, 5, 6]
    assert list(f2g["window_group"]) == [1, 2, 1, 2]


def test_rejects_degenerate_schedules():
    with pytest.raises(ValueError):
        build_astral_frame_tables([])
    # MS1-only (no MS2).
    with pytest.raises(ValueError):
        build_astral_frame_tables([(1, 0.0, 1, None, None, None)])
    # MS2 missing its isolation window.
    with pytest.raises(ValueError):
        build_astral_frame_tables(
            [(1, 0.0, 1, None, None, None), (2, 0.001, 2, None, None, 25.0)]
        )


@pytest.mark.skipif(
    os.environ.get("TIMSIM_ASTRAL_TEMPLATE") is None,
    reason="set TIMSIM_ASTRAL_TEMPLATE + a thermo-enabled connector for the live test",
)
def test_astral_acquisition_builder_writes_db_without_bruker_reference(tmp_path):
    import sqlite3

    from imspy_simulation.timsim.jobs.astral_acquisition import AstralAcquisitionBuilder

    template = os.environ["TIMSIM_ASTRAL_TEMPLATE"]
    out = str(tmp_path / "astral_exp")
    b = AstralAcquisitionBuilder(out, template, num_scans=64, verbose=False)

    # Interface the simulator reads — present, no Bruker reference needed.
    assert b.gradient_length > 0 and b.rt_cycle_length >= 0
    assert b.tdf_writer.helper_handle.mz_lower > 0
    assert b.tdf_writer.helper_handle.im_lower == 0.6
    assert len(b.frame_to_template_scan) == len(b.frame_table)

    # The four acquisition tables are written to synthetic_data.db.
    con = sqlite3.connect(b.synthetics_handle.database_path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"frames", "scans", "dia_ms_ms_info", "dia_ms_ms_windows"} <= tables
    n_frames = con.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    # Frame times are the template's real RTs (non-uniform): more than one distinct gap.
    times = [r[0] for r in con.execute("SELECT time FROM frames ORDER BY frame_id")]
    con.close()
    assert n_frames == len(b.frame_table)
    gaps = {round(times[i + 1] - times[i], 6) for i in range(min(50, len(times) - 1))}
    assert len(gaps) > 1, "Astral RTs should be non-uniform (template-sourced)"


@pytest.mark.skipif(
    os.environ.get("TIMSIM_ASTRAL_TEMPLATE") is None,
    reason="set TIMSIM_ASTRAL_TEMPLATE + a thermo-enabled connector for the live test",
)
def test_astral_builder_nce_override_and_precision(tmp_path):
    from imspy_simulation.timsim.jobs.astral_acquisition import AstralAcquisitionBuilder

    template = os.environ["TIMSIM_ASTRAL_TEMPLATE"]

    # Default: keep the template's genuine per-window NCE (no override).
    base = AstralAcquisitionBuilder(str(tmp_path / "a"), template, num_scans=32, verbose=False)
    assert (base.dia_ms_ms_windows["collision_energy"] > 0).all()

    # Override: every window forced to the single configured NCE.
    over = AstralAcquisitionBuilder(
        str(tmp_path / "b"), template, num_scans=32, collision_energy_nce=29.0, verbose=False
    )
    assert (over.dia_ms_ms_windows["collision_energy"] == 29.0).all()


@pytest.mark.skipif(
    os.environ.get("TIMSIM_ASTRAL_TEMPLATE") is None,
    reason="set TIMSIM_ASTRAL_TEMPLATE + a thermo-enabled connector for the live test",
)
def test_real_template_schedule_builds_tables():
    import imspy_connector

    template = os.environ["TIMSIM_ASTRAL_TEMPLATE"]
    sched = imspy_connector.py_acquisition.PyAcquisitionScheme.thermo_frame_schedule(template)
    ft, win, f2g, f2scan = build_astral_frame_tables(sched, num_scans=1)

    assert len(ft) == len(sched)
    assert len(ft) == len(f2scan)
    # The frame table's times must be exactly the template's recorded RTs, in order.
    assert list(ft["time"]) == [float(r[1]) for r in sched]
    # Real Astral: both MS1 and MS2 frames, several distinct windows.
    assert (ft["ms_type"] == 0).sum() > 0 and (ft["ms_type"] == 9).sum() > 0
    assert len(win) >= 1
    # Every MS2 frame maps to a window group.
    assert len(f2g) == int((ft["ms_type"] == 9).sum())
