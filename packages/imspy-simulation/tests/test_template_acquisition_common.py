"""The instrument-neutral build-from-template primitives live in
``template_acquisition_common`` and are shared by the Astral (.raw) and SCIEX
(.wiff) acquisition builders. These tests lock that contract: the canonical
names exist, the ``astral_acquisition`` back-compat aliases still point at them
(the existing Astral test + any external import depend on it), and SCIEX imports
the same objects.
"""

from imspy_simulation.timsim.jobs import astral_acquisition as A
from imspy_simulation.timsim.jobs import sciex_acquisition as S
from imspy_simulation.timsim.jobs import template_acquisition_common as C


def test_canonical_symbols_exist():
    assert callable(C.build_frame_tables_from_schedule)
    assert callable(C.build_synthetic_scan_table)
    assert C.TemplateHelperHandle is not None
    assert C.TemplateTdfWriterStub is not None


def test_astral_backcompat_aliases_point_at_shared_objects():
    # The function/classes moved out of astral_acquisition; the old names must
    # still resolve to the SAME objects (zero-churn for existing callers/tests).
    assert A.build_astral_frame_tables is C.build_frame_tables_from_schedule
    assert A._AstralHelperHandle is C.TemplateHelperHandle
    assert A._AstralTdfWriterStub is C.TemplateTdfWriterStub


def test_sciex_uses_the_shared_transform():
    # SciexAcquisitionBuilder imports the shared transform (not a private copy).
    assert S.build_frame_tables_from_schedule is C.build_frame_tables_from_schedule
    assert S.TemplateHelperHandle is C.TemplateHelperHandle


def test_stub_handles_carry_the_ranges_jobs_read():
    handle = C.TemplateHelperHandle(100.0, 1700.0, 0.6, 1.6, 451)
    writer = C.TemplateTdfWriterStub(handle)
    assert writer.helper_handle is handle
    assert (handle.mz_lower, handle.mz_upper) == (100.0, 1700.0)
    assert (handle.im_lower, handle.im_upper, handle.num_scans) == (0.6, 1.6, 451)


# --- validate_template_schedule: method-aware scan-schedule sanity gate ---

import pytest


def _regular_dia_schedule(n_cycles=3, n_windows=2):
    """n_cycles of (1 MS1 + n_windows MS2), monotonic RT, regular cycle."""
    rows, scan = [], 0
    for c in range(n_cycles):
        scan += 1; rows.append((scan, float(scan), 1, None, None, None))
        for w in range(n_windows):
            scan += 1; rows.append((scan, float(scan), 2, 500.0 + 20.0 * w, 20.0, 25.0))
    return rows


def test_schedule_gate_accepts_valid_dia():
    assert C.validate_template_schedule(_regular_dia_schedule()) == []


def test_schedule_gate_is_called_by_build():
    # the gate runs at ingestion: a degenerate schedule fails in build_frame_tables too
    with pytest.raises(ValueError, match="no MS2"):
        C.build_frame_tables_from_schedule(
            [(i + 1, float(i), 1, None, None, None) for i in range(5)], num_scans=10
        )


def test_schedule_gate_rejects_all_ms1():
    rows = [(i + 1, float(i), 1, None, None, None) for i in range(5)]
    with pytest.raises(ValueError, match="no MS2"):
        C.validate_template_schedule(rows)


def test_schedule_gate_rejects_ms2_without_isolation():
    rows = [
        (1, 0.0, 1, None, None, None),
        (2, 0.1, 1, None, None, None),
        (3, 0.2, 2, None, None, 25.0),
    ]
    with pytest.raises(ValueError, match="mis-parsed"):
        C.validate_template_schedule(rows)


def test_schedule_gate_rejects_nonfinite_ms2_isolation():
    base = [(1, 0.0, 1, None, None, None), (2, 0.1, 1, None, None, None)]
    for bad in ((float("nan"), 20.0), (float("inf"), 20.0), (-500.0, 20.0), (500.0, 0.0), (500.0, float("nan"))):
        rows = base + [(3, 0.2, 2, bad[0], bad[1], 25.0)]
        with pytest.raises(ValueError, match="non-finite or non-positive|missing"):
            C.validate_template_schedule(rows)


def test_schedule_gate_rejects_single_ms1():
    rows = [(1, 0.0, 1, None, None, None), (2, 0.1, 2, 500.0, 20.0, 25.0)]
    with pytest.raises(ValueError, match="MS1 survey"):
        C.validate_template_schedule(rows)


def test_schedule_gate_rejects_nonmonotonic_rt():
    rows = [
        (1, 0.0, 1, None, None, None),
        (2, 0.1, 2, 500.0, 20.0, 25.0),
        (3, 0.05, 1, None, None, None),   # RT goes backwards
        (4, 0.2, 2, 500.0, 20.0, 25.0),
    ]
    with pytest.raises(ValueError, match="monotonic"):
        C.validate_template_schedule(rows)


def test_schedule_gate_warns_on_ms1_isolation_not_fails():
    # real MS1 events can carry an isolation center -> warn, do NOT hard-fail
    rows = [
        (1, 0.0, 1, None, None, None),
        (2, 0.1, 1, 500.0, 20.0, 25.0),   # MS1 carrying isolation
        (3, 0.2, 2, 500.0, 20.0, 25.0),
        (4, 0.3, 1, None, None, None),
        (5, 0.4, 2, 500.0, 20.0, 25.0),
    ]
    warns = C.validate_template_schedule(rows)
    assert any("MS1 survey" in w and "isolation" in w for w in warns)


def test_schedule_gate_consecutive_ms1_does_not_false_warn():
    # a stray extra MS1 survey (lock-mass / system scan) is a zero-MS2 "cycle" that
    # must be dropped from the regularity check, not read as irregular.
    rows = [
        (1, 0.0, 1, None, None, None),
        (2, 1.0, 2, 500.0, 20.0, 25.0), (3, 2.0, 2, 520.0, 20.0, 25.0),
        (4, 3.0, 1, None, None, None),
        (5, 4.0, 1, None, None, None),   # consecutive MS1 -> zero-MS2 "cycle"
        (6, 5.0, 2, 500.0, 20.0, 25.0), (7, 6.0, 2, 520.0, 20.0, 25.0),
        (8, 7.0, 1, None, None, None),
        (9, 8.0, 2, 500.0, 20.0, 25.0), (10, 9.0, 2, 520.0, 20.0, 25.0),
    ]
    assert not any("irregular DIA cycle" in w for w in C.validate_template_schedule(rows))


def test_schedule_gate_irregular_cycle_warns_then_fails_under_strict():
    rows, scan = [], 0
    for n_windows in (2, 3, 2):   # interior windows-per-cycle varies (2 then 3)
        scan += 1; rows.append((scan, float(scan), 1, None, None, None))
        for w in range(n_windows):
            scan += 1; rows.append((scan, float(scan), 2, 500.0 + 20.0 * w, 20.0, 25.0))
    warns = C.validate_template_schedule(rows)   # default: warn, no raise
    assert any("irregular DIA cycle" in w for w in warns)
    with pytest.raises(ValueError, match="strict_fixed_dia"):
        C.validate_template_schedule(rows, strict_fixed_dia=True)
