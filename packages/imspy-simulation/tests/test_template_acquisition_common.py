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


# --- rt_cycle_length: per-cycle (MS1->MS1), not per-scan spacing (the fix) ---

def _astral_like_schedule():
    """3 cycles of MS1 + 2 MS2 windows; MS1 surveys 1.0 s apart, scans ~0.33 s apart."""
    rows, scan = [], 0
    for c in range(3):
        t0 = float(c)
        scan += 1; rows.append((scan, t0, 1, None, None, None))
        scan += 1; rows.append((scan, t0 + 0.33, 2, 500.0, 20.0, 25.0))
        scan += 1; rows.append((scan, t0 + 0.66, 2, 520.0, 20.0, 27.0))
    return rows


def test_rt_cycle_length_uses_ms1_spacing_not_scan_spacing():
    ft, *_ = C.build_frame_tables_from_schedule(_astral_like_schedule(), num_scans=10)
    # per-scan spacing is ~0.33 s; the true cycle (MS1->MS1) is 1.0 s
    assert abs(C.rt_cycle_length_from_ms1(ft) - 1.0) < 1e-9


def test_rt_cycle_length_raises_on_single_ms1():
    import pytest
    ft, *_ = C.build_frame_tables_from_schedule(
        [(1, 0.0, 1, None, None, None), (2, 0.33, 2, 500.0, 20.0, 25.0)], num_scans=10
    )
    with pytest.raises(ValueError):
        C.rt_cycle_length_from_ms1(ft)


def test_rt_cycle_length_matches_sonar_cycle_time():
    # Waters/SCIEX builders use the authoritative cycle_time_s; confirm it equals the MS1->MS1
    # spacing the helper derives (i.e. the per-cycle interval, not the per-scan window spacing).
    from imspy_simulation.timsim.jobs.waters_acquisition import build_sonar_schedule
    schedule, _c, _n = build_sonar_schedule(
        mz_start=400.0, mz_end=900.0, window_width=20.0, window_step=20.0,
        cycle_time_s=0.5, gradient_length_s=5.0, ce_intercept=5.0, ce_slope_per_mz=0.04,
    )
    ft, *_ = C.build_frame_tables_from_schedule(schedule, num_scans=10)
    assert abs(C.rt_cycle_length_from_ms1(ft) - 0.5) < 1e-6
