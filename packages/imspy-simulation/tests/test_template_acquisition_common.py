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
