"""Spectrum-level AcquisitionWriter tests.

The vendor-neutral writer takes MS1/MS2 *spectra* (not Bruker frames). The
Thermo backend is exercised against a real Astral/Orbitrap ``.raw`` template
when one is supplied via TIMSIM_ASTRAL_TEMPLATE and the connector was built with
the ``thermo`` feature; otherwise the template-dependent tests are skipped. The
shape/protocol checks run unconditionally.
"""
import os

import numpy as np
import pytest

from imspy_simulation.writers import (
    AcquisitionWriter,
    ThermoRawWriter,
    has_thermo,
)

TEMPLATE = os.environ.get("TIMSIM_ASTRAL_TEMPLATE")
_thermo_ready = has_thermo() and TEMPLATE and os.path.exists(TEMPLATE)
needs_thermo = pytest.mark.skipif(
    not _thermo_ready,
    reason="set TIMSIM_ASTRAL_TEMPLATE to a real Astral .raw and build the "
    "connector with --features thermo",
)


class _Spec:
    """Minimal SpectrumLike (no Rust spectrum needed for the protocol checks)."""

    def __init__(self, mz, intensity):
        self._mz = np.asarray(mz, dtype=np.float64)
        self._intensity = np.asarray(intensity, dtype=np.float64)

    @property
    def mz(self):
        return self._mz

    @property
    def intensity(self):
        return self._intensity


@needs_thermo
def test_thermo_writer_round_trips_ms1_and_ms2(tmp_path):
    out = str(tmp_path / "spectrum_out.raw")
    ms1 = _Spec([400.1234, 500.5678, 750.3456], [1.0e5, 2.5e5, 3.3e5])
    ms2 = _Spec([150.05, 333.33, 555.55], [2.0e4, 1.0e5, 3.0e4])

    # Authors only 2 of the template's many slots -> opt into partial finalize.
    with ThermoRawWriter(TEMPLATE, out, allow_partial=True) as w:
        assert isinstance(w, AcquisitionWriter)
        cap_ms1, cap_ms2 = w.capacity
        assert cap_ms1 > 0 and cap_ms2 > 0
        w.write_ms1(ms1, retention_time=0.5)
        w.write_ms2(
            ms2,
            retention_time=1.5,
            isolation_center=500.0,
            isolation_width=4.0,
            collision_energy=28.0,
        )

    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


@needs_thermo
def test_thermo_writer_rejects_finalize_twice(tmp_path):
    out = str(tmp_path / "double_final.raw")
    w = ThermoRawWriter(TEMPLATE, out, allow_partial=True)
    w.write_ms1(_Spec([400.0], [1.0e5]), retention_time=0.5)
    w.finalize()
    with pytest.raises(RuntimeError):
        w.finalize()


@needs_thermo
def test_thermo_writer_rejects_mismatched_peaks(tmp_path):
    out = str(tmp_path / "mismatch.raw")
    w = ThermoRawWriter(TEMPLATE, out)
    with pytest.raises(ValueError):
        w.write_ms1(_Spec([400.0, 500.0], [1.0e5]), retention_time=0.5)


def test_spectrumlike_shape_validation_is_vendor_independent():
    # _peaks must reject ragged/2-D arrays before any vendor backend is touched.
    from imspy_simulation.writers import _peaks

    with pytest.raises(ValueError):
        _peaks(_Spec([400.0, 500.0], [1.0]))
    with pytest.raises(ValueError):
        _peaks(_Spec([[400.0], [500.0]], [[1.0], [2.0]]))
    mz, inten = _peaks(_Spec([400.0, 500.0], [1.0, 2.0]))
    assert mz.dtype == np.float64 and inten.dtype == np.float64
