"""Vendor-neutral, spectrum-level acquisition writers.

The simulator's core currency upstream of vendor rendering is an MS1 or MS2
*spectrum* (a list of ``(m/z, intensity)`` peaks), not a vendor-specific frame.
A Bruker ``TimsFrame`` already bakes in scan/ion-mobility/TOF binning via the
Bruker calibration, so it is an *output* projection — reverse-projecting it into
another vendor's layout would be lossy. These writers instead consume the
spectrum directly and let each vendor backend render it natively:

* :class:`ThermoRawWriter` projects each spectrum onto one scan of a real
  Astral/Orbitrap ``.raw`` template (MS1 -> FTMS profile, MS2 -> ASTMS centroids
  with an isolation window), via the ``imspy_connector`` PyO3 surface.

The common contract is :class:`AcquisitionWriter`: push MS1/MS2 spectra in
acquisition order, then :meth:`AcquisitionWriter.finalize`. Writers are usable
as context managers; leaving the ``with`` block finalizes on success.
"""

from __future__ import annotations

import abc
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SpectrumLike(Protocol):
    """Anything exposing parallel ``mz`` / ``intensity`` peak arrays.

    ``imspy_core.data.MzSpectrum`` satisfies this, but so does any simple
    container, so callers aren't forced to materialise a Rust-backed spectrum
    just to write one scan.
    """

    @property
    def mz(self) -> np.ndarray: ...

    @property
    def intensity(self) -> np.ndarray: ...


def _peaks(spectrum: SpectrumLike) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(mz, intensity)`` as float64 arrays, validating they pair up."""
    mz = np.asarray(spectrum.mz, dtype=np.float64)
    intensity = np.asarray(spectrum.intensity, dtype=np.float64)
    if mz.shape != intensity.shape:
        raise ValueError(
            f"spectrum mz/intensity must be the same shape, got {mz.shape} vs "
            f"{intensity.shape}"
        )
    if mz.ndim != 1:
        raise ValueError(f"spectrum mz/intensity must be 1-D, got {mz.ndim}-D")
    return mz, intensity


class AcquisitionWriter(abc.ABC):
    """Vendor-neutral, spectrum-level acquisition sink.

    Implementations accept MS1/MS2 spectra in acquisition order and render them
    to a concrete vendor format on :meth:`finalize`. The ``retention_time`` is
    acquisition/provenance metadata; whether a given backend writes it depends
    on the format (e.g. the Thermo template preserves its own scan times).
    """

    @abc.abstractmethod
    def write_ms1(self, spectrum: SpectrumLike, retention_time: float) -> None:
        """Append an MS1 spectrum at ``retention_time`` (seconds)."""

    @abc.abstractmethod
    def write_ms2(
        self,
        spectrum: SpectrumLike,
        retention_time: float,
        isolation_center: float,
        isolation_width: float,
        collision_energy: float,
    ) -> None:
        """Append an MS2 spectrum with its isolation window + collision energy."""

    @abc.abstractmethod
    def finalize(self) -> None:
        """Flush the accumulated spectra to disk. Single-shot."""

    # Context-manager sugar: finalize on clean exit, leave the partial output
    # alone on error so it can be inspected.
    def __enter__(self) -> "AcquisitionWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finalize()
        return False


class ThermoRawWriter(AcquisitionWriter):
    """Spectrum-level writer onto a Thermo Astral/Orbitrap ``.raw`` template.

    Wraps ``imspy_connector.py_acquisition.PyThermoRawWriter``: opens a real
    ``.raw`` as a template and authors simulated spectra into its scan slots in
    acquisition order. ``Replace`` (default) rewrites template scans; passing
    ``overlay_merge_tol_ppm`` overlays simulated peaks onto the real template
    signal (real+sim). Requires the connector to be built with the ``thermo``
    feature (see :func:`has_thermo`).

    The template fixes the available MS1/MS2 scan capacity; query it with
    :meth:`capacity` before a run.
    """

    def __init__(
        self,
        template: str,
        out: str,
        overlay_merge_tol_ppm: Optional[float] = None,
    ) -> None:
        acq = _require_thermo()
        self._writer = acq.PyThermoRawWriter.from_template(
            template, out, overlay_merge_tol_ppm
        )
        self._finalized = False

    @property
    def capacity(self) -> tuple[int, int]:
        """Template ``(num_ms1, num_ms2)`` scan capacity."""
        return self._writer.capacity()

    def write_ms1(self, spectrum: SpectrumLike, retention_time: float) -> None:
        mz, intensity = _peaks(spectrum)
        self._writer.write_ms1(retention_time, mz, intensity)

    def write_ms2(
        self,
        spectrum: SpectrumLike,
        retention_time: float,
        isolation_center: float,
        isolation_width: float,
        collision_energy: float,
    ) -> None:
        mz, intensity = _peaks(spectrum)
        self._writer.write_ms2(
            retention_time,
            isolation_center,
            isolation_width,
            collision_energy,
            mz,
            intensity,
        )

    def finalize(self) -> None:
        self._writer.finalize()
        self._finalized = True


def has_thermo() -> bool:
    """Whether the installed ``imspy_connector`` was built with the Thermo
    ``.raw`` writer (the ``thermo`` feature)."""
    try:
        import imspy_connector

        return bool(imspy_connector.py_acquisition.has_thermo())
    except (ImportError, AttributeError):
        return False


def _require_thermo():
    """Return the ``py_acquisition`` module, or raise a helpful error if the
    connector lacks the Thermo writer."""
    import imspy_connector

    acq = imspy_connector.py_acquisition
    if not acq.has_thermo():
        raise RuntimeError(
            "imspy_connector was built without the 'thermo' feature; the Thermo "
            ".raw writer is unavailable. Rebuild the connector with "
            "`maturin build --release --features thermo`."
        )
    return acq
