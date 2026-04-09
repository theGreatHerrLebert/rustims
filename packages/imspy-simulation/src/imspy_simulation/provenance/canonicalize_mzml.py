"""Canonical content hashing for mzML files.

This is the v0 implementation for the mzML side of SIGNING.md §9. It
operates on **content** — the spectra and the per-spectrum metadata
that consumers actually read — not on the XML serialization. The
result is invariant under:

  - whitespace and indentation changes
  - attribute reordering within an element
  - cvParam ordering within a spectrum (we extract by accession, not
    by document order)
  - the optional indexed-mzML wrapper (presence or absence)
  - non-content metadata at the run level (instrument config strings,
    ``dataProcessing`` text, sourceFile listings)

It is sensitive to:

  - any change to a spectrum's m/z or intensity bytes
  - any added, removed, or reindexed spectrum
  - any change to spectrum-level metadata we hash (id, MS level, RT,
    polarity, precursor target/charge/window, ion mobility)

VERSIONING DISCIPLINE
=====================

This file is the **v0** implementation. It is **frozen**: see the same
note in ``canonicalize.py``. Future revisions live in
``canonicalize_mzml_v1.py`` and bump the ``canonicalization_version``
field in ``envelope.MzmlPayload``.

WHAT IS NOT YET SUPPORTED IN V0
================================

  - **numpress compression** (MS:1002312, MS:1002313, MS:1002314).
    Encountered? Raise ``ProvenanceError`` rather than silently
    producing an unstable hash. Phase 1 work.
  - **Run-level metadata** (instrument configuration, dataProcessing
    history, sourceFile pointers). The spectra-only scope is
    deliberate: it is the bytes consumers read.
  - **Chromatograms** (``<chromatogramList>``). Most synthetic mzML
    output we have seen does not include them; if a real-world file
    needs them in the canonical form, that is a v1 follow-up.

The byte separators ``\\x1f`` (Unit Separator) and ``\\x1e`` (Record
Separator) are the same as the SQLite path so all canonicalization
output across the package shares one wire format.
"""

from __future__ import annotations

import base64
import hashlib
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Iterator, Union

from imspy_simulation.provenance.errors import MalformedSidecar, ProvenanceError

PathLike = Union[str, Path]

CANONICALIZATION_VERSION = "v0"

# Domain prefix for the composed mzML content hash. Distinct from the
# .d domain prefix in canonicalize.py so the two cannot collide.
_MZML_CONTENT_DOMAIN = b"timsim.mzml.v0\x1f"

# Byte separators (same as canonicalize.py for cross-format consistency).
US = b"\x1f"  # Unit Separator
RS = b"\x1e"  # Record Separator

# mzML namespace. PSI-MS mzML 1.1.0 uses this URI.
_MZML_NS = "http://psi.hupo.org/ms/mzml"
_NS = {"ms": _MZML_NS}

# PSI-MS controlled-vocabulary accessions we care about. Using accession
# numbers (which are stable across releases) rather than `name` strings
# (which are localized and shortened).
_CV = {
    "ms_level":         "MS:1000511",
    "scan_start_time":  "MS:1000016",
    "positive_scan":    "MS:1000130",
    "negative_scan":    "MS:1000129",
    "selected_ion_mz":  "MS:1000744",
    "charge_state":     "MS:1000041",
    "iso_target":       "MS:1000827",
    "iso_lower_off":    "MS:1000828",
    "iso_upper_off":    "MS:1000829",
    "binary_64":        "MS:1000523",
    "binary_32":        "MS:1000521",
    "no_compression":   "MS:1000576",
    "zlib_compression": "MS:1000574",
    "numpress_linear":  "MS:1002312",
    "numpress_pic":     "MS:1002313",
    "numpress_slof":    "MS:1002314",
    "mz_array":         "MS:1000514",
    "intensity_array":  "MS:1000515",
    "ion_mobility":     "MS:1002476",  # ion mobility drift time
    "ion_mobility_alt": "MS:1003006",  # inverse reduced ion mobility
}

# Streaming chunk size for the file hasher.
_HASH_CHUNK = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Float canonicalization (must match canonicalize.py for consistency)
# ---------------------------------------------------------------------------


def _ieee754_hex(value: float) -> bytes:
    """IEEE 754 big-endian hex of a float, matching the SQLite path's encoding."""
    return struct.pack(">d", float(value)).hex().encode("ascii")


# ---------------------------------------------------------------------------
# cvParam helpers
# ---------------------------------------------------------------------------


def _cv_value(element: ET.Element, accession: str) -> str | None:
    """Find a direct-child cvParam with the given accession; return its value attribute."""
    for child in element.findall("ms:cvParam", _NS):
        if child.get("accession") == accession:
            return child.get("value")
    return None


def _cv_present(element: ET.Element, accession: str) -> bool:
    """True iff any direct-child cvParam has the given accession."""
    for child in element.findall("ms:cvParam", _NS):
        if child.get("accession") == accession:
            return True
    return False


def _cv_value_recursive(element: ET.Element, accession: str) -> str | None:
    """Find a cvParam with the given accession anywhere under ``element``."""
    for child in element.iter(f"{{{_MZML_NS}}}cvParam"):
        if child.get("accession") == accession:
            return child.get("value")
    return None


# ---------------------------------------------------------------------------
# Binary array decoding
# ---------------------------------------------------------------------------


def _decode_binary_array(bda: ET.Element) -> bytes:
    """Decode a single ``<binaryDataArray>`` element to its raw byte content.

    Returns the bytes that were originally compressed/encoded — i.e. the
    array of float32 or float64 values, native-endian-as-stored, after
    base64 and (if applicable) zlib are reversed.

    We do NOT interpret these bytes as floats; the canonical form hashes
    them as a byte string. The precision (32-bit vs 64-bit) is captured
    separately as a tag in the canonical record.
    """
    # numpress compression is not supported in v0.
    for compression_acc in (
        _CV["numpress_linear"],
        _CV["numpress_pic"],
        _CV["numpress_slof"],
    ):
        if _cv_present(bda, compression_acc):
            raise ProvenanceError(
                f"mzml binaryDataArray uses numpress compression "
                f"({compression_acc}); not supported in canonicalize_mzml v0"
            )

    binary_el = bda.find("ms:binary", _NS)
    if binary_el is None:
        raise MalformedSidecar(
            "mzml binaryDataArray is missing the inner <binary> element"
        )
    text = (binary_el.text or "").strip()
    if not text:
        return b""

    try:
        raw = base64.b64decode(text, validate=True)
    except Exception as e:
        raise MalformedSidecar(
            f"mzml binaryDataArray has undecodable base64 content: {e}"
        ) from e

    if _cv_present(bda, _CV["zlib_compression"]):
        try:
            raw = zlib.decompress(raw)
        except zlib.error as e:
            raise MalformedSidecar(
                f"mzml binaryDataArray has undecompressable zlib payload: {e}"
            ) from e
    elif not _cv_present(bda, _CV["no_compression"]):
        # Some converters omit the no-compression cvParam. Treat that
        # as "no compression" but record it in the canonical form so we
        # do not silently confuse missing-cvParam vs explicit-no-compression.
        pass

    return raw


def _array_precision_tag(bda: ET.Element) -> bytes:
    """Return a stable tag for the precision of a binaryDataArray (b'f64', b'f32', b'??')."""
    if _cv_present(bda, _CV["binary_64"]):
        return b"f64"
    if _cv_present(bda, _CV["binary_32"]):
        return b"f32"
    return b"??"


def _array_role(bda: ET.Element) -> str | None:
    """Return 'mz', 'intensity', or None for a binaryDataArray's role."""
    if _cv_present(bda, _CV["mz_array"]):
        return "mz"
    if _cv_present(bda, _CV["intensity_array"]):
        return "intensity"
    return None


# ---------------------------------------------------------------------------
# Spectrum-level extraction
# ---------------------------------------------------------------------------


def _extract_polarity(spectrum: ET.Element) -> bytes:
    if _cv_present(spectrum, _CV["positive_scan"]):
        return b"+"
    if _cv_present(spectrum, _CV["negative_scan"]):
        return b"-"
    return b"?"


def _extract_rt_seconds(spectrum: ET.Element) -> float | None:
    """Pull scan start time from the first <scan> under <scanList>.

    The unit can be ``second`` (UO:0000010) or ``minute`` (UO:0000031);
    we normalize to seconds in the canonical form.
    """
    scan_list = spectrum.find("ms:scanList", _NS)
    if scan_list is None:
        return None
    scan = scan_list.find("ms:scan", _NS)
    if scan is None:
        return None
    for cv in scan.findall("ms:cvParam", _NS):
        if cv.get("accession") == _CV["scan_start_time"]:
            try:
                value = float(cv.get("value", ""))
            except ValueError:
                return None
            unit_acc = cv.get("unitAccession", "")
            if unit_acc == "UO:0000031":  # minute
                value *= 60.0
            return value
    return None


def _extract_ion_mobility(spectrum: ET.Element) -> float | None:
    """Search the spectrum (and its scan elements) for an ion-mobility cvParam."""
    for accession in (_CV["ion_mobility"], _CV["ion_mobility_alt"]):
        v = _cv_value_recursive(spectrum, accession)
        if v is not None:
            try:
                return float(v)
            except ValueError:
                continue
    return None


def _extract_precursor(spectrum: ET.Element) -> dict | None:
    """If the spectrum has a precursor list, return a flat dict of its key fields."""
    plist = spectrum.find("ms:precursorList", _NS)
    if plist is None:
        return None
    precursor = plist.find("ms:precursor", _NS)
    if precursor is None:
        return None

    result: dict = {}
    iso = precursor.find("ms:isolationWindow", _NS)
    if iso is not None:
        for key, accession in (
            ("target", _CV["iso_target"]),
            ("lower", _CV["iso_lower_off"]),
            ("upper", _CV["iso_upper_off"]),
        ):
            v = _cv_value(iso, accession)
            if v is not None:
                try:
                    result[key] = float(v)
                except ValueError:
                    pass

    sil = precursor.find("ms:selectedIonList", _NS)
    if sil is not None:
        sion = sil.find("ms:selectedIon", _NS)
        if sion is not None:
            mz = _cv_value(sion, _CV["selected_ion_mz"])
            if mz is not None:
                try:
                    result["selected_mz"] = float(mz)
                except ValueError:
                    pass
            charge = _cv_value(sion, _CV["charge_state"])
            if charge is not None:
                try:
                    result["charge"] = int(charge)
                except ValueError:
                    pass

    return result if result else None


def _spectrum_record(spectrum: ET.Element) -> bytes:
    """Build the canonical byte record for a single spectrum.

    Field order is fixed and exhaustive: every field that the canonical
    form covers must be either emitted or explicitly absent (with an
    empty value), so two spectra that disagree on whether a field
    exists produce different bytes.
    """
    spec_id = spectrum.get("id", "")
    raw_index = spectrum.get("index", "")
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        raise MalformedSidecar(
            f"mzml spectrum {spec_id!r} has missing or non-integer index attribute"
        )

    ms_level_str = _cv_value(spectrum, _CV["ms_level"]) or ""
    polarity = _extract_polarity(spectrum)
    rt = _extract_rt_seconds(spectrum)
    mob = _extract_ion_mobility(spectrum)
    precursor = _extract_precursor(spectrum)

    # Walk the binary arrays and identify the m/z and intensity arrays.
    # We tolerate any order; we record per-array hashes by role.
    mz_hash = b""
    mz_count = b""
    mz_precision = b""
    intensity_hash = b""
    intensity_count = b""
    intensity_precision = b""

    bdal = spectrum.find("ms:binaryDataArrayList", _NS)
    if bdal is not None:
        for bda in bdal.findall("ms:binaryDataArray", _NS):
            role = _array_role(bda)
            if role is None:
                # Unknown array role (e.g. ion mobility array). Skip in v0.
                continue
            payload = _decode_binary_array(bda)
            tag = _array_precision_tag(bda)
            if tag == b"f64":
                width = 8
            elif tag == b"f32":
                width = 4
            else:
                width = 0
            count = (len(payload) // width) if width else 0
            digest = hashlib.sha256(payload).hexdigest().encode("ascii")
            if role == "mz":
                mz_hash = digest
                mz_count = str(count).encode("ascii")
                mz_precision = tag
            elif role == "intensity":
                intensity_hash = digest
                intensity_count = str(count).encode("ascii")
                intensity_precision = tag

    parts: list[bytes] = []
    def emit(key: bytes, value: bytes) -> None:
        parts.append(US + key + US + value + US)

    emit(b"spec_index", str(index).encode("ascii"))
    emit(b"spec_id", spec_id.encode("utf-8"))
    emit(b"ms_level", ms_level_str.encode("ascii"))
    emit(b"polarity", polarity)
    emit(b"rt_sec", _ieee754_hex(rt) if rt is not None else b"")
    emit(b"mobility", _ieee754_hex(mob) if mob is not None else b"")

    # Precursor block — fixed slot order, empty values when absent.
    if precursor is None:
        emit(b"prec_target", b"")
        emit(b"prec_lower", b"")
        emit(b"prec_upper", b"")
        emit(b"prec_selected", b"")
        emit(b"prec_charge", b"")
    else:
        emit(b"prec_target", _ieee754_hex(precursor["target"]) if "target" in precursor else b"")
        emit(b"prec_lower", _ieee754_hex(precursor["lower"]) if "lower" in precursor else b"")
        emit(b"prec_upper", _ieee754_hex(precursor["upper"]) if "upper" in precursor else b"")
        emit(b"prec_selected", _ieee754_hex(precursor["selected_mz"]) if "selected_mz" in precursor else b"")
        emit(b"prec_charge", str(precursor["charge"]).encode("ascii") if "charge" in precursor else b"")

    emit(b"mz_count", mz_count)
    emit(b"mz_precision", mz_precision)
    emit(b"mz_hash", mz_hash)
    emit(b"int_count", intensity_count)
    emit(b"int_precision", intensity_precision)
    emit(b"int_hash", intensity_hash)

    return b"".join(parts) + RS


# ---------------------------------------------------------------------------
# Top-level canonicalization
# ---------------------------------------------------------------------------


def _iter_spectra_sorted(root: ET.Element) -> Iterator[ET.Element]:
    """Yield spectrum elements sorted by their integer ``index`` attribute.

    Sorted order is what makes the canonical form invariant under any
    document-order rearrangement a converter might apply. Two spectra
    with the same index would be a malformed mzml; we let the duplicate
    surface naturally as identical bytes (which still produces a stable
    hash, just one that does not distinguish them).
    """
    spectra = root.iter(f"{{{_MZML_NS}}}spectrum")
    indexed = []
    for s in spectra:
        try:
            idx = int(s.get("index", ""))
        except (TypeError, ValueError):
            raise MalformedSidecar(
                f"mzml spectrum {s.get('id')!r} has missing or non-integer index"
            )
        indexed.append((idx, s))
    indexed.sort(key=lambda t: t[0])
    for _, s in indexed:
        yield s


def canonicalize_mzml(mzml_path: PathLike) -> bytes:
    """Return the SHA-256 of the canonical content form of an mzML file.

    See module docstring for what is and is not in scope. The hash is
    32 raw bytes (use ``.hex()`` for the conventional string form).
    """
    mzml_path = Path(mzml_path)
    if not mzml_path.is_file():
        raise FileNotFoundError(f"mzml file not found: {mzml_path}")

    try:
        tree = ET.parse(mzml_path)
    except ET.ParseError as e:
        raise MalformedSidecar(f"mzml file is not valid XML: {e}") from e

    root = tree.getroot()

    h = hashlib.sha256()
    h.update(b"TIMSIM-MZML-CANONICAL-v0\x1f")

    spectrum_count = 0
    for spectrum in _iter_spectra_sorted(root):
        h.update(_spectrum_record(spectrum))
        spectrum_count += 1

    h.update(b"\x1fspectrum_count\x1f" + str(spectrum_count).encode("ascii") + b"\x1f")
    return h.digest()


def compose_mzml_content_hash(
    *,
    mzml_hash: bytes,
    config_hash: bytes,
) -> bytes:
    """Compose the per-component hashes into the single content hash that gets signed.

    Layout: ``sha256(_MZML_CONTENT_DOMAIN || mzml_hash || US || config_hash)``.

    The version-tagged domain prefix protects against cross-protocol
    confusion with the .d ``compose_content_hash``.
    """
    if not isinstance(mzml_hash, (bytes, bytearray)) or len(mzml_hash) != 32:
        raise ValueError("mzml_hash must be 32 bytes")
    if not isinstance(config_hash, (bytes, bytearray)) or len(config_hash) != 32:
        raise ValueError("config_hash must be 32 bytes")
    h = hashlib.sha256()
    h.update(_MZML_CONTENT_DOMAIN)
    h.update(bytes(mzml_hash))
    h.update(US)
    h.update(bytes(config_hash))
    return h.digest()
