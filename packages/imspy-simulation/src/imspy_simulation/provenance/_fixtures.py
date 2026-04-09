"""Minimal fixture builders, shipped with the package.

These functions exist so that:

  1. The packaged smoke test (``imspy_simulation.provenance.test_smoke``)
     can run from an installed wheel without needing the test tree.
  2. The pytest fixtures in ``tests/test_provenance/conftest.py`` have a
     single source of truth that is identical to what the smoke test uses.

The functions are deliberately tiny and dependency-free (stdlib only).
They build a Bruker-shaped ``.d`` directory, a synthetic-data DB, and
(for the mzML signing path) a small but representative indexed mzML
file — without going through any real simulator pipeline.

The leading underscore on the module name marks it as private API: it is
not part of the public ``imspy_simulation.provenance`` surface and may
change without notice.
"""

from __future__ import annotations

import base64
import sqlite3
import struct
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal builders for the SQLite tables and the binary peak file.
# ---------------------------------------------------------------------------


def _populate_minimal_tdf(
    conn: sqlite3.Connection,
    *,
    frames: int,
    peaks_per_frame: int,
) -> None:
    """Populate a connection with a minimal but representative TDF schema.

    Tables included (chosen to exercise every value type):
        - GlobalMetadata: TEXT key/value, including a NULL value
        - MzCalibration:  REAL coefficients, including a negative and a tiny float
        - Frames:         INTEGER + REAL columns
        - Segments:       INTEGER columns

    The data is fully deterministic given (frames, peaks_per_frame). No
    randomness, no clocks. Same inputs => byte-identical SQL content.
    """
    cur = conn.cursor()

    cur.execute("CREATE TABLE GlobalMetadata (Key TEXT, Value TEXT);")
    cur.executemany(
        "INSERT INTO GlobalMetadata (Key, Value) VALUES (?, ?);",
        [
            ("InstrumentName", "TimsTOF Pro"),
            ("InstrumentSerialNumber", "synthetic-test-fixture"),
            ("AcquisitionSoftware", "TimSim"),
            ("Description", None),  # exercises NULL
            ("UnicodeKey", "müller-naïve"),  # exercises NFC normalization
        ],
    )

    cur.execute("CREATE TABLE MzCalibration (Id INTEGER, C0 REAL, C1 REAL, C2 REAL);")
    cur.executemany(
        "INSERT INTO MzCalibration (Id, C0, C1, C2) VALUES (?, ?, ?, ?);",
        [
            (1, 0.0, 1.0, -1.5),
            (2, 1e-300, 3.141592653589793, -0.0),  # subnormal, pi, negative zero
        ],
    )

    cur.execute(
        "CREATE TABLE Frames "
        "(Id INTEGER PRIMARY KEY, Time REAL, NumPeaks INTEGER, MsType INTEGER);"
    )
    cur.executemany(
        "INSERT INTO Frames (Id, Time, NumPeaks, MsType) VALUES (?, ?, ?, ?);",
        [(i + 1, 0.5 + i * 0.25, peaks_per_frame, 0) for i in range(frames)],
    )

    cur.execute(
        "CREATE TABLE Segments (Id INTEGER PRIMARY KEY, FirstFrame INTEGER, LastFrame INTEGER);"
    )
    cur.execute(
        "INSERT INTO Segments (Id, FirstFrame, LastFrame) VALUES (?, ?, ?);",
        (1, 1, frames),
    )

    conn.commit()


def _deterministic_tdf_bin(*, frames: int, peaks_per_frame: int) -> bytes:
    """Build a deterministic byte string mimicking analysis.tdf_bin.

    64 leading zero bytes (matches TDFWriter.offset_bytes), then for each
    frame: a frame-id big-endian uint32 followed by ``peaks_per_frame``
    pairs of (mz_int, intensity_int) as big-endian uint32s.

    The exact structure does not matter for the canonicalizer — only that
    it is reproducible. The structure does matter for *tamper tests* that
    target a specific byte offset; those tests use this same layout.
    """
    out = bytearray(b"\x00" * 64)
    for f in range(1, frames + 1):
        out += struct.pack(">I", f)
        for p in range(peaks_per_frame):
            mz_int = 100_000 + p * 17 + f
            int_int = 1_000 + p * 13 + f * 5
            out += struct.pack(">II", mz_int, int_int)
    return bytes(out)


# ---------------------------------------------------------------------------
# Public builders (still under the private _fixtures namespace).
# ---------------------------------------------------------------------------


def make_minimal_d(
    tmp_path: Path,
    *,
    name: str = "test_experiment",
    frames: int = 2,
    peaks_per_frame: int = 10,
    page_size: int | None = None,
) -> Path:
    """Build a minimal but representative .d directory.

    Returns the path to the ``.d`` directory. The structure is::

        {tmp_path}/{name}.d/
            analysis.tdf
            analysis.tdf_bin

    If ``page_size`` is given, the SQLite file is created with that page
    size. This is the lever the §1.1 ``test_page_size_invariance`` test
    uses to prove the canonical hash ignores container layout.
    """
    d_path = Path(tmp_path) / f"{name}.d"
    d_path.mkdir(parents=True, exist_ok=True)

    db_path = d_path / "analysis.tdf"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        if page_size is not None:
            conn.execute(f"PRAGMA page_size = {int(page_size)};")
            # page_size only takes effect after the first write that allocates pages.
            # VACUUM forces a rewrite at the requested page size on an empty DB.
            conn.execute("VACUUM;")
        _populate_minimal_tdf(conn, frames=frames, peaks_per_frame=peaks_per_frame)
    finally:
        conn.close()

    bin_path = d_path / "analysis.tdf_bin"
    bin_path.write_bytes(_deterministic_tdf_bin(frames=frames, peaks_per_frame=peaks_per_frame))

    return d_path


def make_minimal_ground_truth(tmp_path: Path, *, name: str = "synthetic_data.db") -> Path:
    """Build a minimal synthetic_data.db.

    Tables: peptides, proteins. Includes one BLOB column to exercise
    BLOB canonicalization.
    """
    db_path = Path(tmp_path) / name
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE proteins (id INTEGER PRIMARY KEY, accession TEXT, sequence TEXT);"
        )
        cur.executemany(
            "INSERT INTO proteins (id, accession, sequence) VALUES (?, ?, ?);",
            [
                (1, "P12345", "MEEPQSDPSV"),
                (2, "Q67890", "MAGRSGDSDE"),
            ],
        )

        cur.execute(
            "CREATE TABLE peptides "
            "(id INTEGER PRIMARY KEY, sequence TEXT, mass REAL, blob_payload BLOB);"
        )
        cur.executemany(
            "INSERT INTO peptides (id, sequence, mass, blob_payload) VALUES (?, ?, ?, ?);",
            [
                (1, "PEPTIDE", 799.36, b"\x00\x01\x02\x03"),
                (2, "DESIGNS", 749.32, b"\xff\xfe\xfd\xfc"),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Tamper helpers — used by the smoke test and the pytest tamper-detection tests.
# ---------------------------------------------------------------------------


def tamper_byte(path: Path, offset: int) -> None:
    """XOR a single byte at ``offset`` in ``path`` with 1.

    Used to prove that single-byte tampering is detected.
    """
    path = Path(path)
    with open(path, "r+b") as f:
        f.seek(offset)
        b = f.read(1)
        if not b:
            raise ValueError(f"offset {offset} is past EOF for {path}")
        f.seek(offset)
        f.write(bytes([b[0] ^ 1]))


def tamper_sql_value(
    db_path: Path,
    *,
    table: str,
    set_column: str,
    set_value: object,
    where_column: str,
    where_value: object,
) -> None:
    """Run a single UPDATE inside a SQLite file.

    Used by tamper tests that target metadata. Bound parameters only —
    no string interpolation of values.
    """
    # Identifier whitelist: must be a sane name. We don't accept
    # arbitrary SQL here.
    for ident in (table, set_column, where_column):
        if not ident.replace("_", "").isalnum():
            raise ValueError(f"refusing to use suspicious identifier: {ident!r}")
    conn = sqlite3.connect(Path(db_path))
    try:
        sql = f"UPDATE {table} SET {set_column} = ? WHERE {where_column} = ?"
        conn.execute(sql, (set_value, where_value))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# mzML fixture builder
# ---------------------------------------------------------------------------


def make_minimal_mzml(
    tmp_path: Path,
    *,
    name: str = "test_run",
    n_ms1_peaks: int = 3,
    n_ms2_peaks: int = 3,
    indented: bool = True,
) -> Path:
    """Build a minimal but representative indexed mzML file.

    Two spectra: one MS1 (positive scan, 0.5s RT) and one MS2 (positive
    scan, 0.6s RT, precursor m/z 200, charge 2). Both use uncompressed
    64-bit float arrays for m/z and intensity.

    The XML is hand-written so that it round-trips through pyteomics
    cleanly. Whitespace and indentation are explicit so reformatting
    tests can flip ``indented`` to compare an indented and a compact
    version of the SAME logical content (the canonical hash must be
    invariant under that change).

    Returns the path to the written .mzML file.
    """
    # Build the m/z and intensity arrays.
    mz_values = [100.0 + 50.0 * i for i in range(max(n_ms1_peaks, n_ms2_peaks))]
    int_values = [1000.0 + 100.0 * i for i in range(max(n_ms1_peaks, n_ms2_peaks))]

    def _pack(values: list[float], n: int) -> str:
        raw = b"".join(struct.pack("<d", v) for v in values[:n])
        return base64.b64encode(raw).decode("ascii")

    mz_b64_ms1 = _pack(mz_values, n_ms1_peaks)
    int_b64_ms1 = _pack(int_values, n_ms1_peaks)
    mz_b64_ms2 = _pack(mz_values, n_ms2_peaks)
    int_b64_ms2 = _pack(int_values, n_ms2_peaks)

    # Encoded length is the number of base64 characters, not bytes.
    elen_mz_ms1 = len(mz_b64_ms1)
    elen_int_ms1 = len(int_b64_ms1)
    elen_mz_ms2 = len(mz_b64_ms2)
    elen_int_ms2 = len(int_b64_ms2)

    if indented:
        nl, ind = "\n", "  "
    else:
        nl, ind = "", ""

    mzml = (
        f'<?xml version="1.0" encoding="utf-8"?>{nl}'
        f'<indexedmzML xmlns="http://psi.hupo.org/ms/mzml" '
        f'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        f'xsi:schemaLocation="http://psi.hupo.org/ms/mzml '
        f'http://psidev.info/files/ms/mzML/xsd/mzML1.1.0_idx.xsd">{nl}'
        f'{ind}<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0" id="{name}">{nl}'
        f'{ind*2}<cvList count="2">{nl}'
        f'{ind*3}<cv id="MS" fullName="Mass spectrometry ontology" '
        f'URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/>{nl}'
        f'{ind*3}<cv id="UO" fullName="Unit Ontology" '
        f'URI="https://raw.githubusercontent.com/bio-ontologies/unit-ontology/master/unit.obo"/>{nl}'
        f'{ind*2}</cvList>{nl}'
        f'{ind*2}<fileDescription>{nl}'
        f'{ind*3}<fileContent>{nl}'
        f'{ind*4}<cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum"/>{nl}'
        f'{ind*3}</fileContent>{nl}'
        f'{ind*2}</fileDescription>{nl}'
        f'{ind*2}<run id="run1">{nl}'
        f'{ind*3}<spectrumList count="2">{nl}'
        # MS1 spectrum
        f'{ind*4}<spectrum id="scan=1" index="0" defaultArrayLength="{n_ms1_peaks}">{nl}'
        f'{ind*5}<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>{nl}'
        f'{ind*5}<cvParam cvRef="MS" accession="MS:1000130" name="positive scan"/>{nl}'
        f'{ind*5}<scanList count="1">{nl}'
        f'{ind*6}<scan>{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="0.5" '
        f'unitCvRef="UO" unitAccession="UO:0000010" unitName="second"/>{nl}'
        f'{ind*6}</scan>{nl}'
        f'{ind*5}</scanList>{nl}'
        f'{ind*5}<binaryDataArrayList count="2">{nl}'
        f'{ind*6}<binaryDataArray encodedLength="{elen_mz_ms1}">{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>{nl}'
        f'{ind*7}<binary>{mz_b64_ms1}</binary>{nl}'
        f'{ind*6}</binaryDataArray>{nl}'
        f'{ind*6}<binaryDataArray encodedLength="{elen_int_ms1}">{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000515" name="intensity array"/>{nl}'
        f'{ind*7}<binary>{int_b64_ms1}</binary>{nl}'
        f'{ind*6}</binaryDataArray>{nl}'
        f'{ind*5}</binaryDataArrayList>{nl}'
        f'{ind*4}</spectrum>{nl}'
        # MS2 spectrum
        f'{ind*4}<spectrum id="scan=2" index="1" defaultArrayLength="{n_ms2_peaks}">{nl}'
        f'{ind*5}<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2"/>{nl}'
        f'{ind*5}<cvParam cvRef="MS" accession="MS:1000130" name="positive scan"/>{nl}'
        f'{ind*5}<precursorList count="1">{nl}'
        f'{ind*6}<precursor>{nl}'
        f'{ind*7}<isolationWindow>{nl}'
        f'{ind*8}<cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="200.0"/>{nl}'
        f'{ind*8}<cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="0.5"/>{nl}'
        f'{ind*8}<cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="0.5"/>{nl}'
        f'{ind*7}</isolationWindow>{nl}'
        f'{ind*7}<selectedIonList count="1">{nl}'
        f'{ind*8}<selectedIon>{nl}'
        f'{ind*9}<cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="200.0"/>{nl}'
        f'{ind*9}<cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2"/>{nl}'
        f'{ind*8}</selectedIon>{nl}'
        f'{ind*7}</selectedIonList>{nl}'
        f'{ind*6}</precursor>{nl}'
        f'{ind*5}</precursorList>{nl}'
        f'{ind*5}<scanList count="1">{nl}'
        f'{ind*6}<scan>{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="0.6" '
        f'unitCvRef="UO" unitAccession="UO:0000010" unitName="second"/>{nl}'
        f'{ind*6}</scan>{nl}'
        f'{ind*5}</scanList>{nl}'
        f'{ind*5}<binaryDataArrayList count="2">{nl}'
        f'{ind*6}<binaryDataArray encodedLength="{elen_mz_ms2}">{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>{nl}'
        f'{ind*7}<binary>{mz_b64_ms2}</binary>{nl}'
        f'{ind*6}</binaryDataArray>{nl}'
        f'{ind*6}<binaryDataArray encodedLength="{elen_int_ms2}">{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>{nl}'
        f'{ind*7}<cvParam cvRef="MS" accession="MS:1000515" name="intensity array"/>{nl}'
        f'{ind*7}<binary>{int_b64_ms2}</binary>{nl}'
        f'{ind*6}</binaryDataArray>{nl}'
        f'{ind*5}</binaryDataArrayList>{nl}'
        f'{ind*4}</spectrum>{nl}'
        f'{ind*3}</spectrumList>{nl}'
        f'{ind*2}</run>{nl}'
        f'{ind}</mzML>{nl}'
        f'</indexedmzML>{nl}'
    )

    mzml_path = Path(tmp_path) / f"{name}.mzML"
    mzml_path.parent.mkdir(parents=True, exist_ok=True)
    mzml_path.write_text(mzml, encoding="utf-8")
    return mzml_path
