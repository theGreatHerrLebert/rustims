"""Minimal fixture builders, shipped with the package.

These functions exist so that:

  1. The packaged smoke test (``imspy_simulation.provenance.test_smoke``)
     can run from an installed wheel without needing the test tree.
  2. The pytest fixtures in ``tests/test_provenance/conftest.py`` have a
     single source of truth that is identical to what the smoke test uses.

The functions are deliberately tiny and dependency-free (stdlib only).
They build a Bruker-shaped ``.d`` directory and a synthetic-data DB
without going through ``TDFWriter``, which would require a real reference
``.d`` to construct.

The leading underscore on the module name marks it as private API: it is
not part of the public ``imspy_simulation.provenance`` surface and may
change without notice.
"""

from __future__ import annotations

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
