"""Canonical content hashing for TimSim outputs.

This module is the working prototype for SIGNING.md §9: a canonical hash over
mass spectrometry data that survives container-level changes (SQLite VACUUM,
REINDEX, page-size differences) because it operates on *content*, not on
container bytes.

Construction (see plan §3):

    canonical_sql_dump(db) emits a stream of records:

        \\x1ftable\\x1f{name}\\x1f
        \\x1fcol\\x1f{col_name}\\x1f{col_type}\\x1f   (one per column, in cid order)
        \\x1frow\\x1f{val_1}\\x1f{val_2}\\x1f...{val_N}\\x1f\\x1e   (one per row,
                                              ordered by ALL columns ascending)

    Tables are emitted in alphabetical name order. Rows are ordered by all
    columns so insert order, page layout, and rowid quirks do not matter.

    canonicalize_value renders each cell:
        NULL    -> b"\\x00NULL\\x00"
        INTEGER -> decimal ASCII
        REAL    -> 16 hex chars (IEEE 754 big-endian); NaN normalized to
                   0x7ff8000000000000 to avoid payload-bit divergence
        TEXT    -> b"\\x00len{N}\\x00" + UTF-8 bytes after NFC normalization
        BLOB    -> b"\\x00blob{N}\\x00" + lowercase hex

    The whole record stream is fed into a streaming SHA-256.

The byte separators \\x1f (Unit Separator) and \\x1e (Record Separator)
are control bytes that cannot appear in legal SQL identifiers, which
makes the serialization unambiguous without escaping.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
import unicodedata
from pathlib import Path
from typing import Iterator, Union

PathLike = Union[str, Path]

CANONICALIZATION_VERSION = "v0"

# Byte separators. Chosen because they cannot legally appear in SQL
# identifiers and they are not whitespace, so they will not be silently
# normalized by any text-handling layer.
US = b"\x1f"  # Unit Separator — between fields
RS = b"\x1e"  # Record Separator — between rows

# NaN normalization: a single canonical NaN bit pattern. Real MS data
# should never contain NaN; we collapse all NaN payloads to one value
# so platform-specific NaN handling does not break canonicalization.
CANONICAL_NAN_BYTES = struct.pack(">Q", 0x7FF8000000000000)

# Streaming chunk size for the binary file hasher.
_BIN_CHUNK = 1 << 20  # 1 MiB

# Domain prefix for the composed content hash. The version tag protects
# us against future canonicalizer revisions colliding on the same input.
_CONTENT_DOMAIN = b"timsim.v0\x1f"


# ---------------------------------------------------------------------------
# Value canonicalization
# ---------------------------------------------------------------------------


def canonicalize_value(value: object) -> bytes:
    """Render a single SQLite cell value to its canonical byte form.

    See module docstring for the encoding table. This function is the
    only place where type-specific normalization happens; everything
    else in canonicalize.py operates on the bytes this function returns.
    """
    if value is None:
        return b"\x00NULL\x00"

    # bool is a subclass of int — handle it before int.
    if isinstance(value, bool):
        return b"1" if value else b"0"

    if isinstance(value, int):
        return str(value).encode("ascii")

    if isinstance(value, float):
        if math.isnan(value):
            return CANONICAL_NAN_BYTES.hex().encode("ascii")
        # IEEE 754 double, big-endian, 8 bytes -> 16 hex chars.
        return struct.pack(">d", value).hex().encode("ascii")

    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value).encode("utf-8")
        return b"\x00len" + str(len(normalized)).encode("ascii") + b"\x00" + normalized

    if isinstance(value, (bytes, bytearray, memoryview)):
        b = bytes(value)
        return b"\x00blob" + str(len(b)).encode("ascii") + b"\x00" + b.hex().encode("ascii")

    # SQLite has no other storage classes. Anything else is a programming error.
    raise TypeError(f"canonicalize_value: unsupported type {type(value).__name__}")


# ---------------------------------------------------------------------------
# SQL dump canonicalization
# ---------------------------------------------------------------------------


def _list_user_tables(conn: sqlite3.Connection) -> list[str]:
    """Return user table names in alphabetical order, excluding sqlite internals."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name;"
    )
    return [row[0] for row in cur.fetchall()]


def _quote_ident(ident: str) -> str:
    """Quote a SQL identifier as a double-quoted name. Doubles embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _table_columns(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    """Return [(column_name, declared_type)] in declared (cid) order."""
    cur = conn.execute(f"PRAGMA table_info({_quote_ident(table)});")
    cols: list[tuple[int, str, str]] = [(row[0], row[1], row[2] or "") for row in cur.fetchall()]
    cols.sort(key=lambda r: r[0])
    return [(name, declared_type) for _cid, name, declared_type in cols]


def _iter_canonical_sql_records(conn: sqlite3.Connection) -> Iterator[bytes]:
    """Yield the canonical record stream of a SQLite database connection.

    The stream is the input to the SHA-256 hasher. Yielding lets us
    stream rather than build the entire blob in memory.
    """
    for table in _list_user_tables(conn):
        cols = _table_columns(conn, table)
        yield US + b"table" + US + table.encode("utf-8") + US

        for col_name, col_type in cols:
            yield (
                US + b"col" + US
                + col_name.encode("utf-8") + US
                + col_type.encode("utf-8") + US
            )

        if not cols:
            continue

        order_by = ", ".join(_quote_ident(name) for name, _ in cols)
        select_sql = f"SELECT * FROM {_quote_ident(table)} ORDER BY {order_by};"
        cur = conn.execute(select_sql)
        for row in cur:
            parts = [US + b"row"]
            for value in row:
                parts.append(US + canonicalize_value(value))
            parts.append(US + RS)
            yield b"".join(parts)


def canonicalize_sqlite(db_path: PathLike) -> bytes:
    """Return the SHA-256 of the canonical SQL dump of a SQLite file.

    The hash is over *content*, not file bytes. It is invariant under:
        - VACUUM
        - REINDEX
        - PRAGMA user_version changes
        - Different page_size settings
        - Different insert orders for the same logical content

    It is sensitive to:
        - Any difference in row content
        - Any added or removed row
        - Any added or removed column or table
        - Any column type difference
    """
    db_path = Path(db_path)
    # Read-only URI mode so we never accidentally modify the database we're
    # hashing. immutable=1 also tells SQLite to skip locking, which is faster
    # and avoids creating -journal files next to the .d.
    uri = f"file:{db_path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        h = hashlib.sha256()
        for chunk in _iter_canonical_sql_records(conn):
            h.update(chunk)
        return h.digest()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# .d directory hashing
# ---------------------------------------------------------------------------


def _hash_file_streaming(path: Path) -> bytes:
    """SHA-256 a file in chunks. Constant memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_BIN_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.digest()


def canonicalize_d(d_path: PathLike) -> bytes:
    """Return the canonical content hash of a Bruker .d directory.

    Composes ``analysis.tdf_bin`` (raw streaming SHA-256) with
    ``analysis.tdf`` (canonical SQL dump). Returns 32 raw bytes.
    """
    d_path = Path(d_path)
    if not d_path.is_dir():
        raise FileNotFoundError(f".d path is not a directory: {d_path}")

    tdf = d_path / "analysis.tdf"
    tdf_bin = d_path / "analysis.tdf_bin"
    if not tdf.is_file():
        raise FileNotFoundError(f"missing analysis.tdf in {d_path}")
    if not tdf_bin.is_file():
        raise FileNotFoundError(f"missing analysis.tdf_bin in {d_path}")

    bin_hash = _hash_file_streaming(tdf_bin)
    tdf_hash = canonicalize_sqlite(tdf)

    return hashlib.sha256(bin_hash + tdf_hash).digest()


def canonicalize_bytes(data: bytes) -> bytes:
    """Return the SHA-256 of an arbitrary byte string.

    Used for hashing the user's raw config TOML so the attestation
    binds the simulation to the *exact bytes* the user wrote, not a
    re-serialized normalization.
    """
    return hashlib.sha256(data).digest()


def compose_content_hash(
    *,
    d_hash: bytes,
    ground_truth_hash: bytes | None,
    config_hash: bytes,
) -> bytes:
    """Compose the per-component hashes into the single content hash that gets signed.

    Layout:
        sha256(_CONTENT_DOMAIN || d_hash || US || gt_marker || US || config_hash)

    where ``gt_marker`` is the ground-truth hash bytes if present, or
    the literal ``b"none"`` if absent. Using a marker (rather than just
    omitting the field) ensures None and an all-zero ground truth do not
    collide.
    """
    if not isinstance(d_hash, (bytes, bytearray)) or len(d_hash) != 32:
        raise ValueError("d_hash must be 32 bytes")
    if not isinstance(config_hash, (bytes, bytearray)) or len(config_hash) != 32:
        raise ValueError("config_hash must be 32 bytes")
    if ground_truth_hash is None:
        gt_marker = b"none"
    else:
        if not isinstance(ground_truth_hash, (bytes, bytearray)) or len(ground_truth_hash) != 32:
            raise ValueError("ground_truth_hash must be 32 bytes or None")
        gt_marker = bytes(ground_truth_hash)

    h = hashlib.sha256()
    h.update(_CONTENT_DOMAIN)
    h.update(bytes(d_hash))
    h.update(US)
    h.update(gt_marker)
    h.update(US)
    h.update(bytes(config_hash))
    return h.digest()
