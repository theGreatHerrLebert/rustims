"""Canonical hashing tests — §1.1 container invariance + §1.2 determinism.

These tests are the §9 smoking gun. They prove (or refute) the central
claim of the SIGNING.md proposal: that we can canonically hash MS data in
a way that survives container-level changes (SQLite VACUUM, REINDEX,
page-size differences, insert order, PRAGMA mutations) while still
detecting any change in actual content.

If any test in §1.1 fails, the canonicalization design is wrong and we
go back to the plan before writing more code. Until they all pass,
``[provenance] sign = true`` does not become the default in TimSim.

These tests are written *before* canonicalize.py is implemented. They
will fail with NotImplementedError until §3 lands. That is the intended
state of the working tree during test-first development.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest

from imspy_simulation.provenance.canonicalize import (
    CANONICAL_NAN_BYTES,
    canonicalize_d,
    canonicalize_sqlite,
    canonicalize_value,
    compose_content_hash,
)

from .conftest import (
    make_minimal_d,
    reindex_sqlite,
    set_user_version,
    vacuum_sqlite,
)


# ---------------------------------------------------------------------------
# §1.1 Container invariance — the smoking gun
# ---------------------------------------------------------------------------


def test_vacuum_invariance(tmp_path):
    """VACUUM must not change the canonical hash.

    This is the headline §9 claim. If this fails, the canonical form is
    over container bytes, not content, and the entire proposal is unsound.
    """
    d = make_minimal_d(tmp_path)
    before = canonicalize_d(d)
    vacuum_sqlite(d / "analysis.tdf")
    after = canonicalize_d(d)
    assert before == after, (
        f"canonical hash changed under VACUUM: {before.hex()} -> {after.hex()}"
    )


def test_reindex_invariance(tmp_path):
    """REINDEX must not change the canonical hash."""
    d = make_minimal_d(tmp_path)
    before = canonicalize_d(d)
    reindex_sqlite(d / "analysis.tdf")
    after = canonicalize_d(d)
    assert before == after


def test_pragma_user_version_invariance(tmp_path):
    """Mutating PRAGMA user_version must not change the canonical hash.

    PRAGMAs are container metadata. The canonical form must ignore them.
    """
    d = make_minimal_d(tmp_path)
    before = canonicalize_d(d)
    set_user_version(d / "analysis.tdf", 42)
    after = canonicalize_d(d)
    assert before == after


def test_page_size_invariance(tmp_path):
    """Same logical content with different page_size must hash identically."""
    d_default = make_minimal_d(tmp_path / "default")
    d_8k = make_minimal_d(tmp_path / "p8k", page_size=8192)
    h_default = canonicalize_d(d_default)
    h_8k = canonicalize_d(d_8k)
    assert h_default == h_8k, (
        f"hash differs across page sizes: default={h_default.hex()}, 8k={h_8k.hex()}"
    )


def test_insert_order_invariance(tmp_path):
    """Same rows inserted in two different orders must hash identically.

    Builds two databases by hand, inserts the same rows in opposite orders,
    and asserts the canonicalizers produce identical hashes.
    """
    d_a = tmp_path / "a.d"
    d_b = tmp_path / "b.d"
    d_a.mkdir()
    d_b.mkdir()
    (d_a / "analysis.tdf_bin").write_bytes(b"\x00" * 32)
    (d_b / "analysis.tdf_bin").write_bytes(b"\x00" * 32)

    rows = [(1, "alpha", 1.0), (2, "beta", 2.0), (3, "gamma", 3.0)]

    def _build(db_path: Path, ordered_rows: list) -> None:
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE T (id INTEGER, name TEXT, val REAL);")
        conn.executemany("INSERT INTO T VALUES (?, ?, ?);", ordered_rows)
        conn.commit()
        conn.close()

    _build(d_a / "analysis.tdf", rows)
    _build(d_b / "analysis.tdf", list(reversed(rows)))

    assert canonicalize_d(d_a) == canonicalize_d(d_b)


def test_idempotent(tmp_path):
    """Calling canonicalize_d twice on the same .d returns the same bytes."""
    d = make_minimal_d(tmp_path)
    h1 = canonicalize_d(d)
    h2 = canonicalize_d(d)
    assert h1 == h2


# ---------------------------------------------------------------------------
# §1.2 Determinism — float / unicode / blob canonicalization
# ---------------------------------------------------------------------------


def test_canonicalize_value_int():
    assert canonicalize_value(0) == b"0"
    assert canonicalize_value(42) == b"42"
    assert canonicalize_value(-7) == b"-7"
    # Large int — must not use scientific notation.
    assert canonicalize_value(10**18) == b"1000000000000000000"


def test_canonicalize_value_text_nfc():
    """NFC vs NFD encodings of the same string must canonicalize identically."""
    nfc = "müller"  # u with combining diaeresis applied
    import unicodedata
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd  # sanity: they really differ in bytes
    assert canonicalize_value(nfc) == canonicalize_value(nfd)


def test_canonicalize_value_null():
    """NULL has its own distinct canonical form, distinct from any string."""
    null_form = canonicalize_value(None)
    text_form = canonicalize_value("NULL")
    assert null_form != text_form  # don't conflate NULL with the string "NULL"


def test_canonicalize_value_float_bit_exact():
    """REAL values are canonicalized as IEEE 754 big-endian hex (16 chars)."""
    pi = canonicalize_value(3.141592653589793)
    # Should be exactly 16 hex chars (8 bytes -> 16 hex digits).
    assert len(pi) == 16
    # And re-encoding the same value must give the same bytes.
    assert pi == canonicalize_value(3.141592653589793)


def test_canonicalize_value_negative_zero():
    """+0.0 and -0.0 canonicalize differently — IEEE 754 distinguishes them."""
    pos = canonicalize_value(0.0)
    neg = canonicalize_value(-0.0)
    assert pos != neg, "the canonical form must preserve the sign bit"


def test_canonicalize_value_nan_normalized():
    """All NaN payloads collapse to one canonical NaN bit pattern."""
    import math
    nan_a = float("nan")
    # construct a different NaN bit pattern
    nan_b = struct.unpack(">d", struct.pack(">Q", 0x7FFFFFFFFFFFFFFF))[0]
    assert math.isnan(nan_a) and math.isnan(nan_b)
    assert canonicalize_value(nan_a) == canonicalize_value(nan_b)
    assert canonicalize_value(nan_a) == CANONICAL_NAN_BYTES.hex().encode()


def test_canonicalize_value_blob_length_prefixed():
    """BLOBs are length-prefixed so that prefix-collisions cannot happen."""
    short = canonicalize_value(b"\x01\x02")
    long = canonicalize_value(b"\x01\x02\x03\x04")
    # The encoded form of the short blob must NOT be a prefix of the long one,
    # otherwise two different blobs could appear identical at the boundary.
    assert not long.startswith(short)


# ---------------------------------------------------------------------------
# Sensitivity tests — these MUST detect changes (the negative of invariance)
# ---------------------------------------------------------------------------


def test_added_row_changes_hash(tmp_path):
    d = make_minimal_d(tmp_path)
    before = canonicalize_d(d)
    conn = sqlite3.connect(d / "analysis.tdf")
    conn.execute(
        "INSERT INTO GlobalMetadata (Key, Value) VALUES ('Sneaky', 'Injected');"
    )
    conn.commit()
    conn.close()
    after = canonicalize_d(d)
    assert before != after, "added row was not detected"


def test_removed_row_changes_hash(tmp_path):
    d = make_minimal_d(tmp_path)
    before = canonicalize_d(d)
    conn = sqlite3.connect(d / "analysis.tdf")
    conn.execute("DELETE FROM GlobalMetadata WHERE Key='InstrumentName';")
    conn.commit()
    conn.close()
    after = canonicalize_d(d)
    assert before != after, "removed row was not detected"


def test_modified_text_changes_hash(tmp_path):
    d = make_minimal_d(tmp_path)
    before = canonicalize_d(d)
    conn = sqlite3.connect(d / "analysis.tdf")
    conn.execute(
        "UPDATE GlobalMetadata SET Value='FAKE' WHERE Key='InstrumentName';"
    )
    conn.commit()
    conn.close()
    after = canonicalize_d(d)
    assert before != after


def test_modified_real_changes_hash(tmp_path):
    d = make_minimal_d(tmp_path)
    before = canonicalize_d(d)
    conn = sqlite3.connect(d / "analysis.tdf")
    conn.execute("UPDATE MzCalibration SET C0 = 0.000001 WHERE Id=1;")
    conn.commit()
    conn.close()
    after = canonicalize_d(d)
    assert before != after, "tiny REAL change was not detected"


def test_modified_tdf_bin_changes_hash(tmp_path):
    d = make_minimal_d(tmp_path)
    before = canonicalize_d(d)
    bin_path = d / "analysis.tdf_bin"
    data = bytearray(bin_path.read_bytes())
    data[64] ^= 1
    bin_path.write_bytes(bytes(data))
    after = canonicalize_d(d)
    assert before != after


# ---------------------------------------------------------------------------
# Composition test
# ---------------------------------------------------------------------------


def test_compose_content_hash_includes_all_components():
    """The composed content hash must depend on each input component."""
    d = b"\x01" * 32
    g = b"\x02" * 32
    c = b"\x03" * 32
    base = compose_content_hash(d_hash=d, ground_truth_hash=g, config_hash=c)
    # Changing the .d hash must change the composed hash.
    assert base != compose_content_hash(
        d_hash=b"\x09" * 32, ground_truth_hash=g, config_hash=c
    )
    # Changing the ground-truth hash must change it.
    assert base != compose_content_hash(
        d_hash=d, ground_truth_hash=b"\x09" * 32, config_hash=c
    )
    # Changing the config hash must change it.
    assert base != compose_content_hash(
        d_hash=d, ground_truth_hash=g, config_hash=b"\x09" * 32
    )
    # Removing the ground truth must change it (None != all-zero).
    assert base != compose_content_hash(
        d_hash=d, ground_truth_hash=None, config_hash=c
    )
