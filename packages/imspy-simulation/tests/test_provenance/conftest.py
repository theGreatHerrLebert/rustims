"""Test fixtures and helpers for the provenance test suite.

The minimal-fixture builders live in ``imspy_simulation.provenance._fixtures``
so that the packaged smoke test can use them without depending on this
test tree. This conftest re-exports them and adds test-only helpers
(VACUUM/REINDEX/PRAGMA mutators) plus pytest fixtures.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from imspy_simulation.provenance._fixtures import (
    make_minimal_d,
    make_minimal_ground_truth,
    tamper_byte,
    tamper_sql_value,
)

__all__ = [
    "make_minimal_d",
    "make_minimal_ground_truth",
    "tamper_byte",
    "tamper_sql_value",
    "vacuum_sqlite",
    "reindex_sqlite",
    "set_user_version",
]


# ---------------------------------------------------------------------------
# Test-only helpers (not needed by the smoke test, so they stay here).
# ---------------------------------------------------------------------------


def vacuum_sqlite(db_path: Path) -> None:
    """Run VACUUM on a SQLite file. Used by §1.1 invariance tests."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("VACUUM;")
    finally:
        conn.close()


def reindex_sqlite(db_path: Path) -> None:
    """Run REINDEX on a SQLite file. Used by §1.1 invariance tests."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("REINDEX;")
    finally:
        conn.close()


def set_user_version(db_path: Path, version: int) -> None:
    """Mutate PRAGMA user_version. The canonical hash should ignore PRAGMAs."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(f"PRAGMA user_version = {int(version)};")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pytest fixtures (thin wrappers, so tests can take them as arguments)
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_d(tmp_path):
    """A freshly built minimal .d, returned as a Path."""
    return make_minimal_d(tmp_path)


@pytest.fixture
def minimal_ground_truth(tmp_path):
    """A freshly built minimal synthetic_data.db, returned as a Path."""
    return make_minimal_ground_truth(tmp_path)


@pytest.fixture
def minimal_config_bytes() -> bytes:
    """A deterministic, well-formed TimSim TOML config as raw bytes."""
    return (
        b'[paths]\n'
        b'save_path = "/tmp/test_out"\n'
        b'reference_path = "/tmp/blank.d"\n'
        b'fasta_path = "/tmp/proteome.fasta"\n'
        b'\n'
        b'[experiment]\n'
        b'experiment_name = "test_experiment"\n'
        b'acquisition_type = "DIA"\n'
    )
