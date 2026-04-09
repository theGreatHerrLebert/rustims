"""Performance benchmarks for provenance hashing.

These tests are marked ``@pytest.mark.slow`` and are excluded from the
fast inner-loop suite (which targets <1s wall-clock total). Run them
explicitly with::

    pytest tests/test_provenance/test_performance.py -m slow

The budgets are deliberately conservative — at least 4x the measured
real-world throughput on the dev machine — so they catch genuine
regressions without being flaky on a busy CI runner.

Measured baseline (single-threaded SHA-256 + SQLite read on a modern
x86_64 host, page-cached):

    .d size       canonicalize_d  throughput
    -----------  ---------------  ----------
      270 MB       ~475 ms         ~570 MB/s
      580 MB       ~651 ms         ~890 MB/s

The bin file dominates total bytes; the SQL canonical form is small.
The bin path is streaming SHA-256 at near-disk-bandwidth, so the
relationship between size and time is nearly linear. The budgets
below give 4x slack on the SHA-256 portion and a fixed-cost margin
for the SQL portion.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from imspy_simulation.provenance import (
    sign_simulation_output,
    verify_sidecar,
)
from imspy_simulation.provenance.canonicalize import (
    canonicalize_d,
    canonicalize_sqlite,
)
from imspy_simulation.provenance.keys import generate_keypair, write_keypair

from .conftest import make_minimal_d


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_d_with_large_bin(tmp_path: Path, *, bin_size_mb: int, name: str) -> Path:
    """Build a .d whose analysis.tdf_bin is exactly ``bin_size_mb`` MiB.

    The SQLite analysis.tdf is the small minimal fixture; only the bin
    file is bulked up. We write deterministic bytes (a repeating
    pattern) so the test is reproducible without burning entropy on
    /dev/urandom.
    """
    d_path = make_minimal_d(tmp_path, name=name)
    bin_path = d_path / "analysis.tdf_bin"
    chunk = b"\x00\x01\x02\x03" * (256 * 1024)  # 1 MiB chunk
    with open(bin_path, "wb") as f:
        for _ in range(bin_size_mb):
            f.write(chunk)
    return d_path


def _make_d_with_large_table(tmp_path: Path, *, n_rows: int, name: str) -> Path:
    """Build a .d whose Frames table has ``n_rows`` rows.

    Stresses the SQL canonical-dump path (every cell goes through
    canonicalize_value, every row is sorted via ORDER BY all columns).
    """
    d_path = make_minimal_d(tmp_path, name=name)
    conn = sqlite3.connect(d_path / "analysis.tdf")
    try:
        conn.execute("DELETE FROM Frames;")
        rows = [(i + 1, 0.5 + i * 0.01, 100, 0) for i in range(n_rows)]
        conn.executemany(
            "INSERT INTO Frames (Id, Time, NumPeaks, MsType) VALUES (?, ?, ?, ?);",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return d_path


# ---------------------------------------------------------------------------
# Bin-size benchmarks: measures the streaming SHA-256 path
# ---------------------------------------------------------------------------


def test_canonicalize_100mb_bin_under_3s(tmp_path):
    """100 MiB analysis.tdf_bin must canonicalize in under 3 seconds.

    Real measurement: ~180ms on the dev machine. Budget: 3s = ~17x slack.
    """
    d = _make_d_with_large_bin(tmp_path, bin_size_mb=100, name="perf100")
    start = time.perf_counter()
    h = canonicalize_d(d)
    elapsed = time.perf_counter() - start
    assert h is not None and len(h) == 32
    assert elapsed < 3.0, (
        f"100 MiB canonicalize_d took {elapsed:.3f}s, budget 3.0s "
        f"(measured baseline ~0.18s). Likely a regression in the streaming "
        f"hasher or SQL dump path."
    )


def test_canonicalize_500mb_bin_under_10s(tmp_path):
    """500 MiB analysis.tdf_bin must canonicalize in under 10 seconds.

    Real measurement: ~620ms on the dev machine. Budget: 10s = ~16x slack.
    """
    d = _make_d_with_large_bin(tmp_path, bin_size_mb=500, name="perf500")
    start = time.perf_counter()
    h = canonicalize_d(d)
    elapsed = time.perf_counter() - start
    assert h is not None and len(h) == 32
    assert elapsed < 10.0, (
        f"500 MiB canonicalize_d took {elapsed:.3f}s, budget 10.0s "
        f"(measured baseline ~0.62s)."
    )


def test_canonicalize_idempotent_on_large_bin(tmp_path):
    """Two consecutive canonicalize_d calls on a 100 MiB .d return the same hash."""
    d = _make_d_with_large_bin(tmp_path, bin_size_mb=100, name="perf_idem")
    h1 = canonicalize_d(d)
    h2 = canonicalize_d(d)
    assert h1 == h2


# ---------------------------------------------------------------------------
# Table-size benchmarks: measures the SQL canonical-dump path
# ---------------------------------------------------------------------------


def test_canonicalize_50k_row_table_under_3s(tmp_path):
    """A 50k-row Frames table (~larger than any real timsTOF metadata table)
    must canonicalize in under 3 seconds. Stresses the ORDER BY all columns
    + per-cell canonicalize_value path."""
    d = _make_d_with_large_table(tmp_path, n_rows=50_000, name="perf_rows")
    start = time.perf_counter()
    h = canonicalize_d(d)
    elapsed = time.perf_counter() - start
    assert h is not None and len(h) == 32
    assert elapsed < 3.0, (
        f"50k-row canonicalize_d took {elapsed:.3f}s, budget 3.0s. "
        f"Likely a regression in canonicalize_value or the SELECT/ORDER BY "
        f"path."
    )


# ---------------------------------------------------------------------------
# End-to-end signing benchmark
# ---------------------------------------------------------------------------


def test_sign_simulation_output_under_5s_on_100mb_d(tmp_path):
    """Full sign_simulation_output (hash + sign + write sidecar) on a
    100 MiB .d must complete in under 5 seconds.

    The Ed25519 sign step is microseconds; this is essentially the same
    budget as the canonicalize benchmark plus tiny overhead.
    """
    d = _make_d_with_large_bin(tmp_path, bin_size_mb=100, name="perf_sign")
    config = tmp_path / "perf.toml"
    config.write_bytes(b'[experiment]\nexperiment_name = "perf_sign"\n')
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)

    start = time.perf_counter()
    sidecar = sign_simulation_output(
        d_path=d,
        ground_truth_path=None,
        config_path=config,
        experiment_name="perf_sign",
        simulator_version="perf-test",
        private_key_path=key_dir / "signing_key.pem",
    )
    elapsed = time.perf_counter() - start
    assert sidecar.exists()
    assert elapsed < 5.0, (
        f"sign_simulation_output on 100 MiB .d took {elapsed:.3f}s, "
        f"budget 5.0s (sign should be ~hash + tiny constant)."
    )


def test_verify_round_trip_under_5s_on_100mb_d(tmp_path):
    """Sign and then verify a 100 MiB .d under 5 seconds (verify only)."""
    d = _make_d_with_large_bin(tmp_path, bin_size_mb=100, name="perf_verify")
    config = tmp_path / "perf.toml"
    config.write_bytes(b'[experiment]\nexperiment_name = "perf_verify"\n')
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)

    sidecar = sign_simulation_output(
        d_path=d,
        ground_truth_path=None,
        config_path=config,
        experiment_name="perf_verify",
        simulator_version="perf-test",
        private_key_path=key_dir / "signing_key.pem",
    )

    start = time.perf_counter()
    result = verify_sidecar(sidecar)
    elapsed = time.perf_counter() - start
    assert result.overall_ok
    assert elapsed < 5.0, (
        f"verify_sidecar on 100 MiB .d took {elapsed:.3f}s, budget 5.0s."
    )
