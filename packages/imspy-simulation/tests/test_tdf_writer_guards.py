"""Tests for the writer-side invariants in ``imspy_simulation.tdf``.

These exercise the duplicate-``Frames.Id`` guard added alongside the
midiA noise-sampler fix in ``rustdf/src/data/dia.rs``. The guard is the
defence-in-depth that catches the failure shape ('all corrupted rows
collapsed to Id=1') at the writer boundary, in case a future upstream
change reintroduces a class of bug we haven't anticipated.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from imspy_simulation.tdf import validate_frames_id_uniqueness


def test_validate_passes_on_unique_ids():
    df = pd.DataFrame({
        "Id": [1, 2, 3, 4, 5],
        "Time": [0.1, 0.2, 0.3, 0.4, 0.5],
        "MsMsType": [0, 9, 9, 9, 9],
    })
    # Should not raise.
    validate_frames_id_uniqueness(df)


def test_validate_passes_on_empty_frame():
    # Empty frames are not corruption — the writer may legitimately be
    # called before any rows accumulate (e.g. teardown after a failure).
    validate_frames_id_uniqueness(pd.DataFrame())
    validate_frames_id_uniqueness(pd.DataFrame({"Id": []}))


def test_validate_passes_when_id_column_missing():
    # No Id column ⇒ not the Frames table; guard should no-op rather
    # than raising on unrelated tables that happen to flow through.
    df = pd.DataFrame({"foo": [1, 2, 3]})
    validate_frames_id_uniqueness(df)


def test_validate_raises_on_single_duplicate():
    # The exact failure shape of the midiA bug: many rows sharing Id=1.
    df = pd.DataFrame({
        "Id": [1, 1, 1, 2, 3, 4],
        "Time": [0.0, 0.0, 0.0, 0.2, 0.3, 0.4],
        "MsMsType": [-1, -1, -1, 9, 9, 9],
    })
    with pytest.raises(RuntimeError) as exc:
        validate_frames_id_uniqueness(df)
    msg = str(exc.value)
    # Diagnostic must surface the duplicate count and at least one
    # offending Id.
    assert "not unique" in msg
    assert "duplicates" in msg
    assert "1:" in msg or "1," in msg  # Id 1 appears in the dupe report
    # And must point at the upstream root cause so future readers can
    # navigate to the fix.
    assert "rustdf/src/data/dia.rs" in msg


def test_validate_reports_multiple_duplicate_ids():
    # If several different IDs are duplicated, the diagnostic lists
    # them (capped at top-5).
    df = pd.DataFrame({
        "Id": [1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10],
        "Time": list(range(14)),
        "MsMsType": [9] * 14,
    })
    with pytest.raises(RuntimeError) as exc:
        validate_frames_id_uniqueness(df)
    msg = str(exc.value)
    # Both Id=1 and Id=2 should show up in the top-N report.
    assert "1:" in msg or "1," in msg
    assert "2:" in msg or "2," in msg


# ---------------------------------------------------------------------------
# Locked-output diagnostics.
#
# The failure these cover, verbatim from a user running timsim on a VM:
#
#   pandas.errors.DatabaseError: Execution failed on sql
#   'DROP TABLE "MzCalibration"': database is locked
#
# It says nothing about *why*, and the two causes need opposite fixes: a
# second live writer, or a filesystem that cannot take a POSIX lock at all
# (NFS/CIFS/vboxsf/virtiofs — i.e. exactly what you hit when you move a run
# onto a VM and write to a mounted share).
# ---------------------------------------------------------------------------

import sqlite3

from imspy_simulation import tdf as tdf_mod
from imspy_simulation.tdf import TDFWriter, _filesystem_type, _locked_database_error


class _FakeHelperHandle:
    """Minimal stand-in for the reference ``TimsDataset``.

    Supplies only what ``_setup_connections`` reads, so the writer's guards can
    be exercised without a real Bruker `.d`.
    """

    mz_calibration = pd.DataFrame({"Id": [1], "ModeIndex": [0]})
    tims_calibration = pd.DataFrame({"Id": [1], "ModeIndex": [0]})
    global_meta_data_pandas = pd.DataFrame({"Key": ["MzAcqRangeLower"], "Value": ["100"]})
    meta_data = pd.DataFrame({"Id": [1, 2, 3]})

    def get_table(self, name: str) -> pd.DataFrame:
        if name == "Segments":
            return pd.DataFrame({"Id": [1], "FirstFrame": [1], "LastFrame": [0]})
        return pd.DataFrame({"Frame": [1], "WindowGroup": [1]})


def test_locked_output_raises_actionable_error(tmp_path):
    # Simulate the real cause: another writer holding an open write transaction
    # on the output analysis.tdf (a backgrounded or suspended timsim run).
    exp = "RAW.d"
    db = tmp_path / exp / "analysis.tdf"
    db.parent.mkdir(parents=True)
    squatter = sqlite3.connect(str(db))
    squatter.execute("CREATE TABLE MzCalibration (Id INT)")
    squatter.execute("INSERT INTO MzCalibration VALUES (1)")  # holds the write lock
    assert squatter.in_transaction

    try:
        with pytest.raises(RuntimeError) as exc:
            # Short timeout: we want the diagnostic, not a 30 s wait.
            tdf_mod._SQLITE_BUSY_TIMEOUT_S, saved = 0.1, tdf_mod._SQLITE_BUSY_TIMEOUT_S
            try:
                TDFWriter(helper_handle=_FakeHelperHandle(), path=str(tmp_path), exp_name=exp)
            finally:
                tdf_mod._SQLITE_BUSY_TIMEOUT_S = saved
    finally:
        squatter.close()

    msg = str(exc.value)
    # Must name the file, both causes, and the commands that tell them apart.
    assert "database as locked" in msg
    assert str(db) in msg
    assert "ps aux | grep timsim" in msg
    assert "fuser" in msg
    assert "vboxsf" in msg or "NFS" in msg
    assert "--save_path" in msg
    # And must preserve the original SQLite wording so the error stays greppable.
    assert "database is locked" in msg


def test_lock_diagnostic_flags_unsafe_filesystem(monkeypatch, tmp_path):
    monkeypatch.setattr(tdf_mod, "_filesystem_type", lambda _p: "vboxsf")
    msg = str(_locked_database_error(tmp_path / "analysis.tdf", sqlite3.OperationalError("database is locked")))
    assert "'vboxsf'" in msg
    assert "almost certainly the cause" in msg


def test_lock_diagnostic_on_local_filesystem_points_elsewhere(monkeypatch, tmp_path):
    monkeypatch.setattr(tdf_mod, "_filesystem_type", lambda _p: "ext4")
    msg = str(_locked_database_error(tmp_path / "analysis.tdf", sqlite3.OperationalError("database is locked")))
    assert "'ext4'" in msg
    assert "cause (1) or (3) is more likely" in msg


def test_filesystem_type_resolves_a_real_path(tmp_path):
    # Whatever the CI filesystem is, the lookup must return a concrete type
    # rather than blowing up — it only ever enriches a diagnostic.
    fstype = _filesystem_type(tmp_path)
    assert isinstance(fstype, str) and fstype


def test_fresh_output_writes_without_warning(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        writer = TDFWriter(helper_handle=_FakeHelperHandle(), path=str(tmp_path), exp_name="RAW.d")
    tables = {
        r[0] for r in writer.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"MzCalibration", "TimsCalibration", "GlobalMetadata", "Segments"} <= tables
    # LastFrame must be patched from the reference meta data.
    assert writer.conn.execute("SELECT LastFrame FROM Segments").fetchone()[0] == 3


def test_reused_output_folder_warns(tmp_path):
    TDFWriter(helper_handle=_FakeHelperHandle(), path=str(tmp_path), exp_name="RAW.d")
    with pytest.warns(RuntimeWarning, match="already contains an analysis.tdf"):
        TDFWriter(helper_handle=_FakeHelperHandle(), path=str(tmp_path), exp_name="RAW.d")


def test_expect_existing_suppresses_reuse_warning(tmp_path):
    # from_existing / resume legitimately reopen a written .d.
    TDFWriter(helper_handle=_FakeHelperHandle(), path=str(tmp_path), exp_name="RAW.d")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        TDFWriter(helper_handle=_FakeHelperHandle(), path=str(tmp_path),
                  exp_name="RAW.d", expect_existing=True)
