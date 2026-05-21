"""Tests for the writer-side invariants in ``imspy_simulation.tdf``.

These exercise the duplicate-``Frames.Id`` guard added alongside the
midiA noise-sampler fix in ``rustdf/src/data/dia.rs``. The guard is the
defence-in-depth that catches the failure shape ('all corrupted rows
collapsed to Id=1') at the writer boundary, in case a future upstream
change reintroduces a class of bug we haven't anticipated.
"""

from __future__ import annotations

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
