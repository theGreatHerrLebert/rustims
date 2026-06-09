"""Golden parity: the vendor-neutral AcquisitionScheme (via imspy_connector)
must reproduce the Bruker DIA layout the legacy `use_reference_ds_layout` path
copies from the reference `.d` — both the `DiaFrameMsMsWindows` window table and
the per-frame `DiaFrameMsMsInfo` (frame→group) table.

Gated on TIMSIM_BRUKER_DIA_D pointing at a real DIA-PASEF `.d`.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import imspy_connector

acq = imspy_connector.py_acquisition
DIA_D = os.environ.get("TIMSIM_BRUKER_DIA_D")
pytestmark = pytest.mark.skipif(
    not DIA_D, reason="set TIMSIM_BRUKER_DIA_D to a real DIA-PASEF .d folder"
)


def _norm(df, cols):
    return df[cols].astype(float).round(6).sort_values(cols).reset_index(drop=True)


def test_scheme_windows_match_reference():
    scheme = acq.PyAcquisitionScheme.from_bruker_d(DIA_D)
    windows = pd.DataFrame(scheme.to_bruker_windows())

    con = sqlite3.connect(DIA_D + "/analysis.tdf")
    ref = pd.read_sql(
        "SELECT WindowGroup window_group, ScanNumBegin scan_start, ScanNumEnd scan_end, "
        "IsolationMz isolation_mz, IsolationWidth isolation_width, "
        "CollisionEnergy collision_energy FROM DiaFrameMsMsWindows",
        con,
    )
    cols = list(ref.columns)
    assert len(windows) == len(ref)
    pd.testing.assert_frame_equal(_norm(windows, cols), _norm(ref, cols))


def test_scheme_info_matches_reference():
    # The true contract: the scheme's DiaFrameMsMsInfo must reproduce the
    # reference .d's table exactly. (We intentionally do NOT assert equality with
    # the legacy `window_group = index % precursor_every` position formula: that
    # only holds for canonical 1..N references, and where it differs the scheme is
    # the correct one — it preserves the real WindowGroup ids.)
    scheme = acq.PyAcquisitionScheme.from_bruker_d(DIA_D)
    con = sqlite3.connect(DIA_D + "/analysis.tdf")
    num_frames = int(con.execute("SELECT MAX(Id) FROM Frames").fetchone()[0])

    info = pd.DataFrame(scheme.to_bruker_info(num_frames))
    ref = pd.read_sql("SELECT Frame frame, WindowGroup window_group FROM DiaFrameMsMsInfo", con)
    assert len(info) == len(ref)
    pd.testing.assert_frame_equal(
        _norm(info, ["frame", "window_group"]), _norm(ref, ["frame", "window_group"])
    )
    # info is emitted frame-ascending.
    assert info["frame"].is_monotonic_increasing
