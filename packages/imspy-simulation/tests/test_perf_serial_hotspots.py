"""Parity tests for the serial hot-spot fixes (packed-key (scan, tof) dedup, vectorised JSON
helper, persistent tdf_bin handle, stage tracker).

Each fix must reproduce the previous implementation byte-for-byte; the reference
implementations are copied here verbatim from the pre-change code.
"""
import io
import json
import logging
import os
import tempfile

import numpy as np
import pytest

from imspy_simulation.tdf import dedup_scan_tof
from imspy_simulation.utility import python_list_to_json_string


# --------------------------------------------------------------------------- references
def _dedup_reference(scan, tof, intensity):
    scan_tof = np.stack((scan.astype(np.uint32), tof.astype(np.uint32)), axis=1)
    unique_pairs, inverse_indices = np.unique(scan_tof, axis=0, return_inverse=True)
    summed_intensity = np.bincount(inverse_indices.ravel(), weights=intensity)
    unique_scan = unique_pairs[:, 0]
    unique_tof = unique_pairs[:, 1]
    sort_idx = np.lexsort((unique_tof, unique_scan))
    return unique_scan[sort_idx], unique_tof[sort_idx], summed_intensity[sort_idx].astype(np.uint32)


def _json_reference(lst, as_float=True, num_decimals=4):
    if as_float:
        return json.dumps([float(np.round(x, num_decimals)) for x in lst])
    return json.dumps([int(x) for x in lst])


# --------------------------------------------------------------------------- dedup
@pytest.mark.parametrize("seed,n,n_scans,n_tof", [(0, 10, 5, 20), (1, 5_000, 927, 400_000),
                                                   (2, 60_000, 927, 400_000), (3, 3_000, 3, 10)])
def test_dedup_scan_tof_matches_reference(seed, n, n_scans, n_tof):
    rng = np.random.default_rng(seed)
    scan = rng.integers(0, n_scans, n).astype(np.uint32)
    tof = rng.integers(0, n_tof, n).astype(np.uint32)
    intensity = rng.integers(1, 5_000, n).astype(np.uint32)
    ref = _dedup_reference(scan, tof, intensity)
    got = dedup_scan_tof(scan, tof, intensity)
    for r, g in zip(ref, got):
        np.testing.assert_array_equal(r, g)
    assert got[0].dtype == np.uint32 and got[1].dtype == np.uint32 and got[2].dtype == np.uint32
    # order contract: sorted by scan, then tof; no duplicate cells left
    key = got[0].astype(np.uint64) << np.uint64(32) | got[1].astype(np.uint64)
    assert np.all(np.diff(key) > 0)


def test_dedup_scan_tof_sums_duplicates():
    scan = np.array([2, 2, 1, 2], dtype=np.uint32)
    tof = np.array([7, 7, 9, 3], dtype=np.uint32)
    inten = np.array([10, 5, 1, 2], dtype=np.uint32)
    s, t, i = dedup_scan_tof(scan, tof, inten)
    assert s.tolist() == [1, 2, 2]
    assert t.tolist() == [9, 3, 7]
    assert i.tolist() == [1, 2, 15]


def test_dedup_scan_tof_empty():
    e = np.array([], dtype=np.uint32)
    s, t, i = dedup_scan_tof(e, e, e)
    assert len(s) == len(t) == len(i) == 0


def test_dedup_scan_tof_large_tof_values_do_not_collide():
    # tof close to 2**32 must stay separate from scan+1, tof=0
    scan = np.array([0, 1], dtype=np.uint32)
    tof = np.array([np.iinfo(np.uint32).max, 0], dtype=np.uint32)
    inten = np.array([1, 2], dtype=np.uint32)
    s, t, i = dedup_scan_tof(scan, tof, inten)
    assert s.tolist() == [0, 1] and t.tolist() == [np.iinfo(np.uint32).max, 0] and i.tolist() == [1, 2]


# --------------------------------------------------------------------------- json helper
@pytest.mark.parametrize("lst", [
    [], [0.0], [1.23456789, 2.5, 1e-7, 123456.123456], list(np.random.default_rng(0).random(300)),
    np.random.default_rng(1).random(50).astype(np.float32), [1, 2, 3], np.array([0.99995, 0.00005]),
])
def test_json_float_matches_reference(lst):
    assert python_list_to_json_string(lst) == _json_reference(lst)
    assert python_list_to_json_string(lst, num_decimals=2) == _json_reference(lst, num_decimals=2)


@pytest.mark.parametrize("lst", [[], [0], [1, 2, 3], np.array([5, 6, 7], dtype=np.int32),
                                 np.array([1.0, 2.0]), list(range(1000))])
def test_json_int_matches_reference(lst):
    assert python_list_to_json_string(lst, as_float=False) == _json_reference(lst, as_float=False)


def test_json_nan_roundtrip_like_reference():
    lst = [float('nan'), 1.0]
    assert python_list_to_json_string(lst) == _json_reference(lst)


# --------------------------------------------------------------------------- persistent handle
def test_tdf_writer_binary_handle_reopens_and_closes():
    from imspy_simulation.tdf import TDFWriter
    w = TDFWriter.__new__(TDFWriter)  # bypass __init__ (needs a Bruker handle)
    with tempfile.TemporaryDirectory() as d:
        w.binary_file = os.path.join(d, "analysis.tdf_bin")
        w._bin_fh = None
        fh = w._binary_handle()
        fh.write(b"abc")
        assert w._binary_handle() is fh
        w.close_binary()
        assert w._bin_fh is None
        w.close_binary()  # idempotent
        w._binary_handle().write(b"def")
        w.close_binary()
        with open(w.binary_file, "rb") as r:
            assert r.read() == b"abcdef"


# --------------------------------------------------------------------------- stage tracker
def test_stage_tracker_records_and_dumps():
    from imspy_simulation.timsim.simulator import StageTracker
    logger = logging.getLogger("stage-tracker-test")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    st = StageTracker(logger)
    st.begin("A")
    st.lap("first")
    st.lap("second")
    st.begin("B")
    st.annotate(build_s=1.5)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "timings.json")
        st.dump(path)
        payload = json.load(open(path))
    names = [r['stage'] for r in payload['stages']]
    assert names == ["A", "B"]
    a, b = payload['stages']
    assert set(a['laps']) == {"first", "second"}
    assert 'laps' not in b and b['build_s'] == 1.5
    for r in (a, b):
        for k in ('wall_s', 'cpu_s', 'avg_cores', 'peak_rss_gb'):
            assert k in r
    assert "stage done: A" in stream.getvalue() and "stage done: B" in stream.getvalue()
    assert len(st.summary_lines()) >= 3
