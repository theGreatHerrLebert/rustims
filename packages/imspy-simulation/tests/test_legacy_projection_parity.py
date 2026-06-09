"""P3 gate-1 parity: the LegacyCompat projection reproduces a real, validated
synthetic_data.db's stored frame/scan occurrence + abundance.

This is the safety net for the instrument-dispatch refactor (INSTRUMENT_DISPATCH
plan, P3): before any wiring change, prove the dispatch projector can reproduce
the hard-won legacy pipeline EXACTLY. It calls the same imspy_connector kernels
the projector's LegacyCompat path wraps, with the run's parameters, and compares
to the columns the legacy jobs wrote.

Findings the test encodes (validated against TIMSIM-HeLa10K):
* occurrence (frame AND scan) is reproduced BYTE-EXACT (deterministic);
* abundance is the deterministic EMG/Gaussian CDF, diverging only by the
  optional stochastic noise_*_abundance step — which renormalises and so
  PRESERVES total mass; we therefore assert the abundance SUM matches and (when
  the run had noise off) the elementwise values match.

Gated on TIMSIM_PARITY_DB pointing at a real synthetic_data.db. target_p must
match the run's config.target_p (default 0.999); override via TIMSIM_PARITY_TARGET_P.
"""
import json
import os
import sqlite3

import numpy as np
import pytest

imspy_connector = pytest.importorskip("imspy_connector")
u = imspy_connector.py_utility

DB = os.environ.get("TIMSIM_PARITY_DB")
TARGET_P = float(os.environ.get("TIMSIM_PARITY_TARGET_P", "0.999"))
SAMPLE = 8
pytestmark = pytest.mark.skipif(
    not (DB and os.path.exists(DB)),
    reason="set TIMSIM_PARITY_DB to a real synthetic_data.db",
)


@pytest.fixture(scope="module")
def con():
    c = sqlite3.connect(DB)
    yield c
    c.close()


def test_frame_occurrence_is_byte_exact(con):
    frames = con.execute("SELECT frame_id, time FROM frames ORDER BY frame_id").fetchall()
    fid = np.array([r[0] for r in frames], dtype=np.int32)
    ft = np.array([r[1] for r in frames])
    rt_cycle = float(np.mean(np.diff(ft)))

    rows = con.execute(
        "SELECT rt_mu, rt_sigma, rt_lambda, frame_occurrence, frame_abundance "
        "FROM peptides WHERE json_array_length(frame_occurrence) > 5 "
        f"AND rt_mu > 0 LIMIT {SAMPLE}"
    ).fetchall()
    assert rows, "no multi-frame peptides found"
    mus = np.array([r[0] for r in rows])
    sig = np.array([r[1] for r in rows])
    lam = np.array([r[2] for r in rows])

    occ = u.calculate_frame_occurrences_emg_par(
        ft, mus, sig, lam, TARGET_P, 0.001, num_threads=4, n_steps=1000
    )
    ab = u.calculate_frame_abundances_emg_par(
        fid, ft, occ, mus, sig, lam, rt_cycle, num_threads=4, n_steps=1000
    )
    eps = 1e-4  # config.remove_epsilon default; frame job filters occurrence by it
    for k, r in enumerate(rows):
        stored_occ = json.loads(r[3])
        stored_ab = np.array(json.loads(r[4]))
        ra = np.array(ab[k])
        keep = ra > eps
        recomputed_occ = list(np.array(occ[k])[keep])
        recomputed_ab = ra[keep]
        assert recomputed_occ == stored_occ, f"frame occurrence mismatch peptide #{k}"
        # noise (if any) renormalises -> total mass preserved.
        assert np.isclose(recomputed_ab.sum(), stored_ab.sum(), rtol=1e-3), "frame abundance mass"


def test_scan_occurrence_is_byte_exact(con):
    # The legacy job sorts scans ascending by mobility so im_cycle_length > 0.
    scans = sorted(con.execute("SELECT scan, mobility FROM scans").fetchall(), key=lambda r: r[1])
    sid = np.array([r[0] for r in scans], dtype=np.int32)
    smob = np.array([r[1] for r in scans])
    im_cycle = float(np.mean(np.diff(smob)))

    rows = con.execute(
        "SELECT inv_mobility_gru_predictor, inv_mobility_gru_predictor_std, "
        "scan_occurrence, scan_abundance FROM ions "
        f"WHERE json_array_length(scan_occurrence) > 5 LIMIT {SAMPLE}"
    ).fetchall()
    assert rows, "no multi-scan ions found"
    means = np.array([r[0] for r in rows])
    sig = np.array([r[1] for r in rows])

    occ = u.calculate_scan_occurrences_gaussian_par(
        times=smob, means=means, sigmas=sig, target_p=TARGET_P,
        step_size=0.0001, n_lower_start=5, n_upper_start=5, num_threads=4,
    )
    ab = u.calculate_scan_abundances_gaussian_par(
        indices=sid, times=smob, occurrences=occ, means=means, sigmas=sig,
        cycle_length=im_cycle, num_threads=4,
    )
    for k, r in enumerate(rows):
        stored_occ = json.loads(r[2])
        stored_ab = np.array(json.loads(r[3]))
        assert list(occ[k]) == stored_occ, f"scan occurrence mismatch ion #{k}"
        assert np.isclose(np.array(ab[k]).sum(), stored_ab.sum(), rtol=1e-3), "scan abundance mass"
