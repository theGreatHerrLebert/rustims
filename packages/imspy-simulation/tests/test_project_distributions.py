"""P3c: the opt-in projector distribution writer persists the projector output
into the legacy DB columns faithfully (format + start/end + round-trip).

Gated on TIMSIM_PARITY_DB. Operates on a COPY of the DB (never mutates the
source). Proves the writer's plumbing; combined with the parity test (projector
reproduces the legacy kernels) this validates the opt-in path end-to-end.
"""
import json
import os
import shutil
import sqlite3

import numpy as np
import pytest

imspy_connector = pytest.importorskip("imspy_connector")
acq = imspy_connector.py_acquisition

from imspy_simulation.timsim.jobs.project_distributions import (  # noqa: E402
    projector_available,
    write_projected_distributions,
)

DB = os.environ.get("TIMSIM_PARITY_DB")
TARGET_P = float(os.environ.get("TIMSIM_PARITY_TARGET_P", "0.999"))
pytestmark = pytest.mark.skipif(
    not (DB and os.path.exists(DB) and projector_available()),
    reason="set TIMSIM_PARITY_DB to a real synthetic_data.db (and rebuild the connector)",
)


@pytest.fixture
def db_copy(tmp_path):
    dst = str(tmp_path / "synthetic_data.db")
    shutil.copy(DB, dst)
    return dst


def test_writer_persists_projector_output_faithfully(db_copy):
    summary = write_projected_distributions(db_copy, mode="legacy_compat", target_p=TARGET_P)
    assert summary["peptides"] > 0 and summary["ions"] > 0

    con = sqlite3.connect(db_copy)
    # Recompute the projector output directly and confirm the DB columns match it
    # (occurrence exact; abundance to the 4-decimal stored precision; start/end).
    frames = con.execute("SELECT frame_id, time FROM frames ORDER BY frame_id").fetchall()
    fid = [int(r[0]) for r in frames]
    ft = [float(r[1]) for r in frames]
    rt_cycle = float(np.mean(np.diff(ft)))

    peps = con.execute(
        "SELECT peptide_id, rt_mu, rt_sigma, rt_lambda, frame_occurrence, frame_abundance, "
        "frame_occurrence_start, frame_occurrence_end FROM peptides ORDER BY peptide_id LIMIT 12"
    ).fetchall()
    direct = acq.legacy_frame_projection(
        [float(r[1]) for r in peps], [float(r[2]) for r in peps], [float(r[3]) for r in peps],
        fid, ft, rt_cycle, TARGET_P, 0.001, 1e-4, 1000,
    )
    for r, pairs in zip(peps, direct):
        exp_occ = [int(f) for f, _ in pairs]
        exp_ab = [round(float(a), 4) for _, a in pairs]
        assert json.loads(r[4]) == exp_occ, f"peptide {r[0]} occurrence not persisted"
        assert json.loads(r[5]) == exp_ab, f"peptide {r[0]} abundance not persisted"
        assert r[6] == (exp_occ[0] if exp_occ else -1)
        assert r[7] == (exp_occ[-1] if exp_occ else -1)
    con.close()


def test_writer_is_idempotent(db_copy):
    write_projected_distributions(db_copy, mode="legacy_compat", target_p=TARGET_P)
    con = sqlite3.connect(db_copy)
    snap = con.execute(
        "SELECT frame_occurrence, scan_occurrence FROM peptides "
        "JOIN ions USING(peptide_id) ORDER BY peptide_id LIMIT 5"
    ).fetchall()
    con.close()
    write_projected_distributions(db_copy, mode="legacy_compat", target_p=TARGET_P)
    con = sqlite3.connect(db_copy)
    snap2 = con.execute(
        "SELECT frame_occurrence, scan_occurrence FROM peptides "
        "JOIN ions USING(peptide_id) ORDER BY peptide_id LIMIT 5"
    ).fetchall()
    con.close()
    assert snap == snap2, "writing twice must be idempotent"


def test_accurate_mode_not_yet_available(db_copy):
    with pytest.raises(NotImplementedError):
        write_projected_distributions(db_copy, mode="accurate")
