"""Instrument-dispatch P3c: opt-in projector-driven distribution writer.

The Rust frame builder consumes the per-ion/peptide occurrence + abundance
*columns* from ``synthetic_data.db``. This job (re)generates those columns from
the vendor-neutral projector (``imspy_connector`` ``legacy_frame_projection`` /
``legacy_scan_projection`` -> rustdf ``project_time_legacy`` /
``project_mobility_ion_legacy``) and writes them back in the exact legacy
format, so downstream assembly is byte-compatible.

It is **opt-in and default-off**: nothing calls it unless the simulator is run
with the projector flag. When it is NOT called, the legacy
``simulate_frame_distributions_emg`` / ``simulate_scan_distributions_with_variance``
output is used unchanged — the DB-read fallback. This is the safe wiring point
that lets the projector drive distributions without disturbing the hard-won
legacy path until explicitly chosen.

`mode="legacy_compat"` reproduces the legacy kernels exactly (the validated
default-when-opted-in). `mode="accurate"` (event-interval time projection +
per-scan mobility bins) is reserved until the Accurate projector is exposed
through the connector and independently validated.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

import numpy as np

import imspy_connector

from imspy_simulation.utility import python_list_to_json_string

logger = logging.getLogger(__name__)

_acq = imspy_connector.py_acquisition


def projector_available() -> bool:
    """Whether the installed connector exposes the LegacyCompat projector."""
    return hasattr(_acq, "legacy_frame_projection") and hasattr(_acq, "legacy_scan_projection")


def write_projected_distributions(
    db_path: str,
    *,
    mode: str = "legacy_compat",
    target_p: float = 0.999,
    frame_step_size: float = 0.001,
    scan_step_size: float = 0.0001,
    n_steps: int = 1000,
    remove_epsilon: float = 1e-4,
    num_decimals: int = 4,
    num_threads: int = 4,
) -> dict:
    """Regenerate the frame/scan distribution columns in ``db_path`` via the
    projector. Writes ``peptides.frame_occurrence/frame_abundance/
    frame_occurrence_start/frame_occurrence_end`` and ``ions.scan_occurrence/
    scan_abundance`` in the legacy on-disk format.

    Returns a summary dict. Raises if the connector lacks the projector or the
    DB is missing the required columns (``rt_mu`` etc.).
    """
    if mode != "legacy_compat":
        raise NotImplementedError(
            f"projection mode {mode!r} not available yet; only 'legacy_compat' is exposed"
        )
    if not projector_available():
        raise RuntimeError(
            "imspy_connector lacks the LegacyCompat projector (legacy_frame_projection / "
            "legacy_scan_projection); rebuild the connector."
        )

    con = sqlite3.connect(db_path)
    try:
        n_pep = _write_frame_distributions(
            con, target_p, frame_step_size, n_steps, remove_epsilon, num_decimals, num_threads
        )
        n_ion = _write_scan_distributions(con, target_p, scan_step_size, num_decimals, num_threads)
        con.commit()
    finally:
        con.close()

    logger.info(
        "projector distributions written: %d peptides, %d ions (mode=%s)", n_pep, n_ion, mode
    )
    return {"mode": mode, "peptides": n_pep, "ions": n_ion}


def _write_frame_distributions(con, target_p, step_size, n_steps, remove_epsilon, num_decimals, num_threads) -> int:
    frames = con.execute("SELECT frame_id, time FROM frames ORDER BY frame_id").fetchall()
    frame_ids = [int(r[0]) for r in frames]
    frame_times = [float(r[1]) for r in frames]
    if len(frame_times) < 2:
        raise ValueError("frames table needs >= 2 rows to derive rt_cycle_length")
    rt_cycle = float(np.mean(np.diff(frame_times)))

    peps = con.execute(
        "SELECT peptide_id, rt_mu, rt_sigma, rt_lambda FROM peptides ORDER BY peptide_id"
    ).fetchall()
    if not peps:
        return 0
    pep_ids = [int(r[0]) for r in peps]
    out = _acq.legacy_frame_projection(
        [float(r[1]) for r in peps],
        [float(r[2]) for r in peps],
        [float(r[3]) for r in peps],
        frame_ids,
        frame_times,
        rt_cycle,
        target_p,
        step_size,
        remove_epsilon,
        n_steps,
        num_threads,
    )
    cur = con.cursor()
    for pid, pairs in zip(pep_ids, out):
        occ = [int(f) for f, _ in pairs]
        ab = [float(a) for _, a in pairs]
        start = occ[0] if occ else -1
        end = occ[-1] if occ else -1
        cur.execute(
            "UPDATE peptides SET frame_occurrence = ?, frame_abundance = ?, "
            "frame_occurrence_start = ?, frame_occurrence_end = ? WHERE peptide_id = ?",
            (
                python_list_to_json_string(occ, as_float=False),
                python_list_to_json_string(ab, as_float=True, num_decimals=num_decimals),
                start,
                end,
                pid,
            ),
        )
    return len(pep_ids)


def _write_scan_distributions(con, target_p, step_size, num_decimals, num_threads) -> int:
    # Legacy convention: scans ascending by mobility so im_cycle_length > 0.
    scans = sorted(con.execute("SELECT scan, mobility FROM scans").fetchall(), key=lambda r: r[1])
    if len(scans) < 2:
        raise ValueError("scans table needs >= 2 rows to derive im_cycle_length")
    scan_ids = [int(r[0]) for r in scans]
    scan_mob = [float(r[1]) for r in scans]
    im_cycle = float(np.mean(np.diff(scan_mob)))

    ions = con.execute(
        "SELECT ion_id, inv_mobility_gru_predictor, inv_mobility_gru_predictor_std "
        "FROM ions ORDER BY ion_id"
    ).fetchall()
    if not ions:
        return 0
    ion_ids = [int(r[0]) for r in ions]
    # One batched, parallel call (mirrors the legacy job's *_par kernels) rather
    # than a per-ion Python loop.
    out = _acq.legacy_scan_projection_par(
        [float(r[1]) for r in ions],
        [float(r[2]) for r in ions],
        scan_ids,
        scan_mob,
        im_cycle,
        target_p,
        step_size,
        num_threads,
    )
    cur = con.cursor()
    for ion_id, pairs in zip(ion_ids, out):
        occ = [int(s) for s, _ in pairs]
        ab = [float(a) for _, a in pairs]
        cur.execute(
            "UPDATE ions SET scan_occurrence = ?, scan_abundance = ? WHERE ion_id = ?",
            (
                python_list_to_json_string(occ, as_float=False),
                python_list_to_json_string(ab, as_float=True, num_decimals=num_decimals),
                ion_id,
            ),
        )
    return len(ion_ids)
