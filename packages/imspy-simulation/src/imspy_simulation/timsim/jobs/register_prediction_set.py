"""P5a: register the fragment-ion *prediction set*.

The ``fragment_ions`` table is NOT an instrument-neutral trunk artifact — its
rows are predicted at a specific collision energy (from the acquisition) by a
specific intensity model, for a specific acquisition/instrument. P5 makes that
explicit: every run records ONE row in ``prediction_sets`` describing how its
fragment intensities were produced, and stamps ``fragment_ions.prediction_set_id``
so a renderer can later verify the stored fragments match the instrument it is
rendering for (P5b adds that resolution + fail-on-mismatch).

This step is additive and idempotent (safe on checkpoint resume): it does not
change which fragment rows exist or their values.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# Schema version for the prediction-set namespace (bump on shape changes).
PREDICTION_SET_SCHEMA_VERSION = 1

# The single set a run produces is id 0. (Multi-set DBs — re-fragmenting one
# trunk for several instruments — are a later extension; the id space is ready.)
DEFAULT_PREDICTION_SET_ID = 0

# P6d: the activation regime each supported instrument records. The collision
# energy a run STORES (and the renderer APPLIES) is in this unit; the selected
# fragment-intensity model's capability must accept it (see
# fragment_predictor_capability.assert_predictor_supports). Bruker timsTOF is the
# absolute-eV collisional case (unchanged default); Orbitrap Astral is HCD with a
# normalized collision energy (NCE). The CE-encoding stays 'normalized_div100' for
# both — the stored value is CE/100 regardless of unit (verified), so the
# unit-agnostic render keying is unchanged; only the unit label differs.
INSTRUMENT_ACTIVATION = {
    "bruker_timstof": {"activation_method": "hcd", "energy_unit": "ev"},
    "orbitrap_astral": {"activation_method": "hcd", "energy_unit": "nce"},
}


def resolve_instrument_activation(instrument: str) -> tuple[str, str]:
    """Return ``(activation_method, energy_unit)`` for a supported instrument.

    Raises on an unknown instrument rather than silently assuming the Bruker
    eV contract (which would mis-label an instrument's collision energy)."""
    spec = INSTRUMENT_ACTIVATION.get((instrument or "bruker_timstof").lower())
    if spec is None:
        raise ValueError(
            f"unknown instrument '{instrument}'. Supported: "
            f"{sorted(INSTRUMENT_ACTIVATION)}. Each declares the activation "
            f"method and collision-energy unit its run stores."
        )
    return spec["activation_method"], spec["energy_unit"]


def register_prediction_set(
    db_path: str,
    *,
    predictor_model: str | None,
    acquisition_type: str,
    instrument: str = "bruker_timstof",
    activation_method: str = "hcd",
    energy_unit: str = "ev",
    collision_energy_encoding: str = "normalized_div100",
    down_sample_factor: float = 0.0,
) -> int:
    """Create/refresh ``prediction_sets`` and stamp ``fragment_ions``.

    Args:
        db_path: path to ``synthetic_data.db``.
        predictor_model: intensity model name (None -> default prosit timsTOF).
        acquisition_type: 'DIA' or 'DDA'.
        instrument: instrument identity the fragments were predicted for.
        activation_method: dissociation method (collisional -> 'hcd').
        energy_unit: unit of the stored collision energy ('ev' = absolute eV;
            stored normalized as eV/100, hence ``collision_energy_encoding``).
        collision_energy_encoding: how CE is encoded in ``fragment_ions``.
        down_sample_factor: fragment down-sampling fraction used (provenance).

    Returns:
        The prediction_set_id assigned (DEFAULT_PREDICTION_SET_ID).
    """
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()

        # fragment_ions must exist (written by simulate_fragment_intensities).
        tables = {
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "fragment_ions" not in tables:
            logger.warning(
                "register_prediction_set: no fragment_ions table; skipping."
            )
            return DEFAULT_PREDICTION_SET_ID

        # Idempotent: recreate the set table (single-row today).
        cur.execute("DROP TABLE IF EXISTS prediction_sets")
        cur.execute(
            """
            CREATE TABLE prediction_sets (
                prediction_set_id        INTEGER PRIMARY KEY,
                schema_version           INTEGER NOT NULL,
                predictor_model          TEXT,
                instrument               TEXT NOT NULL,
                acquisition_type         TEXT NOT NULL,
                activation_method        TEXT NOT NULL,
                energy_unit              TEXT NOT NULL,
                collision_energy_encoding TEXT NOT NULL,
                down_sample_factor       REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            INSERT INTO prediction_sets (
                prediction_set_id, schema_version, predictor_model, instrument,
                acquisition_type, activation_method, energy_unit,
                collision_energy_encoding, down_sample_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                DEFAULT_PREDICTION_SET_ID,
                PREDICTION_SET_SCHEMA_VERSION,
                predictor_model,
                instrument,
                acquisition_type,
                activation_method,
                energy_unit,
                collision_energy_encoding,
                float(down_sample_factor),
            ),
        )

        # Add the namespace column to fragment_ions if absent (idempotent).
        cols = {
            r[1] for r in cur.execute("PRAGMA table_info(fragment_ions)").fetchall()
        }
        if "prediction_set_id" not in cols:
            cur.execute(
                "ALTER TABLE fragment_ions ADD COLUMN prediction_set_id "
                f"INTEGER NOT NULL DEFAULT {DEFAULT_PREDICTION_SET_ID}"
            )
        else:
            # Resume / re-run: ensure all rows carry the set id.
            cur.execute(
                "UPDATE fragment_ions SET prediction_set_id = ?",
                (DEFAULT_PREDICTION_SET_ID,),
            )

        con.commit()
        logger.info(
            "  prediction set %d registered (instrument=%s, acquisition=%s, "
            "activation=%s, model=%s)",
            DEFAULT_PREDICTION_SET_ID,
            instrument,
            acquisition_type,
            activation_method,
            predictor_model or "prosit(default)",
        )
        return DEFAULT_PREDICTION_SET_ID
    finally:
        con.close()
