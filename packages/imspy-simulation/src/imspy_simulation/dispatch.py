"""Instrument-dispatch foundations (P1) — schema versioning, experiment
conditions, and the identity + intensity contracts.

This is the *additive* foundation for separating the vendor-neutral physical
appearance of peptides/ions from device-specific recording (see
``INSTRUMENT_DISPATCH.md``). Nothing here changes existing simulation behavior:
it adds a schema-version marker, an ``experiment_conditions`` record, a reserved
nullable ``condition_id`` foreign key on ``peptides``/``ions``, and the
deterministic identity/seed + intensity contracts the Rust projector will share.

The mobility-ownership decision (plan §2.3): the DB stores **CCS** (chemistry);
the *render-time instrument profile* owns the drift-gas/temperature/charge → 1/K0
conversion. ``experiment_conditions`` records the conditions the layer-2
quantities (RT, yield, fragmentation) were generated under as *provenance* — it
is **not** the authoritative mobility environment at render time. The default
mobility environment mirrors the timsTOF constants used by
``mscore::chemistry::formulas`` (N2, 31.85 °C, 273.15 offset).
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

# Bump when the on-disk synthetic_data.db layout changes in a way readers must
# branch on. P1 introduces version 1 (pre-P1 DBs have no marker -> treated as 0).
SCHEMA_VERSION = 1

_SCHEMA_META_TABLE = "schema_meta"
_CONDITIONS_TABLE = "experiment_conditions"

# timsTOF mobility-environment defaults (match mscore::chemistry::formulas
# doctests: drift gas N2, temperature 31.85 C, Kelvin offset 273.15).
DEFAULT_DRIFT_GAS = "N2"
DEFAULT_DRIFT_GAS_MASS = 28.013
DEFAULT_TEMPERATURE_C = 31.85
DEFAULT_T_DIFF = 273.15


# --------------------------------------------------------------------------- #
# Experiment conditions (layer-2 provenance, one row per run + reserved FK)
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConditions:
    """Declared conditions the layer-2 appearance was generated under.

    Provenance for the generated RT / charge-yield / fragmentation, plus the
    mobility environment used when 1/K0 was (historically) materialised. One row
    per run by default; per-analyte overrides are supported later via the
    reserved ``condition_id`` FK without a migration.
    """

    condition_id: int = 0
    # Chromatography / source provenance (free-form, model/method identifiers).
    lc_method: str = ""
    source: str = ""
    gradient_length_s: Optional[float] = None
    # Prediction-model provenance.
    ccs_model: str = ""
    rt_model: str = ""
    # Mobility environment (for CCS<->1/K0; NOT authoritative at render time).
    drift_gas: str = DEFAULT_DRIFT_GAS
    drift_gas_mass: float = DEFAULT_DRIFT_GAS_MASS
    temperature_c: float = DEFAULT_TEMPERATURE_C
    t_diff: float = DEFAULT_T_DIFF
    # Reproducibility.
    run_seed: int = 0
    notes: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])

    @classmethod
    def from_row(cls, row: pd.Series) -> "ExperimentConditions":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: row[k] for k in fields if k in row})


# --------------------------------------------------------------------------- #
# Schema versioning
# --------------------------------------------------------------------------- #
def write_schema_version(conn: sqlite3.Connection, version: int = SCHEMA_VERSION) -> None:
    """Stamp the DB with a schema version (idempotent, single-row table)."""
    pd.DataFrame([{"key": "schema_version", "value": int(version)}]).to_sql(
        _SCHEMA_META_TABLE, conn, if_exists="replace", index=False
    )


def read_schema_version(conn: sqlite3.Connection) -> int:
    """Return the stamped schema version, or 0 for a pre-P1 DB without a marker."""
    try:
        df = pd.read_sql(
            f"SELECT value FROM {_SCHEMA_META_TABLE} WHERE key = 'schema_version'", conn
        )
    except Exception:
        return 0
    if df.empty:
        return 0
    return int(df["value"].iloc[0])


# --------------------------------------------------------------------------- #
# experiment_conditions table + reserved condition_id FK
# --------------------------------------------------------------------------- #
def write_experiment_conditions(
    conn: sqlite3.Connection, conditions: ExperimentConditions
) -> int:
    """Write a single experiment-conditions row, returning its ``condition_id``.

    Replaces the table (P1 stores one row per run); the returned id is what a
    future per-analyte override would reference via ``condition_id``.
    """
    conditions.to_frame().to_sql(_CONDITIONS_TABLE, conn, if_exists="replace", index=False)
    conn.commit()
    return conditions.condition_id


def read_experiment_conditions(conn: sqlite3.Connection) -> Optional[ExperimentConditions]:
    """Read the (single) experiment-conditions row, or None if absent."""
    try:
        df = pd.read_sql(f"SELECT * FROM {_CONDITIONS_TABLE}", conn)
    except Exception:
        return None
    if df.empty:
        return None
    return ExperimentConditions.from_row(df.iloc[0])


def ensure_condition_fk(conn: sqlite3.Connection, tables=("peptides", "ions")) -> None:
    """Reserve a nullable ``condition_id`` column on the given tables.

    Additive: existing rows get NULL (interpreted as "the run's single
    experiment_conditions row"). Skips tables that don't exist or already have
    the column, so it is safe to call unconditionally after table creation.
    """
    cur = conn.cursor()
    existing = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in tables:
        if table not in existing:
            continue
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({table})")}
        if "condition_id" in cols:
            continue
        cur.execute(f"ALTER TABLE {table} ADD COLUMN condition_id INTEGER")
    conn.commit()


def initialize_dispatch_schema(
    conn: sqlite3.Connection, conditions: Optional[ExperimentConditions] = None
) -> ExperimentConditions:
    """One-call P1 setup: stamp version, write conditions, reserve the FK.

    Call after the content tables (peptides/ions) exist. Returns the conditions
    actually written (defaults if none supplied).
    """
    conditions = conditions or ExperimentConditions()
    write_schema_version(conn)
    write_experiment_conditions(conn, conditions)
    ensure_condition_fk(conn)
    return conditions


# --------------------------------------------------------------------------- #
# Identity contract — stable ids that survive ordering/batching/migration
# --------------------------------------------------------------------------- #
def analyte_key(sequence: str, charge: int, decoy: bool = False) -> str:
    """Content-addressed analyte id: stable across row reordering and migration.

    Keyed on chemistry (sequence+charge+decoy), never on row index, so the same
    analyte seeds identically regardless of how the DB was built or batched.
    """
    h = hashlib.sha1(f"{sequence}|{int(charge)}|{int(decoy)}".encode()).hexdigest()
    return h[:16]


def profile_key(instrument: str, scheme_provenance: str) -> str:
    """Stable id for an InstrumentProfile (instrument kind + scheme provenance)."""
    h = hashlib.sha1(f"{instrument}|{scheme_provenance}".encode()).hexdigest()
    return h[:16]


def event_key(cycle_index: int, event_in_cycle: int, controller_counter: int = 0) -> str:
    """Deterministic event id.

    Derived from (cycle, position-in-cycle) plus a controller-decision counter,
    NOT emission order — so DDA's dynamically created events (P4) stay
    reproducible. For DIA the controller counter is 0.
    """
    h = hashlib.sha1(
        f"{int(cycle_index)}|{int(event_in_cycle)}|{int(controller_counter)}".encode()
    ).hexdigest()
    return h[:16]


def derive_seed(
    run_seed: int,
    profile_id: str,
    analyte_id: str,
    event_id: str,
    noise_component: str,
) -> int:
    """Counter-based per-(analyte, event, component) seed.

    ``hash(run_seed, profile_id, analyte_id, event_id, noise_component)`` — no
    thread-local / traversal-order RNG, so noise is reproducible independent of
    thread scheduling and the same DB yields stable output per instrument.
    Returns a 64-bit unsigned seed.
    """
    payload = f"{int(run_seed)}|{profile_id}|{analyte_id}|{event_id}|{noise_component}"
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


# --------------------------------------------------------------------------- #
# Intensity conservation contract
# --------------------------------------------------------------------------- #
class IntensityStage(Enum):
    """Where along the conservation chain a spectrum's intensities sit.

    A single conserved quantity flows end-to-end; each stage is a documented
    area-preserving/multiplicative transform of the previous one:

        YIELD            total ions produced for the analyte (layer 2)
          -> TIME        x fraction of yield within the event's time interval
          -> MOBILITY    x fraction within the scan (identity / marginalized for
                         non-IMS instruments)
          -> TRANSMITTED x quadrupole transmission fraction (MS2)
          -> DETECTED    detector response -> counts
          -> CENTROIDED  peak-picked, area-preserving

    A ``RenderedSpectrum`` carries its stage so writers/detector models know
    which transform still needs applying and never double-count.
    """

    YIELD = "yield"
    TIME = "time"
    MOBILITY = "mobility"
    TRANSMITTED = "transmitted"
    DETECTED = "detected"
    CENTROIDED = "centroided"
