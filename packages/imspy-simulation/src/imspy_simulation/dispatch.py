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
import re
import sqlite3
from dataclasses import asdict, dataclass
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
# SQLite column types for the ExperimentConditions fields (so the table can be
# created with a real PRIMARY KEY rather than pandas' typeless `to_sql`).
_CONDITION_COL_TYPES = {
    "condition_id": "INTEGER PRIMARY KEY",
    "lc_method": "TEXT",
    "source": "TEXT",
    "gradient_length_s": "REAL",
    "ccs_model": "TEXT",
    "rt_model": "TEXT",
    "drift_gas": "TEXT",
    "drift_gas_mass": "REAL",
    "temperature_c": "REAL",
    "t_diff": "REAL",
    "run_seed": "INTEGER",
    "notes": "TEXT",
}


def write_experiment_conditions(
    conn: sqlite3.Connection, conditions: ExperimentConditions
) -> int:
    """Write a single experiment-conditions row, returning its ``condition_id``.

    The table is (re)created with ``condition_id`` as a real PRIMARY KEY — the
    value a per-analyte override on ``peptides``/``ions`` references via its
    (soft) ``condition_id`` column. P1 stores one row per run.
    """
    fields = list(_CONDITION_COL_TYPES)
    cols_ddl = ", ".join(f"{f} {_CONDITION_COL_TYPES[f]}" for f in fields)
    placeholders = ", ".join("?" for _ in fields)
    data = asdict(conditions)
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {_CONDITIONS_TABLE}")
    cur.execute(f"CREATE TABLE {_CONDITIONS_TABLE} ({cols_ddl})")
    cur.execute(
        f"INSERT INTO {_CONDITIONS_TABLE} ({', '.join(fields)}) VALUES ({placeholders})",
        [data[f] for f in fields],
    )
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


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_ident(name: str) -> str:
    """Reject anything that isn't a bare SQL identifier (interpolation guard)."""
    if not _IDENT_RE.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


def ensure_condition_fk(conn: sqlite3.Connection, tables=("peptides", "ions")) -> None:
    """Reserve a nullable ``condition_id`` *soft reference* column on the tables.

    "Soft" because SQLite cannot ``ALTER TABLE ... ADD`` an enforced foreign key
    without rebuilding the table; the column references
    ``experiment_conditions.condition_id`` by convention. Existing rows get NULL
    (interpreted as "the run's single experiment_conditions row"). Skips tables
    that don't exist or already have the column, so it is safe to call
    unconditionally after table creation.
    """
    cur = conn.cursor()
    existing = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in tables:
        if table not in existing:
            continue
        table = _safe_ident(table)
        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({table})")}
        if "condition_id" in cols:
            continue
        cur.execute(f"ALTER TABLE {table} ADD COLUMN condition_id INTEGER")
    conn.commit()


def initialize_dispatch_schema(
    conn: sqlite3.Connection, conditions: Optional[ExperimentConditions] = None
) -> ExperimentConditions:
    """One-call P1 setup: stamp version, write conditions, reserve the soft FK.

    Returns the conditions actually written (defaults if none supplied).

    IMPORTANT: call this **after all content tables are finalized**. The legacy
    pipeline writes ``peptides``/``ions`` with pandas ``to_sql(if_exists=
    "replace")``; any such write *after* this call drops the reserved
    ``condition_id`` column. This function is intentionally NOT wired into the
    legacy write path in P1 — it is opt-in foundation only.
    """
    conditions = conditions or ExperimentConditions()
    write_schema_version(conn)
    write_experiment_conditions(conn, conditions)
    ensure_condition_fk(conn)
    return conditions


# --------------------------------------------------------------------------- #
# Identity contract — stable ids that survive ordering/batching/migration
# --------------------------------------------------------------------------- #
def _canonical(*parts) -> bytes:
    """Length-prefixed join so distinct field tuples can't alias.

    ``("a|b", "c")`` and ``("a", "b|c")`` collide under naive ``|`` joining;
    prefixing each part with its byte length makes the encoding injective.
    """
    out = bytearray()
    for p in parts:
        b = str(p).encode("utf-8")
        out += len(b).to_bytes(8, "little")
        out += b
    return bytes(out)


def analyte_key(sequence: str, charge: int, decoy: bool = False) -> str:
    """Content-addressed analyte id (128-bit): stable across reordering/migration.

    Keyed on chemistry (sequence+charge+decoy), never on row index, so the same
    analyte seeds identically regardless of how the DB was built or batched.
    """
    return hashlib.sha256(_canonical(sequence, int(charge), int(decoy))).hexdigest()[:32]


def profile_key(instrument: str, scheme_provenance: str) -> str:
    """Stable 128-bit id for an Instrument (instrument kind + scheme provenance)."""
    return hashlib.sha256(_canonical(instrument, scheme_provenance)).hexdigest()[:32]


def event_key(cycle_index: int, event_in_cycle: int, controller_counter: int = 0) -> str:
    """Deterministic 128-bit event id.

    Derived from (cycle, position-in-cycle) plus a controller-decision counter,
    NOT emission order — so DDA's dynamically created events (P4) stay
    reproducible. For DIA the controller counter is 0.
    """
    return hashlib.sha256(
        _canonical(int(cycle_index), int(event_in_cycle), int(controller_counter))
    ).hexdigest()[:32]


def derive_seed(
    run_seed: int,
    profile_id: str,
    analyte_id: str,
    event_id: str,
    noise_component: str,
    draw: int = 0,
) -> int:
    """Counter-based per-(analyte, event, component, draw) seed.

    ``hash(run_seed, profile_id, analyte_id, event_id, noise_component, draw)``
    over a length-prefixed payload — no thread-local / traversal-order RNG, so
    noise is reproducible independent of thread scheduling and the same DB
    yields stable output per instrument. A single seed fixes ONE draw; when a
    component needs multiple independent draws, increment ``draw`` (do not rely
    on a stateful RNG's local draw order). Returns a 64-bit unsigned seed.
    """
    digest = hashlib.sha256(
        _canonical(int(run_seed), profile_id, analyte_id, event_id, noise_component, int(draw))
    ).digest()
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
