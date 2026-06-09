"""P1 instrument-dispatch foundation: schema versioning, experiment_conditions,
the reserved condition_id FK, and the identity + intensity contracts.

All unconditional (no external data / connector feature needed).
"""
import sqlite3

import pandas as pd
import pytest

from imspy_simulation.dispatch import (
    DEFAULT_DRIFT_GAS,
    SCHEMA_VERSION,
    ExperimentConditions,
    IntensityStage,
    analyte_key,
    derive_seed,
    ensure_condition_fk,
    event_key,
    initialize_dispatch_schema,
    profile_key,
    read_experiment_conditions,
    read_schema_version,
)


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    pd.DataFrame({"peptide_id": [1, 2], "sequence": ["PEPTIDEK", "SAMPLERR"]}).to_sql(
        "peptides", conn, index=False
    )
    pd.DataFrame({"ion_id": [1, 2], "peptide_id": [1, 2]}).to_sql("ions", conn, index=False)
    yield conn
    conn.close()


def test_pre_p1_db_reads_version_zero():
    conn = sqlite3.connect(":memory:")
    assert read_schema_version(conn) == 0  # no marker -> legacy
    assert read_experiment_conditions(conn) is None
    conn.close()


def test_initialize_stamps_version_conditions_and_fk(db):
    cond = initialize_dispatch_schema(db, ExperimentConditions(lc_method="90min", run_seed=7))
    assert read_schema_version(db) == SCHEMA_VERSION
    rt = read_experiment_conditions(db)
    assert rt is not None
    assert rt.lc_method == "90min"
    assert rt.drift_gas == DEFAULT_DRIFT_GAS
    assert rt.run_seed == 7
    for table in ("peptides", "ions"):
        cols = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
        assert "condition_id" in cols
    assert cond.condition_id == 0


def test_ensure_condition_fk_is_idempotent_and_skips_missing(db):
    ensure_condition_fk(db)
    ensure_condition_fk(db)  # second call must not raise (column already present)
    cols = {r[1] for r in db.execute("PRAGMA table_info(peptides)")}
    assert sum(c == "condition_id" for c in cols) == 1
    # missing table is silently skipped
    ensure_condition_fk(db, tables=("does_not_exist",))


def test_conditions_round_trip_preserves_mobility_env(db):
    cond = ExperimentConditions(
        drift_gas="N2", drift_gas_mass=28.013, temperature_c=31.85, t_diff=273.15, ccs_model="deep"
    )
    initialize_dispatch_schema(db, cond)
    rt = read_experiment_conditions(db)
    assert (rt.drift_gas_mass, rt.temperature_c, rt.t_diff) == (28.013, 31.85, 273.15)
    assert rt.ccs_model == "deep"


def test_identity_keys_stable_and_distinct():
    assert analyte_key("PEPTIDEK", 2) == analyte_key("PEPTIDEK", 2)
    assert analyte_key("PEPTIDEK", 2) != analyte_key("PEPTIDEK", 3)
    assert analyte_key("PEPTIDEK", 2, decoy=True) != analyte_key("PEPTIDEK", 2, decoy=False)
    assert profile_key("TimsTofDia", "p") == profile_key("TimsTofDia", "p")
    assert event_key(0, 1) != event_key(0, 1, controller_counter=1)


def test_seed_is_deterministic_and_component_separated():
    a = derive_seed(42, "prof", "an", "ev", "mz_noise")
    b = derive_seed(42, "prof", "an", "ev", "mz_noise")
    c = derive_seed(42, "prof", "an", "ev", "rt_noise")
    assert a == b
    assert a != c
    assert 0 <= a < 2**64


def test_intensity_stage_chain_is_ordered():
    # The conserved-quantity chain, in order.
    assert [s.value for s in IntensityStage] == [
        "yield",
        "time",
        "mobility",
        "transmitted",
        "detected",
        "centroided",
    ]
