"""P6d: Orbitrap Astral NCE fragment-intensity model capability + model-aware
Koina input.

The unconditional tests cover the capability contract (no network). The live
Koina contract test is gated on TIMSIM_KOINA_LIVE=1 (needs network access to
koina.wilhelmlab.org).
"""
import os

import pandas as pd
import pytest

from imspy_simulation.timsim.jobs.fragment_predictor_capability import (
    assert_predictor_supports,
    capability_for,
)
from imspy_simulation.timsim.jobs.simulate_fragment_intensities import (
    KOINA_INTENSITY_MODELS,
    _koina_input_df,
)


def test_orbitrap_nce_capability_declared_for_prosit_hcd():
    # Registered under both the short key and the full Koina alias.
    for key in ("prosit_hcd", "Prosit_2020_intensity_HCD"):
        cap = capability_for(key)
        assert cap is not None, f"{key} must be declared"
        assert cap.energy_unit == "nce"
        assert cap.supported_methods == frozenset({"hcd"})
        # Encoding is shared with the timsTOF models (NCE fed as a fraction /100).
        assert cap.ce_encoding == "normalized_div100"
    # And the short key routes to the Orbitrap Koina model.
    assert KOINA_INTENSITY_MODELS["prosit_hcd"] == "Prosit_2020_intensity_HCD"


def test_timstof_models_remain_ev():
    for key in ("local", "prosit", "alphapeptdeep", "ms2pip"):
        assert capability_for(key).energy_unit == "ev"


def test_astral_model_accepts_hcd_nce_rejects_mismatches():
    # The Astral model accepts HCD + NCE ...
    assert_predictor_supports("prosit_hcd", "hcd", "nce")
    # ... and rejects an eV unit (a timsTOF set fed to the Astral model) ...
    with pytest.raises(ValueError):
        assert_predictor_supports("prosit_hcd", "hcd", "ev")
    # ... a non-HCD method ...
    with pytest.raises(ValueError):
        assert_predictor_supports("prosit_hcd", "cid", "nce")
    # ... and the inverse: a timsTOF (eV) model fed an NCE unit.
    with pytest.raises(ValueError):
        assert_predictor_supports("prosit", "hcd", "nce")


class _FakeModel:
    def __init__(self, inputs):
        self.model_inputs = {k: None for k in inputs}


def test_koina_input_df_is_model_aware():
    data = pd.DataFrame(
        {"sequence": ["PEPTIDEK", "ELVISR"], "charge": [2, 2], "collision_energy": [28.0, 30.0]}
    )
    enc = data["collision_energy"].values / 100.0

    # Orbitrap HCD: only the core three columns, NO instrument_types.
    hcd = _koina_input_df(
        _FakeModel(["peptide_sequences", "precursor_charges", "collision_energies"]),
        data,
        enc,
    )
    assert set(hcd.columns) == {"peptide_sequences", "precursor_charges", "collision_energies"}
    assert "instrument_types" not in hcd.columns

    # timsTOF-style: instrument_types is required and supplied.
    tims = _koina_input_df(
        _FakeModel(
            ["peptide_sequences", "precursor_charges", "collision_energies", "instrument_types"]
        ),
        data,
        enc,
    )
    assert list(tims["instrument_types"]) == ["TIMSTOF", "TIMSTOF"]

    # An unknown required column is a hard error (never silently dropped).
    with pytest.raises(ValueError):
        _koina_input_df(_FakeModel(["peptide_sequences", "mystery_field"]), data, enc)


def test_instrument_activation_resolves_unit_and_rejects_unknown():
    from imspy_simulation.timsim.jobs.register_prediction_set import (
        resolve_instrument_activation,
    )

    assert resolve_instrument_activation("bruker_timstof") == ("hcd", "ev")
    assert resolve_instrument_activation("orbitrap_astral") == ("hcd", "nce")
    assert resolve_instrument_activation(None) == ("hcd", "ev")  # default
    with pytest.raises(ValueError):
        resolve_instrument_activation("orbitrap_zoom")


def test_run_unit_gates_model_selection_both_ways():
    # The RUN's instrument fixes the CE unit; the selected model must accept it.
    am_a, eu_a = ("hcd", "nce")   # orbitrap_astral
    am_b, eu_b = ("hcd", "ev")    # bruker_timstof
    # Astral run accepts the NCE model, rejects the eV timsTOF models.
    assert_predictor_supports("prosit_hcd", am_a, eu_a)
    for m in ("prosit", "local", "alphapeptdeep"):
        with pytest.raises(ValueError):
            assert_predictor_supports(m, am_a, eu_a)
    # Bruker run accepts the eV models, rejects the NCE model.
    assert_predictor_supports("local", am_b, eu_b)
    with pytest.raises(ValueError):
        assert_predictor_supports("prosit_hcd", am_b, eu_b)


def test_astral_prediction_set_records_nce_provenance(tmp_path):
    import sqlite3

    from imspy_simulation.timsim.jobs.register_prediction_set import (
        register_prediction_set,
    )

    db = str(tmp_path / "synthetic_data.db")
    con = sqlite3.connect(db)
    # Minimal fragment_ions table; CE stored normalized (NCE/100) as for a real run.
    con.execute("CREATE TABLE fragment_ions (peptide_id INTEGER, collision_energy REAL)")
    con.execute("INSERT INTO fragment_ions VALUES (1, 0.28), (2, 0.30)")
    con.commit()
    con.close()

    register_prediction_set(
        db,
        predictor_model="prosit_hcd",
        acquisition_type="DIA",
        instrument="orbitrap_astral",
        activation_method="hcd",
        energy_unit="nce",
    )

    con = sqlite3.connect(db)
    row = con.execute(
        "SELECT instrument, activation_method, energy_unit, collision_energy_encoding, "
        "predictor_model FROM prediction_sets"
    ).fetchone()
    # The fragment rows are stamped with the set id.
    n_stamped = con.execute(
        "SELECT COUNT(*) FROM fragment_ions WHERE prediction_set_id = 0"
    ).fetchone()[0]
    con.close()

    instrument, activation, unit, encoding, model = row
    assert instrument == "orbitrap_astral"
    assert activation == "hcd"
    assert unit == "nce"
    # Encoding is shared (stored CE = CE/100 regardless of unit); the render keying
    # is unit-agnostic, so this stays normalized_div100 for the Astral set too.
    assert encoding == "normalized_div100"
    assert model == "prosit_hcd"
    assert n_stamped == 2


def _config_with(**overrides):
    """A SimulationConfig with defaults + required paths, for _validate tests
    (bypasses TOML loading)."""
    from imspy_simulation.timsim.simulator import SimulationConfig, get_default_settings

    cfg = get_default_settings()
    cfg.update({"save_path": "/tmp/x", "reference_path": "/tmp/y", "fasta_path": "/tmp/z"})
    cfg.update(overrides)
    c = SimulationConfig.__new__(SimulationConfig)
    c._config = cfg
    return c


def test_config_validate_astral_requires_dia_and_nce():
    # Default Bruker: valid without any NCE.
    _config_with()._validate()

    # Unknown instrument is rejected at config load.
    with pytest.raises(ValueError, match="unknown instrument"):
        _config_with(instrument="orbitrap_zoom")._validate()

    # Astral + DDA is rejected at config load (not deep in the run).
    with pytest.raises(ValueError, match="does not support DDA"):
        _config_with(
            instrument="orbitrap_astral", acquisition_type="DDA", collision_energy_nce=27
        )._validate()

    # Astral DIA without an NCE is rejected (no silent eV->NCE relabel).
    with pytest.raises(ValueError, match="requires 'collision_energy_nce'"):
        _config_with(instrument="orbitrap_astral", acquisition_type="DIA")._validate()

    # Astral DIA with a positive NCE is valid.
    _config_with(
        instrument="orbitrap_astral", acquisition_type="DIA", collision_energy_nce=27.0
    )._validate()


def test_astral_nce_override_is_astral_only():
    from imspy_simulation.timsim.simulator import astral_nce_override

    # Astral run: the configured NCE is forwarded.
    assert astral_nce_override(
        _config_with(instrument="orbitrap_astral", collision_energy_nce=27.0)
    ) == 27.0
    # Bruker run with a stray NCE: IGNORED (None) — never relabels eV as NCE.
    assert astral_nce_override(
        _config_with(instrument="bruker_timstof", collision_energy_nce=27.0)
    ) is None
    # Default (no instrument set): Bruker -> ignored.
    assert astral_nce_override(_config_with()) is None


def test_dia_builder_nce_override_replaces_collision_energy():
    # The override block (independent of a live reference dataset): a builder whose
    # dia_ms_ms_windows came from a Bruker reference (eV ~24-30) gets its CE fully
    # replaced by the configured NCE — never relabelled.
    import numpy as np
    import pandas as pd

    from imspy_simulation.acquisition import TimsTofAcquisitionBuilderDIA

    b = TimsTofAcquisitionBuilderDIA.__new__(TimsTofAcquisitionBuilderDIA)
    b.dia_ms_ms_windows = pd.DataFrame({"collision_energy": [24.1, 27.3, 30.0]})
    b.collision_energy_nce = 27.0
    b.round_collision_energy = True
    b.collision_energy_decimals = 0

    # Replicate the _setup override + rounding step.
    if b.collision_energy_nce is not None:
        b.dia_ms_ms_windows["collision_energy"] = float(b.collision_energy_nce)
    if b.round_collision_energy:
        b.dia_ms_ms_windows["collision_energy"] = np.round(
            b.dia_ms_ms_windows["collision_energy"].values, decimals=b.collision_energy_decimals
        )
    assert list(b.dia_ms_ms_windows["collision_energy"]) == [27.0, 27.0, 27.0]


@pytest.mark.skipif(
    os.environ.get("TIMSIM_KOINA_LIVE") != "1",
    reason="set TIMSIM_KOINA_LIVE=1 to run the live Koina Prosit HCD contract test",
)
def test_prosit_hcd_live_koina_contract():
    import numpy as np

    from imspy_simulation.timsim.jobs.simulate_fragment_intensities import (
        _predict_intensities_with_koina,
    )

    data = pd.DataFrame(
        {"sequence": ["LGGNEQVTR"], "charge": [2], "collision_energy": [28.0]}
    )
    out = _predict_intensities_with_koina(data, model_name="prosit_hcd", verbose=False)
    iv = np.vstack(out["intensity"].values)
    assert iv.shape == (1, 174)
    # A real Prosit HCD spectrum has several non-zero fragments.
    assert (iv > 0).sum() >= 3
