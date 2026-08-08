"""Pin the OPT-IN mobility-derived collision energy in ``timsim-fragments``.

Two things have to stay true, and neither is safe to assume:

1. **With the option off, nothing moves.** 30 rendered cohort arms exist on disk; a stray metadata
   key or a reordered call would invalidate all of them. The flat-CE schema metadata is therefore
   pinned as a literal.
2. **With the option on, the dedup still holds.** The predictor decodes each ``(sequence, charge)``
   ONCE. That is only legal if collision energy is a function of that key — which it is, because CE
   comes from CCS and CCS comes from the key. These tests exercise the collapse, and require a key
   whose members disagree to RAISE rather than let one isomer's CE stand in for its siblings'.
"""

import numpy as np
import pandas as pd
import pytest

from imspy_simulation.timsim.jobs import fragments as F


# --------------------------------------------------------------------------------------------
# 1. default-off is unchanged
# --------------------------------------------------------------------------------------------

# The metadata every fragment_intensities artifact on disk carries. A literal, not a re-derivation:
# if someone "improves" fragment_schema, this fails instead of silently orphaning the cache.
_HISTORICAL_METADATA = {
    b"timsim.table": b"fragment_intensities",
    b"timsim.schema_version": b"2.0",
    b"timsim.axis": b"measurement",
    b"timsim.producer": b"timsim-fragments",
    b"timsim.fragments.model": b"some-model",
    b"timsim.fragments.collision_energy": b"25.0",
}


def test_flat_ce_schema_metadata_is_byte_for_byte_the_historical_one():
    schema = F.fragment_schema("some-model", 25.0)
    assert schema.metadata == _HISTORICAL_METADATA


def test_per_precursor_ce_adds_metadata_only_when_it_is_actually_used():
    schema = F.fragment_schema("some-model", 25.0, ce_source="/some/ce.parquet")
    assert schema.metadata[b"timsim.fragments.collision_energy_mode"] == b"per-precursor-mobility"
    assert schema.metadata[b"timsim.fragments.collision_energies"] == b"/some/ce.parquet"
    # The flat value is retained (it is what --collision-energy said) but no longer the whole story.
    assert schema.metadata[b"timsim.fragments.collision_energy"] == b"25.0"
    # The column layout must not move either way.
    assert F.fragment_schema("m", 1.0).names == F.fragment_schema("m", 1.0, ce_source="x").names


# --------------------------------------------------------------------------------------------
# 2. the per-key collapse
# --------------------------------------------------------------------------------------------


def test_isomers_sharing_a_key_collapse_to_one_ce_with_zero_spread():
    key_tuples = [("PEPTIDEK", 2), ("SEQUENCER", 3)]
    key2pids = {("PEPTIDEK", 2): [1, 2, 3], ("SEQUENCER", 3): [4]}
    ce = {1: 31.5, 2: 31.5, 3: 31.5, 4: 44.25}
    ces, worst = F.collision_energy_per_key(key_tuples, key2pids, ce, tolerance=1e-9)
    assert ces == [31.5, 44.25]
    assert worst == 0.0


def test_a_key_whose_members_disagree_raises_instead_of_picking_one():
    key_tuples = [("PEPTIDEK", 2)]
    key2pids = {("PEPTIDEK", 2): [1, 2]}
    ce = {1: 31.5, 2: 33.0}  # 1.5 eV apart -- a different scan, i.e. a different ion
    with pytest.raises(ValueError, match="not constant within"):
        F.collision_energy_per_key(key_tuples, key2pids, ce, tolerance=1e-9)
    # Deliberately loosening the tolerance is allowed, and then the spread is reported.
    ces, worst = F.collision_energy_per_key(key_tuples, key2pids, ce, tolerance=2.0)
    assert ces == [31.5] and worst == pytest.approx(1.5)


# --------------------------------------------------------------------------------------------
# 3. end to end through predict_fragment_batches, with the model stubbed out
# --------------------------------------------------------------------------------------------


def _stub_predictor(monkeypatch):
    """Replace the intensity model with a recorder. Returns the list it records calls into."""
    calls = []

    def fake(sequences, charges, collision_energies, model=None):
        seqs = list(sequences)
        calls.append(
            {
                "n": len(seqs),
                "sequences": seqs,
                "charges": [int(c) for c in charges],
                "ces": [float(e) for e in collision_energies],
            }
        )
        pred = np.full((len(seqs), 29, 2, 3), -1.0, dtype=np.float32)
        pred[:, 0, 0, 0] = 0.5  # one y1+ per precursor, above the floor
        return pred, "stub-model"

    monkeypatch.setattr(F, "predict_tensors", fake)
    return calls


def _three_isomers():
    """Three precursors, two of which share a (sequence, charge) key."""
    return pd.DataFrame(
        {
            "precursor_id": [10, 11, 12],
            "sequence": ["PEPTIDEK", "PEPTIDEK", "SEQUENCER"],
            "charge": [2, 2, 3],
        }
    )


def test_prediction_volume_is_identical_with_and_without_per_precursor_ce(monkeypatch):
    prec = _three_isomers()
    calls = _stub_predictor(monkeypatch)

    _p, _s, off = F.predict_fragment_batches(prec, 25.0, verbose=False)
    list(off)
    _p, _s, on = F.predict_fragment_batches(
        prec,
        25.0,
        verbose=False,
        collision_energies={10: 31.5, 11: 31.5, 12: 44.25},
        collision_energies_source="ce.parquet",
    )
    list(on)

    assert len(calls) == 2
    # THE claim: turning the capability on does not add a single model call.
    assert calls[0]["n"] == calls[1]["n"] == 2
    assert calls[0]["sequences"] == calls[1]["sequences"]
    assert calls[0]["charges"] == calls[1]["charges"]
    # ...but the CE the model sees is now per key, not one number for the run.
    assert calls[0]["ces"] == [25.0, 25.0]
    assert calls[1]["ces"] == [31.5, 44.25]


def test_the_fanout_to_precursors_is_unchanged_by_the_ce_mode(monkeypatch):
    prec = _three_isomers()
    _stub_predictor(monkeypatch)

    _p, schema_off, off = F.predict_fragment_batches(prec, 25.0, verbose=False)
    rows_off = pd.concat([b.to_pandas() for b in off])
    _p, schema_on, on = F.predict_fragment_batches(
        prec, 25.0, verbose=False,
        collision_energies={10: 31.5, 11: 31.5, 12: 44.25},
        collision_energies_source="ce.parquet",
    )
    rows_on = pd.concat([b.to_pandas() for b in on])

    # Same precursors, same fragment slots -- only the intensities may differ (here they cannot,
    # because the stub ignores CE).
    assert sorted(rows_off["precursor_id"]) == sorted(rows_on["precursor_id"]) == [10, 11, 12]
    assert schema_off.metadata != schema_on.metadata  # provenance says which mode produced it


def test_a_precursor_missing_from_the_ce_table_raises(monkeypatch):
    prec = _three_isomers()
    _stub_predictor(monkeypatch)
    with pytest.raises(ValueError, match="absent from the collision-energy table"):
        list(
            F.predict_fragment_batches(
                prec, 25.0, verbose=False, collision_energies={10: 31.5, 11: 31.5}
            )[2]
        )


def test_a_key_with_inconsistent_isomer_ces_raises_through_the_public_entry(monkeypatch):
    prec = _three_isomers()
    _stub_predictor(monkeypatch)
    with pytest.raises(ValueError, match="not constant within"):
        list(
            F.predict_fragment_batches(
                prec, 25.0, verbose=False,
                collision_energies={10: 31.5, 11: 40.0, 12: 44.25},
            )[2]
        )


def test_collision_energy_table_round_trips_through_the_loader(tmp_path):
    p = tmp_path / "ce.parquet"
    pd.DataFrame(
        {"precursor_id": np.array([7, 8], dtype="uint64"),
         "scan": np.array([100, 200], dtype="uint32"),
         "collision_energy": [50.75, 47.3]}
    ).to_parquet(p)
    ce = F.load_precursor_collision_energies(p)
    assert ce[7] == pytest.approx(50.75) and ce[8] == pytest.approx(47.3)


# --------------------------------------------------------------------------------------------
# 4. the ramp `timsim-frag-ce` hardcodes must stay v1's ramp
# --------------------------------------------------------------------------------------------


def test_the_frag_ce_default_ramp_still_matches_v1s_dda_pasef_defaults():
    """`timsim-frag-ce`'s CE_BIAS/CE_SLOPE defaults are v1's, copied into Rust. If v1 ever retunes
    its dda-PASEF ramp, this fails and the Rust defaults must follow — otherwise the two halves of
    the simulator would fragment the same ion at two different energies."""
    import inspect

    sched = pytest.importorskip(
        "imspy_simulation.timsim.jobs.dda_selection_scheme"
    ).schedule_precursors
    params = inspect.signature(sched).parameters
    assert params["ce_bias"].default == pytest.approx(54.1984)
    assert params["ce_slope"].default == pytest.approx(-0.0345)


def test_the_activation_policy_is_the_one_v1_drives():
    """The CE any scan maps to, straight from the shared Rust policy — the same object
    `dda_selection_scheme` calls. Pinned across the whole timsTOF mobility ramp."""
    ims = pytest.importorskip("imspy_connector")
    policy = ims.py_acquisition.PyActivationPolicy.bruker_pasef(54.1984, -0.0345)
    scans = list(range(0, 1024))
    got = np.asarray(policy.collision_energies_for_scans(scans))
    want = 54.1984 - 0.0345 * np.asarray(scans, dtype=float)
    assert np.abs(got - want).max() == 0.0
    assert policy.energy_unit == "ev" and policy.activation_method == "hcd"
