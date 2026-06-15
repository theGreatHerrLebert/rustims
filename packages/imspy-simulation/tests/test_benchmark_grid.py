"""The multi-vendor benchmark test grid: config expansion + manifest generation."""

import os

import pytest
import toml

from imspy_simulation.timsim.benchmark.grid import expand, write_configs, manifest_for


def _spec():
    return {
        "save_root": "/tmp/grid",
        "q": 0.01,
        "base": {"fasta_path": "/d/h.fasta", "acquisition_type": "DIA",
                 "num_sample_peptides": 5000, "waters_window_step": None},
        "cells": [
            {"label": "bruker", "engine": "diann",
             "config": {"instrument": "bruker_timstof", "reference_path": "/d/ref.d",
                        "gradient_length": 1800}},
            {"label": "waters", "engine": "diann",
             "config": {"instrument": "waters_synapt_xs", "gradient_length": 1800}},
        ],
    }


def test_expand_merges_base_and_cell_and_adds_paths():
    ex = expand(_spec())
    assert set(ex) == {"bruker", "waters"}
    b = ex["bruker"]
    assert b["instrument"] == "bruker_timstof"          # from cell
    assert b["fasta_path"] == "/d/h.fasta"              # from base
    assert b["num_sample_peptides"] == 5000             # from base
    assert b["save_path"] == "/tmp/grid/bruker"         # derived
    assert b["experiment_name"] == "BRUKER"             # derived from label
    # None-valued base keys (e.g. unset window_step) are dropped (TOML has no null).
    assert "waters_window_step" not in ex["waters"]


def test_expand_rejects_bad_specs():
    with pytest.raises(ValueError):
        expand({"cells": []})  # no cells
    with pytest.raises(ValueError):
        expand({"cells": [{"engine": "diann", "config": {"instrument": "x"}}]})  # no label
    with pytest.raises(ValueError):
        expand({"cells": [{"label": "a", "config": {}}]})  # no instrument
    with pytest.raises(ValueError):
        expand({"cells": [{"label": "a", "config": {"instrument": "x"}},
                          {"label": "a", "config": {"instrument": "y"}}]})  # dup label


def test_write_configs_are_valid_toml(tmp_path):
    ex = expand(_spec())
    paths = write_configs(ex, str(tmp_path))
    assert set(paths) == {"bruker", "waters"}
    loaded = toml.load(paths["bruker"])
    assert loaded["instrument"] == "bruker_timstof"
    assert loaded["save_path"] == "/tmp/grid/bruker"
    assert os.path.exists(paths["waters"])


def test_manifest_points_eval_jsons_under_save_root():
    m = manifest_for(_spec())
    assert m["q"] == 0.01
    by_instr = {v["instrument"]: v for v in m["vendors"]}
    assert by_instr["bruker_timstof"]["eval_json"] == "/tmp/grid/bruker/eval.json"
    assert by_instr["bruker_timstof"]["engine"] == "diann"
    assert by_instr["waters_synapt_xs"]["eval_json"] == "/tmp/grid/waters/eval.json"


def test_manifest_follows_an_overridden_save_path():
    # An explicit save_path must be honoured by the manifest (no drift from the run).
    spec = {"save_root": "/tmp/grid", "cells": [
        {"label": "w", "engine": "diann",
         "config": {"instrument": "waters_synapt_xs", "save_path": "/custom/out"}},
    ]}
    m = manifest_for(spec)
    assert m["vendors"][0]["eval_json"] == "/custom/out/eval.json"


def test_expand_rejects_colliding_save_paths():
    spec = {"save_root": "/tmp/grid", "cells": [
        {"label": "a", "config": {"instrument": "x", "save_path": "/same"}},
        {"label": "b", "config": {"instrument": "y", "save_path": "/same"}},
    ]}
    with pytest.raises(ValueError, match="save_path"):
        expand(spec)


def test_drop_none_is_recursive():
    spec = {"save_root": "/tmp/g", "cells": [
        {"label": "w", "config": {"instrument": "waters_synapt_xs",
                                  "models": {"a": 1, "b": None}}},
    ]}
    cfg = expand(spec)["w"]
    assert cfg["models"] == {"a": 1}  # nested None dropped
