"""TimSim signs its outputs with mzprov, which is now a declared dependency.

Each vendor path's signer is run against mzprov's own minimal fixtures and
the result is checked with mzprov's verifier, so a change on either side that
breaks the attestation shows up here rather than in a user's run.
"""
import logging

import pytest

from mzprov import verify_sidecar
from mzprov._fixtures import make_minimal_d, make_minimal_mzml
from mzprov.verify import find_provenance_for, verify_embedded_d, verify_embedded_mzml

from imspy_simulation.timsim.simulator import (
    emit_provenance_sidecar,
    emit_provenance_sidecar_mzml,
    emit_provenance_sidecar_raw,
)

LOG = logging.getLogger("test_mzprov_signing")


@pytest.fixture
def signing_env(tmp_path, monkeypatch):
    """Isolate the auto-generated signing key from the real home directory."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = tmp_path / "run.toml"
    config.write_text('[experiment]\nname = "run"\n')
    return tmp_path, config


@pytest.mark.parametrize("embed", [False, True])
def test_bruker_d_is_signed_and_verifies(signing_env, embed):
    tmp_path, config = signing_env
    d = make_minimal_d(tmp_path, name="run")
    emit_provenance_sidecar(d, None, config, "run", embed, None, LOG)

    if embed:
        result = verify_embedded_d(d)
    else:
        result = verify_sidecar(tmp_path / "run.provenance.json")
    assert result.overall_ok
    assert result.payload.simulator_name == "TimSim"


@pytest.mark.parametrize("embed", [False, True])
def test_mzml_is_signed_and_verifies(signing_env, embed):
    tmp_path, config = signing_env
    mzml = make_minimal_mzml(tmp_path, name="run")
    emit_provenance_sidecar_mzml(mzml, config, "run", embed, None, LOG)

    if embed:
        result = verify_embedded_mzml(mzml)
    else:
        result = verify_sidecar(mzml.with_name(mzml.stem + ".provenance.json"))
    assert result.overall_ok


def test_thermo_raw_is_signed_and_verifies(signing_env):
    tmp_path, config = signing_env
    raw = tmp_path / "run.raw"
    raw.write_bytes(b"opaque vendor bytes")
    emit_provenance_sidecar_raw(raw, config, "run", None, LOG)

    assert find_provenance_for(raw) is not None
    assert verify_sidecar(tmp_path / "run.provenance.json").overall_ok


def test_signing_failure_never_fails_the_run(signing_env, caplog):
    tmp_path, config = signing_env
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        emit_provenance_sidecar_mzml(tmp_path / "missing.mzML", config, "run", False, None, LOG)
    assert "signing failed (non-fatal)" in caplog.text
