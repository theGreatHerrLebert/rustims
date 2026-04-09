"""Error-path tests — §1.5 of the plan.

Every failure mode must surface a typed exception (or a structured
result), never a stack trace. The CLI maps each typed exception to a
stable exit code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imspy_simulation.provenance import (
    KeyNotFoundError,
    MalformedSidecar,
    MissingArtifact,
    UnknownVersion,
    sign_simulation_output,
    verify_sidecar,
)
from imspy_simulation.provenance.envelope import Sidecar
from imspy_simulation.provenance.keys import (
    generate_keypair,
    load_private_key,
    load_public_key,
    write_keypair,
)

from .conftest import make_minimal_d, make_minimal_ground_truth


# ---------------------------------------------------------------------------
# Key errors
# ---------------------------------------------------------------------------


def test_missing_private_key_raises_key_not_found(tmp_path):
    """Asking sign() to use a private key file that does not exist must raise KeyNotFoundError."""
    d = make_minimal_d(tmp_path)
    config = tmp_path / "c.toml"
    config.write_bytes(b"[experiment]\nexperiment_name = \"x\"\n")

    with pytest.raises(KeyNotFoundError):
        sign_simulation_output(
            d_path=d,
            ground_truth_path=None,
            config_path=config,
            experiment_name="x",
            simulator_version="test",
            private_key_path=tmp_path / "no-such-key.pem",
        )


def test_missing_public_key_override_raises_key_not_found(tmp_path):
    """Verify with --public-key pointing at a missing file → KeyNotFoundError."""
    # Build a valid signed bundle first.
    d = make_minimal_d(tmp_path, name="x")
    config = tmp_path / "c.toml"
    config.write_bytes(b"[experiment]\nexperiment_name = \"x\"\n")
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)
    sidecar = sign_simulation_output(
        d_path=d,
        ground_truth_path=None,
        config_path=config,
        experiment_name="x",
        simulator_version="test",
        private_key_path=key_dir / "signing_key.pem",
    )

    with pytest.raises(KeyNotFoundError):
        verify_sidecar(sidecar, public_key_override=tmp_path / "no-such.pem")


def test_load_private_key_unreadable_raises(tmp_path):
    junk = tmp_path / "junk.pem"
    junk.write_bytes(b"not a real key")
    with pytest.raises(Exception):  # cryptography raises a ValueError or similar
        load_private_key(junk)


def test_load_public_key_unreadable_raises(tmp_path):
    junk = tmp_path / "junk.pem"
    junk.write_bytes(b"not a real public key")
    with pytest.raises(Exception):
        load_public_key(junk)


# ---------------------------------------------------------------------------
# Sidecar parsing errors
# ---------------------------------------------------------------------------


def test_corrupted_sidecar_json_raises_malformed(tmp_path):
    bad = tmp_path / "bad.provenance.json"
    bad.write_bytes(b"this is not json {{{ ")
    with pytest.raises(MalformedSidecar):
        verify_sidecar(bad)


def test_sidecar_root_must_be_object(tmp_path):
    bad = tmp_path / "bad.provenance.json"
    bad.write_bytes(b'["not", "an", "object"]')
    with pytest.raises(MalformedSidecar):
        verify_sidecar(bad)


def test_sidecar_missing_payload_raises_malformed(tmp_path):
    bad = tmp_path / "bad.provenance.json"
    bad.write_bytes(json.dumps({
        "type": "timsim.provenance.v0",
        "signature": "ed25519:base64:AAAA",
        "verifying_key": "ed25519:base64:AAAA",
    }).encode("utf-8"))
    with pytest.raises(MalformedSidecar):
        verify_sidecar(bad)


def test_sidecar_with_unknown_type_raises_unknown_version(tmp_path):
    bad = tmp_path / "bad.provenance.json"
    bad.write_bytes(json.dumps({
        "type": "future.format.v99",
        "payload": {},
        "signature": "ed25519:base64:AAAA",
        "verifying_key": "ed25519:base64:AAAA",
    }).encode("utf-8"))
    with pytest.raises(UnknownVersion):
        verify_sidecar(bad)


def test_sidecar_with_unknown_canonicalization_version_raises(tmp_path):
    """A sidecar with type=v0 but canonicalization_version=v99 must raise UnknownVersion."""
    bad = tmp_path / "bad.provenance.json"
    bad.write_bytes(json.dumps({
        "type": "timsim.provenance.v0",
        "payload": {
            "simulator_name": "TimSim",
            "simulator_version": "test",
            "experiment_name": "x",
            "config_hash": "sha256:0",
            "d_content_hash": "sha256:0",
            "ground_truth_hash": "",
            "content_hash": "sha256:0",
            "timestamp_utc": "2026-01-01T00:00:00.000Z",
            "key_id": "timsim-local-x",
            "canonicalization_version": "v99",
        },
        "signature": "ed25519:base64:AAAA",
        "verifying_key": "ed25519:base64:AAAA",
    }).encode("utf-8"))
    with pytest.raises(UnknownVersion):
        verify_sidecar(bad)


def test_nonexistent_sidecar_raises_malformed(tmp_path):
    """Asking verify_sidecar for a path that does not exist raises MalformedSidecar."""
    with pytest.raises(MalformedSidecar):
        verify_sidecar(tmp_path / "nope.provenance.json")


# ---------------------------------------------------------------------------
# Missing artifact errors
# ---------------------------------------------------------------------------


def test_sidecar_present_but_d_missing_raises_missing_artifact(tmp_path):
    """Sidecar exists, but the .d it references is gone → MissingArtifact."""
    d = make_minimal_d(tmp_path, name="x")
    config = tmp_path / "c.toml"
    config.write_bytes(b"[experiment]\nexperiment_name = \"x\"\n")
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)
    sidecar = sign_simulation_output(
        d_path=d,
        ground_truth_path=None,
        config_path=config,
        experiment_name="x",
        simulator_version="test",
        private_key_path=key_dir / "signing_key.pem",
    )

    import shutil
    shutil.rmtree(d)

    with pytest.raises(MissingArtifact):
        verify_sidecar(sidecar)


def test_sidecar_with_ground_truth_but_db_missing(tmp_path):
    """Sidecar references ground truth but the file is gone → MissingArtifact."""
    d = make_minimal_d(tmp_path, name="x")
    gt = make_minimal_ground_truth(tmp_path)
    config = tmp_path / "c.toml"
    config.write_bytes(b"[experiment]\nexperiment_name = \"x\"\n")
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)
    sidecar = sign_simulation_output(
        d_path=d,
        ground_truth_path=gt,
        config_path=config,
        experiment_name="x",
        simulator_version="test",
        private_key_path=key_dir / "signing_key.pem",
    )

    gt.unlink()

    with pytest.raises(MissingArtifact):
        verify_sidecar(sidecar)


# ---------------------------------------------------------------------------
# Sign-time errors
# ---------------------------------------------------------------------------


def test_sign_with_missing_d_raises_missing_artifact(tmp_path):
    config = tmp_path / "c.toml"
    config.write_bytes(b"[experiment]\nexperiment_name = \"x\"\n")
    with pytest.raises(MissingArtifact):
        sign_simulation_output(
            d_path=tmp_path / "no-such.d",
            ground_truth_path=None,
            config_path=config,
            experiment_name="x",
            simulator_version="test",
            private_key_path=None,
        )


def test_sign_with_missing_config_raises_missing_artifact(tmp_path):
    d = make_minimal_d(tmp_path)
    with pytest.raises(MissingArtifact):
        sign_simulation_output(
            d_path=d,
            ground_truth_path=None,
            config_path=tmp_path / "no-such.toml",
            experiment_name="x",
            simulator_version="test",
            private_key_path=None,
        )
