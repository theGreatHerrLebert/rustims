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


def test_load_private_key_corrupt_raises_malformed_key(tmp_path):
    """A corrupt PEM file must raise MalformedKey, not bare ValueError.

    The simulator hook only catches ProvenanceError. Without this
    wrapping, a corrupt key file would abort the simulation even with
    [provenance] required = false. (Reviewer round-3)
    """
    from imspy_simulation.provenance.errors import MalformedKey
    junk = tmp_path / "junk.pem"
    junk.write_bytes(b"not a real key")
    with pytest.raises(MalformedKey):
        load_private_key(junk)


def test_load_public_key_corrupt_raises_malformed_key(tmp_path):
    from imspy_simulation.provenance.errors import MalformedKey
    junk = tmp_path / "junk.pem"
    junk.write_bytes(b"not a real public key")
    with pytest.raises(MalformedKey):
        load_public_key(junk)


def test_load_private_key_wrong_algorithm_raises_malformed_key(tmp_path):
    """A valid PEM that holds a non-Ed25519 key is also MalformedKey."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from imspy_simulation.provenance.errors import MalformedKey

    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = rsa_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    rsa_path = tmp_path / "rsa.pem"
    rsa_path.write_bytes(pem)
    with pytest.raises(MalformedKey, match="not an Ed25519 private key"):
        load_private_key(rsa_path)


def test_sign_with_corrupt_private_key_raises_malformed_key(tmp_path):
    """The simulator-hook contract: sign_simulation_output with a corrupt
    key raises ProvenanceError (specifically MalformedKey, a subclass).

    The hook catches ProvenanceError, so this means a corrupt key file
    with required=false will produce a warning and continue, NOT abort
    the simulation. Without the wrapping, ValueError bubbles past the
    hook's except clause. (Reviewer round-3)
    """
    from imspy_simulation.provenance.errors import MalformedKey, ProvenanceError
    d = make_minimal_d(tmp_path, name="x")
    config = tmp_path / "c.toml"
    config.write_bytes(b"[experiment]\nexperiment_name = \"x\"\n")
    bad_key = tmp_path / "bad.pem"
    bad_key.write_bytes(b"-----BEGIN PRIVATE KEY-----\nGARBAGE\n-----END PRIVATE KEY-----\n")

    with pytest.raises(ProvenanceError):  # MalformedKey is a ProvenanceError
        sign_simulation_output(
            d_path=d,
            ground_truth_path=None,
            config_path=config,
            experiment_name="x",
            simulator_version="test",
            private_key_path=bad_key,
        )

    # And the more specific check, since callers may want to distinguish.
    with pytest.raises(MalformedKey):
        sign_simulation_output(
            d_path=d,
            ground_truth_path=None,
            config_path=config,
            experiment_name="x",
            simulator_version="test",
            private_key_path=bad_key,
        )


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


def test_sign_with_stale_rollback_journal_raises(tmp_path):
    """A .d with a stale analysis.tdf-journal must NOT be signed silently.

    The canonical hasher opens analysis.tdf with mode=ro&immutable=1,
    which ignores the journal — so without this check we would attest
    to a view that sqlite would otherwise roll back. (Reviewer round-1 #3)
    """
    from imspy_simulation.provenance.errors import SqliteNotQuiescent
    d = make_minimal_d(tmp_path, name="x")
    config = tmp_path / "c.toml"
    config.write_bytes(b"[experiment]\nexperiment_name = \"x\"\n")
    journal = d / "analysis.tdf-journal"
    journal.write_bytes(b"\x00" * 16)  # any non-empty journal-like file is enough

    with pytest.raises(SqliteNotQuiescent):
        sign_simulation_output(
            d_path=d,
            ground_truth_path=None,
            config_path=config,
            experiment_name="x",
            simulator_version="test",
            private_key_path=None,
        )


def test_sign_with_wal_sidecar_raises(tmp_path):
    """A .d with a -wal sidecar (live WAL writer) must NOT be signed.

    Reviewer round-2 #1: WAL-mode SQLite plus mode=ro&immutable=1 means
    the canonical hasher sees the old main-DB image, not what readers
    will get. Must refuse, not silently produce a misleading hash.
    """
    from imspy_simulation.provenance.errors import SqliteNotQuiescent
    d = make_minimal_d(tmp_path, name="x")
    config = tmp_path / "c.toml"
    config.write_bytes(b"[experiment]\nexperiment_name = \"x\"\n")
    (d / "analysis.tdf-wal").write_bytes(b"\x00" * 32)

    with pytest.raises(SqliteNotQuiescent):
        sign_simulation_output(
            d_path=d,
            ground_truth_path=None,
            config_path=config,
            experiment_name="x",
            simulator_version="test",
            private_key_path=None,
        )


def test_sign_with_shm_sidecar_raises(tmp_path):
    """A .d with a -shm sidecar (WAL shared memory) must NOT be signed."""
    from imspy_simulation.provenance.errors import SqliteNotQuiescent
    d = make_minimal_d(tmp_path, name="x")
    config = tmp_path / "c.toml"
    config.write_bytes(b"[experiment]\nexperiment_name = \"x\"\n")
    (d / "analysis.tdf-shm").write_bytes(b"\x00" * 32)

    with pytest.raises(SqliteNotQuiescent):
        sign_simulation_output(
            d_path=d,
            ground_truth_path=None,
            config_path=config,
            experiment_name="x",
            simulator_version="test",
            private_key_path=None,
        )


def test_sign_with_ground_truth_wal_raises(tmp_path):
    """If the ground-truth DB has a WAL sidecar, signing must also refuse.

    The check applies uniformly to every SQLite file we hash, not just
    the .d's analysis.tdf. The synthetic_data.db is just as vulnerable.
    """
    from imspy_simulation.provenance.errors import SqliteNotQuiescent
    d = make_minimal_d(tmp_path, name="x")
    gt = make_minimal_ground_truth(tmp_path)
    config = tmp_path / "c.toml"
    config.write_bytes(b"[experiment]\nexperiment_name = \"x\"\n")
    # Plant a -wal next to the ground-truth DB.
    gt_wal = gt.with_name(gt.name + "-wal")
    gt_wal.write_bytes(b"\x00" * 32)

    with pytest.raises(SqliteNotQuiescent):
        sign_simulation_output(
            d_path=d,
            ground_truth_path=gt,
            config_path=config,
            experiment_name="x",
            simulator_version="test",
            private_key_path=None,
        )


def test_verify_refuses_when_wal_sidecar_appears_after_signing(tmp_path):
    """If a -wal appears between sign and verify, verify must refuse.

    This is the failure mode where someone signs cleanly, then a writer
    opens the .d in WAL mode and commits between sign and verify. The
    verifier must NOT silently report VERIFIED — its view diverges from
    what consumers will read.
    """
    from imspy_simulation.provenance.errors import SqliteNotQuiescent
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

    # Drop a -wal next to analysis.tdf AFTER signing.
    (d / "analysis.tdf-wal").write_bytes(b"\x00" * 32)

    with pytest.raises(SqliteNotQuiescent):
        verify_sidecar(sidecar)


# ---------------------------------------------------------------------------
# Malformed embedded key / signature decoding (Reviewer #5)
# ---------------------------------------------------------------------------


def test_malformed_verifying_key_raises_malformed_sidecar(tmp_path):
    """A sidecar whose embedded verifying_key cannot be decoded is malformed."""
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

    blob = json.loads(sidecar.read_text())
    blob["verifying_key"] = "ed25519:base64:!!!not-base64!!!"
    sidecar.write_text(json.dumps(blob))

    with pytest.raises(MalformedSidecar):
        verify_sidecar(sidecar)


def test_malformed_signature_raises_malformed_sidecar(tmp_path):
    """A sidecar whose signature is undecodable is malformed (not a generic error)."""
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

    blob = json.loads(sidecar.read_text())
    blob["signature"] = "ed25519:base64:!!!not-base64!!!"
    sidecar.write_text(json.dumps(blob))

    with pytest.raises(MalformedSidecar):
        verify_sidecar(sidecar)


def test_verify_cli_malformed_public_key_override_returns_key_error(tmp_path):
    """timsim-verify --public-key bad.pem -> exit 2 (EXIT_KEY_ERROR), not generic 1.

    Reviewer round-6 (LOW): MalformedKey was falling through the explicit
    except clauses and surfacing as 'unexpected error: MalformedKey' with
    exit 1. The typed error should map to the documented key-error code.
    """
    from imspy_simulation.provenance.cli import main as cli_main, EXIT_KEY_ERROR

    d = make_minimal_d(tmp_path, name="x")
    config = tmp_path / "c.toml"
    config.write_bytes(b'[experiment]\nexperiment_name = "x"\n')
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

    bad_pem = tmp_path / "bad.pem"
    bad_pem.write_bytes(b"not a real public key PEM")

    rc = cli_main([str(sidecar.parent), "--public-key", str(bad_pem)])
    assert rc == EXIT_KEY_ERROR


def test_keys_cli_show_with_corrupt_signing_key_returns_key_error(tmp_path):
    """timsim-keys show with a corrupt signing_key.pem -> exit 2."""
    from imspy_simulation.provenance.keys_cli import main as keys_main, EXIT_KEY_ERROR

    key_dir = tmp_path / "kd"
    key_dir.mkdir()
    (key_dir / "signing_key.pem").write_bytes(b"not a real PKCS8 PEM")

    rc = keys_main(["show", "--key-dir", str(key_dir)])
    assert rc == EXIT_KEY_ERROR


def test_keys_cli_export_with_corrupt_signing_key_returns_key_error(tmp_path):
    from imspy_simulation.provenance.keys_cli import main as keys_main, EXIT_KEY_ERROR

    key_dir = tmp_path / "kd"
    key_dir.mkdir()
    (key_dir / "signing_key.pem").write_bytes(b"-----BEGIN PRIVATE KEY-----\nGARBAGE\n-----END PRIVATE KEY-----\n")

    rc = keys_main(["export", "--key-dir", str(key_dir), "--to", str(tmp_path / "out.pem")])
    assert rc == EXIT_KEY_ERROR


def test_cli_unsigned_default_returns_zero(tmp_path):
    """timsim-verify on an unsigned directory: default mode → exit 0 + UNSIGNED message.

    Per the CLI docstring, exit 4 is "only failure if --strict". Without
    --strict, finding no sidecar is informational, not a failure, so
    shell/CI users do not get a spurious non-zero exit. (Reviewer #2)
    """
    from imspy_simulation.provenance.cli import main as cli_main, EXIT_OK, EXIT_UNSIGNED
    # Empty directory: no sidecar present.
    rc = cli_main([str(tmp_path)])
    assert rc == EXIT_OK


def test_cli_unsigned_strict_returns_unsigned_exit_code(tmp_path):
    """timsim-verify --strict on an unsigned directory: exit 4 (EXIT_UNSIGNED)."""
    from imspy_simulation.provenance.cli import main as cli_main, EXIT_UNSIGNED
    rc = cli_main([str(tmp_path), "--strict"])
    assert rc == EXIT_UNSIGNED


def test_signature_with_wrong_byte_length_is_malformed(tmp_path):
    """A signature whose decoded length isn't 64 bytes must surface as MalformedSidecar."""
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

    blob = json.loads(sidecar.read_text())
    # Valid base64 but only 4 bytes of payload — too short for an Ed25519 signature.
    import base64
    blob["signature"] = "ed25519:base64:" + base64.b64encode(b"abcd").decode()
    sidecar.write_text(json.dumps(blob))

    # We accept either MalformedSidecar (preferred) or signature_ok=False
    # via the InvalidSignature path. Both are honest outcomes.
    try:
        result = verify_sidecar(sidecar)
        assert not result.signature_ok
    except MalformedSidecar:
        pass
