"""Sign/verify tests — §1.3 tamper detection + §1.4 cryptographic round-trip.

These tests exercise the full pipeline (canonicalize → sign → write
sidecar → read sidecar → recompute → verify) and the cryptographic
primitive layer in isolation. Tamper-detection tests in §1.3 prove that
real-world tampering at every level we care about is caught and reported
with the correct field name. Crypto round-trip tests in §1.4 prove the
Ed25519 layer behaves as expected under bit flips, truncations, and key
substitution.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from imspy_simulation.provenance import (
    HashMismatch,
    KeyNotFoundError,
    MalformedSidecar,
    ProvenanceError,
    SignatureMismatch,
    sign_simulation_output,
    verify_sidecar,
)
from imspy_simulation.provenance.envelope import Sidecar
from imspy_simulation.provenance.keys import (
    KeyPair,
    derive_key_id,
    generate_keypair,
    load_or_create_keypair,
    write_keypair,
)

from .conftest import (
    make_minimal_d,
    make_minimal_ground_truth,
    tamper_byte,
    tamper_sql_value,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_signed_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build a fully signed minimal bundle in ``tmp_path``.

    Returns (d_path, ground_truth_path, config_path, sidecar_path).
    """
    d_path = make_minimal_d(tmp_path, name="bundle")
    ground_truth = make_minimal_ground_truth(tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(b'[experiment]\nexperiment_name = "bundle"\n')

    # Use a per-test key directory so tests do not pollute the user's home.
    key_dir = tmp_path / "keys"
    keypair = generate_keypair()
    write_keypair(keypair, key_dir)

    sidecar_path = sign_simulation_output(
        d_path=d_path,
        ground_truth_path=ground_truth,
        config_path=config_path,
        experiment_name="bundle",
        simulator_version="test",
        private_key_path=key_dir / "signing_key.pem",
    )
    return d_path, ground_truth, config_path, sidecar_path


# ---------------------------------------------------------------------------
# §1.4 Cryptographic round-trip
# ---------------------------------------------------------------------------


def test_keypair_generation_is_deterministic_on_same_key():
    """A given keypair always derives the same key id."""
    kp = generate_keypair()
    assert kp.key_id == derive_key_id(kp.public_key)
    assert kp.key_id.startswith("timsim-local-")
    assert len(kp.key_id) == len("timsim-local-") + 16  # base32(10 bytes) = 16 chars


def test_distinct_keypairs_have_distinct_ids():
    a = generate_keypair()
    b = generate_keypair()
    assert a.key_id != b.key_id


def test_sign_verify_clean_bundle(tmp_path):
    d, gt, config, sidecar = _build_signed_bundle(tmp_path)
    result = verify_sidecar(sidecar)
    assert result.overall_ok, [(c.name, c.ok) for c in result.checks]
    assert result.signature_ok
    assert all(c.ok for c in result.checks)


def test_sign_verify_disk_round_trip(tmp_path):
    """The sidecar bytes survive a disk round trip without changing meaning."""
    _, _, _, sidecar = _build_signed_bundle(tmp_path)
    raw = sidecar.read_bytes()
    parsed = Sidecar.from_json_bytes(raw)
    assert parsed.payload.experiment_name == "bundle"
    # Re-serializing the parsed sidecar must give bytes that the same
    # parser accepts. (We do not require byte-for-byte equality of the
    # outer envelope — the inner payload is what is signed.)
    Sidecar.from_json_bytes(parsed.to_json_bytes())


def test_signature_bit_flip_fails(tmp_path):
    """Flipping one byte of the signature must cause verification to fail."""
    _, _, _, sidecar = _build_signed_bundle(tmp_path)
    blob = json.loads(sidecar.read_text())
    sig = blob["signature"]
    # The signature is "ed25519:base64:..." — flip a character in the base64.
    prefix, b64 = sig.rsplit(":", 1)
    flipped_char = "B" if b64[5] != "B" else "C"
    blob["signature"] = f"{prefix}:{b64[:5]}{flipped_char}{b64[6:]}"
    sidecar.write_text(json.dumps(blob))

    result = verify_sidecar(sidecar)
    assert not result.signature_ok
    assert not result.overall_ok


def test_signature_truncation_fails(tmp_path):
    """A truncated signature must fail verification, not crash."""
    _, _, _, sidecar = _build_signed_bundle(tmp_path)
    blob = json.loads(sidecar.read_text())
    prefix, b64 = blob["signature"].rsplit(":", 1)
    blob["signature"] = f"{prefix}:{b64[:-4]}"
    sidecar.write_text(json.dumps(blob))

    result = verify_sidecar(sidecar)
    assert not result.signature_ok


def test_payload_modification_fails(tmp_path):
    """If any payload field is mutated post-sign, verification fails."""
    _, _, _, sidecar = _build_signed_bundle(tmp_path)
    blob = json.loads(sidecar.read_text())
    blob["payload"]["experiment_name"] = "stolen"
    sidecar.write_text(json.dumps(blob))

    result = verify_sidecar(sidecar)
    assert not result.signature_ok, (
        "modifying a payload field must invalidate the signature"
    )


def test_substitute_verifying_key_fails(tmp_path):
    """Replacing the embedded verifying key with a different key invalidates the sig."""
    _, _, _, sidecar = _build_signed_bundle(tmp_path)
    blob = json.loads(sidecar.read_text())
    other = generate_keypair()
    from cryptography.hazmat.primitives import serialization
    raw = other.public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    import base64
    blob["verifying_key"] = "ed25519:base64:" + base64.b64encode(raw).decode()
    sidecar.write_text(json.dumps(blob))

    result = verify_sidecar(sidecar)
    assert not result.signature_ok


# ---------------------------------------------------------------------------
# §1.3 Tamper detection — full pipeline
# ---------------------------------------------------------------------------


def test_tamper_tdf_bin_byte_flip(tmp_path):
    d, _, _, sidecar = _build_signed_bundle(tmp_path)
    tamper_byte(d / "analysis.tdf_bin", 64)
    result = verify_sidecar(sidecar)
    assert not result.overall_ok
    bin_check = next(c for c in result.checks if c.name == "d_content_hash")
    assert not bin_check.ok


def test_tamper_global_metadata(tmp_path):
    d, _, _, sidecar = _build_signed_bundle(tmp_path)
    tamper_sql_value(
        d / "analysis.tdf",
        table="GlobalMetadata",
        set_column="Value",
        set_value="FAKE",
        where_column="Key",
        where_value="InstrumentName",
    )
    result = verify_sidecar(sidecar)
    assert not result.overall_ok
    d_check = next(c for c in result.checks if c.name == "d_content_hash")
    assert not d_check.ok


def test_tamper_frame_real_value(tmp_path):
    d, _, _, sidecar = _build_signed_bundle(tmp_path)
    conn = sqlite3.connect(d / "analysis.tdf")
    conn.execute("UPDATE Frames SET Time = 999.0 WHERE Id=1;")
    conn.commit()
    conn.close()
    result = verify_sidecar(sidecar)
    assert not result.overall_ok


def test_tamper_inject_row(tmp_path):
    d, _, _, sidecar = _build_signed_bundle(tmp_path)
    conn = sqlite3.connect(d / "analysis.tdf")
    conn.execute("INSERT INTO GlobalMetadata (Key, Value) VALUES ('Sneaky', 'Injected');")
    conn.commit()
    conn.close()
    result = verify_sidecar(sidecar)
    assert not result.overall_ok


def test_tamper_remove_row(tmp_path):
    d, _, _, sidecar = _build_signed_bundle(tmp_path)
    conn = sqlite3.connect(d / "analysis.tdf")
    conn.execute("DELETE FROM Frames WHERE Id=1;")
    conn.commit()
    conn.close()
    result = verify_sidecar(sidecar)
    assert not result.overall_ok


def test_tamper_ground_truth(tmp_path):
    d, gt, _, sidecar = _build_signed_bundle(tmp_path)
    conn = sqlite3.connect(gt)
    conn.execute("UPDATE peptides SET sequence='HACKED' WHERE id=1;")
    conn.commit()
    conn.close()
    result = verify_sidecar(sidecar)
    assert not result.overall_ok
    gt_check = next(c for c in result.checks if c.name == "ground_truth_hash")
    assert not gt_check.ok


def test_tamper_config_copy(tmp_path):
    """Tampering the COPIED config next to the sidecar must be detected.

    The verifier reads the config copy that sign() placed next to the
    sidecar at sign time, NOT the user's original input. This is the
    file the signature commits to and the file that must be tamper-
    evident on verify.
    """
    d, gt, config, sidecar = _build_signed_bundle(tmp_path)
    # Find the config copy by sidecar-derived filename — the same way
    # the verifier does.
    config_copy = sidecar.parent / sidecar.name.replace(
        ".provenance.json", ".config.toml"
    )
    assert config_copy.is_file(), "sign() should have copied the config next to the sidecar"
    config_copy.write_bytes(b'[experiment]\nexperiment_name = "MUTATED"\n')
    result = verify_sidecar(sidecar)
    assert not result.overall_ok
    cfg_check = next(c for c in result.checks if c.name == "config_hash")
    assert cfg_check.status == "mismatch"


def test_tamper_original_config_does_not_affect_verification(tmp_path):
    """Mutating the user's original input config has no effect on verification.

    The signed bytes are the bytes that were copied at sign time, not
    a live reference to the user's input. This is the right behavior:
    the bundle is self-contained from the moment sign() returns.
    """
    d, gt, config, sidecar = _build_signed_bundle(tmp_path)
    config.write_bytes(b'[experiment]\nexperiment_name = "MUTATED-AFTERWARDS"\n')
    result = verify_sidecar(sidecar)
    assert result.overall_ok
    cfg_check = next(c for c in result.checks if c.name == "config_hash")
    assert cfg_check.status == "ok"


def test_missing_config_copy_is_unchecked_not_tautology(tmp_path):
    """If the config copy is missing and no override is given, the check is UNCHECKED.

    Critically, the verifier must NOT fall back to the signed value
    (which would make the check tautological — comparing the payload
    hash to itself, always passes). overall_ok must be False.
    """
    d, gt, config, sidecar = _build_signed_bundle(tmp_path)
    config_copy = sidecar.parent / sidecar.name.replace(
        ".provenance.json", ".config.toml"
    )
    config_copy.unlink()

    result = verify_sidecar(sidecar)
    assert not result.overall_ok, (
        "missing config must NOT be silently treated as verified"
    )
    cfg_check = next(c for c in result.checks if c.name == "config_hash")
    assert cfg_check.status == "unchecked"
    # The composed content_hash also goes UNCHECKED because we cannot
    # reconstruct it without the config bytes.
    content_check = next(c for c in result.checks if c.name == "content_hash")
    assert content_check.status == "unchecked"


def test_config_override_explicit_path(tmp_path):
    """The --config override path is honored when the verifier is called directly."""
    d, gt, config, sidecar = _build_signed_bundle(tmp_path)
    # Delete the convention copy to prove the override is what matters.
    config_copy = sidecar.parent / sidecar.name.replace(
        ".provenance.json", ".config.toml"
    )
    config_copy.unlink()

    result = verify_sidecar(sidecar, config_path_override=config)
    assert result.overall_ok
    cfg_check = next(c for c in result.checks if c.name == "config_hash")
    assert cfg_check.status == "ok"


def test_config_override_with_wrong_file_fails_loudly(tmp_path):
    """A --config override pointing at unrelated bytes must fail, not pass."""
    d, gt, config, sidecar = _build_signed_bundle(tmp_path)
    wrong = tmp_path / "unrelated.toml"
    wrong.write_bytes(b'[unrelated]\nfield = "wrong"\n')
    result = verify_sidecar(sidecar, config_path_override=wrong)
    assert not result.overall_ok
    cfg_check = next(c for c in result.checks if c.name == "config_hash")
    assert cfg_check.status == "mismatch"


def test_tamper_swap_d_with_other(tmp_path, tmp_path_factory):
    """Replacing the .d entirely (with one from another simulation) is detected.

    The replacement source is built in a *separate* temp directory so the
    verifier's path discovery does not see two candidate .d folders in the
    search root after the swap.
    """
    d, gt, config, sidecar = _build_signed_bundle(tmp_path)
    other_root = tmp_path_factory.mktemp("other")
    other = make_minimal_d(other_root, name="bundle", peaks_per_frame=20)

    import shutil
    shutil.rmtree(d)
    shutil.copytree(other, d)

    result = verify_sidecar(sidecar)
    assert not result.overall_ok


# ---------------------------------------------------------------------------
# Survives a VACUUM in the wild — the operational invariance test
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Vendor-parity case: sign WITHOUT a ground-truth DB
# ---------------------------------------------------------------------------
#
# Real timsTOF acquisitions have no synthetic_data.db — only TimSim outputs do.
# The whole point of the SIGNING.md proposal is that the same scheme should
# work for both. These tests prove that the .d-only path (the case we will
# eventually hand to vendors) is functional end-to-end.


def test_sign_verify_d_only_no_ground_truth(tmp_path):
    """Sign and verify a .d with NO ground-truth DB (vendor-parity case)."""
    d_path = make_minimal_d(tmp_path, name="vendor_like")
    config_path = tmp_path / "vendor.toml"
    config_path.write_bytes(b'[experiment]\nexperiment_name = "vendor_like"\n')

    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)

    sidecar = sign_simulation_output(
        d_path=d_path,
        ground_truth_path=None,  # <-- the vendor case
        config_path=config_path,
        experiment_name="vendor_like",
        simulator_version="vendor-parity-test",
        private_key_path=key_dir / "signing_key.pem",
    )
    assert sidecar.exists()

    result = verify_sidecar(sidecar)
    assert result.overall_ok, [(c.name, c.ok) for c in result.checks]
    assert result.signature_ok

    # The ground_truth_hash check should NOT appear in the result — there
    # is nothing to check.
    check_names = [c.name for c in result.checks]
    assert "ground_truth_hash" not in check_names, (
        "vendor-parity sidecar should not produce a ground_truth_hash check"
    )
    # But d_content_hash, config_hash, and content_hash MUST appear.
    assert "d_content_hash" in check_names
    assert "config_hash" in check_names
    assert "content_hash" in check_names


def test_d_only_payload_distinguishable_from_with_ground_truth(tmp_path):
    """A .d-only attestation must NOT collide with a .d+ground-truth attestation.

    This is a domain-separation property: it must be cryptographically
    impossible to take a vendor signature (no ground truth) and pass it
    off as a TimSim signature (with ground truth) of the same .d.
    """
    d_path = make_minimal_d(tmp_path, name="dual")
    gt = make_minimal_ground_truth(tmp_path)
    config_path = tmp_path / "dual.toml"
    config_path.write_bytes(b'[experiment]\nexperiment_name = "dual"\n')

    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)

    # Sign once with ground truth, once without — same .d, same config, same key.
    sidecar_with = sign_simulation_output(
        d_path=d_path,
        ground_truth_path=gt,
        config_path=config_path,
        experiment_name="dual",
        simulator_version="dual",
        sidecar_path=tmp_path / "with.provenance.json",
        private_key_path=key_dir / "signing_key.pem",
    )
    sidecar_without = sign_simulation_output(
        d_path=d_path,
        ground_truth_path=None,
        config_path=config_path,
        experiment_name="dual",
        simulator_version="dual",
        sidecar_path=tmp_path / "without.provenance.json",
        private_key_path=key_dir / "signing_key.pem",
    )

    blob_with = json.loads(sidecar_with.read_text())
    blob_without = json.loads(sidecar_without.read_text())
    # Content hashes MUST differ — domain separation between the two cases.
    assert blob_with["payload"]["content_hash"] != blob_without["payload"]["content_hash"]
    # And the d_content_hash itself MUST be identical (same .d bytes).
    assert blob_with["payload"]["d_content_hash"] == blob_without["payload"]["d_content_hash"]


def test_vacuum_after_signing_still_verifies(tmp_path):
    """If a user runs VACUUM on analysis.tdf post-signing, verification still passes.

    This is the operational version of test_vacuum_invariance: not just
    'the canonicalizer is invariant', but 'the full sign+verify pipeline
    survives a real container-level operation in the wild'.
    """
    d, _, _, sidecar = _build_signed_bundle(tmp_path)
    conn = sqlite3.connect(d / "analysis.tdf")
    conn.execute("VACUUM;")
    conn.close()
    result = verify_sidecar(sidecar)
    assert result.overall_ok, "VACUUM broke verification — canonical form is wrong"
