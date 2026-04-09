"""Tests for the trust model: --expected-key-id, --require-trusted, registry, timsim-keys CLI.

The trust layer is what converts the Phase 0 prototype from "internally
consistent signing" into "signing with an actual identity claim". These
tests cover:

  - The TrustedKeyRegistry round-trip (load, add, save, find, remove)
  - --expected-key-id pinning (match and mismatch)
  - --require-trusted with key in registry / not in registry
  - The forgery defense: a sidecar that claims a trusted key id but
    embeds DIFFERENT public key bytes must NOT verify
  - Idempotent and conflict-detection behavior of registry.add()
  - The full timsim-keys CLI surface (show, export, trust, list, untrust)

Tests use isolated registry paths via the registry= override so they
do not pollute the user's ~/.config/timsim/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imspy_simulation.provenance import (
    sign_simulation_output,
    verify_sidecar,
)
from imspy_simulation.provenance.errors import ProvenanceError
from imspy_simulation.provenance.keys import (
    derive_key_id,
    generate_keypair,
    write_keypair,
)
from imspy_simulation.provenance.trust import (
    REGISTRY_SCHEMA,
    TrustedKey,
    TrustedKeyRegistry,
    trusted_key_from_pem_file,
    trusted_key_from_public_key,
    trusted_key_from_sidecar_file,
)

from .conftest import make_minimal_d, make_minimal_ground_truth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_signed_bundle_with_keypair(tmp_path: Path, *, name: str = "trustbundle"):
    """Build a signed bundle and return (bundle_paths, keypair, registry_path)."""
    d = make_minimal_d(tmp_path, name=name)
    gt = make_minimal_ground_truth(tmp_path)
    config = tmp_path / f"{name}.toml"
    config.write_bytes(b'[experiment]\nexperiment_name = "' + name.encode() + b'"\n')

    key_dir = tmp_path / "keys"
    keypair = generate_keypair()
    write_keypair(keypair, key_dir)

    sidecar = sign_simulation_output(
        d_path=d,
        ground_truth_path=gt,
        config_path=config,
        experiment_name=name,
        simulator_version="trust-test",
        private_key_path=key_dir / "signing_key.pem",
    )

    return d, gt, config, sidecar, keypair


# ---------------------------------------------------------------------------
# Registry persistence
# ---------------------------------------------------------------------------


def test_registry_load_missing_returns_empty(tmp_path):
    reg = TrustedKeyRegistry.load(tmp_path / "no_such.json")
    assert len(reg) == 0
    assert list(reg) == []


def test_registry_round_trip(tmp_path):
    reg_path = tmp_path / "trusted.json"
    reg = TrustedKeyRegistry.load(reg_path)

    kp = generate_keypair()
    entry = trusted_key_from_public_key(kp.public_key, comment="test entry")
    reg.add(entry)
    reg.save()

    # Reload from a fresh handle.
    reg2 = TrustedKeyRegistry.load(reg_path)
    assert len(reg2) == 1
    assert reg2.find(kp.key_id) is not None
    assert reg2.find(kp.key_id).comment == "test entry"
    assert kp.key_id in reg2


def test_registry_add_idempotent(tmp_path):
    """Adding the same key twice is a no-op (same id, same PEM)."""
    reg = TrustedKeyRegistry.load(tmp_path / "r.json")
    kp = generate_keypair()
    entry = trusted_key_from_public_key(kp.public_key, comment="x")
    reg.add(entry)
    reg.add(entry)  # idempotent
    assert len(reg) == 1


def test_registry_add_conflict_refuses_silent_overwrite(tmp_path):
    """Two keys with the same id but different PEMs is impossible in practice,
    but the registry must refuse to overwrite — never silently trust new bytes
    under an existing id."""
    reg = TrustedKeyRegistry.load(tmp_path / "r.json")
    kp = generate_keypair()
    e1 = trusted_key_from_public_key(kp.public_key, comment="first")
    # Construct a malicious second entry with the same id but different PEM.
    e2 = TrustedKey(
        key_id=kp.key_id,  # same id
        public_key_pem="-----BEGIN PUBLIC KEY-----\nDIFFERENT\n-----END PUBLIC KEY-----\n",
        comment="forged",
        added_at="2026-04-09T00:00:00.000Z",
    )
    reg.add(e1)
    with pytest.raises(ProvenanceError, match="different public key"):
        reg.add(e2)


def test_registry_remove(tmp_path):
    reg = TrustedKeyRegistry.load(tmp_path / "r.json")
    kp = generate_keypair()
    reg.add(trusted_key_from_public_key(kp.public_key, comment="x"))
    assert kp.key_id in reg
    assert reg.remove(kp.key_id) is True
    assert kp.key_id not in reg
    assert reg.remove(kp.key_id) is False  # already gone


def test_registry_schema_is_persisted(tmp_path):
    reg_path = tmp_path / "r.json"
    reg = TrustedKeyRegistry.load(reg_path)
    kp = generate_keypair()
    reg.add(trusted_key_from_public_key(kp.public_key, comment="x"))
    reg.save()
    blob = json.loads(reg_path.read_text())
    assert blob["schema"] == REGISTRY_SCHEMA


def test_registry_load_unknown_schema_raises(tmp_path):
    from imspy_simulation.provenance.errors import MalformedSidecar
    reg_path = tmp_path / "r.json"
    reg_path.write_text(json.dumps({"schema": "future.schema/v99", "keys": []}))
    with pytest.raises(MalformedSidecar):
        TrustedKeyRegistry.load(reg_path)


# ---------------------------------------------------------------------------
# verify_sidecar with --expected-key-id
# ---------------------------------------------------------------------------


def test_verify_with_expected_key_id_match(tmp_path):
    _, _, _, sidecar, keypair = _build_signed_bundle_with_keypair(tmp_path)
    result = verify_sidecar(sidecar, expected_key_id=keypair.key_id)
    assert result.overall_ok
    assert result.trust.status == "ok"
    assert result.trust.was_requested


def test_verify_with_expected_key_id_mismatch(tmp_path):
    _, _, _, sidecar, _ = _build_signed_bundle_with_keypair(tmp_path)
    result = verify_sidecar(
        sidecar, expected_key_id="timsim-local-totally-different-key"
    )
    assert not result.overall_ok
    assert result.trust.status == "id_mismatch"
    assert "expected" in result.trust.detail.lower()


def test_verify_without_pinning_marks_trust_not_requested(tmp_path):
    _, _, _, sidecar, _ = _build_signed_bundle_with_keypair(tmp_path)
    result = verify_sidecar(sidecar)
    assert result.overall_ok
    assert result.trust.status == "not_requested"
    assert result.trust.was_requested is False


def test_expected_key_id_cannot_be_bypassed_by_payload_relabel(tmp_path):
    """The reviewer's bypass must NOT work.

    Scenario:
      1. Sign a bundle with the FORGER's key (key B).
      2. Mutate payload.key_id to a fake "victim" id (key A).
      3. Re-sign the mutated payload with the FORGER's key — math still works.
      4. Caller passes --expected-key-id A.

    Before the fix, this returned VERIFIED + TRUSTED because the trust
    check used payload.key_id (an attacker-controllable signed string)
    instead of deriving the signer id from sidecar.verifying_key.

    After the fix, the consistency check at parse time catches the
    discrepancy and raises MalformedSidecar — payload.key_id (= fake A)
    no longer matches the key id derived from the actual verifying_key
    (= forger B).
    """
    import base64
    from imspy_simulation.provenance.envelope import Payload
    from imspy_simulation.provenance.errors import MalformedSidecar

    _, _, _, sidecar, forger = _build_signed_bundle_with_keypair(tmp_path)
    fake_id = "timsim-local-aaaaaaaaaaaaaaaa"

    blob = json.loads(sidecar.read_text())
    blob["payload"]["key_id"] = fake_id
    payload = Payload.from_dict(blob["payload"])
    new_sig = forger.private_key.sign(payload.to_canonical_json())
    blob["signature"] = "ed25519:base64:" + base64.b64encode(new_sig).decode()
    # verifying_key is still the forger's
    sidecar.write_text(json.dumps(blob))

    with pytest.raises(MalformedSidecar, match="payload.key_id"):
        verify_sidecar(sidecar, expected_key_id=fake_id)


def test_expected_key_id_uses_derived_signer_not_payload_field(tmp_path):
    """A clean bundle: --expected-key-id matches the DERIVED signer key id.

    The derived id and the payload id agree on a clean bundle, so the
    user's experience is unchanged. This test exists to lock down the
    contract: --expected-key-id is compared against the cryptographic
    signer, not against a label.
    """
    from imspy_simulation.provenance.keys import derive_key_id

    _, _, _, sidecar, keypair = _build_signed_bundle_with_keypair(tmp_path)
    derived_id = derive_key_id(keypair.public_key)
    # Sanity: on a clean bundle the two are equal.
    assert derived_id == keypair.key_id

    result = verify_sidecar(sidecar, expected_key_id=derived_id)
    assert result.overall_ok
    assert result.trust.status == "ok"
    assert result.trust.actual_key_id == derived_id


def test_inconsistent_payload_key_id_caught_without_trust_flags(tmp_path):
    """The consistency check is unconditional — it runs even without trust pinning.

    A sidecar where payload.key_id does not match derive(verifying_key)
    is malformed regardless of whether the caller passed --expected-key-id
    or --require-trusted. This guards against forgery attempts even on
    casual ad-hoc verifies.
    """
    import base64
    from imspy_simulation.provenance.envelope import Payload
    from imspy_simulation.provenance.errors import MalformedSidecar

    _, _, _, sidecar, forger = _build_signed_bundle_with_keypair(tmp_path)
    blob = json.loads(sidecar.read_text())
    blob["payload"]["key_id"] = "timsim-local-zzzzzzzzzzzzzzzz"
    payload = Payload.from_dict(blob["payload"])
    new_sig = forger.private_key.sign(payload.to_canonical_json())
    blob["signature"] = "ed25519:base64:" + base64.b64encode(new_sig).decode()
    sidecar.write_text(json.dumps(blob))

    # No trust flags at all — must still be rejected.
    with pytest.raises(MalformedSidecar, match="payload.key_id"):
        verify_sidecar(sidecar)


# ---------------------------------------------------------------------------
# verify_sidecar with --require-trusted
# ---------------------------------------------------------------------------


def test_require_trusted_with_key_in_registry(tmp_path):
    _, _, _, sidecar, keypair = _build_signed_bundle_with_keypair(tmp_path)
    reg_path = tmp_path / "trusted.json"
    reg = TrustedKeyRegistry.load(reg_path)
    reg.add(trusted_key_from_public_key(keypair.public_key, comment="lab"))
    reg.save()

    result = verify_sidecar(
        sidecar, require_trusted=True, trusted_registry_path=reg_path
    )
    assert result.overall_ok
    assert result.trust.status == "ok"


def test_require_trusted_with_key_not_in_registry(tmp_path):
    _, _, _, sidecar, _ = _build_signed_bundle_with_keypair(tmp_path)
    reg_path = tmp_path / "trusted.json"  # never created

    result = verify_sidecar(
        sidecar, require_trusted=True, trusted_registry_path=reg_path
    )
    assert not result.overall_ok
    assert result.trust.status == "not_in_registry"
    assert "registry" in result.trust.detail.lower()


def test_require_trusted_catches_forged_key_id(tmp_path):
    """A sidecar that claims a trusted key id but signed with different bytes is rejected.

    After the consistency-check fix, the forgery is caught at parse time
    as MalformedSidecar BEFORE the trust layer runs (because the lied
    payload.key_id no longer matches the key id derived from the
    embedded verifying_key). This is a STRONGER defense than the
    previous registry-PEM check because it does not require
    --require-trusted to be set — the forgery is rejected unconditionally.
    """
    import base64
    from cryptography.hazmat.primitives import serialization

    from imspy_simulation.provenance.errors import MalformedSidecar

    victim = generate_keypair()
    forger = generate_keypair()

    _, _, _, sidecar, _ = _build_signed_bundle_with_keypair(tmp_path)
    blob = json.loads(sidecar.read_text())

    # Re-point payload.key_id at the VICTIM and re-sign with the FORGER.
    blob["payload"]["key_id"] = victim.key_id
    from imspy_simulation.provenance.envelope import Payload
    payload = Payload.from_dict(blob["payload"])
    forger_signature = forger.private_key.sign(payload.to_canonical_json())
    blob["signature"] = "ed25519:base64:" + base64.b64encode(forger_signature).decode()
    # verifying_key still belongs to the forger.
    forger_raw = forger.public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    blob["verifying_key"] = "ed25519:base64:" + base64.b64encode(forger_raw).decode()
    sidecar.write_text(json.dumps(blob))

    reg_path = tmp_path / "trusted.json"
    reg = TrustedKeyRegistry.load(reg_path)
    reg.add(trusted_key_from_public_key(victim.public_key, comment="victim"))
    reg.save()

    # The consistency check rejects this BEFORE the trust check runs.
    with pytest.raises(MalformedSidecar, match="payload.key_id"):
        verify_sidecar(
            sidecar, require_trusted=True, trusted_registry_path=reg_path
        )

    # And it rejects without --require-trusted too — the consistency
    # check is unconditional.
    with pytest.raises(MalformedSidecar, match="payload.key_id"):
        verify_sidecar(sidecar)


# ---------------------------------------------------------------------------
# trusted_key_from_* helpers
# ---------------------------------------------------------------------------


def test_trusted_key_from_pem_file(tmp_path):
    kp = generate_keypair()
    key_dir = write_keypair(kp, tmp_path / "k")
    entry = trusted_key_from_pem_file(
        key_dir / "verifying_key.pem", comment="from PEM"
    )
    assert entry.key_id == kp.key_id
    assert entry.comment == "from PEM"


def test_trusted_key_from_sidecar_file(tmp_path):
    _, _, _, sidecar, keypair = _build_signed_bundle_with_keypair(tmp_path)
    entry = trusted_key_from_sidecar_file(sidecar, comment="from sidecar")
    assert entry.key_id == keypair.key_id
    assert entry.comment == "from sidecar"


# ---------------------------------------------------------------------------
# timsim-keys CLI
# ---------------------------------------------------------------------------


def test_keys_cli_show(tmp_path, capsys):
    from imspy_simulation.provenance.keys_cli import main as keys_main
    rc = keys_main(["show", "--key-dir", str(tmp_path / "kd")])
    captured = capsys.readouterr()
    assert rc == 0
    assert "key id:" in captured.out
    assert "timsim-local-" in captured.out
    assert "software-rooted" in captured.out.lower()


def test_keys_cli_export_to_file(tmp_path):
    from imspy_simulation.provenance.keys_cli import main as keys_main
    out = tmp_path / "pub.pem"
    rc = keys_main([
        "export",
        "--key-dir", str(tmp_path / "kd"),
        "--to", str(out),
    ])
    assert rc == 0
    pem = out.read_text()
    assert "BEGIN PUBLIC KEY" in pem
    assert "END PUBLIC KEY" in pem


def test_keys_cli_trust_list_untrust_round_trip(tmp_path, capsys):
    """Full registry workflow: trust a sidecar, list it, untrust it, list empty."""
    from imspy_simulation.provenance.keys_cli import main as keys_main

    _, _, _, sidecar, keypair = _build_signed_bundle_with_keypair(tmp_path)
    reg_path = tmp_path / "trusted.json"

    # trust
    rc = keys_main([
        "trust", str(sidecar),
        "--comment", "from-test",
        "--registry", str(reg_path),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "trusted:" in captured.out
    assert keypair.key_id in captured.out

    # list
    rc = keys_main(["list", "--registry", str(reg_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "1 key" in captured.out
    assert keypair.key_id in captured.out
    assert "from-test" in captured.out

    # untrust
    rc = keys_main(["untrust", keypair.key_id, "--registry", str(reg_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "untrusted:" in captured.out
    assert keypair.key_id in captured.out

    # list (should be empty)
    rc = keys_main(["list", "--registry", str(reg_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "empty" in captured.out


def test_keys_cli_trust_pem_file(tmp_path, capsys):
    from imspy_simulation.provenance.keys_cli import main as keys_main
    kp = generate_keypair()
    key_dir = write_keypair(kp, tmp_path / "k")
    reg_path = tmp_path / "trusted.json"

    rc = keys_main([
        "trust", str(key_dir / "verifying_key.pem"),
        "--comment", "from-pem",
        "--registry", str(reg_path),
    ])
    assert rc == 0
    reg = TrustedKeyRegistry.load(reg_path)
    assert kp.key_id in reg


def test_keys_cli_trust_directory_finds_sidecar(tmp_path, capsys):
    from imspy_simulation.provenance.keys_cli import main as keys_main
    _, _, _, sidecar, keypair = _build_signed_bundle_with_keypair(tmp_path)
    reg_path = tmp_path / "trusted.json"

    rc = keys_main([
        "trust", str(sidecar.parent),
        "--comment", "from-dir",
        "--registry", str(reg_path),
    ])
    assert rc == 0
    reg = TrustedKeyRegistry.load(reg_path)
    assert keypair.key_id in reg


def test_keys_cli_untrust_missing_id_returns_4(tmp_path, capsys):
    from imspy_simulation.provenance.keys_cli import main as keys_main, EXIT_KEY_NOT_FOUND
    reg_path = tmp_path / "trusted.json"
    rc = keys_main([
        "untrust", "timsim-local-no-such-key",
        "--registry", str(reg_path),
    ])
    assert rc == EXIT_KEY_NOT_FOUND


def test_keys_cli_trust_requires_comment(tmp_path):
    """The --comment flag is required to prevent thoughtless trust grants."""
    from imspy_simulation.provenance.keys_cli import main as keys_main
    _, _, _, sidecar, _ = _build_signed_bundle_with_keypair(tmp_path)
    reg_path = tmp_path / "trusted.json"
    with pytest.raises(SystemExit):
        keys_main([
            "trust", str(sidecar),
            "--registry", str(reg_path),
        ])


# ---------------------------------------------------------------------------
# timsim-verify CLI integration with trust flags
# ---------------------------------------------------------------------------


def test_verify_cli_expected_key_id_match(tmp_path):
    from imspy_simulation.provenance.cli import main as verify_main, EXIT_OK
    _, _, _, sidecar, keypair = _build_signed_bundle_with_keypair(tmp_path)
    rc = verify_main([
        str(sidecar.parent),
        "--expected-key-id", keypair.key_id,
    ])
    assert rc == EXIT_OK


def test_verify_cli_expected_key_id_mismatch_returns_7(tmp_path):
    from imspy_simulation.provenance.cli import (
        main as verify_main,
        EXIT_KEY_NOT_TRUSTED,
    )
    _, _, _, sidecar, _ = _build_signed_bundle_with_keypair(tmp_path)
    rc = verify_main([
        str(sidecar.parent),
        "--expected-key-id", "timsim-local-totally-different",
    ])
    assert rc == EXIT_KEY_NOT_TRUSTED


def test_verify_cli_require_trusted_without_registry_returns_7(tmp_path):
    """--require-trusted with no registry → EXIT_KEY_NOT_TRUSTED."""
    from imspy_simulation.provenance.cli import (
        main as verify_main,
        EXIT_KEY_NOT_TRUSTED,
    )
    import os

    _, _, _, sidecar, _ = _build_signed_bundle_with_keypair(tmp_path)

    # Point the registry default at an empty location for this test by
    # setting XDG_CONFIG_HOME — verify_main only sees the default path.
    old = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")
    try:
        rc = verify_main([str(sidecar.parent), "--require-trusted"])
    finally:
        if old is None:
            del os.environ["XDG_CONFIG_HOME"]
        else:
            os.environ["XDG_CONFIG_HOME"] = old
    assert rc == EXIT_KEY_NOT_TRUSTED
