"""Tests for the mzML canonicalization, signing, and verification path.

These mirror the .d test suites but operate on mzML files. The mzML
path is for non-Bruker simulators (Synthedia, SMITER, msconvert
outputs, custom converters) — TimSim itself writes Bruker .d.

Each test category exists in both the .d and the mzML world:
  - canonicalization invariance (whitespace, attribute order, indexed
    wrapper) — proves the hash is over content
  - canonicalization sensitivity (peak change, added spectrum,
    precursor change) — proves the hash detects content changes
  - sign + verify round trip
  - tamper detection through the full pipeline
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import pytest

from imspy_simulation.provenance import (
    canonicalize_mzml,
    sign_mzml_output,
    verify_sidecar,
)
from imspy_simulation.provenance._fixtures import make_minimal_mzml
from imspy_simulation.provenance.canonicalize_mzml import compose_mzml_content_hash
from imspy_simulation.provenance.envelope import (
    ATTESTATION_TYPE_MZML,
    MzmlSidecar,
    parse_sidecar,
)
from imspy_simulation.provenance.errors import (
    MalformedSidecar,
    MissingArtifact,
    ProvenanceError,
)
from imspy_simulation.provenance.keys import generate_keypair, write_keypair


# ---------------------------------------------------------------------------
# Canonicalization: invariance
# ---------------------------------------------------------------------------


def test_canonicalize_mzml_idempotent(tmp_path):
    p = make_minimal_mzml(tmp_path)
    h1 = canonicalize_mzml(p)
    h2 = canonicalize_mzml(p)
    assert h1 == h2


def test_canonicalize_mzml_whitespace_invariance(tmp_path):
    """Same logical content, different indentation → same hash."""
    indented = make_minimal_mzml(tmp_path / "i", name="run", indented=True)
    compact = make_minimal_mzml(tmp_path / "c", name="run", indented=False)
    assert indented.read_text() != compact.read_text()  # sanity: bytes differ
    assert canonicalize_mzml(indented) == canonicalize_mzml(compact)


def test_canonicalize_mzml_filename_invariance(tmp_path):
    """The filename is not part of the canonical content; hash unchanged."""
    p1 = make_minimal_mzml(tmp_path / "a", name="alpha")
    p2 = make_minimal_mzml(tmp_path / "b", name="beta")
    # The id attribute on the mzML element is "alpha" vs "beta"; that
    # field is run-level metadata that we deliberately do NOT hash in v0,
    # so the hashes should match. This locks the v0 scope decision.
    assert canonicalize_mzml(p1) == canonicalize_mzml(p2)


def test_canonicalize_mzml_attribute_order_invariance(tmp_path):
    """Reordering XML attributes within an element does not change the hash.

    XML attribute order is not significant per the spec; the canonical
    form must not depend on it.
    """
    p = make_minimal_mzml(tmp_path)
    text = p.read_text()
    h_before = canonicalize_mzml(p)

    # Swap attribute order on the spectrum element. Original is
    # `<spectrum id="..." index="..." defaultArrayLength="...">`.
    swapped = text.replace(
        '<spectrum id="scan=1" index="0" defaultArrayLength="3">',
        '<spectrum defaultArrayLength="3" index="0" id="scan=1">',
    )
    assert swapped != text
    p.write_text(swapped)
    assert canonicalize_mzml(p) == h_before


def test_canonicalize_mzml_extra_whitespace_invariance(tmp_path):
    """Adding whitespace inside text nodes (not inside binary) is invariant."""
    p = make_minimal_mzml(tmp_path)
    h_before = canonicalize_mzml(p)
    text = p.read_text()
    # Add extra whitespace between cvParam elements (XML allows this).
    bloated = text.replace("</cvParam>", "</cvParam>   \n   ")
    p.write_text(bloated)
    assert canonicalize_mzml(p) == h_before


# ---------------------------------------------------------------------------
# Canonicalization: sensitivity (these MUST detect changes)
# ---------------------------------------------------------------------------


def test_canonicalize_mzml_detects_intensity_change(tmp_path):
    """Modifying a single intensity value changes the hash."""
    p = make_minimal_mzml(tmp_path)
    h_before = canonicalize_mzml(p)

    # Re-encode the intensity array with a different first value.
    new_int_values = [9999.0, 1100.0, 1200.0]
    new_bytes = b"".join(struct.pack("<d", v) for v in new_int_values)
    new_b64 = base64.b64encode(new_bytes).decode("ascii")

    # Replace the original first intensity array.
    text = p.read_text()
    original_int_values = [1000.0, 1100.0, 1200.0]
    original_bytes = b"".join(struct.pack("<d", v) for v in original_int_values)
    original_b64 = base64.b64encode(original_bytes).decode("ascii")
    assert original_b64 in text  # sanity
    text = text.replace(original_b64, new_b64, 1)
    p.write_text(text)

    h_after = canonicalize_mzml(p)
    assert h_before != h_after


def test_canonicalize_mzml_detects_added_spectrum(tmp_path):
    """Adding a spectrum changes the hash (and the count tag at the end)."""
    p1 = make_minimal_mzml(tmp_path / "a")
    p2 = make_minimal_mzml(tmp_path / "b")
    # Bump spectrumList count and append a third spectrum to p2.
    text = p2.read_text()
    text = text.replace('spectrumList count="2"', 'spectrumList count="3"')
    insertion = (
        '<spectrum id="scan=3" index="2" defaultArrayLength="3">'
        '<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>'
        '<cvParam cvRef="MS" accession="MS:1000130" name="positive scan"/>'
        '<binaryDataArrayList count="2">'
        '<binaryDataArray encodedLength="32">'
        '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
        '<cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>'
        '<cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>'
        '<binary>AAAAAAAAWUAAAAAAAABZQA==</binary>'
        '</binaryDataArray>'
        '<binaryDataArray encodedLength="32">'
        '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
        '<cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>'
        '<cvParam cvRef="MS" accession="MS:1000515" name="intensity array"/>'
        '<binary>AAAAAAAAWUAAAAAAAABZQA==</binary>'
        '</binaryDataArray>'
        '</binaryDataArrayList>'
        '</spectrum>'
    )
    text = text.replace("</spectrumList>", insertion + "</spectrumList>")
    p2.write_text(text)

    assert canonicalize_mzml(p1) != canonicalize_mzml(p2)


def test_canonicalize_mzml_detects_precursor_charge_change(tmp_path):
    """Changing the precursor charge state on the MS2 spectrum changes the hash."""
    p = make_minimal_mzml(tmp_path)
    h_before = canonicalize_mzml(p)
    text = p.read_text()
    text = text.replace(
        '"charge state" value="2"',
        '"charge state" value="3"',
    )
    p.write_text(text)
    assert canonicalize_mzml(p) != h_before


def test_canonicalize_mzml_detects_rt_change(tmp_path):
    """Changing a spectrum's retention time changes the hash."""
    p = make_minimal_mzml(tmp_path)
    h_before = canonicalize_mzml(p)
    text = p.read_text()
    text = text.replace('value="0.5"', 'value="0.55"', 1)
    p.write_text(text)
    assert canonicalize_mzml(p) != h_before


def test_canonicalize_mzml_detects_polarity_change(tmp_path):
    """Switching positive scan to negative scan changes the hash."""
    p = make_minimal_mzml(tmp_path)
    h_before = canonicalize_mzml(p)
    text = p.read_text()
    text = text.replace(
        '<cvParam cvRef="MS" accession="MS:1000130" name="positive scan"/>',
        '<cvParam cvRef="MS" accession="MS:1000129" name="negative scan"/>',
        1,  # only the first one
    )
    p.write_text(text)
    assert canonicalize_mzml(p) != h_before


# ---------------------------------------------------------------------------
# Cross-check against pyteomics
# ---------------------------------------------------------------------------


def test_canonicalize_mzml_round_trips_through_pyteomics(tmp_path):
    """The fixture mzML must parse cleanly through pyteomics, the
    standard library for the format. This is a cross-validator: if our
    canonicalizer succeeds but pyteomics fails, the fixture is wrong;
    if pyteomics succeeds but ours fails, our parser is wrong.
    """
    pyteomics = pytest.importorskip("pyteomics.mzml")
    p = make_minimal_mzml(tmp_path)

    spectra = []
    with pyteomics.read(str(p)) as reader:
        for s in reader:
            spectra.append({
                "id": s["id"],
                "ms_level": int(s.get("ms level", 0)),
                "n_mz": len(s["m/z array"]),
                "n_int": len(s["intensity array"]),
            })

    assert len(spectra) == 2
    assert spectra[0]["ms_level"] == 1
    assert spectra[1]["ms_level"] == 2
    assert spectra[0]["n_mz"] == 3
    assert spectra[1]["n_mz"] == 3

    # And our canonicalizer also produces a 32-byte hash on the same
    # file without errors.
    h = canonicalize_mzml(p)
    assert len(h) == 32


# ---------------------------------------------------------------------------
# Sign + verify round trip
# ---------------------------------------------------------------------------


def _sign_one(tmp_path: Path, *, with_config: bool = True):
    """Sign a fresh fixture mzML. Returns (mzml_path, sidecar_path, keypair)."""
    mzml = make_minimal_mzml(tmp_path, name="bundle")
    config = None
    if with_config:
        config = tmp_path / "bundle.toml"
        config.write_bytes(b'[experiment]\nexperiment_name = "bundle"\n')
    key_dir = tmp_path / "keys"
    keypair = generate_keypair()
    write_keypair(keypair, key_dir)
    sidecar = sign_mzml_output(
        mzml_path=mzml,
        config_path=config,
        experiment_name="bundle",
        tool_name="TestTool",
        tool_version="0.0.1",
        private_key_path=key_dir / "signing_key.pem",
    )
    return mzml, sidecar, keypair


def test_sign_mzml_writes_sidecar_and_config_copy(tmp_path):
    mzml, sidecar, _ = _sign_one(tmp_path)
    assert sidecar.exists()
    assert sidecar.name == "bundle.provenance.json"
    # Config copy alongside the sidecar.
    config_copy = sidecar.parent / "bundle.config.toml"
    assert config_copy.is_file()


def test_sign_mzml_envelope_type_is_mzml(tmp_path):
    _, sidecar, _ = _sign_one(tmp_path)
    blob = json.loads(sidecar.read_text())
    assert blob["type"] == ATTESTATION_TYPE_MZML
    assert "mzml_content_hash" in blob["payload"]
    assert "tool_name" in blob["payload"]
    # mzml payload must NOT have d-only fields.
    assert "d_content_hash" not in blob["payload"]
    assert "ground_truth_hash" not in blob["payload"]


def test_verify_mzml_clean_succeeds(tmp_path):
    _, sidecar, keypair = _sign_one(tmp_path)
    result = verify_sidecar(sidecar)
    assert result.overall_ok, [(c.name, c.status, c.detail) for c in result.checks]
    assert result.signature_ok
    # Trust check is not requested in this call.
    assert result.trust.status == "not_requested"


def test_verify_mzml_with_expected_key_id_match(tmp_path):
    _, sidecar, keypair = _sign_one(tmp_path)
    result = verify_sidecar(sidecar, expected_key_id=keypair.key_id)
    assert result.overall_ok
    assert result.trust.status == "ok"


def test_verify_mzml_with_expected_key_id_mismatch(tmp_path):
    _, sidecar, _ = _sign_one(tmp_path)
    result = verify_sidecar(sidecar, expected_key_id="timsim-local-wrongwrongwrongwro")
    assert not result.overall_ok
    assert result.trust.status == "id_mismatch"


def test_sign_mzml_no_config_path(tmp_path):
    """Signing with config_path=None must produce a verifiable sidecar."""
    mzml = make_minimal_mzml(tmp_path)
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)
    sidecar = sign_mzml_output(
        mzml_path=mzml,
        config_path=None,
        experiment_name="noconfig",
        private_key_path=key_dir / "signing_key.pem",
    )
    result = verify_sidecar(sidecar)
    assert result.overall_ok


# ---------------------------------------------------------------------------
# Tamper detection through the full pipeline
# ---------------------------------------------------------------------------


def test_tamper_mzml_intensity_after_signing(tmp_path):
    """Modifying an intensity value after signing must fail verification."""
    mzml, sidecar, _ = _sign_one(tmp_path)
    text = mzml.read_text()
    original_int_values = [1000.0, 1100.0, 1200.0]
    original_b64 = base64.b64encode(
        b"".join(struct.pack("<d", v) for v in original_int_values)
    ).decode("ascii")
    new_b64 = base64.b64encode(
        b"".join(struct.pack("<d", v) for v in [9999.0, 1100.0, 1200.0])
    ).decode("ascii")
    text = text.replace(original_b64, new_b64, 1)
    mzml.write_text(text)

    result = verify_sidecar(sidecar)
    assert not result.overall_ok
    mzml_check = next(c for c in result.checks if c.name == "mzml_content_hash")
    assert mzml_check.status == "mismatch"


def test_tamper_mzml_charge_state_after_signing(tmp_path):
    mzml, sidecar, _ = _sign_one(tmp_path)
    text = mzml.read_text()
    text = text.replace('"charge state" value="2"', '"charge state" value="3"')
    mzml.write_text(text)

    result = verify_sidecar(sidecar)
    assert not result.overall_ok


def test_tamper_mzml_via_whitespace_only_does_not_break_verify(tmp_path):
    """Reformatting the mzML after signing must STILL verify (whitespace invariance)."""
    mzml, sidecar, _ = _sign_one(tmp_path)
    text = mzml.read_text()
    # Add extra whitespace inside cvParam closings.
    text = text.replace("</cvParam>", "</cvParam>   \n   ")
    mzml.write_text(text)

    result = verify_sidecar(sidecar)
    assert result.overall_ok, (
        "whitespace-only reformat broke verification — canonical form is wrong"
    )


def test_tamper_config_copy_after_signing(tmp_path):
    _, sidecar, _ = _sign_one(tmp_path)
    config_copy = sidecar.parent / "bundle.config.toml"
    config_copy.write_bytes(b'[experiment]\nexperiment_name = "MUTATED"\n')
    result = verify_sidecar(sidecar)
    assert not result.overall_ok
    cfg_check = next(c for c in result.checks if c.name == "config_hash")
    assert cfg_check.status == "mismatch"


def test_missing_mzml_after_signing_raises_missing_artifact(tmp_path):
    mzml, sidecar, _ = _sign_one(tmp_path)
    mzml.unlink()
    with pytest.raises(MissingArtifact):
        verify_sidecar(sidecar)


# ---------------------------------------------------------------------------
# Cross-format isolation: a .d sidecar must not be parseable as mzml etc.
# ---------------------------------------------------------------------------


def test_parse_sidecar_dispatches_on_type(tmp_path):
    """parse_sidecar() returns the right concrete sidecar type for each blob."""
    _, mzml_sidecar_path, _ = _sign_one(tmp_path)
    parsed = parse_sidecar(mzml_sidecar_path.read_bytes())
    assert isinstance(parsed, MzmlSidecar)
    assert parsed.type == ATTESTATION_TYPE_MZML


def test_dot_d_sidecar_parser_rejects_mzml_type(tmp_path):
    """Sidecar.from_json_bytes() must refuse mzml-typed blobs."""
    from imspy_simulation.provenance.envelope import Sidecar
    from imspy_simulation.provenance.errors import UnknownVersion

    _, mzml_sidecar_path, _ = _sign_one(tmp_path)
    with pytest.raises(UnknownVersion):
        Sidecar.from_json_bytes(mzml_sidecar_path.read_bytes())


# ---------------------------------------------------------------------------
# numpress not yet supported (explicit non-goal in v0)
# ---------------------------------------------------------------------------


def test_canonicalize_mzml_rejects_numpress(tmp_path):
    """numpress compression must surface as an error, not a silent wrong hash."""
    p = make_minimal_mzml(tmp_path)
    text = p.read_text()
    # Replace one of the no-compression cvParams with numpress linear.
    text = text.replace(
        'accession="MS:1000576" name="no compression"',
        'accession="MS:1002312" name="MS-Numpress linear prediction compression"',
        1,
    )
    p.write_text(text)
    with pytest.raises(ProvenanceError, match="numpress"):
        canonicalize_mzml(p)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_verify_cli_finds_sidecar_via_mzml_path(tmp_path):
    """timsim-verify <mzml_path> discovers the sidecar next to the file."""
    from imspy_simulation.provenance.cli import main as cli_main, EXIT_OK

    mzml, _, _ = _sign_one(tmp_path)
    rc = cli_main([str(mzml)])
    assert rc == EXIT_OK


def test_verify_cli_mzml_tamper_returns_5(tmp_path):
    from imspy_simulation.provenance.cli import main as cli_main, EXIT_HASH_MISMATCH

    mzml, sidecar, _ = _sign_one(tmp_path)
    text = mzml.read_text()
    text = text.replace('"charge state" value="2"', '"charge state" value="9"')
    mzml.write_text(text)
    rc = cli_main([str(mzml)])
    assert rc == EXIT_HASH_MISMATCH
