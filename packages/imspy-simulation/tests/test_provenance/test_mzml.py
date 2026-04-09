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


# ---------------------------------------------------------------------------
# Reviewer round-9: every binary array must be hashed (not just mz/intensity)
# ---------------------------------------------------------------------------
#
# Previously the canonicalizer dropped any array whose role was not
# "m/z" or "intensity". That meant ion-mobility, charge, pressure,
# wavelength, and any other consumer-visible binary array could be
# tampered without affecting the hash. The fix hashes every array
# with a stable role label, sorted by role for determinism.


def _build_mzml_with_extra_array(
    tmp_path: Path,
    *,
    name: str,
    extra_role_accession: str,
    extra_role_name: str,
    extra_array_values: list[float],
) -> Path:
    """Build a minimal mzml with one MS1 spectrum that has THREE binary arrays:
    m/z, intensity, and an extra array with the given role accession.
    """
    import base64

    mz_bytes = b"".join(struct.pack("<d", v) for v in [100.0, 200.0, 300.0])
    int_bytes = b"".join(struct.pack("<d", v) for v in [1000.0, 2000.0, 3000.0])
    extra_bytes = b"".join(struct.pack("<d", v) for v in extra_array_values)

    mz_b64 = base64.b64encode(mz_bytes).decode("ascii")
    int_b64 = base64.b64encode(int_bytes).decode("ascii")
    extra_b64 = base64.b64encode(extra_bytes).decode("ascii")

    mzml = f'''<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0" id="{name}">
  <run id="r">
    <spectrumList count="1">
      <spectrum id="scan=1" index="0" defaultArrayLength="3">
        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
        <cvParam cvRef="MS" accession="MS:1000130" name="positive scan"/>
        <binaryDataArrayList count="3">
          <binaryDataArray encodedLength="{len(mz_b64)}">
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>
            <cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>
            <cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>
            <binary>{mz_b64}</binary>
          </binaryDataArray>
          <binaryDataArray encodedLength="{len(int_b64)}">
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>
            <cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>
            <cvParam cvRef="MS" accession="MS:1000515" name="intensity array"/>
            <binary>{int_b64}</binary>
          </binaryDataArray>
          <binaryDataArray encodedLength="{len(extra_b64)}">
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>
            <cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>
            <cvParam cvRef="MS" accession="{extra_role_accession}" name="{extra_role_name}"/>
            <binary>{extra_b64}</binary>
          </binaryDataArray>
        </binaryDataArrayList>
      </spectrum>
    </spectrumList>
  </run>
</mzML>
'''
    p = tmp_path / f"{name}.mzML"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(mzml, encoding="utf-8")
    return p


def test_canonicalize_mzml_detects_ion_mobility_array_tamper(tmp_path):
    """The reviewer-round-9 HIGH bug: tampering an ion-mobility array
    must NOT go undetected. Before the fix, the canonicalizer silently
    dropped any binary array whose role was not m/z or intensity.
    """
    p1 = _build_mzml_with_extra_array(
        tmp_path / "a",
        name="clean",
        extra_role_accession="MS:1003008",
        extra_role_name="raw inverse reduced ion mobility array",
        extra_array_values=[0.85, 0.90, 0.95],
    )
    p2 = _build_mzml_with_extra_array(
        tmp_path / "b",
        name="tampered",
        extra_role_accession="MS:1003008",
        extra_role_name="raw inverse reduced ion mobility array",
        extra_array_values=[0.85, 0.90, 99.99],  # last value tampered
    )
    h1 = canonicalize_mzml(p1)
    h2 = canonicalize_mzml(p2)
    assert h1 != h2, (
        "ion mobility array tamper went undetected — canonicalizer is "
        "still dropping non-mz/intensity binary arrays"
    )


def test_canonicalize_mzml_detects_charge_array_tamper(tmp_path):
    p1 = _build_mzml_with_extra_array(
        tmp_path / "a",
        name="clean",
        extra_role_accession="MS:1000516",
        extra_role_name="charge array",
        extra_array_values=[1.0, 2.0, 3.0],
    )
    p2 = _build_mzml_with_extra_array(
        tmp_path / "b",
        name="tampered",
        extra_role_accession="MS:1000516",
        extra_role_name="charge array",
        extra_array_values=[1.0, 2.0, 9.0],
    )
    assert canonicalize_mzml(p1) != canonicalize_mzml(p2)


def test_canonicalize_mzml_detects_unknown_role_array_tamper(tmp_path):
    """Even an array with an accession we do not have in our known set
    must be hashed (under an "unknown:..." label) so a tamper is caught.
    """
    p1 = _build_mzml_with_extra_array(
        tmp_path / "a",
        name="clean",
        extra_role_accession="MS:9999999",  # not in our known set
        extra_role_name="hypothetical future array",
        extra_array_values=[42.0, 43.0, 44.0],
    )
    p2 = _build_mzml_with_extra_array(
        tmp_path / "b",
        name="tampered",
        extra_role_accession="MS:9999999",
        extra_role_name="hypothetical future array",
        extra_array_values=[42.0, 43.0, 99.0],
    )
    assert canonicalize_mzml(p1) != canonicalize_mzml(p2)


def test_canonicalize_mzml_detects_added_unknown_array(tmp_path):
    """Adding an unknown-role array changes the hash via the array_count tag."""
    p_minimal = make_minimal_mzml(tmp_path / "min")
    p_with_extra = _build_mzml_with_extra_array(
        tmp_path / "ext",
        name="with_extra",
        extra_role_accession="MS:1003008",
        extra_role_name="raw inverse reduced ion mobility array",
        extra_array_values=[0.85, 0.90, 0.95],
    )
    # Both have m/z and intensity arrays in spectrum 0; the second has
    # an extra ion-mobility array. The hashes must differ.
    assert canonicalize_mzml(p_minimal) != canonicalize_mzml(p_with_extra)


def test_canonicalize_mzml_array_order_invariance(tmp_path):
    """If two converters write the same arrays in different document
    order, the canonical form must be invariant. The canonicalizer
    sorts by role label."""
    import base64

    mz_bytes = b"".join(struct.pack("<d", v) for v in [100.0, 200.0])
    int_bytes = b"".join(struct.pack("<d", v) for v in [1000.0, 2000.0])
    mz_b64 = base64.b64encode(mz_bytes).decode("ascii")
    int_b64 = base64.b64encode(int_bytes).decode("ascii")

    def build(arrays_in_order):
        return f'''<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
  <run id="r"><spectrumList count="1">
    <spectrum id="scan=1" index="0" defaultArrayLength="2">
      <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
      <cvParam cvRef="MS" accession="MS:1000130" name="positive scan"/>
      <binaryDataArrayList count="2">
        {arrays_in_order}
      </binaryDataArrayList>
    </spectrum>
  </spectrumList></run>
</mzML>
'''

    mz_array = (
        f'<binaryDataArray encodedLength="{len(mz_b64)}">'
        f'<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
        f'<cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>'
        f'<cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>'
        f'<binary>{mz_b64}</binary>'
        f'</binaryDataArray>'
    )
    int_array = (
        f'<binaryDataArray encodedLength="{len(int_b64)}">'
        f'<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
        f'<cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>'
        f'<cvParam cvRef="MS" accession="MS:1000515" name="intensity array"/>'
        f'<binary>{int_b64}</binary>'
        f'</binaryDataArray>'
    )

    p1 = tmp_path / "mz_first.mzML"
    p1.write_text(build(mz_array + int_array))
    p2 = tmp_path / "int_first.mzML"
    p2.write_text(build(int_array + mz_array))

    assert p1.read_text() != p2.read_text()  # sanity: bytes differ
    assert canonicalize_mzml(p1) == canonicalize_mzml(p2)


def test_canonicalize_mzml_rejects_array_with_no_role_cvparam(tmp_path):
    """A binaryDataArray with NO content-role cvParam (only encoding ones)
    must raise — we never silently produce a hash that ignores it."""
    import base64

    mz_bytes = b"".join(struct.pack("<d", v) for v in [100.0, 200.0])
    mz_b64 = base64.b64encode(mz_bytes).decode("ascii")
    bad_mzml = f'''<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
  <run id="r"><spectrumList count="1">
    <spectrum id="scan=1" index="0" defaultArrayLength="2">
      <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
      <cvParam cvRef="MS" accession="MS:1000130" name="positive scan"/>
      <binaryDataArrayList count="1">
        <binaryDataArray encodedLength="{len(mz_b64)}">
          <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>
          <cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>
          <binary>{mz_b64}</binary>
        </binaryDataArray>
      </binaryDataArrayList>
    </spectrum>
  </spectrumList></run>
</mzML>
'''
    p = tmp_path / "bad.mzML"
    p.write_text(bad_mzml)
    with pytest.raises(MalformedSidecar, match="no cvParam identifying"):
        canonicalize_mzml(p)


# ---------------------------------------------------------------------------
# Reviewer round-9: sidecar discovery uses conventional pairing first
# ---------------------------------------------------------------------------


def test_verify_two_mzml_bundles_in_same_directory(tmp_path):
    """Two signed bundles can coexist in the same directory: the verifier
    uses the conventional {sidecar_stem}.mzML pairing rather than
    requiring the directory to contain a unique mzML."""
    # First bundle.
    mzml_a = make_minimal_mzml(tmp_path, name="sample_a")
    cfg_a = tmp_path / "sample_a.toml"
    cfg_a.write_bytes(b'[experiment]\nexperiment_name = "sample_a"\n')
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)
    sidecar_a = sign_mzml_output(
        mzml_path=mzml_a,
        config_path=cfg_a,
        experiment_name="sample_a",
        private_key_path=key_dir / "signing_key.pem",
    )

    # Second bundle in the SAME directory.
    mzml_b = make_minimal_mzml(tmp_path, name="sample_b", n_ms1_peaks=5)
    cfg_b = tmp_path / "sample_b.toml"
    cfg_b.write_bytes(b'[experiment]\nexperiment_name = "sample_b"\n')
    sidecar_b = sign_mzml_output(
        mzml_path=mzml_b,
        config_path=cfg_b,
        experiment_name="sample_b",
        private_key_path=key_dir / "signing_key.pem",
    )

    # Both sidecars must verify cleanly. Before the fix, both would
    # fail because _find_unique_mzml(parent) saw multiple mzml files.
    result_a = verify_sidecar(sidecar_a)
    result_b = verify_sidecar(sidecar_b)
    assert result_a.overall_ok, [(c.name, c.status) for c in result_a.checks]
    assert result_b.overall_ok, [(c.name, c.status) for c in result_b.checks]


# ---------------------------------------------------------------------------
# Reviewer round-10: integer encoding metadata must affect the canonical hash
# ---------------------------------------------------------------------------
#
# Charge arrays are typically encoded as integers. Before this fix the
# canonicalizer collapsed every non-float encoding to width=0 / count=0
# and emitted the precision tag as b"??" — meaning a converter that
# switched the encoding cvParam from MS:1000519 (i32) to MS:1000522
# (i64) would produce the same hash even though consumers would
# interpret the bytes completely differently. The fix adds explicit
# tags for i32 and i64 with correct widths.


def _build_mzml_with_charge_array_encoding(
    tmp_path: Path,
    *,
    name: str,
    encoding_acc: str,
    encoding_name: str,
    raw_payload: bytes,
) -> Path:
    """Build a tiny mzml with a single charge-array binaryDataArray
    using the given encoding cvParam and raw payload bytes."""
    import base64

    b64 = base64.b64encode(raw_payload).decode("ascii")
    mzml = f'''<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
  <run id="r"><spectrumList count="1">
    <spectrum id="scan=1" index="0" defaultArrayLength="2">
      <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
      <cvParam cvRef="MS" accession="MS:1000130" name="positive scan"/>
      <binaryDataArrayList count="1">
        <binaryDataArray encodedLength="{len(b64)}">
          <cvParam cvRef="MS" accession="{encoding_acc}" name="{encoding_name}"/>
          <cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>
          <cvParam cvRef="MS" accession="MS:1000516" name="charge array"/>
          <binary>{b64}</binary>
        </binaryDataArray>
      </binaryDataArrayList>
    </spectrum>
  </spectrumList></run>
</mzML>
'''
    p = tmp_path / f"{name}.mzML"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(mzml, encoding="utf-8")
    return p


def test_canonicalize_mzml_distinguishes_i32_from_i64_encoding(tmp_path):
    """Same raw bytes, different encoding cvParam (i32 vs i64) → different hash.

    The reviewer's exact reproduction. Before the fix, both encodings
    were collapsed to b"?? "/width=0 and the only thing in the canonical
    record that depended on encoding was a sha256 of the raw payload —
    which is identical because the bytes are identical. The fix adds
    explicit i32 and i64 tags with correct widths, so the precision
    tag and the value count both differ.
    """
    import struct
    raw = struct.pack("<ii", 1, 2)  # 8 bytes interpretable as 2x i32 or 1x i64
    p32 = _build_mzml_with_charge_array_encoding(
        tmp_path / "a",
        name="i32",
        encoding_acc="MS:1000519",
        encoding_name="32-bit integer",
        raw_payload=raw,
    )
    p64 = _build_mzml_with_charge_array_encoding(
        tmp_path / "b",
        name="i64",
        encoding_acc="MS:1000522",
        encoding_name="64-bit integer",
        raw_payload=raw,
    )
    assert canonicalize_mzml(p32) != canonicalize_mzml(p64), (
        "i32-vs-i64 encoding cvParam swap was not detected — the "
        "canonical form is still collapsing integer encodings"
    )


def test_canonicalize_mzml_distinguishes_f32_from_i32_encoding(tmp_path):
    """Same bytes interpreted as f32 vs i32 must hash differently."""
    import struct
    # 8 bytes that could be 2 f32s or 2 i32s — totally different values either way.
    raw = b"\x00\x00\x80\x3f\x00\x00\x00\x40"  # f32: [1.0, 2.0]; i32: small
    p_f = _build_mzml_with_charge_array_encoding(
        tmp_path / "f",
        name="f32",
        encoding_acc="MS:1000521",
        encoding_name="32-bit float",
        raw_payload=raw,
    )
    p_i = _build_mzml_with_charge_array_encoding(
        tmp_path / "i",
        name="i32",
        encoding_acc="MS:1000519",
        encoding_name="32-bit integer",
        raw_payload=raw,
    )
    assert canonicalize_mzml(p_f) != canonicalize_mzml(p_i)


def test_canonicalize_mzml_distinguishes_f64_from_i64_encoding(tmp_path):
    """Same bytes interpreted as f64 vs i64 must hash differently."""
    import struct
    raw = struct.pack("<d", 3.141592653589793)
    p_f = _build_mzml_with_charge_array_encoding(
        tmp_path / "f",
        name="f64",
        encoding_acc="MS:1000523",
        encoding_name="64-bit float",
        raw_payload=raw,
    )
    p_i = _build_mzml_with_charge_array_encoding(
        tmp_path / "i",
        name="i64",
        encoding_acc="MS:1000522",
        encoding_name="64-bit integer",
        raw_payload=raw,
    )
    assert canonicalize_mzml(p_f) != canonicalize_mzml(p_i)


def test_canonicalize_mzml_int_charge_array_tamper_detected(tmp_path):
    """A realistic charge array (int32-encoded) with one byte tampered
    must produce a different hash. This is the primary use case the
    integer-encoding fix unlocks."""
    import struct
    clean = struct.pack("<iii", 1, 2, 3)
    tampered = struct.pack("<iii", 1, 2, 99)
    p_clean = _build_mzml_with_charge_array_encoding(
        tmp_path / "a",
        name="charges_clean",
        encoding_acc="MS:1000519",
        encoding_name="32-bit integer",
        raw_payload=clean,
    )
    p_tampered = _build_mzml_with_charge_array_encoding(
        tmp_path / "b",
        name="charges_tampered",
        encoding_acc="MS:1000519",
        encoding_name="32-bit integer",
        raw_payload=tampered,
    )
    assert canonicalize_mzml(p_clean) != canonicalize_mzml(p_tampered)


def test_canonicalize_mzml_int32_value_count_is_correct(tmp_path):
    """An i32 array with 4 bytes of payload reports value_count = 1
    (4 bytes / 4 bytes per i32). The previous bug reported value_count = 0
    for any non-float encoding because width was 0.

    This is an indirect check: we build two i32 arrays with payloads
    of different LENGTHS (1 value vs 2 values) but where the first
    1-value payload is byte-for-byte identical to the first 4 bytes of
    the 2-value payload. Without per-array value_count in the canonical
    record, only the truncated payload's hash would differ — but with
    value_count emitted, the records are doubly distinguished.

    More importantly: a tampered i32 array of any length produces a
    different hash. This locks in that integer arrays are truly part
    of the canonical form, not just bytes-with-width-0.
    """
    import struct
    one_val = struct.pack("<i", 42)
    two_vals = struct.pack("<ii", 42, 43)
    p1 = _build_mzml_with_charge_array_encoding(
        tmp_path / "a", name="one", encoding_acc="MS:1000519",
        encoding_name="32-bit integer", raw_payload=one_val,
    )
    p2 = _build_mzml_with_charge_array_encoding(
        tmp_path / "b", name="two", encoding_acc="MS:1000519",
        encoding_name="32-bit integer", raw_payload=two_vals,
    )
    assert canonicalize_mzml(p1) != canonicalize_mzml(p2)


def test_verify_falls_back_to_unique_when_convention_misses(tmp_path):
    """If the sidecar's conventional pairing does not exist (e.g. the
    user renamed the mzml after signing) but only one mzml is present,
    the verifier still finds it via the uniqueness fallback."""
    mzml = make_minimal_mzml(tmp_path, name="original")
    cfg = tmp_path / "original.toml"
    cfg.write_bytes(b'[experiment]\nexperiment_name = "original"\n')
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)
    sidecar = sign_mzml_output(
        mzml_path=mzml,
        config_path=cfg,
        experiment_name="original",
        private_key_path=key_dir / "signing_key.pem",
    )

    # User renames the mzml. The conventional pairing
    # original.provenance.json -> original.mzML no longer exists,
    # but renamed.mzML is the only mzml in the dir, so the fallback
    # finds it.
    renamed = mzml.with_name("renamed.mzML")
    mzml.rename(renamed)
    result = verify_sidecar(sidecar)
    assert result.overall_ok
