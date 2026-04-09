"""Real-data integration tests for the provenance system.

Everything else in the test suite uses synthetic fixtures (hand-crafted
SQLite tables and hand-written mzML). Those exercise correctness on
shapes we control, but they don't catch the failure mode "the
canonicalizer hits a real-world cvParam, table, or array we never
anticipated".

This file runs the full sign + verify + tamper cycle against:

  1. A real Bruker timsTOF DDA blank acquisition (from
     /scratch/raw/dda/blanks/), which has 21 SQLite tables, ~4M rows
     in FrameProperties, ~25k Frames, ~218k PASEF rows, 35 GlobalMetadata
     entries, and assorted XML / sqlite sibling files in the .d.
  2. A real msconvert-produced mzML (a sage test fixture pulled from a
     Thermo Q Exactive RAW), which has referenceableParamGroupList,
     softwareList, sourceFile entries with the original RAW SHA-1, and
     real-converter cvParam usage.

Both files are READ-ONLY references; the tests copy / symlink them
into a tempdir so the originals are never modified. Both tests are
marked @pytest.mark.slow and skip cleanly if the source files are
missing — so a CI machine without the local datasets does not break.

The point is correctness, not performance. We only assert that:

  - canonicalize / sign / verify do not raise on real-world shapes
  - clean verify reports VERIFIED
  - tampering produces a hash mismatch that the verifier catches
  - canonicalize is deterministic across two consecutive runs on the
    same real input
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from imspy_simulation.provenance import (
    canonicalize_mzml,
    sign_mzml_output,
    sign_simulation_output,
    verify_sidecar,
)
from imspy_simulation.provenance.canonicalize import canonicalize_d
from imspy_simulation.provenance.keys import generate_keypair, write_keypair


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Real-data paths and skip guards
# ---------------------------------------------------------------------------

REAL_BRUKER_D = Path("/scratch/raw/dda/blanks/K240723_001_S1-A2_1_2772.d")
REAL_MSCONVERT_MZML = Path(
    "/home/administrator/Documents/promotion/sage/tests/LQSRPAAPPAPGPGQLTLR.mzML"
)


def _skip_if_no_real_d():
    if not REAL_BRUKER_D.is_dir():
        pytest.skip(f"real Bruker .d not present at {REAL_BRUKER_D}")
    for child in ("analysis.tdf", "analysis.tdf_bin"):
        if not (REAL_BRUKER_D / child).is_file():
            pytest.skip(f"{REAL_BRUKER_D / child} is missing")


def _skip_if_no_real_mzml():
    if not REAL_MSCONVERT_MZML.is_file():
        pytest.skip(f"real msconvert mzML not present at {REAL_MSCONVERT_MZML}")


# ---------------------------------------------------------------------------
# Helpers: stage real data into a tempdir without copying the big binary
# ---------------------------------------------------------------------------


def _stage_real_d_for_test(tmp_path: Path, *, name: str = "blank") -> Path:
    """Place a writable copy of the real .d in tmp_path.

    analysis.tdf is COPIED (134 MB, ~1 s) so the test can mutate it.
    analysis.tdf_bin (1 GB) is SYMLINKED so the streaming hasher reads
    real bytes without spending 5+ seconds on a copy. The original
    files at /scratch/raw/... are never touched.

    Returns the staged .d path.
    """
    _skip_if_no_real_d()
    dst_d = tmp_path / f"{name}.d"
    dst_d.mkdir()

    shutil.copy(REAL_BRUKER_D / "analysis.tdf", dst_d / "analysis.tdf")
    (dst_d / "analysis.tdf_bin").symlink_to(REAL_BRUKER_D / "analysis.tdf_bin")
    return dst_d


def _stage_real_mzml_for_test(tmp_path: Path, *, name: str = "real_run") -> Path:
    """Copy the real msconvert mzML into tmp_path so the test can mutate it."""
    _skip_if_no_real_mzml()
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    dst = Path(tmp_path) / f"{name}.mzML"
    shutil.copy(REAL_MSCONVERT_MZML, dst)
    return dst


# ---------------------------------------------------------------------------
# Real Bruker .d — canonicalize
# ---------------------------------------------------------------------------


def test_canonicalize_real_bruker_d_succeeds(tmp_path):
    """Hashing a real Bruker .d does not raise and returns 32 bytes.

    The real .d has 21 tables and 4M+ rows in FrameProperties — much
    more variety than any of the synthetic fixtures. If the canonicalizer
    has an unanticipated failure mode (a column type we did not handle,
    a NULL we mis-encode, a SQL identifier with special characters), it
    surfaces here.
    """
    d = _stage_real_d_for_test(tmp_path)
    t0 = time.perf_counter()
    h = canonicalize_d(d)
    elapsed = time.perf_counter() - t0
    assert h is not None and len(h) == 32
    print(f"\n  real .d canonicalize_d: {elapsed:.1f}s -> {h.hex()[:32]}...")


def test_canonicalize_real_bruker_d_is_deterministic(tmp_path):
    """Two consecutive canonicalize_d calls on the real .d return the same bytes."""
    d = _stage_real_d_for_test(tmp_path)
    h1 = canonicalize_d(d)
    h2 = canonicalize_d(d)
    assert h1 == h2


# ---------------------------------------------------------------------------
# Real Bruker .d — sign + verify + tamper
# ---------------------------------------------------------------------------


def test_sign_verify_real_bruker_d(tmp_path):
    """Full pipeline on a real Bruker .d: sign, verify clean, tamper, verify-fails."""
    d = _stage_real_d_for_test(tmp_path)
    config = tmp_path / "real_blank.toml"
    config.write_bytes(b'[experiment]\nexperiment_name = "real_blank"\n')
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)

    # Sign.
    t0 = time.perf_counter()
    sidecar = sign_simulation_output(
        d_path=d,
        ground_truth_path=None,
        config_path=config,
        experiment_name="real_blank",
        simulator_version="real-data-test",
        private_key_path=key_dir / "signing_key.pem",
    )
    sign_elapsed = time.perf_counter() - t0
    assert sidecar.exists()
    print(f"\n  real .d sign_simulation_output: {sign_elapsed:.1f}s")

    # Verify clean.
    t0 = time.perf_counter()
    result = verify_sidecar(sidecar)
    verify_elapsed = time.perf_counter() - t0
    assert result.overall_ok, [(c.name, c.status) for c in result.checks]
    print(f"  real .d verify_sidecar (clean): {verify_elapsed:.1f}s")

    # Tamper a single SQL value in analysis.tdf — this is the COPY,
    # not the source at /scratch/raw, so the original real data is
    # untouched.
    conn = sqlite3.connect(d / "analysis.tdf")
    try:
        conn.execute(
            "UPDATE GlobalMetadata SET Value='REAL-DATA-TAMPER' "
            "WHERE Key='InstrumentName'"
        )
        conn.commit()
    finally:
        conn.close()

    # Verify dirty.
    result = verify_sidecar(sidecar)
    assert not result.overall_ok, "real-data SQL tamper went undetected"
    d_check = next(c for c in result.checks if c.name == "d_content_hash")
    assert d_check.status == "mismatch"
    # Signature should still pass — the payload bytes were never touched.
    assert result.signature_ok


# ---------------------------------------------------------------------------
# Real msconvert mzML — canonicalize
# ---------------------------------------------------------------------------


def test_canonicalize_real_msconvert_mzml_succeeds(tmp_path):
    """Hashing a real msconvert-produced mzML does not raise.

    The sage test fixture is a Thermo Q Exactive scan converted by
    ProteoWizard. It has referenceableParamGroupList, softwareList,
    sourceFile with the original RAW SHA-1, real cvParam usage, and
    other elements that the synthetic fixture doesn't.
    """
    mzml = _stage_real_mzml_for_test(tmp_path)
    h = canonicalize_mzml(mzml)
    assert h is not None and len(h) == 32
    print(f"\n  real mzml canonicalize_mzml -> {h.hex()[:32]}...")


def test_canonicalize_real_msconvert_mzml_is_deterministic(tmp_path):
    mzml = _stage_real_mzml_for_test(tmp_path)
    assert canonicalize_mzml(mzml) == canonicalize_mzml(mzml)


# ---------------------------------------------------------------------------
# Real msconvert mzML — sign + verify + tamper
# ---------------------------------------------------------------------------


def test_sign_verify_real_msconvert_mzml(tmp_path):
    """Full pipeline on a real msconvert mzML: sign, verify, tamper, verify-fails."""
    mzml = _stage_real_mzml_for_test(tmp_path)
    config = tmp_path / "real_msconvert.toml"
    config.write_bytes(b'[experiment]\nexperiment_name = "real_msconvert"\n')
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)

    sidecar = sign_mzml_output(
        mzml_path=mzml,
        config_path=config,
        experiment_name="real_msconvert",
        tool_name="msconvert",
        tool_version="3.0.22204",
        private_key_path=key_dir / "signing_key.pem",
    )
    assert sidecar.exists()

    result = verify_sidecar(sidecar)
    assert result.overall_ok, [(c.name, c.status) for c in result.checks]
    assert result.signature_ok

    # Tamper an arbitrary cvParam in the mzML and re-verify.
    text = mzml.read_text()
    # The fixture has 'positive scan' in the spectrum block; flip it.
    assert '"positive scan"' in text
    text = text.replace('"positive scan"', '"negative scan"', 1)
    text = text.replace(
        'accession="MS:1000130"', 'accession="MS:1000129"', 1
    )
    mzml.write_text(text)

    result = verify_sidecar(sidecar)
    assert not result.overall_ok
    mzml_check = next(c for c in result.checks if c.name == "mzml_content_hash")
    assert mzml_check.status == "mismatch"


def test_canonicalize_real_msconvert_mzml_whitespace_invariance(tmp_path):
    """Reformatting the real mzML with extra whitespace does not change the hash.

    This is the operational version of the whitespace-invariance test
    on synthetic fixtures: it must hold on real converter output too.
    The real msconvert mzML uses self-closing ``<cvParam .../>`` tags,
    so we add whitespace right after every self-closing terminator.
    `/>` is unambiguously a tag terminator in well-formed XML — it
    cannot appear inside text or attribute content — so this mutation
    is purely whitespace.
    """
    mzml1 = _stage_real_mzml_for_test(tmp_path / "a")
    mzml2 = _stage_real_mzml_for_test(tmp_path / "b")
    h_before = canonicalize_mzml(mzml2)
    text = mzml2.read_text()
    text = text.replace("/>", "/>\n  ")
    mzml2.write_text(text)
    # Sanity: bytes really differ now.
    assert mzml1.read_text() != mzml2.read_text()
    # ...but the canonical hash is unchanged.
    h_after = canonicalize_mzml(mzml2)
    assert h_after == h_before
    assert canonicalize_mzml(mzml1) == canonicalize_mzml(mzml2)
