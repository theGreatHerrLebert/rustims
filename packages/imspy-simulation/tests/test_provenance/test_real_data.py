"""Real-data integration tests for the provenance system.

Everything else in the test suite uses synthetic fixtures (hand-crafted
SQLite tables and hand-written mzML). Those exercise correctness on
shapes we control. This file exercises the canonicalize / sign /
verify pipeline against actual real-world data, so we catch the
failure mode "the canonicalizer hits a real cvParam, table, or array
we never anticipated".

The full 2x2 .d matrix is covered:

  +-----+--------------------------------------------------------------+
  |     | DDA                          | DIA                           |
  +-----+--------------------------------------------------------------+
  | Real| K240723_001_S1-A2 blank      | synchro-hela.d (synchroPASEF) |
  | Sim | TIMSIM-DDA-DEMO              | TIMSIM-HeLa10K-001            |
  +-----+------------------------------+-------------------------------+

For mzML the test exercises three real-converter outputs of varying
size and producer, including a real Bruker timsTOF mzML conversion
(the kind of input the user's collaborator's pipeline will produce):

  - sage test fixture (12 KB, Thermo Q Exactive -> ProteoWizard, 64-bit)
  - Resolution50000_32bit.mzML (2.9 MB, 32-bit float precision)
  - 20190227_TIMS2 Yeast Trypsin (51 MB, real Bruker timsTOF conversion)

All tests are marked @pytest.mark.slow and skip cleanly if the source
files are missing — so a CI machine without the local datasets does
not break.

The point is correctness, not performance. We assert that:

  - canonicalize / sign / verify do not raise on real-world shapes
  - clean verify reports VERIFIED
  - tampering produces a hash mismatch the verifier catches
  - canonicalize is deterministic across two consecutive runs
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
#
# .d 2x2 matrix:
#
#                 DDA                            DIA
#  +------+----------------------------+-----------------------------+
#  | Real | K240723 blank (Bruker)     | synchro-hela.d (Bruker)     |
#  | Sim  | TIMSIM-DDA-DEMO            | TIMSIM-HeLa10K-001          |
#  +------+----------------------------+-----------------------------+

REAL_DDA_D = Path("/scratch/raw/dda/blanks/K240723_001_S1-A2_1_2772.d")
REAL_DIA_D = Path("/media/hd02/data/raw/synchro/synchro-hela.d")
SIM_DDA_D = Path("/tmp/timsim_dda_demo/out/TIMSIM-DDA-DEMO/TIMSIM-DDA-DEMO.d")
SIM_DIA_D = Path("/scratch/timsim-demo/TIMSIM-HeLa10K-001/TIMSIM-HeLa10K-001.d")

# mzML diversity:
#   - small Thermo->ProteoWizard, 64-bit float (sage test fixture)
#   - 32-bit float precision (exercises the f32 code path on real data)
#   - real Bruker timsTOF conversion (the kind of file a real
#     collaborator's converter will produce)
THERMO_MSCONVERT_MZML = Path(
    "/home/administrator/Documents/promotion/sage/tests/LQSRPAAPPAPGPGQLTLR.mzML"
)
F32_MZML = Path(
    "/media/hd01/Scrapyard/unlipid/data/07232020_Resolution50000_32bit.mzML"
)
BRUKER_TIMSTOF_MZML = Path(
    "/media/hd02/data/raw/dda/ccs/Raw_Yeast_Trp/"
    "20190227_TIMS2_FlMe_SA_200ng_Yeast_Trypsin_IRT_Fraction_17_A6_01_4229_uncalibrated.mzML"
)


def _skip_if_no_d(d: Path, label: str) -> None:
    if not d.is_dir():
        pytest.skip(f"{label} .d not present at {d}")
    for child in ("analysis.tdf", "analysis.tdf_bin"):
        if not (d / child).is_file():
            pytest.skip(f"{label}: {d / child} is missing")


def _skip_if_no_mzml(p: Path, label: str) -> None:
    if not p.is_file():
        pytest.skip(f"{label} mzml not present at {p}")


# ---------------------------------------------------------------------------
# Helpers: stage real data into a tempdir without copying the big binary
# ---------------------------------------------------------------------------


def _stage_d_for_test(
    tmp_path: Path, source_d: Path, *, name: str
) -> Path:
    """Place a writable copy of a real .d in tmp_path.

    analysis.tdf is COPIED so the test can mutate it (this is the file
    where SQL tampers happen). analysis.tdf_bin is SYMLINKED so the
    streaming hasher reads real bytes without spending seconds on a
    copy. The originals are never touched.

    Returns the staged .d path.
    """
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    dst_d = Path(tmp_path) / f"{name}.d"
    dst_d.mkdir()

    shutil.copy(source_d / "analysis.tdf", dst_d / "analysis.tdf")
    (dst_d / "analysis.tdf_bin").symlink_to(source_d / "analysis.tdf_bin")
    return dst_d


def _stage_mzml_for_test(tmp_path: Path, source: Path, *, name: str) -> Path:
    """Copy a real mzML into tmp_path. Used for the small mzML files
    where copying is cheap; for very large mzMLs (Bruker timsTOF
    conversions can be 600+ MB) we use _stage_mzml_via_symlink instead."""
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    dst = Path(tmp_path) / f"{name}{source.suffix}"
    shutil.copy(source, dst)
    return dst


def _full_d_pipeline(
    tmp_path: Path,
    source_d: Path,
    *,
    label: str,
    name: str,
    tamper_fn,
):
    """Shared body for the four .d cell tests.

    Each cell:
      1. Stages the real .d into a tempdir.
      2. Signs it.
      3. Verifies it (must succeed).
      4. Applies the cell-specific tamper to analysis.tdf (the COPY,
         never the source).
      5. Verifies again (must fail with d_content_hash mismatch).

    The signature must remain valid (the payload bytes are never
    touched), so the tamper isolates the hash-mismatch path from the
    signature path.
    """
    _skip_if_no_d(source_d, label)
    d = _stage_d_for_test(tmp_path, source_d, name=name)
    config = tmp_path / f"{name}.toml"
    config.write_bytes(
        f'[experiment]\nexperiment_name = "{name}"\n'.encode("utf-8")
    )
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)

    t0 = time.perf_counter()
    sidecar = sign_simulation_output(
        d_path=d,
        ground_truth_path=None,
        config_path=config,
        experiment_name=name,
        simulator_version="real-data-test",
        private_key_path=key_dir / "signing_key.pem",
    )
    sign_elapsed = time.perf_counter() - t0
    assert sidecar.exists()

    t0 = time.perf_counter()
    result = verify_sidecar(sidecar)
    verify_elapsed = time.perf_counter() - t0
    assert result.overall_ok, [(c.name, c.status) for c in result.checks]

    print(
        f"\n  {label}: sign={sign_elapsed:.1f}s verify={verify_elapsed:.1f}s "
        f"d_hash={[c for c in result.checks if c.name == 'd_content_hash'][0].expected[:32]}..."
    )

    # Cell-specific tamper.
    tamper_fn(d)

    result = verify_sidecar(sidecar)
    assert not result.overall_ok, f"{label}: SQL tamper went undetected"
    d_check = next(c for c in result.checks if c.name == "d_content_hash")
    assert d_check.status == "mismatch"
    # Signature must still pass — the payload bytes were never touched.
    assert result.signature_ok


# ---------------------------------------------------------------------------
# Cell-specific tamper functions
# ---------------------------------------------------------------------------


def _tamper_global_metadata(d_path: Path) -> None:
    """Generic tamper that works on every Bruker .d schema we know of:
    overwrite GlobalMetadata.InstrumentName."""
    conn = sqlite3.connect(d_path / "analysis.tdf")
    try:
        conn.execute(
            "UPDATE GlobalMetadata SET Value='REAL-DATA-TAMPER' "
            "WHERE Key='InstrumentName'"
        )
        conn.commit()
    finally:
        conn.close()


def _tamper_dia_window(d_path: Path) -> None:
    """DIA-specific tamper: mutate a REAL column in DiaFrameMsMsWindows.

    Real Bruker .d files declare ``DiaFrameMsMsWindows`` as a
    ``WITHOUT ROWID`` table (no implicit ``rowid`` column), while
    TimSim writes it as a regular table. To work on both, we
    update every row by adding a tiny offset to the chosen REAL
    column. The exact magnitude of the change does not matter for
    the test — we only need the canonical hash to differ.

    This exercises the DIA branch of the canonicalizer (DiaFrame*
    tables) in a way the GlobalMetadata tamper cannot.
    """
    conn = sqlite3.connect(d_path / "analysis.tdf")
    try:
        cols = [
            (r[1], r[2])
            for r in conn.execute('PRAGMA table_info("DiaFrameMsMsWindows")').fetchall()
        ]
        target_col = next(
            (name for name, typ in cols if (typ or "").upper().startswith("REAL")),
            None,
        )
        assert target_col is not None, (
            f"DiaFrameMsMsWindows has no REAL column to tamper; columns are {cols}"
        )
        # Add a tiny offset to every row's target column. Works on
        # both regular tables and WITHOUT ROWID tables, no PK lookup
        # needed.
        conn.execute(
            f'UPDATE "DiaFrameMsMsWindows" SET "{target_col}" = "{target_col}" + 0.001'
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# .d 2x2 matrix tests
# ---------------------------------------------------------------------------


def test_real_dda_d_full_pipeline(tmp_path):
    """REAL DDA: real Bruker timsTOF DDA blank acquisition.

    21 tables, 4M+ rows in FrameProperties, 25k Frames, 218k
    PasefFrameMsMsInfo rows, 26k Precursors. Real-Bruker schema.
    Tamper: GlobalMetadata.InstrumentName via SQL UPDATE.
    """
    _full_d_pipeline(
        tmp_path,
        REAL_DDA_D,
        label="REAL DDA",
        name="real_dda",
        tamper_fn=_tamper_global_metadata,
    )


def test_real_dia_d_full_pipeline(tmp_path):
    """REAL DIA: real Bruker timsTOF synchroPASEF acquisition.

    18 tables, 1.5M rows in FrameProperties, 11k Frames, 9k
    DiaFrameMsMsInfo, 4 DiaFrameMsMsWindowGroups, 3708
    DiaFrameMsMsWindows. Real-Bruker schema.
    Tamper: a REAL value in DiaFrameMsMsWindows (DIA-specific).
    """
    _full_d_pipeline(
        tmp_path,
        REAL_DIA_D,
        label="REAL DIA",
        name="real_dia",
        tamper_fn=_tamper_dia_window,
    )


def test_sim_dda_d_full_pipeline(tmp_path):
    """SIM DDA: TimSim DDA output produced from a real DDA reference layout.

    8 tables (TimSim's smaller schema), 16k Frames, 70 PasefFrameMsMsInfo
    rows, 31 Precursors. TimSim-shape schema (no FrameProperties,
    PropertyDefinitions, etc.).
    Tamper: GlobalMetadata.InstrumentName.
    """
    _full_d_pipeline(
        tmp_path,
        SIM_DDA_D,
        label="SIM  DDA",
        name="sim_dda",
        tamper_fn=_tamper_global_metadata,
    )


def test_sim_dia_d_full_pipeline(tmp_path):
    """SIM DIA: TimSim DIA output (HeLa 10K).

    10 tables, 34k Frames, 32k DiaFrameMsMsInfo, 36
    DiaFrameMsMsWindows. TimSim writes ``FrameMsmsInfo`` (lowercase
    'm') where real Bruker writes ``FrameMsMsInfo`` (uppercase 'M');
    both are exercised in this matrix without any special-casing in
    the canonicalizer.
    Tamper: a REAL value in DiaFrameMsMsWindows.
    """
    _full_d_pipeline(
        tmp_path,
        SIM_DIA_D,
        label="SIM  DIA",
        name="sim_dia",
        tamper_fn=_tamper_dia_window,
    )


# ---------------------------------------------------------------------------
# .d determinism (sanity check on the most heterogeneous cell)
# ---------------------------------------------------------------------------


def test_canonicalize_real_dda_d_is_deterministic(tmp_path):
    """Two consecutive canonicalize_d calls on the real DDA .d return
    the same bytes. We pick the real DDA cell because it has the
    widest schema (21 tables, 4M+ rows) and is therefore the strongest
    determinism check we can run."""
    _skip_if_no_d(REAL_DDA_D, "REAL DDA")
    d = _stage_d_for_test(tmp_path, REAL_DDA_D, name="determinism")
    h1 = canonicalize_d(d)
    h2 = canonicalize_d(d)
    assert h1 == h2


def test_canonicalize_real_dia_d_is_deterministic(tmp_path):
    """Same determinism check on the real DIA cell, since DIA tables
    are a different code path."""
    _skip_if_no_d(REAL_DIA_D, "REAL DIA")
    d = _stage_d_for_test(tmp_path, REAL_DIA_D, name="determinism_dia")
    h1 = canonicalize_d(d)
    h2 = canonicalize_d(d)
    assert h1 == h2


# ---------------------------------------------------------------------------
# mzML matrix
# ---------------------------------------------------------------------------


def _full_mzml_pipeline(
    tmp_path: Path,
    source: Path,
    *,
    label: str,
    name: str,
    tool_name: str,
    tool_version: str,
    tamper_fn,
):
    _skip_if_no_mzml(source, label)
    mzml = _stage_mzml_for_test(tmp_path, source, name=name)
    config = tmp_path / f"{name}.toml"
    config.write_bytes(
        f'[experiment]\nexperiment_name = "{name}"\n'.encode("utf-8")
    )
    key_dir = tmp_path / "keys"
    write_keypair(generate_keypair(), key_dir)

    t0 = time.perf_counter()
    sidecar = sign_mzml_output(
        mzml_path=mzml,
        config_path=config,
        experiment_name=name,
        tool_name=tool_name,
        tool_version=tool_version,
        private_key_path=key_dir / "signing_key.pem",
    )
    sign_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = verify_sidecar(sidecar)
    verify_elapsed = time.perf_counter() - t0
    assert result.overall_ok, [(c.name, c.status) for c in result.checks]

    print(f"\n  {label}: sign={sign_elapsed:.1f}s verify={verify_elapsed:.1f}s")

    # Tamper.
    tamper_fn(mzml)

    result = verify_sidecar(sidecar)
    assert not result.overall_ok, f"{label}: tamper went undetected"
    m_check = next(c for c in result.checks if c.name == "mzml_content_hash")
    assert m_check.status == "mismatch"


def _tamper_mzml_charge_state(p: Path) -> None:
    """Flip a precursor charge state in an mzml. Works on most files."""
    text = p.read_text()
    if '"charge state" value="2"' in text:
        text = text.replace('"charge state" value="2"', '"charge state" value="9"', 1)
    elif '"charge state" value="1"' in text:
        text = text.replace('"charge state" value="1"', '"charge state" value="9"', 1)
    elif '"charge state" value="3"' in text:
        text = text.replace('"charge state" value="3"', '"charge state" value="9"', 1)
    else:
        # Fall back to flipping a positive scan to negative scan.
        text = text.replace(
            '"positive scan"', '"negative scan"', 1
        ).replace(
            'accession="MS:1000130"', 'accession="MS:1000129"', 1
        )
    p.write_text(text)


def test_real_thermo_msconvert_mzml_full_pipeline(tmp_path):
    """REAL Thermo + ProteoWizard mzML.

    Sage test fixture: a Q Exactive scan converted by ProteoWizard.
    Has referenceableParamGroupList, softwareList, sourceFile with
    the original RAW SHA-1 — none of which the synthetic fixture
    exercises. 64-bit float precision.
    """
    _full_mzml_pipeline(
        tmp_path,
        THERMO_MSCONVERT_MZML,
        label="REAL Thermo+pwiz mzML",
        name="thermo",
        tool_name="ProteoWizard",
        tool_version="3.0.22204",
        tamper_fn=_tamper_mzml_charge_state,
    )


def test_real_32bit_mzml_full_pipeline(tmp_path):
    """REAL 32-bit precision mzML.

    Exercises the f32 binary array code path on real converter
    output. The other mzML cells in this matrix are 64-bit float.
    """
    _full_mzml_pipeline(
        tmp_path,
        F32_MZML,
        label="REAL f32 mzML",
        name="f32",
        tool_name="unknown",
        tool_version="unknown",
        tamper_fn=_tamper_mzml_charge_state,
    )


def test_real_bruker_timstof_mzml_full_pipeline(tmp_path):
    """REAL Bruker timsTOF mzML.

    A 51 MB real-instrument timsTOF DDA mzML conversion. This is the
    closest thing to what the user's collaborator's mzML simulator
    will produce. Exercises real-converter cvParam usage at scale.
    """
    _full_mzml_pipeline(
        tmp_path,
        BRUKER_TIMSTOF_MZML,
        label="REAL Bruker timsTOF mzML",
        name="bruker_timstof",
        tool_name="Bruker DataAnalysis",
        tool_version="unknown",
        tamper_fn=_tamper_mzml_charge_state,
    )


def test_canonicalize_real_thermo_msconvert_mzml_whitespace_invariance(tmp_path):
    """Reformatting the real msconvert mzML with extra whitespace does
    not change the canonical hash. Operational version of the
    whitespace-invariance claim on real converter output.
    """
    _skip_if_no_mzml(THERMO_MSCONVERT_MZML, "Thermo msconvert mzml")
    mzml1 = _stage_mzml_for_test(tmp_path / "a", THERMO_MSCONVERT_MZML, name="m1")
    mzml2 = _stage_mzml_for_test(tmp_path / "b", THERMO_MSCONVERT_MZML, name="m2")
    h_before = canonicalize_mzml(mzml2)
    text = mzml2.read_text()
    # `/>` is unambiguously a self-closing-tag terminator in well-formed
    # XML — it cannot appear inside text or attribute content — so
    # appending whitespace after it is purely a whitespace mutation.
    text = text.replace("/>", "/>\n  ")
    mzml2.write_text(text)
    assert mzml1.read_text() != mzml2.read_text()
    h_after = canonicalize_mzml(mzml2)
    assert h_after == h_before
    assert canonicalize_mzml(mzml1) == canonicalize_mzml(mzml2)
