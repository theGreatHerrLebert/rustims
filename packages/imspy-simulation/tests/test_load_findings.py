"""Unit tests for timsim/jobs/load_findings.py.

These tests use mock DataFrames that mimic parser output so they can run
without real search-engine result files or a timsTOF reference dataset.
"""

import numpy as np
import pandas as pd
import pytest

from imspy_simulation.timsim.jobs.load_findings import (
    load_findings,
    _parse_findings,
    _deduplicate_findings,
    _build_ions,
    _build_peptides,
    _set_relative_abundance,
    _build_proteins,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parsed_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a mock DataFrame matching parser output format."""
    rng = np.random.RandomState(seed)
    sequences = [f"PEPTIDE{i}C[UNIMOD:4]K" for i in range(n)]
    return pd.DataFrame({
        "sequence": [s.replace("[UNIMOD:4]", "") for s in sequences],
        "sequence_modified": sequences,
        "charge": rng.choice([2, 3, 4], size=n),
        "rt": rng.uniform(10, 50, size=n),  # minutes
        "inverse_mobility": rng.uniform(0.7, 1.3, size=n).astype(np.float32),
        "intensity": rng.uniform(1e3, 1e6, size=n),
        "q_value": rng.uniform(0.0, 0.005, size=n),
        "protein_id": [f"PROT_{i % 3}" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Tests for _parse_findings
# ---------------------------------------------------------------------------

class TestParseFindings:
    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown findings format"):
            _parse_findings("/nonexistent", "unknown_format", 0.01)


# ---------------------------------------------------------------------------
# Tests for _deduplicate_findings
# ---------------------------------------------------------------------------

class TestDeduplicateFindings:
    def test_basic_deduplication(self):
        # Two entries for the same precursor (same seq+charge)
        df = pd.DataFrame({
            "sequence": ["PEPTIDEK", "PEPTIDEK"],
            "sequence_modified": ["PEPTIDEK", "PEPTIDEK"],
            "charge": [2, 2],
            "rt": [20.0, 22.0],
            "rt_seconds": [1200.0, 1320.0],
            "inverse_mobility": [1.0, 1.1],
            "intensity": [1000.0, 2000.0],
            "q_value": [0.001, 0.005],
            "protein_id": ["PROT_A", "PROT_A"],
        })
        _, ions_df = _deduplicate_findings(df, verbose=False)

        assert len(ions_df) == 1
        # Best q_value entry (0.001) should provide RT/IM
        assert ions_df.iloc[0]["rt_seconds"] == 1200.0
        assert ions_df.iloc[0]["inverse_mobility"] == 1.0
        # Intensity should be summed
        assert ions_df.iloc[0]["total_intensity"] == 3000.0

    def test_different_charges_kept(self):
        df = pd.DataFrame({
            "sequence": ["PEPTIDEK", "PEPTIDEK"],
            "sequence_modified": ["PEPTIDEK", "PEPTIDEK"],
            "charge": [2, 3],
            "rt": [20.0, 20.0],
            "rt_seconds": [1200.0, 1200.0],
            "inverse_mobility": [1.0, 0.8],
            "intensity": [1000.0, 500.0],
            "q_value": [0.001, 0.002],
            "protein_id": ["PROT_A", "PROT_A"],
        })
        _, ions_df = _deduplicate_findings(df, verbose=False)
        assert len(ions_df) == 2


# ---------------------------------------------------------------------------
# Tests for _build_ions
# ---------------------------------------------------------------------------

class TestBuildIons:
    def test_filters_by_mz_and_im(self):
        ions_df = pd.DataFrame({
            "sequence_modified": ["PEPTIDEC[UNIMOD:4]K", "PEPTIDEC[UNIMOD:4]K"],
            "sequence": ["PEPTIDECK", "PEPTIDECK"],
            "charge": [2, 2],
            "rt_seconds": [1200.0, 1200.0],
            "inverse_mobility": [0.5, 1.0],  # 0.5 should be filtered if im_lower=0.6
            "total_intensity": [1000.0, 2000.0],
            "protein_id": ["PROT_A", "PROT_A"],
        })

        ions = _build_ions(ions_df, mz_lower=100, mz_upper=2000,
                           im_lower=0.6, im_upper=1.5,
                           inverse_mobility_std_mean=0.009, verbose=False)

        assert len(ions) == 1
        assert ions.iloc[0]["inv_mobility_gru_predictor"] == pytest.approx(1.0, abs=0.01)

    def test_drops_nan_inverse_mobility(self):
        ions_df = pd.DataFrame({
            "sequence_modified": ["PEPTIDEC[UNIMOD:4]K", "AAAC[UNIMOD:4]K"],
            "sequence": ["PEPTIDECK", "AAACK"],
            "charge": [2, 2],
            "rt_seconds": [1200.0, 1200.0],
            "inverse_mobility": [1.0, np.nan],
            "total_intensity": [1000.0, 2000.0],
            "protein_id": ["PROT_A", "PROT_A"],
        })

        ions = _build_ions(ions_df, mz_lower=100, mz_upper=2000,
                           im_lower=0.6, im_upper=1.5,
                           inverse_mobility_std_mean=0.009, verbose=False)

        assert len(ions) == 1


# ---------------------------------------------------------------------------
# Tests for _build_peptides
# ---------------------------------------------------------------------------

class TestBuildPeptides:
    def test_groups_by_sequence(self):
        ions = pd.DataFrame({
            "sequence_modified": ["PEPK", "PEPK", "AAAK"],
            "sequence": ["PEPK", "PEPK", "AAAK"],
            "charge": [2, 3, 2],
            "mz": [300.0, 200.0, 250.0],
            "monoisotopic-mass": [600.0, 600.0, 500.0],
            "observed_intensity": [1000.0, 500.0, 2000.0],
            "inv_mobility_gru_predictor": [1.0, 0.8, 0.9],
            "inv_mobility_gru_predictor_std": [0.009, 0.009, 0.009],
            "retention_time_gru_predictor": [1200.0, 1200.0, 1800.0],
            "protein_id": ["PROT_A", "PROT_A", "PROT_B"],
        })

        peptides = _build_peptides(ions, upscale_factor=100_000, verbose=False)

        assert len(peptides) == 2
        assert "peptide_id" in ions.columns  # should be added in-place
        assert set(peptides["sequence"]) == {"PEPK", "AAAK"}
        assert all(peptides["events"] > 0)

    def test_peptide_id_mapped_to_ions(self):
        ions = pd.DataFrame({
            "sequence_modified": ["PEPK", "PEPK"],
            "sequence": ["PEPK", "PEPK"],
            "charge": [2, 3],
            "mz": [300.0, 200.0],
            "monoisotopic-mass": [600.0, 600.0],
            "observed_intensity": [1000.0, 500.0],
            "inv_mobility_gru_predictor": [1.0, 0.8],
            "inv_mobility_gru_predictor_std": [0.009, 0.009],
            "retention_time_gru_predictor": [1200.0, 1200.0],
            "protein_id": ["PROT_A", "PROT_A"],
        })

        peptides = _build_peptides(ions, upscale_factor=100_000, verbose=False)

        # Both ions should map to the same peptide_id
        assert ions["peptide_id"].nunique() == 1
        assert ions["peptide_id"].iloc[0] == peptides["peptide_id"].iloc[0]


# ---------------------------------------------------------------------------
# Tests for _set_relative_abundance
# ---------------------------------------------------------------------------

class TestSetRelativeAbundance:
    def test_sums_to_one_per_peptide(self):
        ions = pd.DataFrame({
            "sequence_modified": ["PEPK", "PEPK"],
            "sequence": ["PEPK", "PEPK"],
            "charge": [2, 3],
            "mz": [300.0, 200.0],
            "observed_intensity": [750.0, 250.0],
            "inv_mobility_gru_predictor": [1.0, 0.8],
            "inv_mobility_gru_predictor_std": [0.009, 0.009],
            "retention_time_gru_predictor": [1200.0, 1200.0],
            "protein_id": ["PROT_A", "PROT_A"],
            "peptide_id": [0, 0],
            "monoisotopic-mass": [600.0, 600.0],
        })

        peptides = pd.DataFrame({"peptide_id": [0], "sequence": ["PEPK"]})

        result = _set_relative_abundance(ions, peptides)

        assert "relative_abundance" in result.columns
        assert result["relative_abundance"].sum() == pytest.approx(1.0)
        assert result.iloc[0]["relative_abundance"] == pytest.approx(0.75)
        assert result.iloc[1]["relative_abundance"] == pytest.approx(0.25)

    def test_helper_columns_dropped(self):
        ions = pd.DataFrame({
            "sequence_modified": ["PEPK"],
            "sequence": ["PEPK"],
            "charge": [2],
            "mz": [300.0],
            "observed_intensity": [1000.0],
            "inv_mobility_gru_predictor": [1.0],
            "inv_mobility_gru_predictor_std": [0.009],
            "retention_time_gru_predictor": [1200.0],
            "protein_id": ["PROT_A"],
            "peptide_id": [0],
            "monoisotopic-mass": [600.0],
        })

        peptides = pd.DataFrame({"peptide_id": [0], "sequence": ["PEPK"]})

        result = _set_relative_abundance(ions, peptides)

        assert "sequence_modified" not in result.columns
        assert "observed_intensity" not in result.columns
        assert "retention_time_gru_predictor" not in result.columns
        assert "protein_id" not in result.columns


# ---------------------------------------------------------------------------
# Tests for _build_proteins
# ---------------------------------------------------------------------------

class TestBuildProteins:
    def test_unique_proteins(self):
        peptides = pd.DataFrame({
            "protein_id": [0, 0, 1],
            "protein": ["PROT_A", "PROT_A", "PROT_B"],
            "peptide_id": [0, 1, 2],
            "sequence": ["PEPK", "AAAK", "LLLK"],
        })

        proteins = _build_proteins(peptides)

        assert len(proteins) == 2
        assert set(proteins["protein"]) == {"PROT_A", "PROT_B"}
        assert list(proteins.columns) == ["protein_id", "protein", "sequence", "events"]
