"""Unit tests for timsim/jobs/load_findings.py.

Tests cover the standardized findings input format with required columns
(protein, sequence, intensity) and optional columns (charge, rt, im).
"""

import numpy as np
import pandas as pd
import pytest

from imspy_simulation.timsim.jobs.load_findings import (
    load_findings,
    FindingsResult,
    _read_findings,
    _filter_sequences,
    _deduplicate_with_charge,
    _deduplicate_peptides_only,
    _build_ions,
    _build_peptides_from_ions,
    _build_peptides_no_charge,
    _build_proteins,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_SEQUENCES = [
    "PEPTIDEC[UNIMOD:4]K",
    "AAGLTFHEK",
    "LVEQFHC[UNIMOD:4]K",
    "DLGEEHFK",
    "YICDNQDTISSK",
    "SLHTLFGDK",
    "LVNELTEFAK",
    "AEFAEVSK",
    "HLVDEPQNLIK",
    "SHCIAEVENDEMPADLPSLAADFVESK",
]


def _make_full_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """All 6 columns present."""
    rng = np.random.RandomState(seed)
    sequences = [_VALID_SEQUENCES[i % len(_VALID_SEQUENCES)] for i in range(n)]
    return pd.DataFrame({
        "protein": [f"PROT_{i % 3}" for i in range(n)],
        "sequence": sequences,
        "charge": rng.choice([2, 3, 4], size=n),
        "intensity": rng.uniform(1e3, 1e6, size=n),
        "rt": rng.uniform(600, 3000, size=n),
        "im": rng.uniform(0.7, 1.3, size=n),
    })


def _make_minimal_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Only required columns."""
    rng = np.random.RandomState(seed)
    sequences = [_VALID_SEQUENCES[i % len(_VALID_SEQUENCES)] for i in range(n)]
    return pd.DataFrame({
        "protein": [f"PROT_{i % 3}" for i in range(n)],
        "sequence": sequences,
        "intensity": rng.uniform(1e3, 1e6, size=n),
    })


# ---------------------------------------------------------------------------
# Tests for _read_findings
# ---------------------------------------------------------------------------

class TestReadFindings:
    def test_missing_required_columns_raises(self, tmp_path):
        df = pd.DataFrame({"protein": ["P1"], "sequence": ["PEPTIDEK"]})
        path = tmp_path / "bad.tsv"
        df.to_csv(path, sep="\t", index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            _read_findings(str(path))

    def test_reads_tsv_full(self, tmp_path):
        df = _make_full_df(5)
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        result = _read_findings(str(path))
        assert len(result) == 5
        assert "charge" in result.columns
        assert "rt" in result.columns
        assert "im" in result.columns

    def test_reads_csv(self, tmp_path):
        df = _make_full_df(5)
        path = tmp_path / "findings.csv"
        df.to_csv(path, index=False)

        result = _read_findings(str(path))
        assert len(result) == 5

    def test_reads_minimal_columns(self, tmp_path):
        df = _make_minimal_df(5)
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        result = _read_findings(str(path))
        assert len(result) == 5
        assert "charge" not in result.columns
        assert "rt" not in result.columns
        assert "im" not in result.columns

    def test_drops_rows_with_nan(self, tmp_path):
        df = pd.DataFrame({
            "protein": ["P1", "P2"],
            "sequence": ["PEPTIDEK", "AAAK"],
            "intensity": [1000.0, np.nan],
        })
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        result = _read_findings(str(path))
        assert len(result) == 1

    def test_column_name_normalization(self, tmp_path):
        df = pd.DataFrame({
            " Protein ": ["P1"],
            "SEQUENCE": ["PEPTIDEK"],
            "Intensity": [1000.0],
        })
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        result = _read_findings(str(path))
        assert len(result) == 1

    def test_nan_in_optional_charge_drops_row_not_crash(self, tmp_path):
        """Fix #2: NaN charge must be dropped before int cast."""
        df = pd.DataFrame({
            "protein": ["P1", "P2", "P3"],
            "sequence": ["PEPTIDEK", "AAAK", "LLLK"],
            "intensity": [1000.0, 2000.0, 3000.0],
            "charge": [2.0, np.nan, 3.0],
        })
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        result = _read_findings(str(path))
        assert len(result) == 2
        assert result["charge"].dtype == int


# ---------------------------------------------------------------------------
# Tests for _filter_sequences
# ---------------------------------------------------------------------------

class TestFilterSequences:
    def test_drops_invalid_sequences(self):
        df = pd.DataFrame({
            "protein": ["P1", "P2"],
            "sequence": ["PEPTIDEK", "NOT123VALID"],
            "intensity": [1000.0, 2000.0],
        })
        result = _filter_sequences(df, verbose=False)
        assert len(result) == 1
        assert result.iloc[0]["sequence"] == "PEPTIDEK"


# ---------------------------------------------------------------------------
# Tests for _deduplicate_with_charge
# ---------------------------------------------------------------------------

class TestDeduplicateWithCharge:
    def test_basic_deduplication(self):
        df = pd.DataFrame({
            "protein": ["PROT_A", "PROT_A"],
            "sequence": ["PEPTIDEK", "PEPTIDEK"],
            "charge": [2, 2],
            "intensity": [1000.0, 2000.0],
            "rt": [1200.0, 1320.0],
            "im": [1.0, 1.1],
        })
        result = _deduplicate_with_charge(df, has_rt=True, has_im=True, verbose=False)

        assert len(result) == 1
        assert result.iloc[0]["intensity"] == 3000.0
        expected_rt = (1200.0 * 1000 + 1320.0 * 2000) / 3000.0
        assert result.iloc[0]["rt"] == pytest.approx(expected_rt, rel=1e-4)

    def test_different_charges_kept(self):
        df = pd.DataFrame({
            "protein": ["PROT_A", "PROT_A"],
            "sequence": ["PEPTIDEK", "PEPTIDEK"],
            "charge": [2, 3],
            "intensity": [1000.0, 500.0],
            "rt": [1200.0, 1200.0],
            "im": [1.0, 0.8],
        })
        result = _deduplicate_with_charge(df, has_rt=True, has_im=True, verbose=False)
        assert len(result) == 2

    def test_without_rt_and_im(self):
        df = pd.DataFrame({
            "protein": ["PROT_A", "PROT_A"],
            "sequence": ["PEPTIDEK", "PEPTIDEK"],
            "charge": [2, 2],
            "intensity": [1000.0, 2000.0],
        })
        result = _deduplicate_with_charge(df, has_rt=False, has_im=False, verbose=False)
        assert len(result) == 1
        assert result.iloc[0]["intensity"] == 3000.0
        assert "rt" not in result.columns
        assert "im" not in result.columns


# ---------------------------------------------------------------------------
# Tests for _deduplicate_peptides_only
# ---------------------------------------------------------------------------

class TestDeduplicatePeptidesOnly:
    def test_merges_duplicates(self):
        df = pd.DataFrame({
            "protein": ["PROT_A", "PROT_A"],
            "sequence": ["PEPTIDEK", "PEPTIDEK"],
            "intensity": [1000.0, 2000.0],
            "rt": [1200.0, 1320.0],
        })
        result = _deduplicate_peptides_only(df, has_rt=True, verbose=False)

        assert len(result) == 1
        assert result.iloc[0]["intensity"] == 3000.0

    def test_keeps_different_sequences(self):
        df = pd.DataFrame({
            "protein": ["PROT_A", "PROT_B"],
            "sequence": ["PEPTIDEK", "AAAK"],
            "intensity": [1000.0, 500.0],
        })
        result = _deduplicate_peptides_only(df, has_rt=False, verbose=False)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests for _build_ions
# ---------------------------------------------------------------------------

class TestBuildIons:
    def test_filters_by_mz_and_im(self):
        deduped = pd.DataFrame({
            "sequence": ["PEPTIDEC[UNIMOD:4]K", "PEPTIDEC[UNIMOD:4]K"],
            "charge": [2, 2],
            "rt": [1200.0, 1200.0],
            "im": [0.5, 1.0],
            "intensity": [1000.0, 2000.0],
            "protein": ["PROT_A", "PROT_A"],
        })

        ions = _build_ions(deduped, mz_lower=100, mz_upper=2000,
                           im_lower=0.6, im_upper=1.5,
                           inverse_mobility_std_mean=0.009,
                           has_im=True, verbose=False)

        assert len(ions) == 1
        assert ions.iloc[0]["inv_mobility_gru_predictor"] == pytest.approx(1.0, abs=0.01)

    def test_without_im(self):
        deduped = pd.DataFrame({
            "sequence": ["PEPTIDEC[UNIMOD:4]K"],
            "charge": [2],
            "intensity": [1000.0],
            "protein": ["PROT_A"],
        })

        ions = _build_ions(deduped, mz_lower=100, mz_upper=2000,
                           im_lower=0.6, im_upper=1.5,
                           inverse_mobility_std_mean=0.009,
                           has_im=False, verbose=False)

        assert len(ions) == 1
        assert "inv_mobility_gru_predictor" not in ions.columns

    def test_relative_abundance(self):
        deduped = pd.DataFrame({
            "sequence": ["PEPK", "PEPK"],
            "charge": [2, 3],
            "intensity": [750.0, 250.0],
            "protein": ["PROT_A", "PROT_A"],
        })

        ions = _build_ions(deduped, mz_lower=100, mz_upper=2000,
                           im_lower=0.6, im_upper=1.5,
                           inverse_mobility_std_mean=0.009,
                           has_im=False, verbose=False)

        assert len(ions) == 2
        assert ions["relative_abundance"].sum() == pytest.approx(1.0)
        assert ions.iloc[0]["relative_abundance"] == pytest.approx(0.75)
        assert ions.iloc[1]["relative_abundance"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Tests for _build_peptides_from_ions
# ---------------------------------------------------------------------------

class TestBuildPeptidesFromIons:
    def test_groups_by_sequence(self):
        deduped = pd.DataFrame({
            "sequence": ["PEPK", "PEPK", "AAAK"],
            "charge": [2, 3, 2],
            "intensity": [1000.0, 500.0, 2000.0],
            "protein": ["PROT_A", "PROT_A", "PROT_B"],
            "rt": [1200.0, 1200.0, 1800.0],
        })
        ions = pd.DataFrame({
            "sequence": ["PEPK", "PEPK", "AAAK"],
            "charge": [2, 3, 2],
            "mz": [300.0, 200.0, 250.0],
            "relative_abundance": [0.667, 0.333, 1.0],
        })

        peptides = _build_peptides_from_ions(ions, deduped,
                                              upscale_factor=100_000,
                                              intensity_multiplier=1.0,
                                              has_rt=True, verbose=False)

        assert len(peptides) == 2
        assert "peptide_id" in ions.columns
        assert all(peptides["events"] > 0)
        assert "retention_time_gru_predictor" in peptides.columns

    def test_no_orphan_peptides(self):
        """Fix #1 regression: peptides must only include sequences with surviving ions."""
        deduped = pd.DataFrame({
            "sequence": ["PEPK", "AAAK"],
            "charge": [2, 2],
            "intensity": [1000.0, 2000.0],
            "protein": ["PROT_A", "PROT_B"],
        })
        # Simulate that AAAK was filtered out during ion building (e.g., mz out of range)
        ions = pd.DataFrame({
            "sequence": ["PEPK"],
            "charge": [2],
            "mz": [300.0],
            "relative_abundance": [1.0],
        })

        peptides = _build_peptides_from_ions(ions, deduped,
                                              upscale_factor=100_000,
                                              intensity_multiplier=1.0,
                                              has_rt=False, verbose=False)

        assert len(peptides) == 1
        assert peptides.iloc[0]["sequence"] == "PEPK"


# ---------------------------------------------------------------------------
# Tests for _build_peptides_no_charge
# ---------------------------------------------------------------------------

class TestBuildPeptidesNoCharge:
    def test_each_row_is_peptide(self):
        deduped = pd.DataFrame({
            "sequence": ["PEPK", "AAAK"],
            "intensity": [1000.0, 2000.0],
            "protein": ["PROT_A", "PROT_B"],
        })

        peptides = _build_peptides_no_charge(deduped, upscale_factor=100_000,
                                              intensity_multiplier=1.0,
                                              has_rt=False, verbose=False)

        assert len(peptides) == 2
        # RT column always present (defaults to 0.0) for Rust column order compatibility
        assert (peptides["retention_time_gru_predictor"] == 0.0).all()

    def test_with_rt(self):
        deduped = pd.DataFrame({
            "sequence": ["PEPK"],
            "intensity": [1000.0],
            "protein": ["PROT_A"],
            "rt": [1200.0],
        })

        peptides = _build_peptides_no_charge(deduped, upscale_factor=100_000,
                                              intensity_multiplier=1.0,
                                              has_rt=True, verbose=False)

        assert len(peptides) == 1
        assert "retention_time_gru_predictor" in peptides.columns


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


# ---------------------------------------------------------------------------
# Integration tests for load_findings flags
# ---------------------------------------------------------------------------

class TestLoadFindingsFlags:
    def test_full_input_flags(self, tmp_path):
        df = _make_full_df(5)
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        result = load_findings(
            str(path), rt_lower=0, rt_upper=5000,
            mz_lower=100, mz_upper=2000,
            im_lower=0.5, im_upper=1.5,
            upscale_factor=100_000,
            inverse_mobility_std_mean=0.009, intensity_multiplier=1.0, verbose=False,
        )

        assert isinstance(result, FindingsResult)
        assert result.has_rt is True
        assert result.has_charge is True
        assert result.has_im is True
        assert result.ions is not None
        assert len(result.peptides) > 0
        assert len(result.proteins) > 0

    def test_minimal_input_flags(self, tmp_path):
        df = _make_minimal_df(5)
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        result = load_findings(
            str(path), rt_lower=0, rt_upper=5000,
            mz_lower=100, mz_upper=2000,
            im_lower=0.5, im_upper=1.5,
            upscale_factor=100_000,
            inverse_mobility_std_mean=0.009, intensity_multiplier=1.0, verbose=False,
        )

        assert result.has_rt is False
        assert result.has_charge is False
        assert result.has_im is False
        assert result.ions is None
        assert len(result.peptides) > 0

    def test_charge_without_im(self, tmp_path):
        df = _make_minimal_df(5)
        df["charge"] = [2, 3, 2, 3, 4]
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        result = load_findings(
            str(path), rt_lower=0, rt_upper=5000,
            mz_lower=100, mz_upper=2000,
            im_lower=0.5, im_upper=1.5,
            upscale_factor=100_000,
            inverse_mobility_std_mean=0.009, intensity_multiplier=1.0, verbose=False,
        )

        assert result.has_charge is True
        assert result.has_im is False
        assert result.ions is not None
        assert "inv_mobility_gru_predictor" not in result.ions.columns

    def test_im_without_charge_ignored(self, tmp_path):
        """IM without charge is meaningless — should be ignored."""
        df = _make_minimal_df(5)
        df["im"] = [1.0, 1.1, 0.9, 1.2, 0.8]
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        result = load_findings(
            str(path), rt_lower=0, rt_upper=5000,
            mz_lower=100, mz_upper=2000,
            im_lower=0.5, im_upper=1.5,
            upscale_factor=100_000,
            inverse_mobility_std_mean=0.009, intensity_multiplier=1.0, verbose=False,
        )

        assert result.has_charge is False
        assert result.has_im is False
        assert result.ions is None


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

class TestRegressions:
    def test_no_orphan_peptides_after_mz_filtering(self, tmp_path):
        """Fix #1: peptides with no surviving ions must be pruned."""
        # One peptide whose ions fall inside mz range, one whose fall outside
        df = pd.DataFrame({
            "protein": ["P1", "P2"],
            "sequence": ["PEPTIDEC[UNIMOD:4]K", "AAGLTFHEK"],
            "charge": [2, 2],
            "intensity": [1000.0, 2000.0],
            "rt": [1200.0, 1800.0],
            "im": [1.0, 1.0],
        })
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        result = load_findings(
            str(path), rt_lower=0, rt_upper=5000,
            # Tight mz range that only includes one peptide
            mz_lower=490, mz_upper=550,
            im_lower=0.5, im_upper=1.5,
            upscale_factor=100_000,
            inverse_mobility_std_mean=0.009, intensity_multiplier=1.0, verbose=False,
        )

        # Every peptide should have at least one ion
        peptide_seqs = set(result.peptides["sequence"])
        ion_seqs = set(result.ions["sequence"])
        assert peptide_seqs == ion_seqs

    def test_nan_charge_does_not_crash(self, tmp_path):
        """Fix #2: NaN in optional charge must be dropped before int cast."""
        df = pd.DataFrame({
            "protein": ["P1", "P2", "P3"],
            "sequence": ["PEPTIDEC[UNIMOD:4]K", "AAGLTFHEK", "DLGEEHFK"],
            "intensity": [1000.0, 2000.0, 3000.0],
            "charge": [2.0, np.nan, 3.0],
            "rt": [600.0, 1200.0, 1800.0],
            "im": [1.0, 1.0, 1.0],
        })
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        # Should not raise — the NaN row is dropped
        result = load_findings(
            str(path), rt_lower=0, rt_upper=5000,
            mz_lower=100, mz_upper=2000,
            im_lower=0.5, im_upper=1.5,
            upscale_factor=100_000,
            inverse_mobility_std_mean=0.009, intensity_multiplier=1.0, verbose=False,
        )
        assert result.has_charge is True
        assert len(result.ions) == 2

    def test_all_invalid_sequences_raises(self, tmp_path):
        """Fix #3: fully invalid input raises a clear error."""
        df = pd.DataFrame({
            "protein": ["P1", "P2"],
            "sequence": ["123INVALID", "ALSO!!!BAD"],
            "intensity": [1000.0, 2000.0],
        })
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        with pytest.raises(ValueError, match="No valid rows remain"):
            load_findings(
                str(path), rt_lower=0, rt_upper=5000,
                mz_lower=100, mz_upper=2000,
                im_lower=0.5, im_upper=1.5,
                upscale_factor=100_000,
                inverse_mobility_std_mean=0.009, intensity_multiplier=1.0, verbose=False,
            )

    def test_all_ions_filtered_raises(self, tmp_path):
        """Fix #3: all ions outside mz range raises a clear error."""
        df = pd.DataFrame({
            "protein": ["P1"],
            "sequence": ["PEPTIDEC[UNIMOD:4]K"],
            "charge": [2],
            "intensity": [1000.0],
        })
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        with pytest.raises(ValueError, match="No ions survive"):
            load_findings(
                str(path), rt_lower=0, rt_upper=5000,
                # mz range that excludes everything
                mz_lower=10, mz_upper=20,
                im_lower=0.5, im_upper=1.5,
                upscale_factor=100_000,
                inverse_mobility_std_mean=0.009, intensity_multiplier=1.0, verbose=False,
            )

    def test_intensity_multiplier_changes_events(self, tmp_path):
        """Intensity multiplier must actually scale event counts."""
        df = pd.DataFrame({
            "protein": ["P1", "P2"],
            "sequence": ["PEPTIDEC[UNIMOD:4]K", "AAGLTFHEK"],
            "charge": [2, 2],
            "intensity": [1000.0, 2000.0],
        })
        path = tmp_path / "findings.tsv"
        df.to_csv(path, sep="\t", index=False)

        kwargs = dict(
            rt_lower=0, rt_upper=5000,
            mz_lower=100, mz_upper=2000,
            im_lower=0.5, im_upper=1.5,
            upscale_factor=100_000,
            inverse_mobility_std_mean=0.009, verbose=False,
        )

        r1 = load_findings(str(path), intensity_multiplier=1.0, **kwargs)
        r10 = load_findings(str(path), intensity_multiplier=10.0, **kwargs)

        events_1x = r1.peptides["events"].sum()
        events_10x = r10.peptides["events"].sum()

        assert events_10x > events_1x
        assert events_10x == pytest.approx(events_1x * 10, rel=0.01)


# ---------------------------------------------------------------------------
# reference_median: shared event-scaling denominator preserves cross-sample ratios
# ---------------------------------------------------------------------------

class TestReferenceMedian:
    """events = intensity / median * upscale. With the per-sample median, two
    conditions whose intensity distributions differ rescale cross-sample ratios
    by median_j/median_i. A shared reference_median removes that."""

    SHARED = "SHAREDPEPTIDEK"          # present in both samples
    _FILLERS = ["AAAAAAAK", "DDDDDDDK", "EEEEEEEK"]

    def _events(self, tmp_path, name, shared_int, filler_int, reference_median):
        # no charge column -> simplest (no_charge) path; wide ranges so nothing is filtered
        df = pd.DataFrame({
            "protein": ["P0"] + [f"P{i+1}" for i in range(len(self._FILLERS))],
            "sequence": [self.SHARED] + self._FILLERS,
            "intensity": [float(shared_int)] + [float(filler_int)] * len(self._FILLERS),
        })
        path = tmp_path / f"{name}.tsv"
        df.to_csv(path, sep="\t", index=False)
        res = load_findings(str(path), rt_lower=0.0, rt_upper=1e9, mz_lower=0.0,
                            mz_upper=1e9, im_lower=0.0, im_upper=2.0,
                            upscale_factor=100_000, inverse_mobility_std_mean=0.009,
                            intensity_multiplier=1.0, verbose=False,
                            reference_median=reference_median)
        pep = res.peptides.set_index("sequence")
        return float(pep.loc[self.SHARED, "events"])

    def test_shared_reference_preserves_ratio(self, tmp_path):
        # A: shared=100, fillers=50 (median 50); B: shared=300, fillers=200 (median 200)
        # seed ratio B/A for the shared peptide = 3.0
        ea = self._events(tmp_path, "A", 100, 50, reference_median=100.0)
        eb = self._events(tmp_path, "B", 300, 200, reference_median=100.0)
        assert eb / ea == pytest.approx(3.0, rel=0.01)

    def test_per_sample_median_distorts_ratio(self, tmp_path):
        # legacy behaviour (reference_median=None): ratio scaled by median_A/median_B = 50/200
        ea = self._events(tmp_path, "A", 100, 50, reference_median=None)
        eb = self._events(tmp_path, "B", 300, 200, reference_median=None)
        assert eb / ea == pytest.approx(3.0 * 50.0 / 200.0, rel=0.01)  # = 0.75, NOT 3.0

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_reference_raises(self, tmp_path, bad):
        with pytest.raises(ValueError, match="reference_median"):
            self._events(tmp_path, "x", 100, 50, reference_median=bad)
