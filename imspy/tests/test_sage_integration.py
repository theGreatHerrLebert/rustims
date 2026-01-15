"""
Integration tests for Sage weighted scoring.

These tests verify that:
1. Weighted scoring with uniform (1.0) intensities produces identical results
   to unweighted scoring (validates the new code path)
2. The imspy -> sagepy pipeline works end-to-end
"""

import pytest
import numpy as np
import tempfile
import os
from numpy.testing import assert_array_almost_equal

# Sagepy imports
from sagepy.core import (
    SageSearchConfiguration,
    EnzymeBuilder,
    IndexedDatabase,
    Scorer,
    ScoreType,
    PredictedIntensityStore,
    Tolerance,
    SpectrumProcessor,
    RawSpectrum,
    Precursor,
    Representation,
)

# Imspy imports
from imspy.algorithm.intensity.sage_interface import (
    PredictionRequest as ImspyPredictionRequest,
    PredictionResult as ImspyPredictionResult,
    predict_intensities_for_sage,
    write_intensity_file,
    read_intensity_file,
    write_predictions_for_database,
    ION_KIND_B,
    ION_KIND_Y,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_fasta():
    """Sample FASTA content for testing."""
    return """>sp|P00000|TEST_HUMAN Test protein
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH
>sp|P00001|TEST2_HUMAN Another test protein
PEPTIDEKAAANOTHERPEPTIDEKSEQMENCEKWITHMODSK
"""


@pytest.fixture
def sample_spectra():
    """Sample spectra for scoring tests."""
    # Create simple test spectra
    spectra_data = [
        {
            "id": "spectrum_1",
            "precursor_mz": 500.25,
            "precursor_charge": 2,
            "precursor_intensity": 1e6,
            "mz": np.array([147.11, 246.18, 347.23, 448.28, 549.33, 650.38], dtype=np.float64),
            "intensity": np.array([1000.0, 2000.0, 5000.0, 3000.0, 4000.0, 1500.0], dtype=np.float64),
            "rt": 300.0,
        },
        {
            "id": "spectrum_2",
            "precursor_mz": 600.30,
            "precursor_charge": 3,
            "precursor_intensity": 5e5,
            "mz": np.array([175.12, 288.20, 401.28, 514.37, 627.45], dtype=np.float64),
            "intensity": np.array([800.0, 1500.0, 3500.0, 2500.0, 1200.0], dtype=np.float64),
            "rt": 450.0,
        },
    ]
    return spectra_data


@pytest.fixture
def indexed_database(sample_fasta):
    """Create an IndexedDatabase from sample FASTA."""
    config = SageSearchConfiguration(
        fasta=sample_fasta,
        static_mods={"C": "[UNIMOD:4]"},
        variable_mods={"M": ["[UNIMOD:35]"]},
        enzyme_builder=EnzymeBuilder(
            missed_cleavages=2,
            min_len=7,
            max_len=30,
            cleave_at="KR",
            restrict="P",
            c_terminal=True,
        ),
        generate_decoys=True,
        decoy_tag="DECOY_",
        bucket_size=8192,
    )
    return config.generate_indexed_database()


@pytest.fixture
def uniform_intensity_store(indexed_database):
    """Create a uniform intensity store for the database."""
    peptide_sequences = indexed_database.peptides_as_string()
    peptide_lengths = [len(seq) for seq in peptide_sequences]

    return PredictedIntensityStore.uniform(
        peptide_lengths=peptide_lengths,
        max_charge=3,
        ion_kinds=[ION_KIND_B, ION_KIND_Y],
    )


@pytest.fixture
def scorer_params():
    """Common scorer parameters."""
    return {
        "precursor_tolerance": Tolerance(da=(-10.0, 10.0)),
        "fragment_tolerance": Tolerance(da=(-0.5, 0.5)),
        "min_matched_peaks": 4,
        "report_psms": 10,
    }


def create_processed_spectrum(spectrum_data, processor=None):
    """Helper to create a ProcessedSpectrum from test data."""
    # Create precursor with isolation window (Tolerance object)
    isolation_window = Tolerance(da=(-1.5, 1.5))

    precursor = Precursor(
        mz=spectrum_data["precursor_mz"],
        intensity=spectrum_data.get("precursor_intensity"),
        charge=spectrum_data["precursor_charge"],
        isolation_window=isolation_window,
    )

    raw = RawSpectrum(
        file_id=0,
        spec_id=spectrum_data["id"],
        total_ion_current=float(np.sum(spectrum_data["intensity"])),
        precursors=[precursor],
        mz=spectrum_data["mz"],
        intensity=spectrum_data["intensity"],
        representation=Representation("centroid"),
        scan_start_time=spectrum_data["rt"],
        ion_injection_time=100.0,
        ms_level=2,
    )

    if processor is None:
        processor = SpectrumProcessor(take_top_n=150)

    return processor.process(raw)


# =============================================================================
# Test: Uniform Weighted Scoring Matches Unweighted Scoring
# =============================================================================

class TestUniformWeightedScoring:
    """
    Test that weighted scoring with uniform (1.0) intensities produces
    identical results to unweighted scoring.

    This validates that the new weighted code path is mathematically correct.
    """

    def test_uniform_store_creation(self, indexed_database):
        """Test that uniform intensity store is created correctly."""
        peptide_sequences = indexed_database.peptides_as_string()
        peptide_lengths = [len(seq) for seq in peptide_sequences]

        store = PredictedIntensityStore.uniform(
            peptide_lengths=peptide_lengths,
            max_charge=3,
            ion_kinds=[ION_KIND_B, ION_KIND_Y],
        )

        assert store.peptide_count == len(peptide_sequences)
        assert store.max_charge == 3
        # ion_kinds may be returned as bytes or list depending on binding
        ion_kinds = store.ion_kinds
        if isinstance(ion_kinds, bytes):
            ion_kinds = list(ion_kinds)
        assert ion_kinds == [ION_KIND_B, ION_KIND_Y]

    def test_uniform_store_returns_one(self, indexed_database, uniform_intensity_store):
        """Test that uniform store returns 1.0 for all queries."""
        store = uniform_intensity_store

        # Query several random peptides
        for peptide_idx in [0, 1, min(5, store.peptide_count - 1)]:
            peptide_len = len(indexed_database.peptides_as_string()[peptide_idx])

            for ion_kind in [ION_KIND_B, ION_KIND_Y]:
                for position in range(min(3, peptide_len - 1)):
                    for charge in range(1, 4):
                        intensity = store.get_intensity_or_default(
                            peptide_idx, peptide_len, ion_kind, position, charge
                        )
                        assert intensity == 1.0, \
                            f"Expected 1.0, got {intensity} for peptide {peptide_idx}, pos {position}, charge {charge}"

    def test_hyperscore_vs_weighted_hyperscore(
        self, indexed_database, uniform_intensity_store, scorer_params, sample_spectra
    ):
        """
        Test that hyperscore and weightedhyperscore produce identical results
        when using uniform (1.0) intensities.
        """
        # Create scorers
        scorer_unweighted = Scorer(
            score_type=ScoreType("hyperscore"),
            **scorer_params,
        )

        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        # Process and score each spectrum
        processor = SpectrumProcessor(
            take_top_n=150,
        )

        for spec_data in sample_spectra:
            processed = create_processed_spectrum(spec_data, processor)

            # Score without intensity store (old path)
            features_unweighted = scorer_unweighted.score(indexed_database, processed)

            # Score with uniform intensity store (new path)
            features_weighted = scorer_weighted.score(
                indexed_database, processed, intensity_store=uniform_intensity_store
            )

            # Compare results
            assert len(features_unweighted) == len(features_weighted), \
                f"Different number of PSMs: {len(features_unweighted)} vs {len(features_weighted)}"

            for f_old, f_new in zip(features_unweighted, features_weighted):
                # Hyperscores should be identical
                assert abs(f_old.hyperscore - f_new.hyperscore) < 1e-5, \
                    f"Hyperscore mismatch: {f_old.hyperscore} vs {f_new.hyperscore}"

                # Same peptide should be matched
                assert f_old.peptide_idx == f_new.peptide_idx, \
                    f"Different peptide: {f_old.peptide_idx} vs {f_new.peptide_idx}"

                # Same number of matched peaks
                assert f_old.matched_peaks == f_new.matched_peaks, \
                    f"Different peak count: {f_old.matched_peaks} vs {f_new.matched_peaks}"

    def test_openmshyperscore_vs_weighted(
        self, indexed_database, uniform_intensity_store, scorer_params, sample_spectra
    ):
        """
        Test that openmshyperscore and weightedopenmshyperscore produce
        identical results when using uniform intensities.
        """
        scorer_unweighted = Scorer(
            score_type=ScoreType("openmshyperscore"),
            **scorer_params,
        )

        scorer_weighted = Scorer(
            score_type=ScoreType("weightedopenmshyperscore"),
            **scorer_params,
        )

        processor = SpectrumProcessor(take_top_n=150)

        for spec_data in sample_spectra:
            processed = create_processed_spectrum(spec_data, processor)

            features_unweighted = scorer_unweighted.score(indexed_database, processed)
            features_weighted = scorer_weighted.score(
                indexed_database, processed, intensity_store=uniform_intensity_store
            )

            assert len(features_unweighted) == len(features_weighted)

            for f_old, f_new in zip(features_unweighted, features_weighted):
                assert abs(f_old.hyperscore - f_new.hyperscore) < 1e-5, \
                    f"OpenMS hyperscore mismatch: {f_old.hyperscore} vs {f_new.hyperscore}"


# =============================================================================
# Test: Imspy Predictions with Sagepy
# =============================================================================

class TestImspySagepyIntegration:
    """
    Test the integration between imspy's intensity predictions and sagepy's
    weighted scoring.
    """

    def test_write_and_load_sagi_file(self, indexed_database):
        """Test writing predictions from imspy and loading in sagepy."""
        peptide_sequences = indexed_database.peptides_as_string()
        peptide_lengths = [len(seq) for seq in peptide_sequences]

        # Create mock predictions (uniform for simplicity)
        predictions = []
        for pep_len in peptide_lengths:
            n_positions = pep_len - 1
            pred = np.ones((2, n_positions, 2), dtype=np.float32)
            predictions.append(pred)

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            temp_path = f.name

        try:
            # Write using imspy
            write_intensity_file(
                temp_path,
                predictions,
                peptide_lengths,
                max_charge=2,
                ion_kinds=[ION_KIND_B, ION_KIND_Y],
            )

            # Load using sagepy
            store = PredictedIntensityStore(temp_path)

            assert store.peptide_count == len(peptide_sequences)
            assert store.max_charge == 2
            # ion_kinds may be returned as bytes or list depending on binding
            ion_kinds = store.ion_kinds
            if isinstance(ion_kinds, bytes):
                ion_kinds = list(ion_kinds)
            assert ion_kinds == [ION_KIND_B, ION_KIND_Y]

            # Verify values
            for peptide_idx in range(min(3, store.peptide_count)):
                intensity = store.get_intensity_or_default(
                    peptide_idx, peptide_lengths[peptide_idx], ION_KIND_B, 0, 1
                )
                assert intensity == 1.0

        finally:
            os.unlink(temp_path)

    def test_imspy_format_compatibility(self, indexed_database):
        """Test that imspy's output format is compatible with sagepy."""
        peptide_sequences = indexed_database.peptides_as_string()
        peptide_lengths = [len(seq) for seq in peptide_sequences]

        # Create varied predictions to test the full range
        predictions = []
        for i, pep_len in enumerate(peptide_lengths):
            n_positions = pep_len - 1
            # Create predictable but varied values
            pred = np.zeros((2, n_positions, 2), dtype=np.float32)
            for ion_idx in range(2):  # B, Y
                for pos in range(n_positions):
                    for charge in range(2):
                        # Value depends on position and charge
                        pred[ion_idx, pos, charge] = (pos + 1) / n_positions * (charge + 1) / 2
            predictions.append(pred)

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            temp_path = f.name

        try:
            # Write with imspy format
            write_intensity_file(
                temp_path,
                predictions,
                peptide_lengths,
                max_charge=2,
                ion_kinds=[ION_KIND_B, ION_KIND_Y],
            )

            # Load with sagepy
            store = PredictedIntensityStore(temp_path)

            # Verify specific values
            for peptide_idx in range(min(5, store.peptide_count)):
                pep_len = peptide_lengths[peptide_idx]
                n_positions = pep_len - 1

                for pos in range(min(3, n_positions)):
                    for charge in [1, 2]:
                        expected_b = (pos + 1) / n_positions * charge / 2
                        expected_y = (pos + 1) / n_positions * charge / 2

                        actual_b = store.get_intensity_or_default(
                            peptide_idx, pep_len, ION_KIND_B, pos, charge
                        )
                        actual_y = store.get_intensity_or_default(
                            peptide_idx, pep_len, ION_KIND_Y, pos, charge
                        )

                        assert abs(actual_b - expected_b) < 1e-5, \
                            f"B ion mismatch at peptide {peptide_idx}, pos {pos}, charge {charge}"
                        assert abs(actual_y - expected_y) < 1e-5, \
                            f"Y ion mismatch at peptide {peptide_idx}, pos {pos}, charge {charge}"

        finally:
            os.unlink(temp_path)


# =============================================================================
# Test: Full Pipeline with Prosit (slow, requires TensorFlow)
# =============================================================================

class TestFullPipelineWithProsit:
    """
    End-to-end tests using actual Prosit predictions.
    These tests are slow and require TensorFlow.
    """

    @pytest.mark.slow
    def test_prosit_to_sagepy_pipeline(self, indexed_database):
        """Test full pipeline: Prosit prediction -> .sagi file -> sagepy scoring."""
        try:
            peptide_sequences = indexed_database.peptides_as_string()
            peptide_lengths = [len(seq) for seq in peptide_sequences]

            # Only predict for a small subset (first 10 target peptides)
            # to keep the test fast
            target_peptides = [
                (i, seq) for i, seq in enumerate(peptide_sequences)
                if "DECOY_" not in seq and len(seq) <= 25
            ][:10]

            if not target_peptides:
                pytest.skip("No suitable peptides for prediction")

            indices = [t[0] for t in target_peptides]
            sequences = [t[1] for t in target_peptides]
            charges = [2] * len(sequences)  # All charge 2

            # Predict with Prosit
            result = predict_intensities_for_sage(
                sequences=sequences,
                charges=charges,
                peptide_indices=indices,
                max_fragment_charge=2,
                verbose=False,
            )

            # Verify predictions
            assert len(result.intensities) == len(sequences)
            for seq, intensity in zip(sequences, result.intensities):
                seq_len = len(seq)
                assert intensity.shape == (2, seq_len - 1, 2)
                assert np.all(intensity >= 0)
                assert np.all(intensity <= 1)

            # Write to file
            with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
                temp_path = f.name

            try:
                write_predictions_for_database(
                    temp_path,
                    result,
                    num_peptides=len(peptide_sequences),
                    peptide_lengths=peptide_lengths,
                    aggregation='max_charge',
                    default_value=1.0,  # Uniform for non-predicted peptides
                )

                # Load with sagepy
                store = PredictedIntensityStore(temp_path)

                assert store.peptide_count == len(peptide_sequences)

                # Verify a predicted peptide has non-uniform values
                for idx in indices[:3]:
                    pep_len = peptide_lengths[idx]
                    intensity = store.get_intensity_or_default(
                        idx, pep_len, ION_KIND_Y, 0, 1
                    )
                    # Should be a valid intensity (may or may not be 1.0)
                    assert 0 <= intensity <= 1.0

            finally:
                os.unlink(temp_path)

        except ImportError:
            pytest.skip("TensorFlow not available")

    @pytest.mark.slow
    def test_weighted_scoring_with_prosit(
        self, indexed_database, scorer_params, sample_spectra
    ):
        """Test weighted scoring with actual Prosit predictions."""
        try:
            peptide_sequences = indexed_database.peptides_as_string()
            peptide_lengths = [len(seq) for seq in peptide_sequences]

            # Predict for target peptides
            target_peptides = [
                (i, seq) for i, seq in enumerate(peptide_sequences)
                if "DECOY_" not in seq and len(seq) <= 25
            ][:20]

            if len(target_peptides) < 5:
                pytest.skip("Not enough peptides for prediction")

            indices = [t[0] for t in target_peptides]
            sequences = [t[1] for t in target_peptides]
            charges = [2] * len(sequences)

            # Predict
            result = predict_intensities_for_sage(
                sequences=sequences,
                charges=charges,
                peptide_indices=indices,
                max_fragment_charge=2,
                verbose=False,
            )

            with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
                temp_path = f.name

            try:
                write_predictions_for_database(
                    temp_path,
                    result,
                    num_peptides=len(peptide_sequences),
                    peptide_lengths=peptide_lengths,
                )

                store = PredictedIntensityStore(temp_path)

                # Score with weighted scoring
                scorer = Scorer(
                    score_type=ScoreType("weightedhyperscore"),
                    **scorer_params,
                )

                processor = SpectrumProcessor(take_top_n=150)

                for spec_data in sample_spectra:
                    processed = create_processed_spectrum(spec_data, processor)

                    features = scorer.score(
                        indexed_database, processed, intensity_store=store
                    )

                    # Should return some results
                    # (may be empty if no good matches for test spectra)
                    for f in features:
                        assert f.hyperscore >= 0
                        assert f.matched_peaks >= 0

            finally:
                os.unlink(temp_path)

        except ImportError:
            pytest.skip("TensorFlow not available")


# =============================================================================
# Test: Full Pipeline with Real 4P FASTA
# =============================================================================

class TestPipeline4PFASTA:
    """
    Comprehensive integration tests using the real 4P.fasta file containing:
    - ADH1_YEAST (Alcohol dehydrogenase 1)
    - ALBU_BOVIN (Bovine serum albumin)
    - ENO1_YEAST (Enolase 1)
    - PYGM_RABIT (Glycogen phosphorylase)

    These tests validate the full intensity prediction pipeline with realistic
    protein sequences.
    """

    FASTA_PATH = "/Users/davidteschner/Promotion/timsim/4P.fasta"

    @pytest.fixture
    def fasta_content(self):
        """Read the 4P FASTA file content."""
        import os
        if not os.path.exists(self.FASTA_PATH):
            pytest.skip(f"4P FASTA file not found at {self.FASTA_PATH}")
        with open(self.FASTA_PATH, 'r') as f:
            return f.read()

    @pytest.fixture
    def indexed_db_4p(self, fasta_content):
        """Create an IndexedDatabase from the 4P FASTA."""
        config = SageSearchConfiguration(
            fasta=fasta_content,
            static_mods={"C": "[UNIMOD:4]"},
            variable_mods={"M": ["[UNIMOD:35]"]},
            enzyme_builder=EnzymeBuilder(
                missed_cleavages=2,
                min_len=7,
                max_len=30,
                cleave_at="KR",
                restrict="P",
                c_terminal=True,
            ),
            generate_decoys=True,
            decoy_tag="DECOY_",
            bucket_size=8192,
        )
        return config.generate_indexed_database()

    def test_database_construction(self, indexed_db_4p):
        """Test that database is built correctly from 4P FASTA."""
        peptides = list(indexed_db_4p.peptides_as_string())

        # Should have a reasonable number of peptides
        # 4 proteins with trypsin digestion should generate many peptides
        # With decoy generation, we expect roughly 2x the target peptides
        assert len(peptides) > 100, f"Expected many peptides (with decoys), got {len(peptides)}"

        # Verify decoy identification using db._peptides[i].decoy attribute
        # This is the CORRECT way to identify decoys - not by checking the sequence string
        db_peptides = indexed_db_4p._peptides
        targets = sum(1 for p in db_peptides if not p.decoy)
        decoys = sum(1 for p in db_peptides if p.decoy)

        assert targets > 0, "Expected target peptides"
        assert decoys > 0, "Expected decoy peptides (generate_decoys=True)"
        assert targets == decoys, f"Should have equal targets ({targets}) and decoys ({decoys})"

        # Verify index alignment between peptides_as_string() and _peptides
        for i in range(min(10, len(peptides))):
            assert peptides[i] == db_peptides[i].sequence, \
                f"Index mismatch at {i}: '{peptides[i]}' vs '{db_peptides[i].sequence}'"

        # Check peptide length distribution (all peptides should be valid)
        from imspy.algorithm.intensity.sage_interface import remove_unimod_annotation
        lengths = [len(remove_unimod_annotation(p)) for p in peptides]
        assert min(lengths) >= 7, "All peptides should be >= 7 AA"
        assert max(lengths) <= 30, "All peptides should be <= 30 AA"

    def test_uniform_weighted_matches_unweighted_4p(self, indexed_db_4p, scorer_params):
        """Test uniform weighted scoring matches unweighted with real database."""
        peptides = indexed_db_4p.peptides_as_string()
        peptide_lengths = [len(p) for p in peptides]

        # Create uniform store
        store = PredictedIntensityStore.uniform(
            peptide_lengths=peptide_lengths,
            max_charge=2,
            ion_kinds=[ION_KIND_B, ION_KIND_Y],
        )

        # Create scorers
        scorer_unweighted = Scorer(
            score_type=ScoreType("hyperscore"),
            **scorer_params,
        )
        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        # Create a synthetic spectrum that should match some peptides
        # Use m/z values that span a realistic range
        spec_data = {
            "id": "test_4p_spectrum",
            "precursor_mz": 650.35,
            "precursor_charge": 2,
            "precursor_intensity": 1e6,
            "mz": np.array([
                147.11, 175.12, 248.13, 276.15, 347.19,
                405.22, 476.26, 520.28, 591.32, 662.36,
                763.41, 834.45, 905.49, 976.53, 1047.57
            ], dtype=np.float64),
            "intensity": np.array([
                1200.0, 800.0, 2500.0, 1800.0, 4000.0,
                3200.0, 5500.0, 4800.0, 3500.0, 2800.0,
                2200.0, 1600.0, 1000.0, 600.0, 400.0
            ], dtype=np.float64),
            "rt": 1200.0,
        }

        processor = SpectrumProcessor(take_top_n=150)
        processed = create_processed_spectrum(spec_data, processor)

        # Score both ways
        features_unweighted = scorer_unweighted.score(indexed_db_4p, processed)
        features_weighted = scorer_weighted.score(
            indexed_db_4p, processed, intensity_store=store
        )

        assert len(features_unweighted) == len(features_weighted)

        for f_old, f_new in zip(features_unweighted, features_weighted):
            assert abs(f_old.hyperscore - f_new.hyperscore) < 1e-5, \
                f"Hyperscore mismatch: {f_old.hyperscore} vs {f_new.hyperscore}"

    @pytest.mark.slow
    def test_full_prosit_pipeline_4p(self, indexed_db_4p):
        """
        Test the complete Prosit prediction pipeline with real 4P proteins.
        This tests a significant number of peptides.
        """
        try:
            peptides = indexed_db_4p.peptides_as_string()
            peptide_lengths = [len(p) for p in peptides]

            # Get target peptides suitable for Prosit
            from imspy.algorithm.intensity.sage_interface import remove_unimod_annotation

            target_peptides = []
            for i, seq in enumerate(peptides):
                if "DECOY_" in seq:
                    continue
                # Check actual sequence length (without modifications)
                unmod_len = len(remove_unimod_annotation(seq))
                if unmod_len <= 30:
                    target_peptides.append((i, seq))

            # Predict for all suitable target peptides
            if len(target_peptides) < 10:
                pytest.skip("Not enough suitable peptides")

            indices = [t[0] for t in target_peptides]
            sequences = [t[1] for t in target_peptides]

            # Predict with multiple charge states
            all_indices = []
            all_sequences = []
            all_charges = []

            for idx, seq in zip(indices, sequences):
                for charge in [2, 3]:
                    all_indices.append(idx)
                    all_sequences.append(seq)
                    all_charges.append(charge)

            print(f"\nPredicting intensities for {len(all_sequences)} "
                  f"peptide-charge combinations...")

            result = predict_intensities_for_sage(
                sequences=all_sequences,
                charges=all_charges,
                peptide_indices=all_indices,
                max_fragment_charge=2,
                batch_size=2048,
                verbose=True,
            )

            # Verify predictions
            assert len(result.intensities) == len(all_sequences)

            # Check all predictions are valid
            for i, (seq, intensity) in enumerate(zip(all_sequences, result.intensities)):
                unmod_len = len(remove_unimod_annotation(seq))
                expected_shape = (2, unmod_len - 1, 2)  # [B/Y, positions, charges]
                assert intensity.shape == expected_shape, \
                    f"Wrong shape for peptide {i}: {intensity.shape} vs {expected_shape}"
                assert np.all(intensity >= 0), f"Negative intensities for peptide {i}"
                assert np.all(intensity <= 1), f"Intensities > 1 for peptide {i}"

            # Write to file
            with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
                temp_path = f.name

            try:
                write_predictions_for_database(
                    temp_path,
                    result,
                    num_peptides=len(peptides),
                    peptide_lengths=peptide_lengths,
                    aggregation='max_charge',
                    default_value=1.0,
                )

                # Load and validate
                store = PredictedIntensityStore(temp_path)

                assert store.peptide_count == len(peptides)
                assert store.max_charge == 2

                # Check that predicted peptides have non-trivial values
                predicted_non_uniform = 0
                for idx in indices[:20]:
                    pep_len = peptide_lengths[idx]
                    for pos in range(min(3, pep_len - 1)):
                        intensity = store.get_intensity_or_default(
                            idx, pep_len, ION_KIND_Y, pos, 1
                        )
                        if abs(intensity - 1.0) > 0.01:
                            predicted_non_uniform += 1

                assert predicted_non_uniform > 0, \
                    "Expected some non-uniform intensities from Prosit"

                print(f"\nSuccess: {len(target_peptides)} peptides predicted, "
                      f"{predicted_non_uniform} non-uniform values found")

            finally:
                os.unlink(temp_path)

        except ImportError:
            pytest.skip("TensorFlow not available")

    @pytest.mark.slow
    def test_pipeline_class_4p(self, fasta_content):
        """Test the IntensityPredictionPipeline class with 4P FASTA."""
        try:
            from imspy.algorithm.intensity.pipeline import (
                IntensityPredictionPipeline,
                PipelineConfig,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                # Write FASTA to temp file
                fasta_path = os.path.join(temp_dir, "4P.fasta")
                with open(fasta_path, 'w') as f:
                    f.write(fasta_content)

                # Create pipeline
                config = PipelineConfig(
                    charges=[2, 3],
                    max_fragment_charge=2,
                    collision_energy=25.0,
                )

                pipeline = IntensityPredictionPipeline(
                    fasta_path=fasta_path,
                    output_dir=temp_dir,
                    config=config,
                )

                # Build database
                db = pipeline.build_database()
                peptides = list(db.peptides_as_string())

                print(f"\nBuilt database with {len(peptides)} peptides")

                # Get summary before prediction
                summary = pipeline.get_summary()
                assert summary['total_peptides'] == len(peptides)
                assert summary['predictions_available'] is False

                # Predict intensities
                result = pipeline.predict_intensities(
                    batch_size=2048,
                    verbose=True,
                )

                assert result is not None
                assert len(result.intensities) > 0

                # Save store
                sagi_path = pipeline.save_intensity_store()

                assert sagi_path.exists()

                # Load store
                store = pipeline.load_intensity_store()

                assert store.peptide_count == len(peptides)

                # Get final summary
                summary = pipeline.get_summary()
                assert summary['predictions_available'] is True
                assert summary['sagi_file_exists'] is True
                assert summary['num_predictions'] > 0

                print(f"\nPipeline summary: {summary}")

        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")

    @pytest.mark.slow
    def test_weighted_scoring_with_prosit_4p(self, indexed_db_4p, scorer_params):
        """Test weighted scoring with Prosit predictions on 4P database."""
        try:
            from imspy.algorithm.intensity.sage_interface import remove_unimod_annotation

            peptides = indexed_db_4p.peptides_as_string()
            peptide_lengths = [len(p) for p in peptides]

            # Get target peptides
            target_peptides = []
            for i, seq in enumerate(peptides):
                if "DECOY_" in seq:
                    continue
                unmod_len = len(remove_unimod_annotation(seq))
                if unmod_len <= 30:
                    target_peptides.append((i, seq))

            if len(target_peptides) < 10:
                pytest.skip("Not enough suitable peptides")

            # Predict with Prosit
            indices = [t[0] for t in target_peptides]
            sequences = [t[1] for t in target_peptides]
            charges = [2] * len(sequences)

            result = predict_intensities_for_sage(
                sequences=sequences,
                charges=charges,
                peptide_indices=indices,
                max_fragment_charge=2,
                verbose=False,
            )

            # Write predictions
            with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
                temp_path = f.name

            try:
                write_predictions_for_database(
                    temp_path,
                    result,
                    num_peptides=len(peptides),
                    peptide_lengths=peptide_lengths,
                )

                store = PredictedIntensityStore(temp_path)

                # Create scorers
                scorer_weighted = Scorer(
                    score_type=ScoreType("weightedhyperscore"),
                    **scorer_params,
                )

                scorer_unweighted = Scorer(
                    score_type=ScoreType("hyperscore"),
                    **scorer_params,
                )

                # Create test spectrum
                spec_data = {
                    "id": "test_weighted_4p",
                    "precursor_mz": 550.28,
                    "precursor_charge": 2,
                    "precursor_intensity": 2e6,
                    "mz": np.array([
                        175.12, 262.15, 333.19, 446.27, 547.32,
                        662.35, 775.43, 876.48, 989.56, 1090.61
                    ], dtype=np.float64),
                    "intensity": np.array([
                        1500.0, 2800.0, 4200.0, 5500.0, 4800.0,
                        3500.0, 2200.0, 1400.0, 800.0, 400.0
                    ], dtype=np.float64),
                    "rt": 900.0,
                }

                processor = SpectrumProcessor(take_top_n=150)
                processed = create_processed_spectrum(spec_data, processor)

                # Score with weighted and unweighted
                features_weighted = scorer_weighted.score(
                    indexed_db_4p, processed, intensity_store=store
                )
                features_unweighted = scorer_unweighted.score(indexed_db_4p, processed)

                # Both should produce results (may be different)
                print(f"\nUnweighted PSMs: {len(features_unweighted)}")
                print(f"Weighted PSMs: {len(features_weighted)}")

                # Verify weighted scores are valid
                for f in features_weighted:
                    assert f.hyperscore >= 0
                    assert f.matched_peaks >= 0

                # With Prosit predictions, weighted scores might differ from unweighted
                # This is expected behavior - the test validates the pipeline works

            finally:
                os.unlink(temp_path)

        except ImportError:
            pytest.skip("TensorFlow not available")
