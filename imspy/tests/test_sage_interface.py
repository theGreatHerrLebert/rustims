"""
Tests for Sage intensity prediction interface.

These tests verify the functionality of the sage_interface module including:
- PredictionRequest/PredictionResult dataclasses
- Prosit to Sage format transformation
- .sagi binary file I/O
- Aggregation and validation functions
"""

import pytest
import numpy as np
import tempfile
import os
from numpy.testing import assert_array_equal, assert_array_almost_equal

from imspy.algorithm.intensity.sage_interface import (
    PredictionRequest,
    PredictionResult,
    remove_unimod_annotation,
    _prosit_to_sage_format,
    predict_intensities_for_sage,
    validate_prediction_result,
    aggregate_predictions_by_peptide,
    write_intensity_file,
    read_intensity_file,
    get_intensity_from_file,
    create_uniform_predictions,
    write_predictions_for_database,
    ION_KIND_A,
    ION_KIND_B,
    ION_KIND_Y,
    DEFAULT_ION_KINDS,
    SAGI_MAGIC,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_sequences():
    """Simple unmodified peptide sequences."""
    return ["PEPTIDEK", "ANOTHERK", "TESTSEQK"]


@pytest.fixture
def modified_sequences():
    """Peptide sequences with UNIMOD modifications."""
    return [
        "PEPTC[UNIMOD:4]IDEK",
        "ANOTHERM[UNIMOD:35]K",
        "TESTS[UNIMOD:21]EQK",
    ]


@pytest.fixture
def simple_charges():
    """Simple charge states."""
    return [2, 3, 2]


@pytest.fixture
def simple_peptide_indices():
    """Simple peptide indices."""
    return [0, 1, 2]


@pytest.fixture
def mock_prosit_output():
    """Mock Prosit output array shape [29, 2, 3] (positions, ion_type, charge)."""
    # Create predictable values for testing
    output = np.zeros((29, 2, 3), dtype=np.float32)
    # Y ions (index 0): fill with 0.1 * position
    for pos in range(29):
        for charge in range(3):
            output[pos, 0, charge] = 0.1 * (pos + 1) / 29  # Y ions
            output[pos, 1, charge] = 0.2 * (pos + 1) / 29  # B ions
    return output


@pytest.fixture
def sample_prediction_result(simple_sequences, simple_charges, simple_peptide_indices):
    """Sample PredictionResult for testing."""
    intensities = []
    for seq in simple_sequences:
        seq_len = len(seq)
        n_positions = seq_len - 1
        # Create random but reproducible intensities
        np.random.seed(42)
        intensity = np.random.rand(2, n_positions, 2).astype(np.float32)
        intensities.append(intensity)

    return PredictionResult(
        peptide_indices=np.array(simple_peptide_indices, dtype=np.int64),
        charges=np.array(simple_charges, dtype=np.int32),
        intensities=intensities,
        ion_kinds=[ION_KIND_B, ION_KIND_Y],
        max_fragment_charge=2,
    )


# =============================================================================
# Test remove_unimod_annotation
# =============================================================================

class TestRemoveUnimodAnnotation:
    """Tests for remove_unimod_annotation function."""

    def test_no_modification(self):
        """Test sequence without modifications."""
        assert remove_unimod_annotation("PEPTIDEK") == "PEPTIDEK"

    def test_single_modification(self):
        """Test sequence with single modification."""
        assert remove_unimod_annotation("PEPTC[UNIMOD:4]IDEK") == "PEPTCIDEK"

    def test_multiple_modifications(self):
        """Test sequence with multiple modifications."""
        seq = "AC[UNIMOD:4]M[UNIMOD:35]PEPTIDEK"
        assert remove_unimod_annotation(seq) == "ACMPEPTIDEK"

    def test_different_mod_ids(self):
        """Test various UNIMOD IDs."""
        assert remove_unimod_annotation("S[UNIMOD:21]EQ") == "SEQ"
        assert remove_unimod_annotation("M[UNIMOD:35]ET") == "MET"
        assert remove_unimod_annotation("K[UNIMOD:1]AA") == "KAA"


# =============================================================================
# Test PredictionRequest
# =============================================================================

class TestPredictionRequest:
    """Tests for PredictionRequest dataclass."""

    def test_create_from_arrays(self, simple_sequences, simple_charges, simple_peptide_indices):
        """Test creating PredictionRequest from arrays."""
        request = PredictionRequest(
            sequences=np.array(simple_sequences),
            charges=np.array(simple_charges, dtype=np.int32),
            peptide_indices=np.array(simple_peptide_indices, dtype=np.int64),
        )
        assert len(request) == 3
        assert request.sequences[0] == "PEPTIDEK"
        assert request.charges[1] == 3

    def test_to_dataframe(self, simple_sequences, simple_charges, simple_peptide_indices):
        """Test conversion to DataFrame."""
        request = PredictionRequest(
            sequences=np.array(simple_sequences),
            charges=np.array(simple_charges, dtype=np.int32),
            peptide_indices=np.array(simple_peptide_indices, dtype=np.int64),
        )
        df = request.to_dataframe()
        assert 'sequence' in df.columns
        assert 'charge' in df.columns
        assert 'peptide_idx' in df.columns
        assert len(df) == 3

    def test_from_dataframe(self, simple_sequences, simple_charges, simple_peptide_indices):
        """Test creating from DataFrame."""
        import pandas as pd
        df = pd.DataFrame({
            'sequence': simple_sequences,
            'charge': simple_charges,
            'peptide_idx': simple_peptide_indices,
        })
        request = PredictionRequest.from_dataframe(df)
        assert len(request) == 3
        assert list(request.sequences) == simple_sequences

    def test_from_sequences_expands_charges(self):
        """Test from_sequences expands for multiple charge states."""
        sequences = ["PEPTIDE", "ANOTHER"]
        charges = [2, 3]
        request = PredictionRequest.from_sequences(sequences, charges)

        # Should have 4 entries (2 sequences x 2 charges)
        assert len(request) == 4
        assert list(request.sequences) == ["PEPTIDE", "PEPTIDE", "ANOTHER", "ANOTHER"]
        assert list(request.charges) == [2, 3, 2, 3]
        assert list(request.peptide_indices) == [0, 0, 1, 1]


# =============================================================================
# Test PredictionResult
# =============================================================================

class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_create_result(self, sample_prediction_result):
        """Test creating PredictionResult."""
        assert len(sample_prediction_result) == 3
        assert sample_prediction_result.max_fragment_charge == 2
        assert sample_prediction_result.ion_kinds == [ION_KIND_B, ION_KIND_Y]

    def test_default_ion_kinds(self):
        """Test default ion kinds are B and Y."""
        result = PredictionResult(
            peptide_indices=np.array([0]),
            charges=np.array([2]),
            intensities=[np.zeros((2, 5, 2), dtype=np.float32)],
        )
        assert result.ion_kinds == [ION_KIND_B, ION_KIND_Y]

    def test_intensity_shapes(self, sample_prediction_result, simple_sequences):
        """Test intensity array shapes match sequences."""
        for seq, intensity in zip(simple_sequences, sample_prediction_result.intensities):
            expected_positions = len(seq) - 1
            assert intensity.shape == (2, expected_positions, 2)


# =============================================================================
# Test _prosit_to_sage_format
# =============================================================================

class TestPrositToSageFormat:
    """Tests for Prosit to Sage format transformation."""

    def test_basic_transformation(self, mock_prosit_output):
        """Test basic format transformation."""
        seq_len = 10
        result = _prosit_to_sage_format(mock_prosit_output, seq_len, max_fragment_charge=2)

        # Expected shape: [2 ion_kinds, 9 positions, 2 charges]
        assert result.shape == (2, seq_len - 1, 2)

    def test_ion_order_swap(self, mock_prosit_output):
        """Test that Y,B is swapped to B,Y."""
        seq_len = 10
        result = _prosit_to_sage_format(mock_prosit_output, seq_len, max_fragment_charge=2)

        # In mock_prosit_output: index 0 = Y (0.1*), index 1 = B (0.2*)
        # After transformation: index 0 = B, index 1 = Y
        # B ions should have higher values (0.2*) at index 0
        # Y ions should have lower values (0.1*) at index 1
        assert result[0, 0, 0] > result[1, 0, 0]  # B > Y at first position

    def test_position_trimming(self, mock_prosit_output):
        """Test that positions are trimmed to sequence length."""
        for seq_len in [7, 15, 25]:
            result = _prosit_to_sage_format(mock_prosit_output, seq_len, max_fragment_charge=2)
            assert result.shape[1] == seq_len - 1

    def test_charge_trimming(self, mock_prosit_output):
        """Test charge dimension trimming."""
        seq_len = 10
        for max_charge in [1, 2, 3]:
            result = _prosit_to_sage_format(mock_prosit_output, seq_len, max_fragment_charge=max_charge)
            assert result.shape[2] == max_charge

    def test_negative_values_zeroed(self):
        """Test that negative values (masked) are set to zero."""
        prosit_output = np.full((29, 2, 3), -1.0, dtype=np.float32)
        result = _prosit_to_sage_format(prosit_output, 10, max_fragment_charge=2)
        assert np.all(result >= 0)

    def test_values_clipped_to_one(self):
        """Test that values > 1 are clipped."""
        prosit_output = np.full((29, 2, 3), 2.0, dtype=np.float32)
        result = _prosit_to_sage_format(prosit_output, 10, max_fragment_charge=2)
        assert np.all(result <= 1.0)

    def test_output_dtype(self, mock_prosit_output):
        """Test output is float32."""
        result = _prosit_to_sage_format(mock_prosit_output, 10, max_fragment_charge=2)
        assert result.dtype == np.float32


# =============================================================================
# Test validate_prediction_result
# =============================================================================

class TestValidatePredictionResult:
    """Tests for validate_prediction_result function."""

    def test_valid_result(self, simple_sequences, simple_charges, simple_peptide_indices):
        """Test validation passes for correct result."""
        request = PredictionRequest(
            sequences=np.array(simple_sequences),
            charges=np.array(simple_charges, dtype=np.int32),
            peptide_indices=np.array(simple_peptide_indices, dtype=np.int64),
        )

        intensities = [
            np.zeros((2, len(seq) - 1, 2), dtype=np.float32)
            for seq in simple_sequences
        ]

        result = PredictionResult(
            peptide_indices=np.array(simple_peptide_indices, dtype=np.int64),
            charges=np.array(simple_charges, dtype=np.int32),
            intensities=intensities,
        )

        assert validate_prediction_result(request, result) is True

    def test_length_mismatch(self, simple_sequences, simple_charges, simple_peptide_indices):
        """Test validation fails for length mismatch."""
        request = PredictionRequest(
            sequences=np.array(simple_sequences),
            charges=np.array(simple_charges, dtype=np.int32),
            peptide_indices=np.array(simple_peptide_indices, dtype=np.int64),
        )

        # Only provide 2 intensities instead of 3
        result = PredictionResult(
            peptide_indices=np.array([0, 1], dtype=np.int64),
            charges=np.array([2, 3], dtype=np.int32),
            intensities=[np.zeros((2, 5, 2), dtype=np.float32)] * 2,
        )

        with pytest.raises(ValueError, match="Length mismatch"):
            validate_prediction_result(request, result)

    def test_shape_mismatch(self, simple_sequences, simple_charges, simple_peptide_indices):
        """Test validation fails for intensity shape mismatch."""
        request = PredictionRequest(
            sequences=np.array(simple_sequences),
            charges=np.array(simple_charges, dtype=np.int32),
            peptide_indices=np.array(simple_peptide_indices, dtype=np.int64),
        )

        # Wrong number of positions
        intensities = [np.zeros((2, 5, 2), dtype=np.float32)] * 3

        result = PredictionResult(
            peptide_indices=np.array(simple_peptide_indices, dtype=np.int64),
            charges=np.array(simple_charges, dtype=np.int32),
            intensities=intensities,
        )

        with pytest.raises(ValueError, match="shape mismatch"):
            validate_prediction_result(request, result)


# =============================================================================
# Test aggregate_predictions_by_peptide
# =============================================================================

class TestAggregatePredictions:
    """Tests for aggregate_predictions_by_peptide function."""

    def test_max_charge_aggregation(self):
        """Test max_charge aggregation strategy."""
        # Same peptide at charges 2 and 3
        result = PredictionResult(
            peptide_indices=np.array([0, 0, 1], dtype=np.int64),
            charges=np.array([2, 3, 2], dtype=np.int32),
            intensities=[
                np.full((2, 5, 2), 0.5, dtype=np.float32),  # peptide 0, charge 2
                np.full((2, 5, 2), 0.8, dtype=np.float32),  # peptide 0, charge 3
                np.full((2, 5, 2), 0.3, dtype=np.float32),  # peptide 1, charge 2
            ],
        )

        aggregated = aggregate_predictions_by_peptide(result, aggregation='max_charge')

        assert len(aggregated) == 2
        # Peptide 0 should have charge 3 prediction (0.8)
        assert_array_almost_equal(aggregated[0], np.full((2, 5, 2), 0.8, dtype=np.float32))
        # Peptide 1 should have charge 2 prediction (0.3)
        assert_array_almost_equal(aggregated[1], np.full((2, 5, 2), 0.3, dtype=np.float32))

    def test_mean_aggregation(self):
        """Test mean aggregation strategy."""
        result = PredictionResult(
            peptide_indices=np.array([0, 0], dtype=np.int64),
            charges=np.array([2, 3], dtype=np.int32),
            intensities=[
                np.full((2, 5, 2), 0.4, dtype=np.float32),
                np.full((2, 5, 2), 0.6, dtype=np.float32),
            ],
        )

        aggregated = aggregate_predictions_by_peptide(result, aggregation='mean')

        assert len(aggregated) == 1
        # Mean of 0.4 and 0.6 should be 0.5
        assert_array_almost_equal(aggregated[0], np.full((2, 5, 2), 0.5, dtype=np.float32))

    def test_first_aggregation(self):
        """Test first aggregation strategy."""
        result = PredictionResult(
            peptide_indices=np.array([0, 0], dtype=np.int64),
            charges=np.array([2, 3], dtype=np.int32),
            intensities=[
                np.full((2, 5, 2), 0.4, dtype=np.float32),
                np.full((2, 5, 2), 0.6, dtype=np.float32),
            ],
        )

        aggregated = aggregate_predictions_by_peptide(result, aggregation='first')

        assert_array_almost_equal(aggregated[0], np.full((2, 5, 2), 0.4, dtype=np.float32))

    def test_unknown_aggregation_raises(self):
        """Test that unknown aggregation strategy raises ValueError."""
        result = PredictionResult(
            peptide_indices=np.array([0], dtype=np.int64),
            charges=np.array([2], dtype=np.int32),
            intensities=[np.zeros((2, 5, 2), dtype=np.float32)],
        )

        with pytest.raises(ValueError, match="Unknown aggregation"):
            aggregate_predictions_by_peptide(result, aggregation='invalid')


# =============================================================================
# Test create_uniform_predictions
# =============================================================================

class TestCreateUniformPredictions:
    """Tests for create_uniform_predictions function."""

    def test_basic_creation(self):
        """Test basic uniform prediction creation."""
        peptide_lengths = [8, 10, 12]
        predictions = create_uniform_predictions(peptide_lengths)

        assert len(predictions) == 3
        for pred, pep_len in zip(predictions, peptide_lengths):
            assert pred.shape == (2, pep_len - 1, 2)
            assert np.all(pred == 1.0)

    def test_custom_value(self):
        """Test uniform predictions with custom value."""
        predictions = create_uniform_predictions([8], value=0.5)
        assert np.all(predictions[0] == 0.5)

    def test_custom_ion_kinds(self):
        """Test uniform predictions with custom ion kinds."""
        predictions = create_uniform_predictions(
            [8],
            ion_kinds=[ION_KIND_B, ION_KIND_Y, ION_KIND_A],
            max_fragment_charge=3
        )
        assert predictions[0].shape == (3, 7, 3)

    def test_dtype(self):
        """Test output dtype is float32."""
        predictions = create_uniform_predictions([8])
        assert predictions[0].dtype == np.float32


# =============================================================================
# Test .sagi file I/O
# =============================================================================

class TestSagiFileIO:
    """Tests for .sagi binary file reading and writing."""

    def test_write_and_read_roundtrip(self):
        """Test write/read roundtrip preserves data."""
        peptide_lengths = [8, 10, 12]
        predictions = [
            np.random.rand(2, pep_len - 1, 2).astype(np.float32)
            for pep_len in peptide_lengths
        ]

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            temp_path = f.name

        try:
            # Write
            write_intensity_file(temp_path, predictions, peptide_lengths)

            # Read
            data = read_intensity_file(temp_path)

            assert data['peptide_count'] == 3
            assert data['max_charge'] == 2
            assert data['ion_kinds'] == [ION_KIND_B, ION_KIND_Y]
            assert len(data['offsets']) == 3

            # Verify individual peptide data
            for i, (pred, pep_len) in enumerate(zip(predictions, peptide_lengths)):
                loaded = get_intensity_from_file(data, i, pep_len)
                assert_array_almost_equal(pred, loaded)
        finally:
            os.unlink(temp_path)

    def test_write_custom_ion_kinds(self):
        """Test writing with custom ion kinds."""
        predictions = [np.random.rand(3, 7, 2).astype(np.float32)]
        peptide_lengths = [8]
        ion_kinds = [ION_KIND_B, ION_KIND_Y, ION_KIND_A]

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            temp_path = f.name

        try:
            write_intensity_file(temp_path, predictions, peptide_lengths, ion_kinds=ion_kinds)
            data = read_intensity_file(temp_path)
            assert data['ion_kinds'] == ion_kinds
        finally:
            os.unlink(temp_path)

    def test_write_custom_max_charge(self):
        """Test writing with custom max charge."""
        predictions = [np.random.rand(2, 7, 3).astype(np.float32)]
        peptide_lengths = [8]

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            temp_path = f.name

        try:
            write_intensity_file(temp_path, predictions, peptide_lengths, max_charge=3)
            data = read_intensity_file(temp_path)
            assert data['max_charge'] == 3
        finally:
            os.unlink(temp_path)

    def test_invalid_magic_raises(self):
        """Test reading file with invalid magic raises error."""
        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            f.write(b'\x00\x00\x00\x00')  # Invalid magic
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid magic"):
                read_intensity_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_shape_validation(self):
        """Test that write validates prediction shapes."""
        # Wrong number of positions for peptide length
        predictions = [np.random.rand(2, 10, 2).astype(np.float32)]  # 10 positions
        peptide_lengths = [8]  # Should have 7 positions

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="expected 7 positions"):
                write_intensity_file(temp_path, predictions, peptide_lengths)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# =============================================================================
# Test write_predictions_for_database
# =============================================================================

class TestWritePredictionsForDatabase:
    """Tests for write_predictions_for_database function."""

    def test_fills_missing_with_uniform(self):
        """Test that missing peptides get uniform predictions."""
        # Result only has prediction for peptide 1
        result = PredictionResult(
            peptide_indices=np.array([1], dtype=np.int64),
            charges=np.array([2], dtype=np.int32),
            intensities=[np.full((2, 9, 2), 0.5, dtype=np.float32)],
        )

        peptide_lengths = [8, 10, 12]  # 3 peptides total

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            temp_path = f.name

        try:
            write_predictions_for_database(
                temp_path,
                result,
                num_peptides=3,
                peptide_lengths=peptide_lengths,
                default_value=1.0,
            )

            data = read_intensity_file(temp_path)

            # Peptide 0: should be uniform (1.0)
            pred0 = get_intensity_from_file(data, 0, peptide_lengths[0])
            assert np.all(pred0 == 1.0)

            # Peptide 1: should be our prediction (0.5)
            pred1 = get_intensity_from_file(data, 1, peptide_lengths[1])
            assert_array_almost_equal(pred1, np.full((2, 9, 2), 0.5, dtype=np.float32))

            # Peptide 2: should be uniform (1.0)
            pred2 = get_intensity_from_file(data, 2, peptide_lengths[2])
            assert np.all(pred2 == 1.0)
        finally:
            os.unlink(temp_path)

    def test_aggregates_multiple_charges(self):
        """Test that multiple charge predictions are aggregated."""
        # Same peptide at two charge states
        result = PredictionResult(
            peptide_indices=np.array([0, 0], dtype=np.int64),
            charges=np.array([2, 3], dtype=np.int32),
            intensities=[
                np.full((2, 7, 2), 0.4, dtype=np.float32),  # charge 2
                np.full((2, 7, 2), 0.8, dtype=np.float32),  # charge 3
            ],
        )

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            temp_path = f.name

        try:
            write_predictions_for_database(
                temp_path,
                result,
                num_peptides=1,
                peptide_lengths=[8],
                aggregation='max_charge',
            )

            data = read_intensity_file(temp_path)
            pred = get_intensity_from_file(data, 0, 8)
            # Should have charge 3 prediction (0.8)
            assert_array_almost_equal(pred, np.full((2, 7, 2), 0.8, dtype=np.float32))
        finally:
            os.unlink(temp_path)


# =============================================================================
# Integration test with Prosit (optional, requires TensorFlow)
# =============================================================================

class TestPrositIntegration:
    """Integration tests with actual Prosit model.

    These tests are marked as slow and may be skipped in CI.
    """

    @pytest.mark.slow
    def test_predict_intensities_for_sage(self):
        """Test full prediction pipeline with Prosit."""
        try:
            result = predict_intensities_for_sage(
                sequences=["PEPTIDEK", "ANOTHERK"],
                charges=[2, 3],
                peptide_indices=[0, 1],
                max_fragment_charge=2,
                verbose=False,
            )

            assert len(result) == 2
            assert result.intensities[0].shape == (2, 7, 2)  # 8 AA -> 7 positions
            assert result.intensities[1].shape == (2, 7, 2)  # 8 AA -> 7 positions

            # Values should be normalized 0-1
            for intensity in result.intensities:
                assert np.all(intensity >= 0)
                assert np.all(intensity <= 1)

        except ImportError:
            pytest.skip("TensorFlow not available")

    @pytest.mark.slow
    def test_predict_with_modifications(self):
        """Test prediction with modified sequences."""
        try:
            result = predict_intensities_for_sage(
                sequences=["PEPTC[UNIMOD:4]IDEK", "M[UNIMOD:35]YPEPTIDE"],
                charges=[2, 2],
                peptide_indices=[0, 1],
                verbose=False,
            )

            assert len(result) == 2
            # PEPTCIDEK = 9 AA -> 8 positions
            assert result.intensities[0].shape == (2, 8, 2)
            # MYPEPTIDE = 9 AA -> 8 positions
            assert result.intensities[1].shape == (2, 8, 2)

        except ImportError:
            pytest.skip("TensorFlow not available")

    @pytest.mark.slow
    def test_full_pipeline_roundtrip(self):
        """Test full pipeline: predict -> write -> read -> verify."""
        try:
            sequences = ["PEPTIDEK", "ANOTHERK", "TESTSEQK"]
            charges = [2, 3, 2]
            indices = [0, 1, 2]
            peptide_lengths = [len(s) for s in sequences]

            # Predict
            result = predict_intensities_for_sage(
                sequences=sequences,
                charges=charges,
                peptide_indices=indices,
                verbose=False,
            )

            with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
                temp_path = f.name

            try:
                # Write
                write_predictions_for_database(
                    temp_path,
                    result,
                    num_peptides=3,
                    peptide_lengths=peptide_lengths,
                )

                # Read and verify
                data = read_intensity_file(temp_path)
                assert data['peptide_count'] == 3

                for i, pep_len in enumerate(peptide_lengths):
                    loaded = get_intensity_from_file(data, i, pep_len)
                    assert loaded.shape == (2, pep_len - 1, 2)

            finally:
                os.unlink(temp_path)

        except ImportError:
            pytest.skip("TensorFlow not available")
