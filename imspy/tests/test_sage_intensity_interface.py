"""
Unit tests for the Sage intensity prediction interface.

These tests validate the integration between imspy's Prosit predictor
and Sage's intensity prediction format without requiring the actual
Prosit model (mocked where necessary).

Run tests:
    pytest tests/test_sage_intensity_interface.py -v
"""

import pytest
import numpy as np
import tempfile
import os

from imspy.algorithm.intensity.sage_interface import (
    remove_unimod_annotation,
    _prosit_to_sage_format,
    PredictionRequest,
    PredictionResult,
    validate_prediction_result,
    aggregate_predictions_by_peptide,
    write_intensity_file,
    read_intensity_file,
    get_intensity_from_file,
    create_uniform_predictions,
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
    return ["PEPTIDE", "SEQUENCE", "ANOTHERK"]


@pytest.fixture
def modified_sequences():
    """Peptide sequences with UNIMOD modifications."""
    return [
        "PEPTC[UNIMOD:4]IDE",  # Carbamidomethyl
        "PEPTM[UNIMOD:35]IDE",  # Oxidation
        "AC[UNIMOD:4]M[UNIMOD:35]PEPTIDEK",  # Multiple mods
        "PEPS[UNIMOD:21]TIDE",  # Phosphorylation
    ]


@pytest.fixture
def prosit_raw_output():
    """Mock Prosit output shape: [29, 2, 3] for a single peptide."""
    # Shape: [max_positions=29, ion_types=2 (Y, B), charges=3]
    return np.random.rand(29, 2, 3).astype(np.float32)


@pytest.fixture
def sample_prediction_result():
    """Create a sample PredictionResult for testing."""
    return PredictionResult(
        peptide_indices=np.array([0, 0, 1, 1], dtype=np.int64),
        charges=np.array([2, 3, 2, 3], dtype=np.int32),
        intensities=[
            np.random.rand(2, 6, 2).astype(np.float32),  # PEPTIDE (7 AA)
            np.random.rand(2, 6, 2).astype(np.float32),  # PEPTIDE (7 AA)
            np.random.rand(2, 7, 2).astype(np.float32),  # SEQUENCE (8 AA)
            np.random.rand(2, 7, 2).astype(np.float32),  # SEQUENCE (8 AA)
        ],
        ion_kinds=[ION_KIND_B, ION_KIND_Y],
        max_fragment_charge=2,
    )


# =============================================================================
# Test: UNIMOD Annotation Removal
# =============================================================================

class TestRemoveUnimodAnnotation:
    """Test UNIMOD annotation removal from peptide sequences."""

    def test_unmodified_sequence_unchanged(self):
        """Unmodified sequences should remain unchanged."""
        seq = "PEPTIDEK"
        assert remove_unimod_annotation(seq) == "PEPTIDEK"

    def test_single_modification_removed(self):
        """Single UNIMOD modification should be removed."""
        seq = "PEPTC[UNIMOD:4]IDEK"
        assert remove_unimod_annotation(seq) == "PEPTCIDEK"

    def test_multiple_modifications_removed(self):
        """Multiple UNIMOD modifications should all be removed."""
        seq = "AC[UNIMOD:4]M[UNIMOD:35]PEPTIDEK"
        assert remove_unimod_annotation(seq) == "ACMPEPTIDEK"

    def test_modification_at_start(self):
        """Modification at sequence start should be removed."""
        seq = "M[UNIMOD:35]PEPTIDEK"
        assert remove_unimod_annotation(seq) == "MPEPTIDEK"

    def test_modification_at_end(self):
        """Modification at sequence end should be removed."""
        seq = "PEPTIDEM[UNIMOD:35]"
        assert remove_unimod_annotation(seq) == "PEPTIDEM"

    def test_empty_sequence(self):
        """Empty sequence should return empty string."""
        assert remove_unimod_annotation("") == ""

    def test_phosphorylation_removed(self):
        """Phosphorylation modification should be removed."""
        seq = "PEPS[UNIMOD:21]TIDE"
        assert remove_unimod_annotation(seq) == "PEPSTIDE"

    def test_large_unimod_id_removed(self):
        """UNIMOD IDs with multiple digits should be removed."""
        seq = "PEPTIDE[UNIMOD:1234]K"
        assert remove_unimod_annotation(seq) == "PEPTIDEK"


# =============================================================================
# Test: Prosit to Sage Format Conversion
# =============================================================================

class TestPrositToSageFormat:
    """Test conversion of Prosit output to Sage format."""

    def test_output_shape_correct(self, prosit_raw_output):
        """Output shape should be [n_ion_kinds, seq_len-1, max_charge]."""
        sequence_length = 10  # 10 AA peptide
        max_charge = 2

        result = _prosit_to_sage_format(prosit_raw_output, sequence_length, max_charge)

        assert result.shape == (2, 9, 2)  # [2 ion types, 9 positions, 2 charges]

    def test_ion_order_swapped(self, prosit_raw_output):
        """Prosit Y,B order should be swapped to B,Y for Sage."""
        sequence_length = 8
        max_charge = 2

        result = _prosit_to_sage_format(prosit_raw_output, sequence_length, max_charge)

        # Prosit: index 0 = Y ions, index 1 = B ions
        # Sage:   index 0 = B ions, index 1 = Y ions
        # After transpose: [ion_type, position, charge]
        # B ions should be first in result
        assert result.shape[0] == 2

    def test_negative_values_replaced(self):
        """Negative values (masked positions) should be replaced with 0."""
        prosit_output = np.array([
            [[-1.0, -1.0, -1.0], [0.5, 0.6, 0.7]],  # pos 0
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],     # pos 1
        ], dtype=np.float32)

        result = _prosit_to_sage_format(prosit_output, sequence_length=3, max_fragment_charge=2)

        assert np.all(result >= 0), "Negative values should be replaced with 0"

    def test_values_clipped_to_one(self):
        """Values greater than 1 should be clipped to 1."""
        prosit_output = np.array([
            [[1.5, 2.0, 0.5], [0.5, 0.6, 0.7]],
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
        ], dtype=np.float32)

        result = _prosit_to_sage_format(prosit_output, sequence_length=3, max_fragment_charge=2)

        assert np.all(result <= 1), "Values should be clipped to 1"

    def test_output_dtype_float32(self, prosit_raw_output):
        """Output should be float32."""
        result = _prosit_to_sage_format(prosit_raw_output, sequence_length=10, max_fragment_charge=2)
        assert result.dtype == np.float32

    def test_charge_dimension_trimmed(self, prosit_raw_output):
        """Output should trim charge dimension to max_fragment_charge."""
        result = _prosit_to_sage_format(prosit_raw_output, sequence_length=10, max_fragment_charge=1)
        assert result.shape[2] == 1  # Only charge 1

    def test_short_peptide(self):
        """Short peptide should produce correctly sized output."""
        prosit_output = np.random.rand(29, 2, 3).astype(np.float32)
        result = _prosit_to_sage_format(prosit_output, sequence_length=5, max_fragment_charge=2)

        assert result.shape == (2, 4, 2)  # 5 AA peptide has 4 fragment positions


# =============================================================================
# Test: PredictionRequest
# =============================================================================

class TestPredictionRequest:
    """Test PredictionRequest dataclass and methods."""

    def test_from_sequences_basic(self, simple_sequences):
        """Test creation from simple sequences."""
        request = PredictionRequest.from_sequences(simple_sequences, charges=[2, 3])

        assert len(request) == len(simple_sequences) * 2  # 2 charges per sequence

    def test_from_sequences_charges_expanded(self, simple_sequences):
        """Charges should be expanded for each sequence."""
        request = PredictionRequest.from_sequences(simple_sequences, charges=[2, 3])

        # Each sequence should have entries for charge 2 and charge 3
        expected_charges = [2, 3] * len(simple_sequences)
        assert list(request.charges) == expected_charges

    def test_from_sequences_indices_correct(self, simple_sequences):
        """Peptide indices should map back to original sequence order."""
        request = PredictionRequest.from_sequences(simple_sequences, charges=[2, 3])

        # Indices should repeat for each charge
        expected_indices = [0, 0, 1, 1, 2, 2]
        assert list(request.peptide_indices) == expected_indices

    def test_to_dataframe(self, simple_sequences):
        """Test conversion to DataFrame."""
        request = PredictionRequest.from_sequences(simple_sequences, charges=[2])
        df = request.to_dataframe()

        assert len(df) == len(simple_sequences)
        assert list(df.columns) == ['sequence', 'charge', 'peptide_idx']

    def test_from_dataframe_roundtrip(self, simple_sequences):
        """Test roundtrip through DataFrame."""
        original = PredictionRequest.from_sequences(simple_sequences, charges=[2, 3])
        df = original.to_dataframe()
        restored = PredictionRequest.from_dataframe(df)

        assert len(restored) == len(original)
        np.testing.assert_array_equal(restored.peptide_indices, original.peptide_indices)
        np.testing.assert_array_equal(restored.charges, original.charges)

    def test_single_charge(self, simple_sequences):
        """Test with single charge state."""
        request = PredictionRequest.from_sequences(simple_sequences, charges=[2])

        assert len(request) == len(simple_sequences)
        assert all(c == 2 for c in request.charges)


# =============================================================================
# Test: Prediction Result Validation
# =============================================================================

class TestValidatePredictionResult:
    """Test validation of PredictionResult against PredictionRequest."""

    def test_valid_result_passes(self):
        """Valid result should pass validation."""
        request = PredictionRequest(
            sequences=np.array(["PEPTIDE", "SEQUENCE"]),
            charges=np.array([2, 2], dtype=np.int32),
            peptide_indices=np.array([0, 1], dtype=np.int64),
        )
        result = PredictionResult(
            peptide_indices=np.array([0, 1], dtype=np.int64),
            charges=np.array([2, 2], dtype=np.int32),
            intensities=[
                np.random.rand(2, 6, 2).astype(np.float32),  # 7 AA
                np.random.rand(2, 7, 2).astype(np.float32),  # 8 AA
            ],
            ion_kinds=[ION_KIND_B, ION_KIND_Y],
            max_fragment_charge=2,
        )

        assert validate_prediction_result(request, result) is True

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        request = PredictionRequest(
            sequences=np.array(["PEPTIDE", "SEQUENCE"]),
            charges=np.array([2, 2], dtype=np.int32),
            peptide_indices=np.array([0, 1], dtype=np.int64),
        )
        result = PredictionResult(
            peptide_indices=np.array([0], dtype=np.int64),
            charges=np.array([2], dtype=np.int32),
            intensities=[np.random.rand(2, 6, 2).astype(np.float32)],
            ion_kinds=[ION_KIND_B, ION_KIND_Y],
            max_fragment_charge=2,
        )

        with pytest.raises(ValueError, match="Length mismatch"):
            validate_prediction_result(request, result)

    def test_indices_mismatch_raises(self):
        """Mismatched peptide indices should raise ValueError."""
        request = PredictionRequest(
            sequences=np.array(["PEPTIDE"]),
            charges=np.array([2], dtype=np.int32),
            peptide_indices=np.array([0], dtype=np.int64),
        )
        result = PredictionResult(
            peptide_indices=np.array([1], dtype=np.int64),  # Wrong index
            charges=np.array([2], dtype=np.int32),
            intensities=[np.random.rand(2, 6, 2).astype(np.float32)],
            ion_kinds=[ION_KIND_B, ION_KIND_Y],
            max_fragment_charge=2,
        )

        with pytest.raises(ValueError, match="Peptide indices don't match"):
            validate_prediction_result(request, result)

    def test_intensity_shape_mismatch_raises(self):
        """Wrong intensity shape should raise ValueError."""
        request = PredictionRequest(
            sequences=np.array(["PEPTIDE"]),
            charges=np.array([2], dtype=np.int32),
            peptide_indices=np.array([0], dtype=np.int64),
        )
        result = PredictionResult(
            peptide_indices=np.array([0], dtype=np.int64),
            charges=np.array([2], dtype=np.int32),
            intensities=[np.random.rand(2, 10, 2).astype(np.float32)],  # Wrong positions
            ion_kinds=[ION_KIND_B, ION_KIND_Y],
            max_fragment_charge=2,
        )

        with pytest.raises(ValueError, match="Intensity shape mismatch"):
            validate_prediction_result(request, result)


# =============================================================================
# Test: Prediction Aggregation
# =============================================================================

class TestAggregatePredictionsByPeptide:
    """Test aggregation of multi-charge predictions to per-peptide predictions."""

    def test_max_charge_aggregation(self, sample_prediction_result):
        """Max charge aggregation should keep highest charge prediction."""
        aggregated = aggregate_predictions_by_peptide(
            sample_prediction_result,
            aggregation='max_charge'
        )

        # Should have one entry per unique peptide index
        assert len(aggregated) == 2  # Peptides 0 and 1

    def test_first_aggregation(self, sample_prediction_result):
        """First aggregation should keep first occurrence."""
        aggregated = aggregate_predictions_by_peptide(
            sample_prediction_result,
            aggregation='first'
        )

        assert len(aggregated) == 2

    def test_mean_aggregation(self, sample_prediction_result):
        """Mean aggregation should average across charges."""
        aggregated = aggregate_predictions_by_peptide(
            sample_prediction_result,
            aggregation='mean'
        )

        assert len(aggregated) == 2
        # Result should be float32
        assert all(v.dtype == np.float32 for v in aggregated.values())

    def test_single_entry_per_peptide(self):
        """Single entry per peptide should be preserved."""
        result = PredictionResult(
            peptide_indices=np.array([0, 1], dtype=np.int64),
            charges=np.array([2, 2], dtype=np.int32),
            intensities=[
                np.random.rand(2, 6, 2).astype(np.float32),
                np.random.rand(2, 7, 2).astype(np.float32),
            ],
            ion_kinds=[ION_KIND_B, ION_KIND_Y],
            max_fragment_charge=2,
        )

        aggregated = aggregate_predictions_by_peptide(result, aggregation='max_charge')
        assert len(aggregated) == 2

    def test_unknown_aggregation_raises(self, sample_prediction_result):
        """Unknown aggregation strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            aggregate_predictions_by_peptide(sample_prediction_result, aggregation='unknown')


# =============================================================================
# Test: File I/O (V1 Format)
# =============================================================================

class TestIntensityFileIO:
    """Test writing and reading .sagi intensity files."""

    def test_write_and_read_roundtrip(self):
        """Test that written file can be read back correctly."""
        predictions = [
            np.random.rand(2, 6, 2).astype(np.float32),  # 7 AA peptide
            np.random.rand(2, 9, 2).astype(np.float32),  # 10 AA peptide
        ]
        peptide_lengths = [7, 10]

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            tmp_path = f.name

        try:
            write_intensity_file(tmp_path, predictions, peptide_lengths)
            file_data = read_intensity_file(tmp_path)

            assert file_data['peptide_count'] == 2
            assert file_data['max_charge'] == 2
            assert file_data['ion_kinds'] == [ION_KIND_B, ION_KIND_Y]
        finally:
            os.unlink(tmp_path)

    def test_magic_number_correct(self):
        """Written file should have correct magic number."""
        predictions = [np.random.rand(2, 5, 2).astype(np.float32)]
        peptide_lengths = [6]

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            tmp_path = f.name

        try:
            write_intensity_file(tmp_path, predictions, peptide_lengths)

            with open(tmp_path, 'rb') as f:
                import struct
                magic = struct.unpack('<I', f.read(4))[0]
                assert magic == SAGI_MAGIC
        finally:
            os.unlink(tmp_path)

    def test_get_intensity_from_file(self):
        """Test extracting intensity for specific peptide."""
        pred1 = np.arange(24, dtype=np.float32).reshape(2, 6, 2)  # Predictable values
        pred2 = np.arange(24, 60, dtype=np.float32).reshape(2, 9, 2)
        predictions = [pred1, pred2]
        peptide_lengths = [7, 10]

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            tmp_path = f.name

        try:
            write_intensity_file(tmp_path, predictions, peptide_lengths)
            file_data = read_intensity_file(tmp_path)

            # Get first peptide's intensities
            intensity = get_intensity_from_file(file_data, peptide_idx=0, peptide_len=7)
            np.testing.assert_array_almost_equal(intensity, pred1)

            # Get second peptide's intensities
            intensity2 = get_intensity_from_file(file_data, peptide_idx=1, peptide_len=10)
            np.testing.assert_array_almost_equal(intensity2, pred2)
        finally:
            os.unlink(tmp_path)

    def test_invalid_magic_raises(self):
        """File with invalid magic number should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            f.write(b'\x00\x00\x00\x00')  # Invalid magic
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid magic number"):
                read_intensity_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_custom_ion_kinds(self):
        """Test writing with custom ion kinds."""
        predictions = [np.random.rand(3, 5, 2).astype(np.float32)]  # 3 ion kinds
        peptide_lengths = [6]
        custom_ion_kinds = [ION_KIND_B, ION_KIND_Y, 0]  # B, Y, A ions

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            tmp_path = f.name

        try:
            write_intensity_file(
                tmp_path,
                predictions,
                peptide_lengths,
                ion_kinds=custom_ion_kinds
            )
            file_data = read_intensity_file(tmp_path)

            assert file_data['ion_kinds'] == custom_ion_kinds
        finally:
            os.unlink(tmp_path)

    def test_invalid_peptide_length_raises(self):
        """Mismatched peptide length should raise ValueError."""
        predictions = [np.random.rand(2, 6, 2).astype(np.float32)]  # 7 AA
        peptide_lengths = [10]  # Wrong length

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="expected .* positions"):
                write_intensity_file(tmp_path, predictions, peptide_lengths)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# =============================================================================
# Test: Uniform Predictions
# =============================================================================

class TestCreateUniformPredictions:
    """Test creation of uniform (constant) intensity predictions."""

    def test_correct_shapes(self):
        """Output shapes should match peptide lengths."""
        peptide_lengths = [7, 10, 15]
        predictions = create_uniform_predictions(peptide_lengths)

        assert len(predictions) == 3
        assert predictions[0].shape == (2, 6, 2)   # 7-1 = 6 positions
        assert predictions[1].shape == (2, 9, 2)   # 10-1 = 9 positions
        assert predictions[2].shape == (2, 14, 2)  # 15-1 = 14 positions

    def test_default_value_is_one(self):
        """Default uniform value should be 1.0."""
        predictions = create_uniform_predictions([10])

        assert np.all(predictions[0] == 1.0)

    def test_custom_value(self):
        """Custom uniform value should be used."""
        predictions = create_uniform_predictions([10], value=0.5)

        assert np.all(predictions[0] == 0.5)

    def test_dtype_float32(self):
        """Output should be float32."""
        predictions = create_uniform_predictions([10])

        assert predictions[0].dtype == np.float32

    def test_custom_ion_kinds(self):
        """Custom ion kinds should affect shape."""
        predictions = create_uniform_predictions(
            [10],
            ion_kinds=[ION_KIND_B, ION_KIND_Y, 0],  # 3 ion kinds
        )

        assert predictions[0].shape[0] == 3

    def test_custom_max_charge(self):
        """Custom max fragment charge should affect shape."""
        predictions = create_uniform_predictions([10], max_fragment_charge=3)

        assert predictions[0].shape[2] == 3


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_predictions_list(self):
        """Empty prediction list should be writable."""
        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            tmp_path = f.name

        try:
            write_intensity_file(tmp_path, [], [])
            file_data = read_intensity_file(tmp_path)
            assert file_data['peptide_count'] == 0
        finally:
            os.unlink(tmp_path)

    def test_short_peptide_two_aa(self):
        """2 AA peptide (1 position) should work."""
        predictions = [np.random.rand(2, 1, 2).astype(np.float32)]
        peptide_lengths = [2]

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            tmp_path = f.name

        try:
            write_intensity_file(tmp_path, predictions, peptide_lengths)
            file_data = read_intensity_file(tmp_path)
            assert file_data['peptide_count'] == 1
        finally:
            os.unlink(tmp_path)

    def test_max_charge_one(self):
        """Single charge state should work."""
        predictions = [np.random.rand(2, 6, 1).astype(np.float32)]
        peptide_lengths = [7]

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            tmp_path = f.name

        try:
            write_intensity_file(tmp_path, predictions, peptide_lengths, max_charge=1)
            file_data = read_intensity_file(tmp_path)
            assert file_data['max_charge'] == 1
        finally:
            os.unlink(tmp_path)


# =============================================================================
# Test: PredictionResult
# =============================================================================

class TestPredictionResult:
    """Test PredictionResult dataclass."""

    def test_length(self, sample_prediction_result):
        """Length should match number of intensities."""
        assert len(sample_prediction_result) == 4

    def test_default_ion_kinds(self):
        """Default ion kinds should be B and Y."""
        result = PredictionResult(
            peptide_indices=np.array([0]),
            charges=np.array([2]),
            intensities=[np.random.rand(2, 6, 2).astype(np.float32)],
        )

        assert result.ion_kinds == [ION_KIND_B, ION_KIND_Y]

    def test_default_max_fragment_charge(self):
        """Default max fragment charge should be 2."""
        result = PredictionResult(
            peptide_indices=np.array([0]),
            charges=np.array([2]),
            intensities=[np.random.rand(2, 6, 2).astype(np.float32)],
        )

        assert result.max_fragment_charge == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
