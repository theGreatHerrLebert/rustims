"""
Sage intensity prediction interface.

This module provides the interface between imspy's Prosit intensity predictor
and Sage's binary intensity file format (.sagi) for weighted scoring.

The workflow is:
1. Create a PredictionRequest from Sage's IndexedDatabase peptides
2. Call predict_intensities_for_sage() to get predictions via Prosit
3. Write predictions to .sagi file using write_intensity_file()
4. Sage loads the file during search for weighted scoring
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING, Union
import struct
import re

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .collision_energy import CollisionEnergyModel


# Ion kind codes for Sage format
ION_KIND_A = 0
ION_KIND_B = 1
ION_KIND_C = 2
ION_KIND_X = 3
ION_KIND_Y = 4
ION_KIND_Z = 5

# Default ion kinds (B and Y ions)
DEFAULT_ION_KINDS = [ION_KIND_B, ION_KIND_Y]

# Default collision energy (normalized, will be divided by 100 internally)
DEFAULT_COLLISION_ENERGY = 35.0

# Magic number for .sagi file format
SAGI_MAGIC = 0x49474153  # "SAGI" in little-endian


def remove_unimod_annotation(sequence: str) -> str:
    """Remove UNIMOD annotations from a peptide sequence.

    Args:
        sequence: Peptide sequence with UNIMOD notation, e.g., "PEPTC[UNIMOD:4]IDEK"

    Returns:
        Sequence without modifications, e.g., "PEPTCIDEK"
    """
    pattern = r'\[UNIMOD:\d+\]'
    return re.sub(pattern, '', sequence)


@dataclass
class PredictionRequest:
    """Batch of peptides for intensity prediction.

    This class holds the input data for intensity prediction, typically
    extracted from Sage's IndexedDatabase.

    Attributes:
        sequences: Modified peptide sequences with UNIMOD notation
        charges: Precursor charge states (parallel to sequences)
        peptide_indices: IndexedDatabase peptide indices for mapping predictions back
    """
    sequences: NDArray  # dtype: str/object
    charges: NDArray  # dtype: int32
    peptide_indices: NDArray  # dtype: int64

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy inspection/manipulation."""
        return pd.DataFrame({
            'sequence': self.sequences,
            'charge': self.charges,
            'peptide_idx': self.peptide_indices,
        })

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'PredictionRequest':
        """Create PredictionRequest from a DataFrame.

        Args:
            df: DataFrame with columns 'sequence', 'charge', 'peptide_idx'

        Returns:
            PredictionRequest instance
        """
        return cls(
            sequences=df['sequence'].values,
            charges=df['charge'].astype(np.int32).values,
            peptide_indices=df['peptide_idx'].astype(np.int64).values,
        )

    @classmethod
    def from_sequences(
        cls,
        sequences: List[str],
        charges: List[int] = [2, 3],
    ) -> 'PredictionRequest':
        """Create PredictionRequest from a list of sequences.

        Each sequence is expanded for all specified charge states.

        Args:
            sequences: List of peptide sequences with UNIMOD modifications
            charges: List of precursor charges to predict for each peptide

        Returns:
            PredictionRequest with expanded (sequence, charge) pairs
        """
        n_sequences = len(sequences)
        n_charges = len(charges)

        expanded_sequences = np.repeat(sequences, n_charges)
        expanded_charges = np.tile(charges, n_sequences).astype(np.int32)
        expanded_indices = np.repeat(np.arange(n_sequences), n_charges).astype(np.int64)

        return cls(
            sequences=expanded_sequences,
            charges=expanded_charges,
            peptide_indices=expanded_indices,
        )

    def __len__(self) -> int:
        return len(self.sequences)


@dataclass
class PredictionResult:
    """Predicted intensities for a batch of peptides.

    Attributes:
        peptide_indices: IndexedDatabase peptide indices (must match request)
        charges: Precursor charges (must match request)
        intensities: Fragment intensities, one array per peptide.
            Each array has shape [n_ion_kinds, seq_len-1, max_fragment_charge]
            Values should be normalized (0.0 to 1.0)
        ion_kinds: Ion type codes used (e.g., [1, 4] for B and Y)
        max_fragment_charge: Maximum fragment charge state predicted
    """
    peptide_indices: NDArray
    charges: NDArray
    intensities: List[NDArray]
    ion_kinds: List[int] = field(default_factory=lambda: DEFAULT_ION_KINDS.copy())
    max_fragment_charge: int = 2

    def __len__(self) -> int:
        return len(self.intensities)


def _prosit_to_sage_format(
    prosit_output: NDArray,
    sequence_length: int,
    max_fragment_charge: int = 2,
) -> NDArray:
    """Transform Prosit output to Sage's expected format.

    Prosit output shape: [29, 2, 3] → [position, ion_type(Y,B), charge]
    Sage expected shape: [2, seq_len-1, max_charge] → [ion_kind(B,Y), position, charge]

    Args:
        prosit_output: Raw Prosit prediction array, shape [29, 2, 3]
        sequence_length: Actual peptide length (to trim padding)
        max_fragment_charge: Maximum fragment charge to include (1-3)

    Returns:
        Transformed array in Sage format, shape [2, seq_len-1, max_fragment_charge]
    """
    n_positions = sequence_length - 1

    # Trim to actual positions (Prosit pads to 29)
    pred = prosit_output[:n_positions, :, :]

    # Transpose: [position, ion_type, charge] → [ion_type, position, charge]
    pred = np.transpose(pred, (1, 0, 2))

    # Swap Y,B → B,Y: Prosit uses [0]=Y, [1]=B; Sage wants [0]=B, [1]=Y
    pred = pred[[1, 0], :, :]

    # Trim charge dimension if needed
    pred = pred[:, :, :max_fragment_charge]

    # Ensure values are 0-1 normalized, replace -1 (masked) with 0
    pred = np.where(pred < 0, 0, pred)
    pred = np.clip(pred, 0, 1).astype(np.float32)

    return pred


def predict_intensities_for_sage(
    sequences: List[str],
    charges: List[int],
    peptide_indices: List[int],
    collision_energies: Optional[List[float]] = None,
    collision_energy_model: Optional['CollisionEnergyModel'] = None,
    precursor_mz: Optional[List[float]] = None,
    ion_kinds: List[int] = None,
    max_fragment_charge: int = 2,
    batch_size: int = 2048,
    verbose: bool = True,
) -> PredictionResult:
    """Predict fragment ion intensities using Prosit for Sage integration.

    This function wraps the Prosit 2023 timsTOF predictor and transforms
    the output to Sage's expected format for the .sagi binary file.

    Collision energy can be specified in three ways (in order of priority):
    1. Explicit `collision_energies` list (per-peptide)
    2. `collision_energy_model` + `precursor_mz` (m/z-dependent)
    3. Default constant value (35.0 NCE)

    Args:
        sequences: Peptide sequences with UNIMOD modifications
        charges: Precursor charge states (parallel to sequences)
        peptide_indices: Indices for mapping back (preserved in output)
        collision_energies: Explicit collision energies per peptide (takes priority)
        collision_energy_model: m/z-dependent CE model (requires precursor_mz)
        precursor_mz: Precursor m/z values (required when using collision_energy_model)
        ion_kinds: Ion type codes to predict (default: [1, 4] for B, Y)
        max_fragment_charge: Maximum fragment charge to predict (1-3)
        batch_size: Batch size for Prosit prediction
        verbose: Whether to show progress

    Returns:
        PredictionResult with intensities in Sage format

    Example:
        >>> # With explicit collision energies
        >>> result = predict_intensities_for_sage(
        ...     sequences=["PEPTIDEK", "ANOTHERK"],
        ...     charges=[2, 3],
        ...     peptide_indices=[0, 1],
        ...     collision_energies=[30.0, 35.0],
        ... )

        >>> # With m/z-dependent CE model
        >>> from imspy.algorithm.intensity.collision_energy import CollisionEnergyModel
        >>> ce_model = CollisionEnergyModel.default()
        >>> result = predict_intensities_for_sage(
        ...     sequences=["PEPTIDEK", "ANOTHERK"],
        ...     charges=[2, 3],
        ...     peptide_indices=[0, 1],
        ...     collision_energy_model=ce_model,
        ...     precursor_mz=[450.25, 600.33],
        ... )
    """
    # Lazy import to avoid loading TensorFlow at module import time
    from .predictors import Prosit2023TimsTofWrapper

    if ion_kinds is None:
        ion_kinds = DEFAULT_ION_KINDS.copy()

    # Determine collision energies
    if collision_energies is None:
        if collision_energy_model is not None:
            if precursor_mz is None:
                raise ValueError(
                    "precursor_mz is required when using collision_energy_model"
                )
            collision_energies = collision_energy_model.predict_batch(
                np.array(precursor_mz)
            ).tolist()
        else:
            collision_energies = [DEFAULT_COLLISION_ENERGY] * len(sequences)

    # Validate inputs
    assert len(sequences) == len(charges) == len(peptide_indices), \
        "sequences, charges, and peptide_indices must have the same length"
    assert 1 <= max_fragment_charge <= 3, \
        "max_fragment_charge must be between 1 and 3"

    # Initialize Prosit model
    prosit = Prosit2023TimsTofWrapper(verbose=verbose)

    # Predict intensities (don't flatten - keep 3D shape)
    # Note: Prosit expects charges as numpy array for tf.one_hot
    raw_predictions = prosit.predict_intensities(
        sequences=sequences,
        charges=np.array(charges, dtype=np.int32),
        collision_energies=collision_energies,
        batch_size=batch_size,
        flatten=False,
    )

    # Transform each prediction to Sage format
    intensities = []
    for seq, pred in zip(sequences, raw_predictions):
        seq_len = len(remove_unimod_annotation(seq))
        transformed = _prosit_to_sage_format(pred, seq_len, max_fragment_charge)
        intensities.append(transformed)

    return PredictionResult(
        peptide_indices=np.array(peptide_indices, dtype=np.int64),
        charges=np.array(charges, dtype=np.int32),
        intensities=intensities,
        ion_kinds=ion_kinds,
        max_fragment_charge=max_fragment_charge,
    )


def predict_intensities_for_sage_from_request(
    request: PredictionRequest,
    collision_energies: Optional[List[float]] = None,
    collision_energy_model: Optional['CollisionEnergyModel'] = None,
    precursor_mz: Optional[List[float]] = None,
    ion_kinds: List[int] = None,
    max_fragment_charge: int = 2,
    batch_size: int = 2048,
    verbose: bool = True,
) -> PredictionResult:
    """Predict intensities from a PredictionRequest.

    Convenience wrapper around predict_intensities_for_sage.

    Collision energy can be specified in three ways (in order of priority):
    1. Explicit `collision_energies` list (per-peptide)
    2. `collision_energy_model` + `precursor_mz` (m/z-dependent)
    3. Default constant value (35.0 NCE)

    Args:
        request: PredictionRequest containing sequences, charges, and indices
        collision_energies: Explicit per-peptide collision energies (takes priority)
        collision_energy_model: m/z-dependent CE model (requires precursor_mz)
        precursor_mz: Precursor m/z values (required when using collision_energy_model)
        ion_kinds: Ion type codes (default: [1, 4] for B, Y)
        max_fragment_charge: Maximum fragment charge (default: 2)
        batch_size: Batch size for prediction
        verbose: Whether to show progress

    Returns:
        PredictionResult with intensities in Sage format
    """
    return predict_intensities_for_sage(
        sequences=list(request.sequences),
        charges=list(request.charges),
        peptide_indices=list(request.peptide_indices),
        collision_energies=collision_energies,
        collision_energy_model=collision_energy_model,
        precursor_mz=precursor_mz,
        ion_kinds=ion_kinds,
        max_fragment_charge=max_fragment_charge,
        batch_size=batch_size,
        verbose=verbose,
    )


def validate_prediction_result(
    request: PredictionRequest,
    result: PredictionResult,
) -> bool:
    """Validate that a PredictionResult matches its PredictionRequest.

    Args:
        request: The original prediction request
        result: The prediction result to validate

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails with details about the mismatch
    """
    if len(request) != len(result.intensities):
        raise ValueError(
            f"Length mismatch: request has {len(request)}, "
            f"result has {len(result.intensities)}"
        )

    if not np.array_equal(request.peptide_indices, result.peptide_indices):
        raise ValueError("Peptide indices don't match")

    if not np.array_equal(request.charges, result.charges):
        raise ValueError("Charges don't match")

    # Validate intensity shapes
    for i, (seq, intensity) in enumerate(zip(request.sequences, result.intensities)):
        seq_len = len(remove_unimod_annotation(seq))
        expected_shape = (
            len(result.ion_kinds),
            seq_len - 1,
            result.max_fragment_charge
        )
        if intensity.shape != expected_shape:
            raise ValueError(
                f"Intensity shape mismatch at index {i}: "
                f"expected {expected_shape}, got {intensity.shape}"
            )

    return True


def aggregate_predictions_by_peptide(
    result: PredictionResult,
    aggregation: str = 'max_charge',
) -> dict:
    """Aggregate predictions to one per peptide_idx.

    When the same peptide is predicted at multiple charge states, this
    function aggregates them to a single prediction per peptide index.

    Args:
        result: PredictionResult with potentially multiple entries per peptide
        aggregation: Aggregation strategy:
            - 'max_charge': Keep prediction for highest precursor charge
            - 'mean': Average across charges
            - 'first': Keep first occurrence

    Returns:
        Dict mapping peptide_idx to aggregated intensity array
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for idx, charge, intensity in zip(
        result.peptide_indices,
        result.charges,
        result.intensities
    ):
        grouped[idx].append((charge, intensity))

    aggregated = {}
    for idx, predictions in grouped.items():
        if aggregation == 'max_charge':
            max_charge = max(p[0] for p in predictions)
            aggregated[idx] = next(p[1] for p in predictions if p[0] == max_charge)
        elif aggregation == 'mean':
            aggregated[idx] = np.mean([p[1] for p in predictions], axis=0).astype(np.float32)
        elif aggregation == 'first':
            aggregated[idx] = predictions[0][1]
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation}")

    return aggregated


def write_intensity_file(
    path: str,
    predictions: List[NDArray],
    peptide_lengths: List[int],
    max_charge: int = 2,
    ion_kinds: List[int] = None,
) -> None:
    """Write predicted intensities to Sage binary format (.sagi).

    Args:
        path: Output file path (typically .sagi extension)
        predictions: One array per peptide, shape [n_ion_kinds, peptide_len-1, max_charge]
            Values should be normalized intensities (0.0 to 1.0)
        peptide_lengths: Length of each peptide (for validation)
        max_charge: Maximum fragment charge state (default: 2)
        ion_kinds: Ion type codes (default: [1, 4] for B and Y ions)

    Example:
        >>> predictions = [
        ...     np.random.rand(2, 6, 2).astype(np.float32),  # 7 AA peptide
        ...     np.random.rand(2, 9, 2).astype(np.float32),  # 10 AA peptide
        ... ]
        >>> write_intensity_file("predictions.sagi", predictions, [7, 10])
    """
    if ion_kinds is None:
        ion_kinds = DEFAULT_ION_KINDS.copy()

    # Validate predictions
    for i, (pred, pep_len) in enumerate(zip(predictions, peptide_lengths)):
        expected_positions = pep_len - 1
        if pred.shape[1] != expected_positions:
            raise ValueError(
                f"Prediction {i}: expected {expected_positions} positions "
                f"for peptide length {pep_len}, got {pred.shape[1]}"
            )

    with open(path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', SAGI_MAGIC))  # magic "SAGI"
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<Q', len(predictions)))  # peptide_count
        f.write(struct.pack('<B', max_charge))
        f.write(struct.pack('<B', len(ion_kinds)))
        for k in ion_kinds:
            f.write(struct.pack('<B', k))

        # Calculate offsets
        offsets = []
        current_offset = 0
        for pred in predictions:
            offsets.append(current_offset)
            current_offset += pred.size * 4  # f32 = 4 bytes

        # Write offsets
        for off in offsets:
            f.write(struct.pack('<Q', off))

        # Write data
        for pred in predictions:
            # Ensure correct memory layout and dtype
            f.write(pred.astype('<f4').tobytes())


def read_intensity_file(path: str) -> dict:
    """Read predicted intensities from Sage binary format (.sagi).

    Args:
        path: Path to .sagi file

    Returns:
        Dict with keys:
            - peptide_count: int
            - max_charge: int
            - ion_kinds: list of int
            - offsets: numpy.ndarray of uint64
            - data: numpy.ndarray of float32 (raw data buffer)
    """
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != SAGI_MAGIC:
            raise ValueError(f"Invalid magic number: {hex(magic)}, expected {hex(SAGI_MAGIC)}")

        version = struct.unpack('<I', f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")

        peptide_count = struct.unpack('<Q', f.read(8))[0]
        max_charge = struct.unpack('<B', f.read(1))[0]
        ion_kind_count = struct.unpack('<B', f.read(1))[0]
        ion_kinds = [struct.unpack('<B', f.read(1))[0] for _ in range(ion_kind_count)]

        offsets = np.frombuffer(f.read(peptide_count * 8), dtype='<u8')
        data = np.frombuffer(f.read(), dtype='<f4')

        return {
            'peptide_count': peptide_count,
            'max_charge': max_charge,
            'ion_kinds': ion_kinds,
            'offsets': offsets,
            'data': data,
        }


def get_intensity_from_file(
    file_data: dict,
    peptide_idx: int,
    peptide_len: int,
) -> NDArray:
    """Extract intensity array for a specific peptide from loaded file data.

    Args:
        file_data: Dict returned by read_intensity_file()
        peptide_idx: Index of the peptide
        peptide_len: Length of the peptide sequence

    Returns:
        Intensity array with shape [n_ion_kinds, peptide_len-1, max_charge]
    """
    n_ion_kinds = len(file_data['ion_kinds'])
    max_charge = file_data['max_charge']
    n_positions = peptide_len - 1
    expected_size = n_ion_kinds * n_positions * max_charge

    offset = int(file_data['offsets'][peptide_idx])
    start_idx = offset // 4  # Convert byte offset to float32 index

    flat_data = file_data['data'][start_idx:start_idx + expected_size]
    return flat_data.reshape(n_ion_kinds, n_positions, max_charge)


def create_uniform_predictions(
    peptide_lengths: List[int],
    ion_kinds: List[int] = None,
    max_fragment_charge: int = 2,
    value: float = 1.0,
) -> List[NDArray]:
    """Create uniform (constant) predictions for all peptides.

    Useful for creating baseline/fallback predictions or for decoys.

    Args:
        peptide_lengths: Length of each peptide
        ion_kinds: Ion type codes (default: [1, 4] for B, Y)
        max_fragment_charge: Maximum fragment charge
        value: Uniform value to use (default: 1.0)

    Returns:
        List of uniform prediction arrays
    """
    if ion_kinds is None:
        ion_kinds = DEFAULT_ION_KINDS.copy()

    predictions = []
    for pep_len in peptide_lengths:
        n_positions = pep_len - 1
        pred = np.full(
            (len(ion_kinds), n_positions, max_fragment_charge),
            value,
            dtype=np.float32
        )
        predictions.append(pred)

    return predictions


def write_predictions_for_database(
    output_path: str,
    result: PredictionResult,
    num_peptides: int,
    peptide_lengths: List[int],
    aggregation: str = 'max_charge',
    default_value: float = 1.0,
) -> None:
    """Write predictions to .sagi file, handling missing peptides with defaults.

    This is the main entry point for creating a complete .sagi file from
    prediction results. Missing peptides (e.g., decoys) get uniform predictions.

    Args:
        output_path: Path for output .sagi file
        result: PredictionResult from predict_intensities_for_sage()
        num_peptides: Total number of peptides in the database
        peptide_lengths: Length of each peptide in database order
        aggregation: How to aggregate multi-charge predictions
        default_value: Value for uniform predictions on missing peptides

    Example:
        >>> # After prediction
        >>> result = predict_intensities_for_sage(sequences, charges, indices)
        >>> write_predictions_for_database(
        ...     "predictions.sagi",
        ...     result,
        ...     num_peptides=len(all_peptides),
        ...     peptide_lengths=[len(p) for p in all_peptides],
        ... )
    """
    # Aggregate to one prediction per peptide index
    aggregated = aggregate_predictions_by_peptide(result, aggregation)

    # Build complete prediction list in peptide index order
    predictions = []
    for idx in range(num_peptides):
        pep_len = peptide_lengths[idx]
        if idx in aggregated:
            predictions.append(aggregated[idx])
        else:
            # Missing peptide (e.g., decoy) - use uniform prediction
            pred = np.full(
                (len(result.ion_kinds), pep_len - 1, result.max_fragment_charge),
                default_value,
                dtype=np.float32
            )
            predictions.append(pred)

    # Write to file
    write_intensity_file(
        output_path,
        predictions,
        peptide_lengths,
        max_charge=result.max_fragment_charge,
        ion_kinds=result.ion_kinds,
    )


class IntensityPredictionPipeline:
    """High-level pipeline for generating intensity predictions for Sage weighted scoring.

    This class provides a simplified interface for:
    1. Extracting peptide sequences from an IndexedDatabase
    2. Predicting fragment ion intensities using Prosit
    3. Creating a PredictedIntensityStore for weighted scoring
    4. Saving/loading intensity predictions

    Example:
        >>> from sagepy.core import SageSearchConfiguration, EnzymeBuilder
        >>> config = SageSearchConfiguration(fasta=fasta_str, ...)
        >>> indexed_db = config.generate_indexed_database()
        >>>
        >>> pipeline = IntensityPredictionPipeline(indexed_db)
        >>> pipeline.predict_intensities(charges=[2, 3])
        >>> store = pipeline.get_intensity_store()
        >>> pipeline.save_intensity_store("predictions.sagi")

    IMPORTANT: For production use with FDR estimation, you MUST include decoys
    in the predictions (exclude_decoys=False). Using uniform intensities for
    decoys will inflate their scores and break FDR estimation.
    """

    def __init__(self, indexed_db, expected_mods: List[str] = None):
        """Initialize pipeline with an indexed database.

        Args:
            indexed_db: Sage IndexedDatabase containing target and decoy peptides.
            expected_mods: List of UNIMOD strings to include in peptide sequences,
                e.g., ["[UNIMOD:4]", "[UNIMOD:35]"]. If None, uses plain sequences.
        """
        self._indexed_db = indexed_db
        self._expected_mods = expected_mods
        if expected_mods:
            self._peptide_sequences = indexed_db.peptides_as_unimod_string(expected_mods)
        else:
            self._peptide_sequences = indexed_db.peptides_as_string()
        self._peptide_lengths = [len(remove_unimod_annotation(seq)) for seq in self._peptide_sequences]
        self._predictions = None
        self._ion_kinds = DEFAULT_ION_KINDS.copy()
        self._max_fragment_charge = 2
        self._store = None

    @property
    def num_peptides(self) -> int:
        """Total number of peptides in the database."""
        return len(self._peptide_sequences)

    @property
    def num_targets(self) -> int:
        """Number of target peptides."""
        return sum(1 for i in range(self.num_peptides) if not self._indexed_db[i].decoy)

    @property
    def num_decoys(self) -> int:
        """Number of decoy peptides."""
        return sum(1 for i in range(self.num_peptides) if self._indexed_db[i].decoy)

    def predict_intensities(
        self,
        charges: List[int] = [2, 3],
        exclude_decoys: bool = False,
        collision_energy: float = DEFAULT_COLLISION_ENERGY,
        batch_size: int = 2048,
        verbose: bool = True,
    ) -> None:
        """Predict fragment ion intensities for all peptides.

        Args:
            charges: Precursor charge states to predict for each peptide.
            exclude_decoys: If True, decoys get uniform 1.0 intensities (NOT recommended
                for production - breaks FDR estimation).
            collision_energy: Normalized collision energy for predictions.
            batch_size: Batch size for Prosit model.
            verbose: Whether to show progress.
        """
        from tqdm import tqdm

        if exclude_decoys:
            import warnings
            warnings.warn(
                "exclude_decoys=True will produce unfair FDR estimates. "
                "For production use, set exclude_decoys=False.",
                UserWarning
            )

        # Collect peptides to predict with progress bar
        sequences_to_predict = []
        charges_to_predict = []
        indices_to_predict = []

        peptide_iter = enumerate(self._peptide_sequences)
        if verbose:
            peptide_iter = tqdm(
                list(peptide_iter),
                desc="Collecting peptides",
                unit="peptides"
            )

        for idx, seq in peptide_iter:
            is_decoy = self._indexed_db[idx].decoy

            if exclude_decoys and is_decoy:
                continue

            for charge in charges:
                sequences_to_predict.append(seq)
                charges_to_predict.append(charge)
                indices_to_predict.append(idx)

        if verbose:
            n_predictions = len(sequences_to_predict)
            print(f"Predicting intensities for {n_predictions} peptide-charge combinations...")

        # Run predictions
        collision_energies = [collision_energy] * len(sequences_to_predict)

        result = predict_intensities_for_sage(
            sequences=sequences_to_predict,
            charges=charges_to_predict,
            peptide_indices=indices_to_predict,
            collision_energies=collision_energies,
            ion_kinds=self._ion_kinds,
            max_fragment_charge=self._max_fragment_charge,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Aggregate by peptide index
        if verbose:
            print("Aggregating predictions by peptide...")
        aggregated = aggregate_predictions_by_peptide(result, aggregation='max_charge')

        # Build complete prediction list with uniform fallback for missing peptides
        self._predictions = []
        peptide_range = range(self.num_peptides)
        if verbose:
            peptide_range = tqdm(
                peptide_range,
                desc="Building intensity store",
                unit="peptides"
            )

        for idx in peptide_range:
            pep_len = self._peptide_lengths[idx]
            if idx in aggregated:
                self._predictions.append(aggregated[idx])
            else:
                # Missing peptide - use uniform 1.0
                pred = np.full(
                    (len(self._ion_kinds), pep_len - 1, self._max_fragment_charge),
                    1.0,
                    dtype=np.float32
                )
                self._predictions.append(pred)

        # Clear cached store
        self._store = None

    def get_intensity_store(self):
        """Get a PredictedIntensityStore for use with Sage scoring.

        Returns:
            PredictedIntensityStore instance.

        Raises:
            ValueError: If predict_intensities() hasn't been called yet.
        """
        if self._predictions is None:
            raise ValueError("No predictions available. Call predict_intensities() first.")

        if self._store is not None:
            return self._store

        from sagepy.core import PredictedIntensityStore

        # Create store using uniform method as base, then we'll need to write to file
        # and reload since there's no direct constructor from arrays
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            write_intensity_file(
                tmp_path,
                self._predictions,
                self._peptide_lengths,
                max_charge=self._max_fragment_charge,
                ion_kinds=self._ion_kinds,
            )
            self._store = PredictedIntensityStore(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return self._store

    def save_intensity_store(self, path: str) -> None:
        """Save predictions to a .sagi file.

        Args:
            path: Output file path.

        Raises:
            ValueError: If predict_intensities() hasn't been called yet.
        """
        if self._predictions is None:
            raise ValueError("No predictions available. Call predict_intensities() first.")

        write_intensity_file(
            path,
            self._predictions,
            self._peptide_lengths,
            max_charge=self._max_fragment_charge,
            ion_kinds=self._ion_kinds,
        )

    @classmethod
    def load_intensity_store(cls, path: str):
        """Load a PredictedIntensityStore from a .sagi file.

        Args:
            path: Path to .sagi file.

        Returns:
            PredictedIntensityStore instance.
        """
        from sagepy.core import PredictedIntensityStore
        return PredictedIntensityStore(path)


# =============================================================================
# V2 Key-Based Intensity Prediction Support
# =============================================================================

def predict_intensities_v2(
    sequences: List[str],
    charges: List[int],
    collision_energies: Optional[List[float]] = None,
    collision_energy_model: Optional['CollisionEnergyModel'] = None,
    precursor_mz: Optional[List[float]] = None,
    ion_kinds: List[int] = None,
    max_fragment_charge: int = 2,
    batch_size: int = 2048,
    verbose: bool = True,
) -> dict:
    """Predict fragment ion intensities using Prosit, returning V2 key-based format.

    This function generates predictions keyed by (raw_sequence, precursor_charge)
    which is decoupled from database peptide indices. This enables:
    - Database chunking for parallel processing
    - Prediction reuse across different database configurations
    - Flexible parallel workflows

    Collision energy can be specified in three ways (in order of priority):
    1. Explicit `collision_energies` list (per-peptide)
    2. `collision_energy_model` + `precursor_mz` (m/z-dependent)
    3. Default constant value (35.0 NCE)

    Args:
        sequences: Peptide sequences (may include UNIMOD modifications)
        charges: Precursor charge states (parallel to sequences)
        collision_energies: Explicit collision energies per peptide (takes priority)
        collision_energy_model: m/z-dependent CE model (requires precursor_mz)
        precursor_mz: Precursor m/z values (required when using collision_energy_model)
        ion_kinds: Ion type codes (default: [1, 4] for B, Y ions)
        max_fragment_charge: Maximum fragment charge to predict (1-3)
        batch_size: Batch size for Prosit prediction
        verbose: Whether to show progress

    Returns:
        dict[tuple[str, int], np.ndarray]: Map from (raw_sequence, precursor_charge)
            to intensity tensor with shape [n_ion_kinds, seq_len-1, max_fragment_charge]

    Example:
        >>> # With constant CE (default)
        >>> predictions = predict_intensities_v2(
        ...     sequences=["PEPTIDEK", "ANOTHERK"],
        ...     charges=[2, 3],
        ... )
        >>> predictions[("PEPTIDEK", 2)].shape  # (2, 7, 2) for 8 AA peptide

        >>> # With m/z-dependent CE
        >>> from imspy.algorithm.intensity.collision_energy import CollisionEnergyModel
        >>> ce_model = CollisionEnergyModel.default()
        >>> predictions = predict_intensities_v2(
        ...     sequences=["PEPTIDEK", "ANOTHERK"],
        ...     charges=[2, 3],
        ...     collision_energy_model=ce_model,
        ...     precursor_mz=[450.25, 600.33],
        ... )
    """
    from .predictors import Prosit2023TimsTofWrapper

    if ion_kinds is None:
        ion_kinds = DEFAULT_ION_KINDS.copy()

    # Determine collision energies
    if collision_energies is None:
        if collision_energy_model is not None:
            if precursor_mz is None:
                raise ValueError(
                    "precursor_mz is required when using collision_energy_model"
                )
            collision_energies = collision_energy_model.predict_batch(
                np.array(precursor_mz)
            ).tolist()
        else:
            collision_energies = [DEFAULT_COLLISION_ENERGY] * len(sequences)

    assert len(sequences) == len(charges), "sequences and charges must have same length"
    assert 1 <= max_fragment_charge <= 3, "max_fragment_charge must be 1-3"

    # Initialize Prosit model
    prosit = Prosit2023TimsTofWrapper(verbose=verbose)

    # Predict intensities
    raw_predictions = prosit.predict_intensities(
        sequences=sequences,
        charges=np.array(charges, dtype=np.int32),
        collision_energies=collision_energies,
        batch_size=batch_size,
        flatten=False,
    )

    # Transform to V2 format: (modified_sequence, charge) -> intensity array
    # Key is the FULL sequence with UNIMOD annotations (mod-aware)
    predictions = {}
    for seq, charge, pred in zip(sequences, charges, raw_predictions):
        # Use raw sequence length for array sizing, but keep full seq as key
        raw_seq = remove_unimod_annotation(seq)
        seq_len = len(raw_seq)
        transformed = _prosit_to_sage_format(pred, seq_len, max_fragment_charge)
        # Key is the ORIGINAL sequence with UNIMOD (modification-aware)
        predictions[(seq, charge)] = transformed

    return predictions


class IntensityPredictionPipelineV2:
    """V2 pipeline for generating key-based intensity predictions.

    This class uses the V2 format where predictions are keyed by (modified_sequence, charge)
    instead of peptide index. The key includes UNIMOD annotations for modification-aware
    predictions. This decouples predictions from database order and enables:
    - Database chunking for parallel processing
    - Modification-aware intensity predictions
    - Flexible parallel workflows

    Example:
        >>> from sagepy.core import SageSearchConfiguration, EnzymeBuilder
        >>> config = SageSearchConfiguration(fasta=fasta_str, ...)
        >>> indexed_db = config.generate_indexed_database()
        >>>
        >>> pipeline = IntensityPredictionPipelineV2(indexed_db)
        >>> pipeline.predict_intensities(charges=[2, 3])
        >>> store = pipeline.get_intensity_store()
        >>> pipeline.save_intensity_store("predictions_v2.sagi")

    IMPORTANT: For production use with FDR estimation, you MUST include decoys
    in the predictions (exclude_decoys=False). Using uniform intensities for
    decoys will inflate their scores and break FDR estimation.
    """

    def __init__(self, indexed_db, expected_mods: List[str] = None):
        """Initialize pipeline with an indexed database.

        Args:
            indexed_db: Sage IndexedDatabase containing target and decoy peptides.
            expected_mods: List of UNIMOD strings to include in peptide sequences,
                e.g., ["[UNIMOD:4]", "[UNIMOD:35]"]. If None, uses plain sequences.
                For V2 modification-aware predictions, this should include all
                static and variable modifications.
        """
        self._indexed_db = indexed_db
        self._expected_mods = expected_mods
        if expected_mods:
            self._peptide_sequences = indexed_db.peptides_as_unimod_string(expected_mods)
        else:
            self._peptide_sequences = indexed_db.peptides_as_string()
        self._predictions = None
        self._ion_kinds = DEFAULT_ION_KINDS.copy()
        self._max_fragment_charge = 2
        self._store = None

    @property
    def num_peptides(self) -> int:
        """Total number of peptides in the database."""
        return len(self._peptide_sequences)

    @property
    def num_targets(self) -> int:
        """Number of target peptides."""
        return sum(1 for i in range(self.num_peptides) if not self._indexed_db[i].decoy)

    @property
    def num_decoys(self) -> int:
        """Number of decoy peptides."""
        return sum(1 for i in range(self.num_peptides) if self._indexed_db[i].decoy)

    def predict_intensities(
        self,
        charges: List[int] = [2, 3],
        exclude_decoys: bool = False,
        collision_energy: Optional[float] = None,
        collision_energy_model: Optional['CollisionEnergyModel'] = None,
        batch_size: int = 2048,
        verbose: bool = True,
    ) -> None:
        """Predict fragment ion intensities for all unique (sequence, charge) pairs.

        Collision energy can be specified in two ways:
        1. `collision_energy`: Constant CE for all peptides (default: 35.0 NCE)
        2. `collision_energy_model`: m/z-dependent CE model

        Args:
            charges: Precursor charge states to predict for each unique sequence.
            exclude_decoys: If True, decoys get uniform 1.0 intensities (NOT recommended
                for production - breaks FDR estimation).
            collision_energy: Constant collision energy for predictions.
            collision_energy_model: m/z-dependent CE model (computes CE from precursor m/z).
            batch_size: Batch size for Prosit model.
            verbose: Whether to show progress.
        """
        from tqdm import tqdm
        from imspy.data.peptide import PeptideSequence

        if exclude_decoys:
            import warnings
            warnings.warn(
                "exclude_decoys=True will produce unfair FDR estimates. "
                "For production use, set exclude_decoys=False.",
                UserWarning
            )

        # Collect unique (modified_sequence, charge) pairs
        # Key is the FULL sequence with UNIMOD annotations (mod-aware)
        unique_pairs = set()
        decoy_sequences = set()

        peptide_iter = enumerate(self._peptide_sequences)
        if verbose:
            peptide_iter = tqdm(
                list(peptide_iter),
                desc="Collecting unique sequences",
                unit="peptides"
            )

        for idx, seq in peptide_iter:
            is_decoy = self._indexed_db[idx].decoy

            if is_decoy:
                decoy_sequences.add(seq)

            if exclude_decoys and is_decoy:
                continue

            # Use FULL modified sequence as key (mod-aware)
            for charge in charges:
                unique_pairs.add((seq, charge))

        # Convert to lists for prediction
        sequences_to_predict = []
        charges_to_predict = []

        for mod_seq, charge in unique_pairs:
            sequences_to_predict.append(mod_seq)
            charges_to_predict.append(charge)

        if verbose:
            n_unique = len(unique_pairs)
            print(f"Unique (sequence, charge) pairs to predict: {n_unique}")

        # Determine collision energies
        if collision_energy_model is not None:
            # Use fast Rust parallel computation for m/z-dependent CE
            if verbose:
                print(f"Using m/z-dependent CE model: {collision_energy_model}")
            from imspy_connector import py_peptide
            collision_energies = py_peptide.calculate_collision_energies_parallel(
                sequences_to_predict,
                charges_to_predict,
                collision_energy_model.intercept,
                collision_energy_model.slope
            )
        else:
            # Use constant CE
            ce = collision_energy if collision_energy is not None else DEFAULT_COLLISION_ENERGY
            collision_energies = [ce] * len(sequences_to_predict)

        self._predictions = predict_intensities_v2(
            sequences=sequences_to_predict,
            charges=charges_to_predict,
            collision_energies=collision_energies,
            ion_kinds=self._ion_kinds,
            max_fragment_charge=self._max_fragment_charge,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Clear cached store
        self._store = None

        if verbose:
            print(f"Generated {len(self._predictions)} V2 predictions")

    def get_intensity_store(self):
        """Get a V2 PredictedIntensityStore for use with Sage scoring.

        Returns:
            PredictedIntensityStore instance (V2 key-based format).

        Raises:
            ValueError: If predict_intensities() hasn't been called yet.
        """
        if self._predictions is None:
            raise ValueError("No predictions available. Call predict_intensities() first.")

        if self._store is not None:
            return self._store

        from sagepy.core import PredictedIntensityStore

        # Create V2 store directly from predictions dict
        self._store = PredictedIntensityStore.from_predictions_v2(
            predictions=self._predictions,
            max_charge=self._max_fragment_charge,
            ion_kinds=self._ion_kinds,
        )

        return self._store

    def save_intensity_store(self, path: str) -> None:
        """Save V2 predictions to a .sagi file.

        Args:
            path: Output file path.

        Raises:
            ValueError: If predict_intensities() hasn't been called yet.
        """
        if self._predictions is None:
            raise ValueError("No predictions available. Call predict_intensities() first.")

        # Get the store and save it
        store = self.get_intensity_store()
        store.write(path)

    @classmethod
    def load_intensity_store(cls, path: str):
        """Load a PredictedIntensityStore from a .sagi file.

        The loaded store can be either V1 or V2 format - it auto-detects.

        Args:
            path: Path to .sagi file.

        Returns:
            PredictedIntensityStore instance.
        """
        from sagepy.core import PredictedIntensityStore
        return PredictedIntensityStore(path)
