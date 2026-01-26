"""
Fragment Ion Intensity Predictors.

This module provides predictors for fragment ion intensities using Koina
(remote Prosit models) as the primary prediction method.

Classes:
    - Prosit2023TimsTofWrapper: Wrapper for Prosit intensity prediction via Koina
    - IonIntensityPredictor: Abstract base class for intensity predictors
"""

import os
import re
from typing import List, Tuple, Optional

from numpy.typing import NDArray
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from tqdm import tqdm

from imspy_predictors.intensity.utility import (
    post_process_predicted_fragment_spectra,
    reshape_dims,
    seq_to_index,
)

from imspy_core.data import PeptideProductIonSeriesCollection, PeptideSequence


# Lazy import for sagepy (optional dependency, requires imspy-search)
def _get_sagepy_utils():
    """Lazy import of sagepy utilities. Requires imspy-search package."""
    try:
        from sagepy.core.scoring import associate_fragment_ions_with_prosit_predicted_intensities, Psm
        return associate_fragment_ions_with_prosit_predicted_intensities, Psm
    except ImportError:
        raise ImportError(
            "sagepy is required for PSM-based predictions. "
            "Install imspy-search package for this functionality."
        )


# Lazy import for simulation utility (optional, requires imspy-simulation)
def _get_flatten_prosit_array():
    """Lazy import of flatten_prosit_array. Requires imspy-simulation package."""
    try:
        from imspy_simulation.utility import flatten_prosit_array
        return flatten_prosit_array
    except ImportError:
        raise ImportError(
            "flatten_prosit_array requires imspy-simulation package."
        )


def predict_intensities_prosit(
        psm_collection: List,
        calibrate_collision_energy: bool = True,
        verbose: bool = False,
        num_threads: int = -1,
) -> None:
    """
    Predict the fragment ion intensities using Prosit via Koina.

    Note: This function requires sagepy (via imspy-search package).

    Args:
        psm_collection: a list of peptide-spectrum matches (sagepy Psm objects)
        calibrate_collision_energy: whether to calibrate the collision energy
        verbose: whether to print progress
        num_threads: number of threads to use

    Returns:
        None, the fragment ion intensities are stored in the PeptideSpectrumMatch objects
    """
    associate_fragment_ions_with_prosit_predicted_intensities, Psm = _get_sagepy_utils()

    # check if num_threads is -1, if so, use all available threads
    if num_threads == -1:
        num_threads = os.cpu_count()

    # the intensity predictor model
    prosit_model = Prosit2023TimsTofWrapper(verbose=False)

    # sample for collision energy calibration
    sample = list(sorted(psm_collection, key=lambda x: x.hyperscore, reverse=True))[:int(2 ** 11)]

    if calibrate_collision_energy:
        collision_energy_calibration_factor, _ = get_collision_energy_calibration_factor(
            list(filter(lambda match: match.decoy is not True, sample)),
            prosit_model,
            verbose=verbose
        )

    else:
        collision_energy_calibration_factor = 0.0

    for ps in psm_collection:
        ps.collision_energy_calibrated = ps.collision_energy + collision_energy_calibration_factor

    intensity_pred = prosit_model.predict_intensities(
        [p.sequence_modified for p in psm_collection],
        np.array([p.charge for p in psm_collection]),
        [p.collision_energy_calibrated for p in psm_collection],
        batch_size=2048,
        flatten=True,
    )

    psm_collection_intensity = associate_fragment_ions_with_prosit_predicted_intensities(
        psm_collection, intensity_pred, num_threads=num_threads
    )

    # calculate the spectral similarity metrics
    for psm, psm_intensity in tqdm(zip(psm_collection, psm_collection_intensity),
                                                      desc='Calc spectral similarity metrics', ncols=100, disable=not verbose):
        psm.prosit_predicted_intensities = psm_intensity.prosit_predicted_intensities


def get_collision_energy_calibration_factor(
        sample: List,
        model: 'Prosit2023TimsTofWrapper',
        lower: int = -30,
        upper: int = 30,
        verbose: bool = False,
) -> Tuple[float, List[float]]:
    """
    Get the collision energy calibration factor for a given sample.

    Note: This function requires sagepy (via imspy-search package).

    Args:
        sample: a list of PeptideSpectrumMatch objects (sagepy Psm objects)
        model: a Prosit2023TimsTofWrapper object
        lower: lower bound for the search
        upper: upper bound for the search
        verbose: whether to print progress

    Returns:
        Tuple[float, List[float]]: the collision energy calibration factor and the angle similarities
    """
    associate_fragment_ions_with_prosit_predicted_intensities, Psm = _get_sagepy_utils()

    cos_target, cos_decoy = [], []

    if verbose:
        print(f"Searching for collision energy calibration factor between {lower} and {upper} ...")

    for i in tqdm(range(lower, upper), disable=not verbose, desc='calibrating CE', ncols=100):
        I = model.predict_intensities(
            [p.sequence for p in sample],
            np.array([p.charge for p in sample]),
            [p.collision_energy + i for p in sample],
            batch_size=2048,
            flatten=True
        )

        psm_i = associate_fragment_ions_with_prosit_predicted_intensities(sample, I)
        target = list(filter(lambda x: not x.decoy, psm_i))
        decoy = list(filter(lambda x: x.decoy, psm_i))

        cos_target.append((i, np.mean([x.spectral_angle_similarity for x in target])))
        cos_decoy.append((i, np.mean([x.spectral_angle_similarity for x in decoy])))

    calibration_factor, similarities = cos_target[np.argmax([x[1] for x in cos_target])][0], [x[1] for x in cos_target]

    if verbose:
        print(f"Best calibration factor: {calibration_factor}, with similarity: {np.max(np.round(similarities, 2))}")

    return calibration_factor, similarities


def remove_unimod_annotation(sequence: str) -> str:
    """
    Remove the unimod annotation from a peptide sequence.
    Args:
        sequence: a peptide sequence

    Returns:
        str: the peptide sequence without unimod annotation
    """

    pattern = r'\[UNIMOD:\d+\]'
    return re.sub(pattern, '', sequence)


class IonIntensityPredictor(ABC):
    """
    Abstract interface for simulation of fragment ion intensities.
    """

    def __init__(self):
        pass

    @abstractmethod
    def simulate_ion_intensities(self, sequences: list[str], charges: list[int], collision_energies) -> NDArray:
        pass

    @abstractmethod
    def simulate_ion_intensities_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class Prosit2023TimsTofWrapper(IonIntensityPredictor):
    """
    Wrapper for the Prosit 2023 TIMS-TOF predictor using Koina.

    This wrapper uses the Koina API to access Prosit models remotely,
    eliminating the need for local TensorFlow installation.

    Args:
        verbose: Whether to print progress during prediction
        model_name: Name identifier for the predictor
        use_koina: If True (default), use Koina API. If False, try local model.
    """

    KOINA_MODEL_NAME = "Prosit_2023_intensity_timsTOF"

    def __init__(
        self,
        verbose: bool = True,
        model_name: str = 'deep_ion_intensity_predictor',
        use_koina: bool = True,
    ):
        super().__init__()

        self.verbose = verbose
        self.model_name = model_name
        self.use_koina = use_koina
        self._koina_model = None

    def _get_koina_model(self):
        """Lazy load Koina model."""
        if self._koina_model is None:
            from imspy_predictors.koina_models import ModelFromKoina
            self._koina_model = ModelFromKoina(model_name=self.KOINA_MODEL_NAME)
        return self._koina_model

    def simulate_ion_intensities_pandas_batched(
            self,
            data: pd.DataFrame,
            batch_size_tf_ds: int = 1024,
            batch_size: int = int(4e5),
            divide_collision_energy_by: float = 1e2,
    ) -> pd.DataFrame:

        tables = []

        batch_counter = 0
        num_batches = max(1, int(np.ceil(len(data) / batch_size))) if len(data) > 0 else 0
        for batch_indices in tqdm(np.array_split(data.index, num_batches) if num_batches > 0 else [],
                                  total=num_batches,
                                  desc='Simulating intensities', ncols=100, disable=not self.verbose):

            batch = data.loc[batch_indices].reset_index(drop=True)
            data_pred = self.simulate_ion_intensities_pandas(batch, batch_size=batch_size_tf_ds,
                                                             divide_collision_energy_by=divide_collision_energy_by)

            tables.append(data_pred)
            batch_counter += 1

        return pd.concat(tables)

    def simulate_ion_intensities_pandas(
            self,
            data: pd.DataFrame,
            batch_size: int = 512,
            divide_collision_energy_by: float = 1e2,
            verbose: bool = False,
            flatten: bool = False,
    ) -> pd.DataFrame:
        flatten_prosit_array = _get_flatten_prosit_array()

        if verbose:
            print("Generating Prosit compatible input data...")

        data['collision_energy'] = data.apply(lambda r: r.collision_energy / divide_collision_energy_by, axis=1)
        data['sequence_length'] = data.apply(lambda r: len(remove_unimod_annotation(r.sequence)), axis=1)

        # Use Koina for prediction
        I_pred = self._predict_with_koina(
            data.sequence.tolist(),
            data.charge.tolist(),
            data.collision_energy.tolist(),
            batch_size=batch_size,
        )

        data['intensity_raw'] = list(I_pred)
        I_pred = np.squeeze(reshape_dims(post_process_predicted_fragment_spectra(data)))

        if flatten:
            I_pred = np.vstack([flatten_prosit_array(r) for r in I_pred])

        data['intensity'] = list(I_pred)

        return data

    def _predict_with_koina(
            self,
            sequences: List[str],
            charges: List[int],
            collision_energies: List[float],
            batch_size: int = 512,
    ) -> NDArray:
        """Predict intensities using Koina API."""
        koina_model = self._get_koina_model()

        # Prepare input DataFrame for Koina
        input_df = pd.DataFrame({
            'peptide_sequences': sequences,
            'precursor_charges': charges,
            'collision_energies': collision_energies,
            'instrument_types': ['TIMSTOF'] * len(sequences),
        })

        # Get predictions from Koina
        result = koina_model.predict(input_df)

        # Extract intensities from result
        # Koina returns a DataFrame with intensities column
        if 'intensities' in result.columns:
            intensities = np.vstack(result['intensities'].values)
        else:
            # Fallback: try to extract from first numeric column
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                intensities = result[numeric_cols].values
            else:
                raise ValueError("Could not extract intensities from Koina response")

        return intensities

    def predict_intensities(
            self,
            sequences: List[str],
            charges: List[int],
            collision_energies: List[float],
            divide_collision_energy_by: float = 1e2,
            batch_size: int = 512,
            flatten: bool = False,
    ) -> List[NDArray]:
        flatten_prosit_array = _get_flatten_prosit_array()

        sequences_unmod = [remove_unimod_annotation(s) for s in sequences]
        sequence_length = [len(s) for s in sequences_unmod]
        collision_energies_norm = [ce / divide_collision_energy_by for ce in collision_energies]

        # Use Koina for prediction
        I_pred = self._predict_with_koina(
            sequences,
            list(charges) if isinstance(charges, np.ndarray) else charges,
            collision_energies_norm,
            batch_size=batch_size,
        )

        I_pred = list(I_pred)
        I_pred = np.squeeze(reshape_dims(post_process_predicted_fragment_spectra(pd.DataFrame({
            'sequence': sequences,
            'charge': charges,
            'collision_energy': collision_energies,
            'sequence_length': sequence_length,
            'intensity_raw': I_pred,
        }))))

        if flatten:
            I_pred = np.vstack([flatten_prosit_array(r) for r in I_pred])

        return I_pred

    def simulate_ion_intensities(
            self,
            sequences: List[str],
            charges: List[int],
            collision_energies: List[float],
            divide_collision_energy_by: float = 1e2,
            batch_size: int = 512,
    ) -> List[PeptideProductIonSeriesCollection]:
        flatten_prosit_array = _get_flatten_prosit_array()

        sequences_unmod = [remove_unimod_annotation(s) for s in sequences]
        sequence_length = [len(s) for s in sequences_unmod]
        collision_energies_norm = [ce / divide_collision_energy_by for ce in collision_energies]

        # Use Koina for prediction
        I_pred = self._predict_with_koina(
            sequences_unmod,
            list(charges) if isinstance(charges, np.ndarray) else charges,
            collision_energies_norm,
            batch_size=batch_size,
        )

        I_pred = list(I_pred)
        I_pred = np.squeeze(reshape_dims(post_process_predicted_fragment_spectra(pd.DataFrame({
            'sequence': sequences,
            'charge': charges,
            'collision_energy': collision_energies,
            'sequence_length': sequence_length,
            'intensity_raw': I_pred,
        }))))

        intensities = np.vstack([flatten_prosit_array(r) for r in I_pred])
        peptide_sequences = [PeptideSequence(s) for s in sequences]
        ion_collections = []

        for peptide, charge, intensity in zip(peptide_sequences, charges, intensities):
            series = peptide.associate_fragment_ion_series_with_prosit_intensities(
                intensity,
                charge
            )
            ion_collections.append(series)

        return ion_collections


def predict_fragment_intensities_with_koina(
        model_name: str,
        data: pd.DataFrame,
        seq_col: str = 'sequence',
        charge_col: str = 'charge',
        ce_col: str = 'collision_energy',
        verbose: bool = False,
) -> pd.DataFrame:
    """
    Predict fragment ion intensities with Koina.
    Args:
        model_name: Model name for Koina fragment intensity prediction.
        data: DataFrame with peptide sequences.
        seq_col: Column name for peptide sequences in data.
        charge_col: Column name for precursor charges in data.
        ce_col: Column name for collision energies in data.
        verbose: Verbosity.

    Returns:
        pd.DataFrame: DataFrame with with columns ['peptide_sequences', 'precursor_charges', 'collision_energies',
            'instrument_types', 'intensities', 'mz', 'annotation'],
            last three are output columns.
            intensities are min-max normalized by base intensity,
            annotation is in format of b'b5+2'
    """
    from imspy_predictors.koina_models import ModelFromKoina
    intensity_model = ModelFromKoina(model_name=model_name)
    inputs = data.copy()
    if 'instrument_types' not in inputs.columns:
        inputs['instrument_types'] = 'TIMSTOF'
    inputs.rename(columns={'peptide_sequences': seq_col,
                           'precursor_charges': charge_col,
                           'collision_energies': ce_col}, inplace=True)
    intensity = intensity_model.predict(inputs)

    if verbose:
        print(f"[DEBUG] Koina model {model_name} predicted fragment intensity for {len(intensity)} peptides. Columns: {intensity.columns}")

    return intensity


# =============================================================================
# Local PyTorch Intensity Predictor (PROSPECT fine-tuned)
# =============================================================================

def get_model_path(relative_path: str):
    """Get path to model file in pretrained or checkpoints directory."""
    from pathlib import Path
    package_dir = Path(__file__).parent.parent

    # Try pretrained directory first (src/imspy_predictors/pretrained)
    pretrained_path = package_dir / 'pretrained' / relative_path
    if pretrained_path.exists():
        return pretrained_path

    # Also try without subdirectory prefix (e.g., 'intensity/best_model.pt' instead of 'timstof_intensity/best_model.pt')
    simple_name = relative_path.replace('timstof_', '')
    pretrained_simple = package_dir / 'pretrained' / simple_name
    if pretrained_simple.exists():
        return pretrained_simple

    # Try checkpoints directory (legacy)
    model_path = package_dir / 'checkpoints' / relative_path
    if model_path.exists():
        return model_path

    # Try package root checkpoints (packages/imspy-predictors/checkpoints)
    package_root = package_dir.parent.parent
    root_path = package_root / 'checkpoints' / relative_path
    if root_path.exists():
        return root_path

    # Return pretrained path (preferred) even if not found
    return pretrained_path


def load_deep_intensity_predictor(map_location: Optional[str] = None):
    """
    Load a pretrained intensity predictor model.

    This loads the PROSPECT fine-tuned transformer model trained on
    timsTOF MS2 data.

    Args:
        map_location: Device to load model to ('cpu', 'cuda', etc.)

    Returns:
        Loaded UnifiedPeptideModel with intensity head
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for local intensity prediction. Install with: pip install torch")

    from imspy_predictors.models import UnifiedPeptideModel

    model_path = get_model_path('timstof_intensity/best_model.pt')
    if not model_path.exists():
        raise FileNotFoundError(
            f"Intensity model not found at {model_path}. "
            "Please ensure the model checkpoint is installed."
        )

    model = UnifiedPeptideModel.from_pretrained(
        str(model_path),
        tasks=['intensity'],
        map_location=map_location
    )
    model.eval()
    return model


class DeepPeptideIntensityPredictor(IonIntensityPredictor):
    """
    High-level wrapper for local intensity prediction using PyTorch.

    Uses the PROSPECT fine-tuned transformer model for timsTOF MS2 prediction.

    Args:
        model: Pre-loaded model (optional, will load default if None)
        tokenizer: Tokenizer for sequences (optional, will load default if None)
        verbose: Whether to print progress
        device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)

    Example:
        >>> predictor = DeepPeptideIntensityPredictor()
        >>> intensities = predictor.predict_intensities(
        ...     sequences=["PEPTIDE", "SEQUENCE"],
        ...     charges=[2, 3],
        ...     collision_energies=[30.0, 30.0]
        ... )
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        verbose: bool = True,
        device: Optional[str] = None,
    ):
        try:
            import torch
            self._torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for DeepPeptideIntensityPredictor. "
                "Install with: pip install torch"
            )

        super().__init__()

        # Load tokenizer
        if tokenizer is None:
            from imspy_predictors.utilities.tokenizers import ProformaTokenizer
            self.tokenizer = ProformaTokenizer.with_defaults()
        else:
            self.tokenizer = tokenizer

        # Load model
        if model is None:
            self.model = load_deep_intensity_predictor()
        else:
            self.model = model

        self.verbose = verbose

        # Set device
        if device is None:
            if self._torch.cuda.is_available():
                self._device = 'cuda'
            elif hasattr(self._torch.backends, 'mps') and self._torch.backends.mps.is_available():
                self._device = 'mps'
            else:
                self._device = 'cpu'
        else:
            self._device = device

        self.model = self.model.to(self._device)
        self.model.eval()

    def _preprocess(
        self,
        sequences: List[str],
        charges: List[int],
        collision_energies: List[float],
    ):
        """Prepare inputs for the model."""
        # Get max_seq_len from model
        max_seq_len = getattr(self.model, 'max_seq_len', 50)
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'max_seq_len'):
            max_seq_len = self.model.encoder.max_seq_len

        # Tokenize sequences
        result = self.tokenizer(sequences, padding=True, return_tensors='pt')
        tokens = result['input_ids']

        # Pad to max length
        if tokens.shape[1] < max_seq_len:
            padding = self._torch.zeros(
                tokens.shape[0], max_seq_len - tokens.shape[1], dtype=self._torch.long
            )
            tokens = self._torch.cat([tokens, padding], dim=1)
        elif tokens.shape[1] > max_seq_len:
            tokens = tokens[:, :max_seq_len]

        # Charge as integer tensor (model handles one-hot internally)
        charge_tensor = self._torch.tensor(charges, dtype=self._torch.long)

        # Collision energies (already normalized in calling function)
        ce_tensor = self._torch.tensor(collision_energies, dtype=self._torch.float32)

        return tokens, charge_tensor, ce_tensor

    def simulate_ion_intensities_pandas_batched(
        self,
        data: pd.DataFrame,
        batch_size_tf_ds: int = 1024,
        batch_size: int = int(4e5),
        divide_collision_energy_by: float = 1e2,
    ) -> pd.DataFrame:
        """
        Predict intensities for a DataFrame in batches.

        Args:
            data: DataFrame with 'sequence', 'charge', 'collision_energy' columns
            batch_size_tf_ds: Batch size for model inference
            batch_size: Batch size for data chunking
            divide_collision_energy_by: CE normalization factor (default 100)

        Returns:
            DataFrame with 'intensity' column containing (seq_len-1, 2, 3) arrays
        """
        tables = []

        num_batches = max(1, int(np.ceil(len(data) / batch_size))) if len(data) > 0 else 0
        for batch_indices in tqdm(
            np.array_split(data.index, num_batches) if num_batches > 0 else [],
            total=num_batches,
            desc='Simulating intensities (local)',
            ncols=100,
            disable=not self.verbose
        ):
            batch = data.loc[batch_indices].reset_index(drop=True)
            data_pred = self.simulate_ion_intensities_pandas(
                batch,
                batch_size=batch_size_tf_ds,
                divide_collision_energy_by=divide_collision_energy_by
            )
            tables.append(data_pred)

        return pd.concat(tables)

    def simulate_ion_intensities_pandas(
        self,
        data: pd.DataFrame,
        batch_size: int = 512,
        divide_collision_energy_by: float = 1e2,
        verbose: bool = False,
        flatten: bool = False,
    ) -> pd.DataFrame:
        """
        Predict intensities for a DataFrame.

        Args:
            data: DataFrame with 'sequence', 'charge', 'collision_energy' columns
            batch_size: Batch size for model inference
            divide_collision_energy_by: CE normalization factor
            verbose: Print progress
            flatten: Flatten output to 174-dim vectors

        Returns:
            DataFrame with 'intensity' column
        """
        sequences = data['sequence'].tolist()
        charges = data['charge'].tolist()
        collision_energies = (data['collision_energy'] / divide_collision_energy_by).tolist()

        intensities = self._predict_batch(
            sequences, charges, collision_energies, batch_size=batch_size
        )

        # Post-process to Prosit format (seq_len-1, 2, 3)
        data = data.copy()
        # Normalize collision_energy like old Prosit code - required for Rust frame builder key matching
        data['collision_energy'] = data['collision_energy'] / divide_collision_energy_by
        data['sequence_length'] = data['sequence'].apply(lambda s: len(remove_unimod_annotation(s)))
        data['intensity_raw'] = list(intensities)

        # Process through Prosit post-processing pipeline
        processed = post_process_predicted_fragment_spectra(data)

        # Reshape from flat (174,) to (29, 6) then to (29, 2, 3)
        # Layout: (29, 6) where 6 = [y+1, y+2, y+3, b+1, b+2, b+3]
        # Target: (29, 2, 3) where dim1 = [y, b] and dim2 = [+1, +2, +3]
        #
        # Prosit flat format groups by position:
        #   [y1+1, y1+2, y1+3, b1+1, b1+2, b1+3, y2+1, y2+2, ...]
        #
        # After reshape_dims to (29, 6), each row is:
        #   [y+1, y+2, y+3, b+1, b+2, b+3] for that position
        #
        # We need (29, 2, 3) where:
        #   arr[pos, 0, :] = [y+1, y+2, y+3]  (y ions)
        #   arr[pos, 1, :] = [b+1, b+2, b+3]  (b ions)
        #
        # C-order (default) reshape achieves this:
        #   6 values -> arr[0,:] gets first 3, arr[1,:] gets next 3
        I_pred = reshape_dims(processed)  # (batch, 29, 6)
        if I_pred.ndim == 2:
            # Single sample: (29, 6) -> (29, 2, 3)
            I_pred = I_pred.reshape(29, 2, 3)  # C-order (default)
            I_pred = [I_pred]
        else:
            # Batch: (batch, 29, 6) -> list of (29, 2, 3)
            I_pred = [arr.reshape(29, 2, 3) for arr in I_pred]

        if flatten:
            flatten_prosit_array = _get_flatten_prosit_array()
            # Convert to list of 1D arrays for DataFrame compatibility
            I_pred = [flatten_prosit_array(r) for r in I_pred]

        data['intensity'] = I_pred

        return data

    def _predict_batch(
        self,
        sequences: List[str],
        charges: List[int],
        collision_energies: List[float],
        batch_size: int = 512,
    ) -> np.ndarray:
        """Run batched prediction."""
        all_intensities = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            batch_charges = charges[i:i + batch_size]
            batch_ces = collision_energies[i:i + batch_size]

            tokens, charge_onehot, ce_tensor = self._preprocess(
                batch_seqs, batch_charges, batch_ces
            )

            tokens = tokens.to(self._device)
            charge_onehot = charge_onehot.to(self._device)
            ce_tensor = ce_tensor.to(self._device)

            with self._torch.no_grad():
                outputs = self.model(
                    tokens,
                    charge=charge_onehot,
                    collision_energy=ce_tensor,
                )

            # Extract intensity output
            if 'intensity' in outputs:
                intensities = outputs['intensity'].cpu().numpy()
            else:
                # Fallback: use first output
                intensities = list(outputs.values())[0].cpu().numpy()

            all_intensities.append(intensities)

        return np.vstack(all_intensities)

    def predict_intensities(
        self,
        sequences: List[str],
        charges: List[int],
        collision_energies: List[float],
        divide_collision_energy_by: float = 1e2,
        batch_size: int = 512,
        flatten: bool = False,
    ) -> List[NDArray]:
        """
        Predict fragment intensities.

        Args:
            sequences: Peptide sequences (with UNIMOD modifications)
            charges: Precursor charges
            collision_energies: Collision energies
            divide_collision_energy_by: CE normalization factor
            batch_size: Batch size
            flatten: Flatten to 174-dim vectors

        Returns:
            List of intensity arrays
        """
        flatten_prosit_array = _get_flatten_prosit_array()

        sequences_unmod = [remove_unimod_annotation(s) for s in sequences]
        sequence_length = [len(s) for s in sequences_unmod]
        collision_energies_norm = [ce / divide_collision_energy_by for ce in collision_energies]

        I_pred = self._predict_batch(
            sequences,
            list(charges) if isinstance(charges, np.ndarray) else charges,
            collision_energies_norm,
            batch_size=batch_size,
        )

        I_pred = list(I_pred)
        processed = post_process_predicted_fragment_spectra(pd.DataFrame({
            'sequence': sequences,
            'charge': charges,
            'collision_energy': collision_energies,
            'sequence_length': sequence_length,
            'intensity_raw': I_pred,
        }))

        # Reshape to (batch, 29, 6) then to (29, 2, 3) per sample
        I_pred = reshape_dims(processed)  # (batch, 29, 6) or (29, 6)
        if I_pred.ndim == 2:
            # Single sample: (29, 6) -> (29, 2, 3)
            I_pred = [I_pred.reshape(29, 2, 3)]
        else:
            # Batch: (batch, 29, 6) -> list of (29, 2, 3)
            I_pred = [arr.reshape(29, 2, 3) for arr in I_pred]

        if flatten:
            I_pred = np.vstack([flatten_prosit_array(r) for r in I_pred])

        return I_pred

    def simulate_ion_intensities(
        self,
        sequences: List[str],
        charges: List[int],
        collision_energies: List[float],
        divide_collision_energy_by: float = 1e2,
        batch_size: int = 512,
    ) -> List:
        """
        Predict fragment intensities and return as PeptideProductIonSeriesCollection.

        Args:
            sequences: Peptide sequences (with UNIMOD modifications)
            charges: Precursor charges
            collision_energies: Collision energies
            divide_collision_energy_by: CE normalization factor
            batch_size: Batch size

        Returns:
            List of PeptideProductIonSeriesCollection objects
        """
        flatten_prosit_array = _get_flatten_prosit_array()

        sequences_unmod = [remove_unimod_annotation(s) for s in sequences]
        sequence_length = [len(s) for s in sequences_unmod]
        collision_energies_norm = [ce / divide_collision_energy_by for ce in collision_energies]

        I_pred = self._predict_batch(
            sequences,
            list(charges) if isinstance(charges, np.ndarray) else charges,
            collision_energies_norm,
            batch_size=batch_size,
        )

        I_pred = list(I_pred)
        processed = post_process_predicted_fragment_spectra(pd.DataFrame({
            'sequence': sequences,
            'charge': charges,
            'collision_energy': collision_energies,
            'sequence_length': sequence_length,
            'intensity_raw': I_pred,
        }))

        # Reshape to (batch, 29, 6) then to (29, 2, 3) per sample
        I_pred = reshape_dims(processed)  # (batch, 29, 6) or (29, 6)
        if I_pred.ndim == 2:
            # Single sample: (29, 6) -> (29, 2, 3)
            I_pred = [I_pred.reshape(29, 2, 3)]
        else:
            # Batch: (batch, 29, 6) -> list of (29, 2, 3)
            I_pred = [arr.reshape(29, 2, 3) for arr in I_pred]

        intensities = np.vstack([flatten_prosit_array(r) for r in I_pred])
        peptide_sequences = [PeptideSequence(s) for s in sequences]
        ion_collections = []

        for peptide, charge, intensity in zip(peptide_sequences, charges, intensities):
            series = peptide.associate_fragment_ion_series_with_prosit_intensities(
                intensity,
                charge
            )
            ion_collections.append(series)

        return ion_collections
