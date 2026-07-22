"""
Fragment Ion Intensity Predictors.

This module provides predictors for fragment ion intensities using Koina
(remote Prosit models) as the primary prediction method.

Classes:
    - Prosit2023TimsTofWrapper: Wrapper for Prosit intensity prediction via Koina
    - IonIntensityPredictor: Abstract base class for intensity predictors
"""

import os
from typing import List, Tuple, Optional

from numpy.typing import NDArray
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from tqdm import tqdm

from imspy_predictors.intensity.utility import (
    post_process_predicted_fragment_spectra,
    reshape_dims,
)

from imspy_core.data import PeptideProductIonSeriesCollection, PeptideSequence
from imspy_core.utility import remove_unimod_annotation
from imspy_predictors.utility import InMemoryCheckpoint

# Lazy imports for optional dependencies
from imspy_predictors.lazy_imports import (
    get_sagepy_fragment_utils,
    get_simulation_flatten_prosit,
)

# NOTE: `masked_spectral_distance` (from imspy_predictors.losses, which imports
# torch unconditionally) is intentionally NOT imported here. Importing this module
# must stay torch-free so the Koina (remote) intensity path — Prosit2023TimsTofWrapper,
# predict_fragment_intensities_with_koina — loads without torch. The loss is used
# only in the training loops (fine_tune_model), where it is imported at point-of-use
# alongside the already-deferred torch DataLoader import.


# -----------------------------------------------------------------------------
# Fragment label helpers (used by DeepPeptideIntensityPredictor.fine_tune_psms)
# -----------------------------------------------------------------------------

def _ion_to_text(ion) -> str:
    """Coerce sagepy's IonType enum / string to ``"b"`` or ``"y"``."""
    text = str(ion).lower()
    if text in ("b", "y"):
        return text
    if text in ("iontype(b)", "iontype(y)"):
        return text[-2]
    if text.endswith(" b") or text.endswith(".b"):
        return "b"
    if text.endswith(" y") or text.endswith(".y"):
        return "y"
    return text[-1:]


def observed_fragments_to_intensity_target(
    sequence: str,
    precursor_charge: int,
    fragments,
) -> np.ndarray:
    """Build a native Prosit-layout target vector from observed Sage fragments.

    Output layout is ordinal-major:
    ``[y1+1, y1+2, y1+3, b1+1, b1+2, b1+3, y2+1, ...]``. Impossible
    fragments are marked ``-1`` so ``masked_spectral_distance`` ignores
    them; valid but unmatched fragments remain zero (so the model learns
    "this ion is possible but wasn't observed").
    """
    sequence_length = len(remove_unimod_annotation(sequence))
    target = np.zeros(174, dtype=np.float32)

    target_3d = target.reshape(29, 6)
    max_frag_pos = max(sequence_length - 1, 0)
    if max_frag_pos < 29:
        target_3d[max_frag_pos:, :] = -1.0
    for ion_charge in range(1, 4):
        if ion_charge > int(precursor_charge):
            target_3d[:, ion_charge - 1] = -1.0
            target_3d[:, ion_charge + 2] = -1.0

    intensities = np.asarray(fragments.intensities, dtype=np.float32)
    if intensities.size == 0:
        return target
    max_intensity = float(np.max(intensities))
    if max_intensity <= 0:
        return target

    for ion_type, ordinal, charge, intensity in zip(
        fragments.ion_types,
        fragments.fragment_ordinals,
        fragments.charges,
        intensities,
    ):
        ion = _ion_to_text(ion_type)
        ordinal = int(ordinal)
        charge = int(charge)
        if ion not in ("b", "y"):
            continue
        if not (1 <= ordinal <= 29 and 1 <= charge <= 3):
            continue
        slot = (ordinal - 1) * 6 + (charge - 1 if ion == "y" else 3 + charge - 1)
        if target[slot] >= 0:
            target[slot] = float(intensity) / max_intensity
    return target


def _ion_to_text(ion) -> str:
    text = str(ion).lower()
    if text in ("b", "y"):
        return text
    if text in ("iontype(b)", "iontype(y)"):
        return text[-2]
    if text.endswith(" b") or text.endswith(".b"):
        return "b"
    if text.endswith(" y") or text.endswith(".y"):
        return "y"
    return text[-1:]


def observed_fragments_to_intensity_target(
    sequence: str,
    precursor_charge: int,
    fragments,
) -> np.ndarray:
    """Build a native Prosit-layout target vector from observed Sage fragments.

    Output layout is ordinal-major:
    [y1+1, y1+2, y1+3, b1+1, b1+2, b1+3, y2+1, ...].
    Impossible fragments are marked -1 for masked spectral loss; valid but
    unmatched fragments remain zero.
    """
    sequence_length = len(remove_unimod_annotation(sequence))
    target = np.zeros(174, dtype=np.float32)

    target_3d = target.reshape(29, 6)
    max_frag_pos = max(sequence_length - 1, 0)
    if max_frag_pos < 29:
        target_3d[max_frag_pos:, :] = -1.0
    for ion_charge in range(1, 4):
        if ion_charge > int(precursor_charge):
            target_3d[:, ion_charge - 1] = -1.0
            target_3d[:, ion_charge + 2] = -1.0

    intensities = np.asarray(fragments.intensities, dtype=np.float32)
    if intensities.size == 0:
        return target
    max_intensity = float(np.max(intensities))
    if max_intensity <= 0:
        return target

    for ion_type, ordinal, charge, intensity in zip(
        fragments.ion_types,
        fragments.fragment_ordinals,
        fragments.charges,
        intensities,
    ):
        ion = _ion_to_text(ion_type)
        ordinal = int(ordinal)
        charge = int(charge)
        if ion not in ("b", "y"):
            continue
        if not (1 <= ordinal <= 29 and 1 <= charge <= 3):
            continue
        slot = (ordinal - 1) * 6 + (charge - 1 if ion == "y" else 3 + charge - 1)
        if target[slot] >= 0:
            target[slot] = float(intensity) / max_intensity
    return target


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
    associate_fragment_ions_with_prosit_predicted_intensities, Psm = get_sagepy_fragment_utils()

    # check if num_threads is -1, if so, use all available threads
    if num_threads == -1:
        num_threads = os.cpu_count()

    # the intensity predictor model
    prosit_model = Prosit2023TimsTofWrapper(verbose=False)

    # Calibrate one absolute NCE for the run (calibrate_nce drops decoys and
    # caps the sample internally). The model conditions on a per-run NCE, so
    # every PSM is predicted at that single value -- not observed CE + offset.
    if calibrate_collision_energy:
        calibration = calibrate_nce(prosit_model, psm_collection, verbose=verbose)
        calibrated_nce = float(calibration["best_nce"])
        for ps in psm_collection:
            ps.collision_energy_calibrated = calibrated_nce
    else:
        for ps in psm_collection:
            ps.collision_energy_calibrated = ps.collision_energy

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


def calibrate_nce(
        model,
        psms: List,
        nce_grid: Optional[List[int]] = None,
        per_charge: bool = False,
        max_sample: int = 2048,
        verbose: bool = False,
) -> dict:
    """Calibrate the absolute normalized collision energy (NCE) for a run.

    The intensity model conditions on a per-run NCE scalar -- it was fine-tuned
    on ``collision_energy_aligned_normed`` (domain ~7-43). Calibration sweeps
    absolute NCE values, predicts every PSM at each, and returns the value that
    maximizes the mean predicted-vs-observed spectral angle.

    This is an *absolute* sweep, NOT an offset on the observed collision energy:
    the observed CE (e.g. the Bruker mobility-ramped value) is a different
    physical quantity and must not be added to. One NCE is returned per run.

    Note: This function requires sagepy (via the imspy-search package).

    Args:
        model: an intensity predictor exposing ``predict_intensities(sequences,
            charges, collision_energies, batch_size=, flatten=True)`` -- e.g.
            Prosit2023TimsTofWrapper or DeepPeptideIntensityPredictor.
        psms: sagepy Psm objects carrying observed fragments. Decoys are dropped.
        nce_grid: absolute NCE values to sweep (default ``range(15, 51)``).
        per_charge: also report the best NCE separately per precursor charge.
        max_sample: cap the calibration sample; if exceeded, the highest-scoring
            PSMs (by hyperscore) are kept.
        verbose: whether to print progress.

    Returns:
        dict: ``{best_nce, curve: [(nce, mean_spectral_angle), ...], n_psms}``;
        also ``per_charge: {charge: best_nce}`` when ``per_charge`` is True.
    """
    associate_fragment_ions_with_prosit_predicted_intensities, _ = get_sagepy_fragment_utils()

    if nce_grid is None:
        nce_grid = list(range(15, 51))
    nce_grid = [int(x) for x in nce_grid]

    targets = [p for p in psms if not getattr(p, "decoy", False)]
    if not targets:
        raise ValueError("calibrate_nce: no target PSMs to calibrate on")
    if max_sample and len(targets) > max_sample:
        targets = sorted(targets, key=lambda p: getattr(p, "hyperscore", 0.0),
                         reverse=True)[:max_sample]

    def _sweep(sample):
        # sequence_modified (not sequence) -- the prediction is mod-aware.
        seqs = [p.sequence_modified for p in sample]
        chgs = np.array([p.charge for p in sample])
        curve = []
        for nce in tqdm(nce_grid, disable=not verbose, desc="calibrating NCE", ncols=100):
            intensities = model.predict_intensities(
                seqs, chgs, [float(nce)] * len(sample),
                batch_size=2048, flatten=True,
            )
            scored = associate_fragment_ions_with_prosit_predicted_intensities(
                sample, intensities
            )
            sa = float(np.mean([x.spectral_angle_similarity for x in scored]))
            curve.append((int(nce), sa))
        best = curve[int(np.argmax([s for _, s in curve]))][0]
        return int(best), curve

    best_nce, curve = _sweep(targets)
    result = {"best_nce": best_nce, "curve": curve, "n_psms": len(targets)}

    if per_charge:
        pc = {}
        for z in sorted({int(p.charge) for p in targets}):
            sub = [p for p in targets if int(p.charge) == z]
            if len(sub) < 100:
                continue
            pc[z], _ = _sweep(sub)
        result["per_charge"] = pc

    if verbose:
        print(f"calibrate_nce: best NCE = {best_nce} "
              f"(mean spectral angle {max(s for _, s in curve):.4f}, "
              f"n = {len(targets)})")
    return result


def get_collision_energy_calibration_factor(
        sample: List,
        model: 'Prosit2023TimsTofWrapper',
        verbose: bool = False,
) -> Tuple[float, List[float]]:
    """DEPRECATED -- use :func:`calibrate_nce`.

    BEHAVIOR CHANGED. This previously returned an *offset* added to each PSM's
    ``collision_energy`` and -- a bug -- calibrated on the unmodified sequence.
    It now delegates to :func:`calibrate_nce` and returns the **absolute** best
    NCE. Callers must set ``collision_energy_calibrated = best_nce`` directly,
    NOT ``collision_energy + factor``.

    Returns:
        Tuple[float, List[float]]: the absolute best NCE and the per-NCE mean
        spectral angles.
    """
    calibration = calibrate_nce(model, sample, verbose=verbose)
    return float(calibration["best_nce"]), [sa for _, sa in calibration["curve"]]


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
        flatten_prosit_array = get_simulation_flatten_prosit()

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
        flatten_prosit_array = get_simulation_flatten_prosit()

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
        flatten_prosit_array = get_simulation_flatten_prosit()

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
    """Get path to model file in pretrained or checkpoints directory.

    Falls back to downloading from GitHub Releases via
    :func:`imspy_predictors.pretrained.hub.ensure_model` when no local
    copy is found.
    """
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

    # Fall back to downloading & caching via hub
    from imspy_predictors.pretrained.hub import ensure_model, MODELS
    # Try the path as-is first (e.g. "intensity/best_model.pt")
    if simple_name in MODELS:
        return ensure_model(simple_name)
    if relative_path in MODELS:
        return ensure_model(relative_path)

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
    from imspy_predictors.utility import require_torch
    torch = require_torch("local intensity prediction")

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
        from imspy_predictors.utility import require_torch
        self._torch = require_torch("DeepPeptideIntensityPredictor (local intensity model)")

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
            flatten_prosit_array = get_simulation_flatten_prosit()
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

    def fine_tune_model(
        self,
        data: pd.DataFrame,
        batch_size: int = 64,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        patience: int = 5,
        divide_collision_energy_by: float = 1e2,
        verbose: bool = False,
    ) -> None:
        """
        Fine-tune the native intensity model on observed 174-vector targets.

        Args:
            data: DataFrame with columns: sequence, charge, collision_energy,
                intensity_target. The target must use the native ordinal-major
                174-vector layout and mark impossible ions as -1.
            batch_size: Training batch size
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            divide_collision_energy_by: CE normalization factor
            verbose: Whether to print progress
        """
        assert 'sequence' in data.columns, 'Data must contain column "sequence"'
        assert 'charge' in data.columns, 'Data must contain column "charge"'
        assert 'collision_energy' in data.columns, 'Data must contain column "collision_energy"'
        assert 'intensity_target' in data.columns, 'Data must contain column "intensity_target"'

        from torch.utils.data import DataLoader, TensorDataset
        from imspy_predictors.losses import masked_spectral_distance

        if len(data) < 2:
            if verbose:
                print("Skipping intensity fine-tune: need at least two PSMs")
            return

        sequences = data.sequence.tolist()
        charges = data.charge.astype(np.int64).tolist()
        collision_energies = (data.collision_energy.astype(float) / divide_collision_energy_by).tolist()
        targets = np.vstack(data.intensity_target.to_numpy()).astype(np.float32)
        if targets.shape != (len(data), 174):
            raise ValueError(f"intensity_target must have shape (n, 174), got {targets.shape}")

        tokens, charge_tensor, ce_tensor = self._preprocess(
            sequences,
            charges,
            collision_energies,
        )
        tokens = tokens.to(self._device)
        charge_tensor = charge_tensor.to(self._device)
        ce_tensor = ce_tensor.to(self._device)
        target_tensor = self._torch.tensor(targets, dtype=self._torch.float32, device=self._device)

        n = len(sequences)
        n_train = max(1, int(0.8 * n))
        if n_train >= n:
            n_train = n - 1
        indices = self._torch.randperm(n, device=self._device)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_dataset = TensorDataset(
            tokens[train_idx],
            charge_tensor[train_idx],
            ce_tensor[train_idx],
            target_tensor[train_idx],
        )
        val_dataset = TensorDataset(
            tokens[val_idx],
            charge_tensor[val_idx],
            ce_tensor[val_idx],
            target_tensor[val_idx],
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        self.model.train()
        optimizer = self._torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = self._torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, min_lr=1e-6
        )
        checkpoint = InMemoryCheckpoint(patience=patience)

        history = {"epochs": [], "train_loss": [], "val_loss": []}
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for tokens_b, charge_b, ce_b, target_b in train_loader:
                optimizer.zero_grad()
                outputs = self.model(
                    tokens_b,
                    charge=charge_b,
                    collision_energy=ce_b,
                )
                pred = outputs['intensity'] if 'intensity' in outputs else list(outputs.values())[0]
                loss = masked_spectral_distance(target_b, pred)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            self.model.eval()
            val_loss = 0.0
            with self._torch.no_grad():
                for tokens_b, charge_b, ce_b, target_b in val_loader:
                    outputs = self.model(
                        tokens_b,
                        charge=charge_b,
                        collision_energy=ce_b,
                    )
                    pred = outputs['intensity'] if 'intensity' in outputs else list(outputs.values())[0]
                    val_loss += masked_spectral_distance(target_b, pred).item()
            val_loss /= max(len(val_loader), 1)
            scheduler.step(val_loss)

            history["epochs"].append(epoch)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))

            if verbose and epoch % 5 == 0:
                print(
                    f"Epoch {epoch}: intensity train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f}"
                )

            if checkpoint.step(val_loss, self.model):
                if verbose:
                    print(f"Early stopping intensity fine-tune at epoch {epoch}")
                break

        checkpoint.restore(self.model)
        self.model.eval()
        self._finetune_history = history

    def fine_tune_psms(
        self,
        psm_collection: List,
        batch_size: int = 64,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        patience: int = 5,
        verbose: bool = False,
    ) -> None:
        """Fine-tune the native intensity model from Sage PSM observed fragments."""
        rows = []
        for psm in psm_collection:
            sequence = psm.sequence_modified if not psm.decoy else psm.sequence_decoy_modified
            target = observed_fragments_to_intensity_target(
                sequence,
                psm.charge,
                psm.sage_feature.fragments,
            )
            if np.any(target > 0):
                rows.append({
                    'sequence': sequence,
                    'charge': int(psm.charge),
                    'collision_energy': float(psm.collision_energy),
                    'intensity_target': target,
                })

        if len(rows) < 2:
            if verbose:
                print("Skipping intensity fine-tune: no usable fragment targets")
            return

        self.fine_tune_model(
            pd.DataFrame(rows),
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
            verbose=verbose,
        )

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
        flatten_prosit_array = get_simulation_flatten_prosit()

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
        flatten_prosit_array = get_simulation_flatten_prosit()

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


    # -------------------------------------------------------------------
    # Fine-tuning
    # -------------------------------------------------------------------

    def fine_tune_model(
        self,
        data: pd.DataFrame,
        batch_size: int = 64,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        patience: int = 5,
        divide_collision_energy_by: float = 1e2,
        verbose: bool = False,
    ) -> None:
        """Fine-tune the native intensity model on observed 174-vec targets.

        Required ``data`` columns:
            ``sequence`` (str, UNIMOD-bracket modified),
            ``charge`` (int),
            ``collision_energy`` (float),
            ``intensity_target`` (np.ndarray shape (174,) — observed
            intensities in the canonical ordinal-major layout; impossible
            ions marked -1; unobserved valid ions = 0).
        """
        assert 'sequence' in data.columns, 'Data must contain column "sequence"'
        assert 'charge' in data.columns, 'Data must contain column "charge"'
        assert 'collision_energy' in data.columns, 'Data must contain column "collision_energy"'
        assert 'intensity_target' in data.columns, 'Data must contain column "intensity_target"'

        from torch.utils.data import DataLoader, TensorDataset
        from imspy_predictors.losses import masked_spectral_distance

        if len(data) < 2:
            if verbose:
                print("Skipping intensity fine-tune: need at least two PSMs")
            return

        sequences = data.sequence.tolist()
        charges = data.charge.astype(np.int64).tolist()
        collision_energies = (
            data.collision_energy.astype(float) / divide_collision_energy_by
        ).tolist()
        targets = np.vstack(data.intensity_target.to_numpy()).astype(np.float32)
        if targets.shape != (len(data), 174):
            raise ValueError(
                f"intensity_target must have shape (n, 174), got {targets.shape}"
            )

        tokens, charge_tensor, ce_tensor = self._preprocess(
            sequences, charges, collision_energies,
        )
        tokens = tokens.to(self._device)
        charge_tensor = charge_tensor.to(self._device)
        ce_tensor = ce_tensor.to(self._device)
        target_tensor = self._torch.tensor(
            targets, dtype=self._torch.float32, device=self._device,
        )

        n = len(sequences)
        # Group-aware (peptide × charge) split: same (modseq, charge) → same
        # fold. PSM-level random split would leak — the predictor is
        # deterministic per (sequence, charge, CE) so identical inputs in
        # train and val collapse val loss to the instrument's intensity-
        # noise floor, not the model's generalization.
        group_keys = np.array([f"{s}_{int(c)}" for s, c in zip(sequences, charges)])
        uniq, inv = np.unique(group_keys, return_inverse=True)
        n_groups = len(uniq)
        n_val_groups = max(1, int(n_groups * 0.2))
        if n_val_groups >= n_groups:
            n_val_groups = n_groups - 1
        rng_np = np.random.default_rng(42)
        perm_groups = rng_np.permutation(n_groups)
        val_groups = set(perm_groups[:n_val_groups].tolist())
        mask_val = np.fromiter((g in val_groups for g in inv),
                                   dtype=bool, count=n)
        val_idx   = self._torch.from_numpy(np.flatnonzero(mask_val)).to(self._device)
        train_idx = self._torch.from_numpy(np.flatnonzero(~mask_val)).to(self._device)
        if verbose:
            print(f"[intens-ft] {n} PSMs ({n_groups:,} unique (modseq,charge)) → "
                  f"train {len(train_idx):,}, val {len(val_idx):,} "
                  f"(val groups: {n_val_groups:,})")

        train_ds = TensorDataset(
            tokens[train_idx], charge_tensor[train_idx],
            ce_tensor[train_idx], target_tensor[train_idx],
        )
        val_ds = TensorDataset(
            tokens[val_idx], charge_tensor[val_idx],
            ce_tensor[val_idx], target_tensor[val_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        self.model.train()
        optimizer = self._torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = self._torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, min_lr=1e-6,
        )
        checkpoint = InMemoryCheckpoint(patience=patience)

        history = {"epochs": [], "train_loss": [], "val_loss": []}
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for tokens_b, charge_b, ce_b, target_b in train_loader:
                optimizer.zero_grad()
                outputs = self.model(
                    tokens_b, charge=charge_b, collision_energy=ce_b,
                )
                pred = (outputs['intensity'] if 'intensity' in outputs
                          else list(outputs.values())[0])
                loss = masked_spectral_distance(target_b, pred)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            self.model.eval()
            val_loss = 0.0
            with self._torch.no_grad():
                for tokens_b, charge_b, ce_b, target_b in val_loader:
                    outputs = self.model(
                        tokens_b, charge=charge_b, collision_energy=ce_b,
                    )
                    pred = (outputs['intensity'] if 'intensity' in outputs
                              else list(outputs.values())[0])
                    val_loss += masked_spectral_distance(target_b, pred).item()
            val_loss /= max(len(val_loader), 1)
            scheduler.step(val_loss)

            history["epochs"].append(epoch)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))

            if verbose and epoch % 5 == 0:
                print(
                    f"Epoch {epoch}: intensity train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f}"
                )

            if checkpoint.step(val_loss, self.model):
                if verbose:
                    print(f"Early stopping intensity fine-tune at epoch {epoch}")
                break

        checkpoint.restore(self.model)
        self.model.eval()
        self._finetune_history = history

    def fine_tune_psms(
        self,
        psm_collection: List,
        batch_size: int = 64,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        patience: int = 5,
        verbose: bool = False,
    ) -> None:
        """Fine-tune the intensity model on a list of sagepy PSM objects.

        Matches the signature ``sagepy-rescore`` calls
        (``fine_tune_psms(psms, batch_size=..., verbose=...)``). Labels are
        built from each PSM's matched fragments via
        :func:`observed_fragments_to_intensity_target`. CE is read from
        ``psm.collision_energy`` (rescore wrapper should have set the
        per-tile value beforehand).
        """
        rows = []
        for psm in psm_collection:
            sequence = (psm.sequence_modified if not psm.decoy
                          else psm.sequence_decoy_modified)
            target = observed_fragments_to_intensity_target(
                sequence,
                psm.charge,
                psm.sage_feature.fragments,
            )
            if np.any(target > 0):
                rows.append({
                    'sequence': sequence,
                    'charge': int(psm.charge),
                    'collision_energy': float(psm.collision_energy),
                    'intensity_target': target,
                })

        if len(rows) < 2:
            if verbose:
                print("Skipping intensity fine-tune: no usable fragment targets")
            return

        self.fine_tune_model(
            pd.DataFrame(rows),
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
            verbose=verbose,
        )
