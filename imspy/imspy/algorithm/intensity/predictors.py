import os
import re
from typing import List, Tuple

from numpy.typing import NDArray
import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from sagepy.core.scoring import associate_fragment_ions_with_prosit_predicted_intensities, Psm
from tqdm import tqdm

from imspy.algorithm.utility import get_model_path
from .utility import (generate_prosit_intensity_prediction_dataset, unpack_dict,
                                               post_process_predicted_fragment_spectra, reshape_dims)

from imspy.data.peptide import PeptideProductIonSeriesCollection, PeptideSequence
from imspy.simulation.utility import flatten_prosit_array

def predict_intensities_prosit(
        psm_collection: List[Psm],
        calibrate_collision_energy: bool = True,
        verbose: bool = False,
        num_threads: int = -1,
) -> None:
    """
    Predict the fragment ion intensities using Prosit.
    Args:
        psm_collection: a list of peptide-spectrum matches
        calibrate_collision_energy: whether to calibrate the collision energy
        verbose: whether to print progress
        num_threads: number of threads to use

    Returns:
        None, the fragment ion intensities are stored in the PeptideSpectrumMatch objects
    """
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

    # Use raw sequence (without UNIMOD) - the Prosit model uses ALPHABET_UNMOD
    # which doesn't support modified tokens like C[UNIMOD:4].
    intensity_pred = prosit_model.predict_intensities(
        [p.sequence for p in psm_collection],
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
        sample: List[Psm],
        model: 'Prosit2023TimsTofWrapper',
        lower: int = -30,
        upper: int = 30,
        verbose: bool = False,
) -> Tuple[float, List[float]]:
    """
    Get the collision energy calibration factor for a given sample.
    Args:
        sample: a list of PeptideSpectrumMatch objects
        model: a Prosit2023TimsTofWrapper object
        lower: lower bound for the search
        upper: upper bound for the search
        verbose: whether to print progress

    Returns:
        Tuple[float, List[float]]: the collision energy calibration factor and the angle similarities
    """
    cos_target, cos_decoy = [], []

    if verbose:
        print(f"Searching for collision energy calibration factor between {lower} and {upper} ...")

    for i in tqdm(range(lower, upper), disable=not verbose, desc='calibrating CE', ncols=100):
        # Use raw sequence (without UNIMOD) - the Prosit model uses ALPHABET_UNMOD
        # which doesn't support modified tokens like C[UNIMOD:4]. Using sequence_modified
        # would cause modified residues to be encoded as index 0, corrupting predictions.
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


def get_calibrated_ce_model(
        sample: List[Psm],
        model: 'Prosit2023TimsTofWrapper' = None,
        lower: int = -30,
        upper: int = 30,
        verbose: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Get a calibrated CE model by:
    1. Finding the optimal CE offset that maximizes spectral angle similarity
    2. Fitting a linear model on (CE_actual + offset) vs m/z

    This provides the best CE model for intensity prediction since it:
    - Corrects for systematic bias between instrument CE and Prosit's expected CE
    - Captures the m/z-dependent relationship after correction

    Args:
        sample: a list of PeptideSpectrumMatch objects (targets only recommended)
        model: a Prosit2023TimsTofWrapper object (optional, will create if not provided)
        lower: lower bound for offset search
        upper: upper bound for offset search
        verbose: whether to print progress

    Returns:
        Tuple[float, float, float, float]: (intercept, slope, optimal_offset, best_similarity)
            - intercept: CE model intercept
            - slope: CE model slope (CE per m/z unit)
            - optimal_offset: the calibration offset found
            - best_similarity: the spectral angle similarity at optimal offset
    """
    if len(sample) == 0:
        raise ValueError("Sample cannot be empty")

    # Filter to targets only
    targets = [p for p in sample if not getattr(p, 'decoy', False)]
    if len(targets) == 0:
        raise ValueError("No target PSMs in sample")

    if verbose:
        print(f"Calibrating CE model with {len(targets)} target PSMs...")

    # Create model if not provided
    if model is None:
        model = Prosit2023TimsTofWrapper(verbose=False)

    # Step 1: Find optimal CE offset
    optimal_offset, similarities = get_collision_energy_calibration_factor(
        targets, model, lower=lower, upper=upper, verbose=verbose
    )
    best_similarity = max(similarities)

    # Step 2: Fit linear model on (CE_actual + offset) vs m/z
    mz_values = np.array([p.mono_mz_calculated for p in targets])
    ce_corrected = np.array([p.collision_energy + optimal_offset for p in targets])

    # Linear fit: ce_corrected = intercept + slope * mz
    coeffs = np.polyfit(mz_values, ce_corrected, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    if verbose:
        print(f"Calibrated CE model: CE = {intercept:.4f} + {slope:.6f} * m/z")
        print(f"  Optimal offset: {optimal_offset}")
        print(f"  Best spectral angle similarity: {best_similarity:.4f}")

        # Show fit quality
        ce_predicted = intercept + slope * mz_values
        residuals = ce_corrected - ce_predicted
        print(f"  Linear fit R²: {1 - np.var(residuals) / np.var(ce_corrected):.4f}")
        print(f"  Residual std: {np.std(residuals):.4f}")

    return intercept, slope, optimal_offset, best_similarity


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


def _disable_gpu_for_prosit():
    """Disable GPU for Prosit predictions on macOS only.

    TensorFlow Metal (Apple Silicon GPU) produces numerically different results
    compared to CPU execution for this model. On macOS, we force CPU-only execution.

    On Linux with CUDA, GPU execution works correctly and provides significant speedup.
    """
    import platform
    if platform.system() == 'Darwin':  # macOS only
        try:
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError:
            # GPU visibility must be set before GPUs are initialized
            # If we're here, GPUs were already initialized - this is fine if
            # we already disabled them in a previous call
            pass


def load_prosit_2023_timsTOF_predictor():
    """ Get a pretrained deep predictor model
    This model was downloaded from ZENODO: https://zenodo.org/records/8211811
    PAPER : https://doi.org/10.1101/2023.07.17.549401

    Note: GPU is disabled for this model because TensorFlow Metal produces
    numerically different (incorrect) results. See _disable_gpu_for_prosit().

    Returns:
        The pretrained deep predictor model
    """
    _disable_gpu_for_prosit()
    return tf.saved_model.load(get_model_path('intensity/Prosit2023TimsTOFPredictor'))


class IonIntensityPredictor(ABC):
    """
    ABSTRACT INTERFACE for simulation of ion-mobility apex value
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
    Wrapper for the Prosit 2023 TIMS-TOF predictor
    """
    def __init__(self, verbose: bool = True, model_name: str = 'deep_ion_intensity_predictor'):
        super().__init__()

        self.verbose = verbose
        self.model_name = model_name
        self.model = load_prosit_2023_timsTOF_predictor()

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

            # TODO: Save the data_pred to a database per batch, so that we don't have to keep everything in memory
            """
            if batch_counter == 0:
                handle.create_table('intensity_predictions', data_pred)
            """

            tables.append(data_pred)
            batch_counter += 1

        return pd.concat(tables)

    def simulate_ion_intensities_pandas(self, data: pd.DataFrame, batch_size: int = 512,
                                        divide_collision_energy_by: float = 1e2,
                                        verbose: bool = False, flatten: bool = False) -> pd.DataFrame:

        if verbose:
            print("Generating Prosit compatible input data...")

        data['collision_energy'] = data.apply(lambda r: r.collision_energy / divide_collision_energy_by, axis=1)
        data['sequence_length'] = data.apply(lambda r: len(remove_unimod_annotation(r.sequence)), axis=1)
        # Strip UNIMOD annotations for Prosit - ALPHABET_UNMOD doesn't support modified tokens
        data['sequence_unmod'] = data.sequence.apply(remove_unimod_annotation)

        tf_ds = (generate_prosit_intensity_prediction_dataset(
            data.sequence_unmod,  # Use unmodified sequences for Prosit
            data.charge,
            np.expand_dims(data.collision_energy, 1)).batch(batch_size))

        # Map the unpacking function over the dataset
        ds_unpacked = tf_ds.map(unpack_dict)

        intensity_predictions = []

        # Iterate over the dataset and call the model with unpacked inputs
        for peptides_in, precursor_charge_in, collision_energy_in in tqdm(ds_unpacked, desc='Predicting intensities',
                                                                          total=len(data) // batch_size + 1, ncols=100,
                                                                          disable=not verbose):
            model_input = [peptides_in, precursor_charge_in, collision_energy_in]
            model_output = self.model(model_input).numpy()
            intensity_predictions.append(model_output)

        I_pred = list(np.vstack(intensity_predictions))
        data['intensity_raw'] = I_pred
        I_pred = np.squeeze(reshape_dims(post_process_predicted_fragment_spectra(data)))

        if flatten:
            I_pred = np.vstack([flatten_prosit_array(r) for r in I_pred])

        data['intensity'] = list(I_pred)

        return data

    def predict_intensities(
            self,
            sequences: List[str],
            charges: List[int],
            collision_energies: List[float],
            divide_collision_energy_by: float = 1e2,
            batch_size: int = 512,
            flatten: bool = False,
    ) -> List[NDArray]:
        # Strip UNIMOD annotations - the Prosit model uses ALPHABET_UNMOD which
        # doesn't support modified tokens like C[UNIMOD:4]. Using sequences with
        # UNIMOD would cause modified residues to be encoded as index 0.
        sequences_unmod = [remove_unimod_annotation(s) for s in sequences]
        sequence_length = [len(s) for s in sequences_unmod]
        collision_energies_norm = [ce / divide_collision_energy_by for ce in collision_energies]

        tf_ds = generate_prosit_intensity_prediction_dataset(
            sequences_unmod,  # Use unmodified sequences for Prosit
            charges,
            np.expand_dims(collision_energies_norm, 1)).batch(batch_size)

        ds_unpacked = tf_ds.map(unpack_dict)

        intensity_predictions = []
        for peptides_in, precursor_charge_in, collision_energy_in in tqdm(ds_unpacked, desc='Predicting intensities',
                                                                          total=len(sequences) // batch_size + 1,
                                                                          ncols=100,
                                                                          disable=not self.verbose):
            model_input = [peptides_in, precursor_charge_in, collision_energy_in]
            model_output = self.model(model_input).numpy()
            intensity_predictions.append(model_output)

        I_pred = list(np.vstack(intensity_predictions))
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
        sequences_unmod = [remove_unimod_annotation(s) for s in sequences]
        sequence_length = [len(s) for s in sequences_unmod]
        collision_energies_norm = [ce / divide_collision_energy_by for ce in collision_energies]

        tf_ds = generate_prosit_intensity_prediction_dataset(
            sequences_unmod,
            charges,
            np.expand_dims(collision_energies_norm, 1)).batch(batch_size)

        ds_unpacked = tf_ds.map(unpack_dict)

        intensity_predictions = []
        for peptides_in, precursor_charge_in, collision_energy_in in tqdm(ds_unpacked, desc='Predicting intensities',
                                                                          total=len(sequences) // batch_size + 1, ncols=100,
                                                                          disable=not self.verbose):
            model_input = [peptides_in, precursor_charge_in, collision_energy_in]
            model_output = self.model(model_input).numpy()
            intensity_predictions.append(model_output)

        I_pred = list(np.vstack(intensity_predictions))
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
    from ..koina_models.access_models import ModelFromKoina
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