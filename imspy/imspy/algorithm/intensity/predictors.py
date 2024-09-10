import os
import re
from typing import List, Tuple

from numpy.typing import NDArray
import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from sagepy.core.scoring import associate_fragment_ions_with_prosit_predicted_intensities, PeptideSpectrumMatch
from tqdm import tqdm

from imspy.algorithm.utility import get_model_path
from .utility import (generate_prosit_intensity_prediction_dataset, unpack_dict,
                                               post_process_predicted_fragment_spectra, reshape_dims, beta_score)

from imspy.data.peptide import PeptideProductIonSeriesCollection, PeptideSequence

from imspy.simulation.utility import flatten_prosit_array

def predict_intensities_prosit(
        psm_collection: List[PeptideSpectrumMatch],
        calibrate_collision_energy: bool = True,
        verbose: bool = False,
        num_threads: int = -1,
) -> None:
    """
    Predict the fragment ion intensities using Prosit.
    Args:
        psm_collection: a list of peptide-spectrum matches
        calibrate_collision_energy: whether to calibrate the collision energy
        verbose:
        num_threads:

    Returns:

    """
    # check if num_threads is -1, if so, use all available threads
    if num_threads == -1:
        num_threads = os.cpu_count()

    # the intensity predictor model
    prosit_model = Prosit2023TimsTofWrapper(verbose=False)

    # sample for collision energy calibration
    sample = list(sorted(psm_collection, key=lambda x: x.hyper_score, reverse=True))[:int(2 ** 11)]

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
        [p.sequence for p in psm_collection],
        np.array([p.charge for p in psm_collection]),
        [p.collision_energy_calibrated for p in psm_collection],
        batch_size=2048,
        flatten=True,
    )

    associate_fragment_ions_with_prosit_predicted_intensities(psm_collection, intensity_pred, num_threads=num_threads)

    for ps in psm_collection:
        ps.beta_score = beta_score(ps.fragments_observed, ps.fragments_predicted)


def get_collision_energy_calibration_factor(
        sample: List[PeptideSpectrumMatch],
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
        Tuple[float, List[float]]: the collision energy calibration factor and the cosine similarities
    """
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

        cos_target.append((i, np.mean([x.cosine_similarity for x in target])))
        cos_decoy.append((i, np.mean([x.cosine_similarity for x in decoy])))

    return cos_target[np.argmax([x[1] for x in cos_target])][0], [x[1] for x in cos_target]



def remove_unimod_annotation(sequence: str) -> str:
    """Remove [UNIMOD:N] annotations from the sequence."""
    pattern = r'\[UNIMOD:\d+\]'
    return re.sub(pattern, '', sequence)


def load_prosit_2023_timsTOF_predictor():
    """ Get a pretrained deep predictor model
    This model was downloaded from ZENODO: https://zenodo.org/records/8211811
    PAPER : https://doi.org/10.1101/2023.07.17.549401
    Returns:
        The pretrained deep predictor model
    """
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
        for batch_indices in tqdm(np.array_split(data.index, np.ceil(len(data) / batch_size)),
                                  total=int(np.ceil(len(data) / batch_size)),
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

        tf_ds = (generate_prosit_intensity_prediction_dataset(
            data.sequence,
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
        sequences_unmod = [remove_unimod_annotation(s) for s in sequences]
        sequence_length = [len(s) for s in sequences_unmod]
        collision_energies_norm = [ce / divide_collision_energy_by for ce in collision_energies]

        tf_ds = generate_prosit_intensity_prediction_dataset(
            sequences,
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
