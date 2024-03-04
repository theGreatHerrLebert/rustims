import re

from numpy.typing import NDArray
import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from tqdm import tqdm

from imspy.algorithm.utility import get_model_path
from imspy.algorithm.intensity.utility import (generate_prosit_intensity_prediction_dataset, unpack_dict,
                                               post_process_predicted_fragment_spectra, reshape_dims)

from imspy.simulation.utility import flatten_prosit_array


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
    return tf.saved_model.load(get_model_path('Prosit2023TimsTOFPredictor'))


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
                                        divide_collision_energy_by: float = 1e2, verbose: bool = False) -> pd.DataFrame:

        if verbose:
            print("Generating Prosit compatible input data...")

        data['sequence_unmod'] = data.apply(lambda r: remove_unimod_annotation(r.sequence), axis=1)
        data['collision_energy'] = data.apply(lambda r: r.collision_energy / divide_collision_energy_by, axis=1)
        data['sequence_length'] = data.apply(lambda r: len(r.sequence_unmod), axis=1)

        tf_ds = (generate_prosit_intensity_prediction_dataset(
            data.sequence_unmod,
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

        data['intensity'] = list(I_pred)

        return data

    def simulate_ion_intensities(
            self,
            sequences: list[str],
            charges: list[int],
            collision_energies: list[float],
            divide_collision_energy_by: float = 1e2,
            batch_size: int = 512,
    ) -> NDArray:
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

        return np.vstack([flatten_prosit_array(r) for r in I_pred])
