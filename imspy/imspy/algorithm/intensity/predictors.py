from numpy.typing import NDArray
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from tqdm import tqdm

from imspy.algorithm.utility import get_model_path
from imspy.simulation.utility import remove_unimod_annotation, reshape_dims
from imspy.algorithm.intensity.utility import generate_prosit_intensity_prediction_dataset, unpack_dict, post_process_predicted_fragment_spectra


def load_prosit_2023_timsTOF_predictor():
    """ Get a pretrained deep predictor model
    This model was downloaded from ZENODO: https://zenodo.org/records/8211811
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
    def __init__(self, verbose: bool = True, model_name: str = 'deep_ion_intensity_predictor'):
        super().__init__()

        self.verbose = verbose
        self.model_name = model_name
        self.model = load_prosit_2023_timsTOF_predictor()

    def simulate_ion_intensities(self, sequences: list[str], charges: list[int], collision_energies) -> NDArray:
        pass

    def simulate_ion_intensities_pandas(self, data: pd.DataFrame) -> pd.DataFrame:

        data['sequence_unmod'] = data.apply(lambda r: remove_unimod_annotation(r.sequence), axis=1)
        data['collision_energy'] = data.apply(lambda r: r.collision_energy / 1e3, axis=1)
        data['sequence_length'] = data.apply(lambda r: len(r.sequence_unmod), axis=1)

        tf_ds = generate_prosit_intensity_prediction_dataset(
            data.sequence_unmod,
            data.charge,
            np.expand_dims(data.collision_energy, 1)
        ).batch(512)

        # Map the unpacking function over the dataset
        ds_unpacked = tf_ds.map(unpack_dict)

        intensity_predictions = []

        # Iterate over the dataset and call the model with unpacked inputs
        for peptides_in, precursor_charge_in, collision_energy_in in tqdm(ds_unpacked, desc='Predicting intensities',
                                                                          total=len(data) // 512 + 1,
                                                                          silent=not self.verbose):
            model_input = [peptides_in, precursor_charge_in, collision_energy_in]
            model_output = self.model(model_input).numpy()
            intensity_predictions.append(model_output)

        I_pred = list(np.vstack(intensity_predictions))
        data['intensity_raw'] = I_pred
        I_pred = np.squeeze(reshape_dims(post_process_predicted_fragment_spectra(data)))

        data['intensity'] = list(I_pred)

        return data
