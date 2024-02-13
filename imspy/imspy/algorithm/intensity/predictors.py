from numpy.typing import NDArray
import pandas as pd
from abc import ABC, abstractmethod
import tensorflow as tf

from imspy.algorithm.utility import get_model_path


def load_prosit_2019_predictor() -> tf.keras.models.Model:
    """ Get a pretrained deep predictor model

    Returns:
        The pretrained deep predictor model
    """
    return tf.keras.models.load_model(get_model_path('Prosit2019Predictor'))


class IonIntensityPredictor(ABC):
    """
    ABSTRACT INTERFACE for simulation of ion-mobility apex value
    """

    def __init__(self):
        pass

    @abstractmethod
    def simulate_ion_intensities(self, sequences: list[str], charges: list[int]) -> NDArray:
        pass

    @abstractmethod
    def simulate_ion_intensities_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

