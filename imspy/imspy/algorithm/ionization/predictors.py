import pandas as pd
import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod
from numpy.typing import NDArray

from tqdm import tqdm
from tensorflow.keras.models import load_model

from imspy.algorithm.utility import get_model_path
from imspy.utility import tokenize_unimod_sequence

import imspy_connector
ims = imspy_connector.py_chemistry


def load_deep_charge_state_predictor() -> tf.keras.models.Model:
    """ Get a pretrained deep predictor model

    Returns:
        The pretrained deep predictor model
    """
    path = get_model_path('charge/chargepred-26-05-2024.keras')

    custom_objects = {
        'GRUChargeStatePredictor': GRUChargeStatePredictor
    }

    return load_model(path, custom_objects=custom_objects)


def charge_state_distribution_from_sequence_rust(sequence: str, max_charge: int = None,
                                                 charge_probability: float = None) -> NDArray:
    return np.array(ims.simulate_charge_state_for_sequence(sequence, max_charge, charge_probability))


def charge_state_distributions_from_sequences_rust(sequences: list[str], n_threads: int = 4, max_charge: int = None,
                                                   charge_probability: float = None) -> NDArray:
    return np.array(ims.simulate_charge_states_for_sequences(sequences, n_threads, max_charge, charge_probability))


class PeptideChargeStateDistribution(ABC):
    """
    ABSTRACT INTERFACE for ionization simulation of peptides
    """

    def __init__(self):
        pass

    @abstractmethod
    def simulate_ionizations(self, sequences: list[str]) -> np.array:
        pass

    @abstractmethod
    def simulate_charge_state_distribution_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class BinomialChargeStateDistributionModel(PeptideChargeStateDistribution, ABC):

    def __init__(self,
                 charged_probability: float = .8,
                 max_charge: int = 4,
                 normalize: bool = True
                 ):
        self.charged_probability = charged_probability
        self.max_charge = max_charge
        self.normalize = normalize

    def simulate_ionizations(self, sequences: list[str]) -> np.array:
        return charge_state_distributions_from_sequences_rust(sequences, max_charge=self.max_charge,
                                                              charge_probability=self.charged_probability)

    def simulate_charge_state_distribution_pandas(self, data: pd.DataFrame, min_charge_contrib: float = .005) -> pd.DataFrame:
        probabilities = charge_state_distributions_from_sequences_rust(data.sequence.values, max_charge=self.max_charge,
                                                                       charge_probability=self.charged_probability)

        r_table = []

        for charges, (_, row) in tqdm(zip(probabilities, data.iterrows()), desc='flatmap charges', ncols=80,
                                      total=len(probabilities)):
            if self.normalize:
                kept_prob_mass = np.sum(np.where(charges[1:]>= min_charge_contrib, charges[1:], 0))
                charges[1:] = charges[1:] / kept_prob_mass
            for i, charge in enumerate(charges[1:], start=1):
                if charge >= min_charge_contrib:
                    r_table.append({'peptide_id': row.peptide_id, 'charge': i, 'relative_abundance': charge})

        return pd.DataFrame(r_table)


class DeepChargeStateDistribution(PeptideChargeStateDistribution, ABC):

    def __init__(self,
                 model: 'GRUChargeStatePredictor',
                 tokenizer: tf.keras.preprocessing.text.Tokenizer,
                 allowed_charges: NDArray = np.array([1, 2, 3, 4]),
                 name: str = 'gru_predictor',
                 verbose: bool = True
                 ):
        super(DeepChargeStateDistribution, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.allowed_charges = allowed_charges
        self.name = name
        self.verbose = verbose

    def _preprocess_sequences(self, sequences: list[str], pad_len: int = 50) -> np.array:
        char_tokens = [tokenize_unimod_sequence(seq) for seq in sequences]
        char_tokens = self.tokenizer.texts_to_sequences(char_tokens)
        char_tokens = tf.keras.preprocessing.sequence.pad_sequences(char_tokens, pad_len, padding='post')
        return char_tokens

    def simulate_ionizations(self, sequences: list[str], batch_size: int = 1024) -> NDArray:
        tokens = self._preprocess_sequences(sequences)
        tf_ds = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)

        probabilities = self.model.predict(tf_ds, verbose=self.verbose)

        c_list = []

        for p in probabilities:
            c_list.append(np.random.choice(range(1, len(p) + 1), 1, p=p)[0])

        return np.array(c_list)

    def simulate_charge_state_distribution_pandas(self,
                                                  data: pd.DataFrame,
                                                  charge_state_one_probability: float = 0.1,
                                                  batch_size: int = 1024,
                                                  min_charge_contrib: float = .005) -> pd.DataFrame:
        """
        Simulate charge state distribution for a pandas DataFrame containing peptide sequences

        Args:
            data: pandas DataFrame containing peptide sequences
            charge_state_one_probability: probability of charge state 1
            batch_size: batch size for prediction
            min_charge_contrib: minimum relative abundance of a charge state to be included in the output

        Returns:
            pandas DataFrame containing simulated charge state distributions
        """
        assert 0 <= charge_state_one_probability <= 1, f'charge_state_one_probability must be in [0, 1], was: {charge_state_one_probability}'

        tokens = self._preprocess_sequences(data.sequence.values)
        tf_ds = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)

        probabilities = self.model.predict(tf_ds, verbose=self.verbose)

        r_table = []

        # add charge state 1 probability
        probabilities[:, 0] = probabilities[:, 0] + charge_state_one_probability
        # normalize
        probabilities = probabilities / np.expand_dims(np.sum(probabilities, axis=1), axis=1)

        for charges, (_, row) in tqdm(zip(probabilities, data.iterrows()), desc='flatmap charges', ncols=80,
                                      total=len(probabilities)):

            for i, charge in enumerate(charges, start=1):
                if charge >= min_charge_contrib:
                    r_table.append({'peptide_id': row.peptide_id, 'charge': i, 'relative_abundance': charge})

        return pd.DataFrame(r_table)


@tf.keras.saving.register_keras_serializable()
class GRUChargeStatePredictor(tf.keras.models.Model):

    def __init__(self,
                 num_tokens,
                 max_charge=4,
                 seq_len=50,
                 emb_dim=128,
                 gru_1=128,
                 gru_2=64,
                 rdo=0.0,
                 do=0.2):
        super(GRUChargeStatePredictor, self).__init__()

        self.num_tokens = num_tokens
        self.max_charge = max_charge
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.gru_1 = gru_1
        self.gru_2 = gru_2
        self.rdo = rdo
        self.do = do

        self.emb = tf.keras.layers.Embedding(input_dim=num_tokens + 1, output_dim=emb_dim, input_length=seq_len)
        self.gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_1, return_sequences=True, name='GRU1'))
        self.gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_2, return_sequences=False, name='GRU2', recurrent_dropout=rdo))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name='Dense1', kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu', name='Dense2', kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))
        self.dropout = tf.keras.layers.Dropout(do, name='Dropout')
        self.out = tf.keras.layers.Dense(max_charge, activation='softmax', name='Output')

    def build(self, input_shape):
        self.emb.build((input_shape[0], self.seq_len))
        self.gru1.build((input_shape[0], self.seq_len, self.emb.output_dim))
        gru1_output_dim = self.gru1.forward_layer.units + self.gru1.backward_layer.units
        self.gru2.build((input_shape[0], self.seq_len, gru1_output_dim))
        gru2_output_dim = self.gru2.forward_layer.units + self.gru2.backward_layer.units
        self.dense1.build((input_shape[0], gru2_output_dim))
        self.dense2.build((input_shape[0], self.dense1.units))
        self.dropout.build((input_shape[0], self.dense1.units))
        self.out.build((input_shape[0], self.dense2.units))
        super(GRUChargeStatePredictor, self).build(input_shape)

    def call(self, inputs, training=False):
        seq = inputs
        x_recurrent = self.gru2(self.gru1(self.emb(seq)))
        d1 = self.dropout(self.dense1(x_recurrent), training=training)
        return self.out(self.dense2(d1))

    def get_config(self):
        config = super(GRUChargeStatePredictor, self).get_config()
        config.update({
            "num_tokens": self.num_tokens,
            "max_charge": self.max_charge,
            "seq_len": self.seq_len,
            "emb_dim": self.emb_dim,
            "gru_1": self.gru_1,
            "gru_2": self.gru_2,
            "rdo": self.rdo,
            "do": self.do,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
