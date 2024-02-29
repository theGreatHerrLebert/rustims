import numpy as np
import pandas as pd
import tensorflow as tf
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from imspy.core.chemistry.mobility import ccs_to_k0, one_over_k0_to_ccs
from scipy.optimize import curve_fit
from imspy.utility import tokenize_unimod_sequence
from imspy.algorithm.utility import get_model_path


def load_deep_ccs_predictor() -> tf.keras.models.Model:
    """ Get a pretrained deep predictor model

    Returns:
        The pretrained deep predictor model
    """
    return tf.keras.models.load_model(get_model_path('DeepCCSPredictor'))


class PeptideIonMobilityApex(ABC):
    """
    ABSTRACT INTERFACE for simulation of ion-mobility apex value
    """

    def __init__(self):
        pass

    @abstractmethod
    def simulate_ion_mobilities(self, sequences: list[str], charges: list[int]) -> NDArray:
        pass

    @abstractmethod
    def simulate_ion_mobilities_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


def get_sqrt_slopes_and_intercepts(mz: np.ndarray, charge: np.ndarray,
                                   ccs: np.ndarray, fit_charge_state_one: bool = False) -> (np.ndarray, np.ndarray):
    """

    Args:
        mz:
        charge:
        ccs:
        fit_charge_state_one:

    Returns:

    """

    if fit_charge_state_one:
        slopes, intercepts = [], []
    else:
        slopes, intercepts = [0.0], [0.0]

    if fit_charge_state_one:
        c_begin = 1
    else:
        c_begin = 2

    for c in range(c_begin, 5):
        def fit_func(x, a, b):
            return a * np.sqrt(x) + b

        triples = list(filter(lambda x: x[1] == c, zip(mz, charge, ccs)))

        mz_tmp, charge_tmp = np.array([x[0] for x in triples]), np.array([x[1] for x in triples])
        ccs_tmp = np.array([x[2] for x in triples])

        popt, _ = curve_fit(fit_func, mz_tmp, ccs_tmp)

        slopes.append(popt[0])
        intercepts.append(popt[1])

    return np.array(slopes, np.float32), np.array(intercepts, np.float32)


class ProjectToInitialSqrtCCS(tf.keras.layers.Layer):
    """
    Simple sqrt regression layer, calculates ccs value as linear mapping from mz, charge -> ccs
    """

    def __init__(self, slopes, intercepts):
        super(ProjectToInitialSqrtCCS, self).__init__()
        self.slopes = tf.constant([slopes])
        self.intercepts = tf.constant([intercepts])

    def call(self, inputs):
        mz, charge = inputs[0], inputs[1]
        # since charge is one-hot encoded, can use it to gate linear prediction by charge state
        return tf.expand_dims(tf.reduce_sum((self.slopes * tf.sqrt(mz) + self.intercepts) * tf.squeeze(charge), axis=1),
                              1)


class GRUCCSPredictor(tf.keras.models.Model):
    """
    Deep Learning model combining initial linear fit with sequence based features, both scalar and complex
    """

    def __init__(self, slopes, intercepts, num_tokens,
                 seq_len=50,
                 emb_dim=128,
                 gru_1=128,
                 gru_2=64,
                 rdo=0.0,
                 do=0.2):
        super(GRUCCSPredictor, self).__init__()
        self.__seq_len = seq_len

        self.initial = ProjectToInitialSqrtCCS(slopes, intercepts)

        self.emb = tf.keras.layers.Embedding(input_dim=num_tokens + 1, output_dim=emb_dim, input_length=seq_len)

        self.gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_1, return_sequences=True,
                                                                      name='GRU1'))

        self.gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_2, return_sequences=False,
                                                                      name='GRU2',
                                                                      recurrent_dropout=rdo))

        self.dense1 = tf.keras.layers.Dense(128, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))

        self.dropout = tf.keras.layers.Dropout(do)

        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        """
        :param inputs: should contain: (mz, charge_one_hot, seq_as_token_indices)
        """
        # get inputs
        mz, charge, seq = inputs[0], inputs[1], inputs[2]
        # sequence learning
        x_recurrent = self.gru2(self.gru1(self.emb(seq)))
        # concat to feed to dense layers
        concat = tf.keras.layers.Concatenate()([charge, x_recurrent])
        # regularize
        d1 = self.dropout(self.dense1(concat))
        d2 = self.dense2(d1)
        # combine simple linear hypotheses with deep part
        return self.initial([mz, charge]) + self.out(d2), self.out(d2)


class DeepPeptideIonMobilityApex(PeptideIonMobilityApex):
    def __init__(self, model: GRUCCSPredictor, tokenizer: tf.keras.preprocessing.text.Tokenizer,
                 verbose: bool = False,
                 name: str = 'gru_predictor'):
        super(DeepPeptideIonMobilityApex, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.verbose = verbose

    def _preprocess_sequences(self, sequences: list[str], pad_len: int = 50) -> NDArray:
        char_tokens = [tokenize_unimod_sequence(sequence) for sequence in sequences]
        char_tokens = self.tokenizer.texts_to_sequences(char_tokens)
        char_tokens = tf.keras.preprocessing.sequence.pad_sequences(char_tokens, pad_len, padding='post')
        return char_tokens

    def simulate_ion_mobilities(self,
                                sequences: list[str],
                                charges: list[int],
                                mz: list[float],
                                batch_size: int = 1024) -> NDArray:
        tokenized_sequences = self._preprocess_sequences(sequences)

        # prepare masses, charges, sequences
        m = np.expand_dims(mz, 1)
        charges_one_hot = tf.one_hot(np.array(charges) - 1, 4)

        ds = tf.data.Dataset.from_tensor_slices(((m, charges_one_hot, tokenized_sequences), np.zeros_like(mz))).batch(batch_size)
        ccs, _ = self.model.predict(ds, verbose=self.verbose)

        return np.array([1 / ccs_to_k0(c, m, z) for c, m, z in zip(ccs, mz, charges)])

    def simulate_ion_mobilities_pandas(self, data: pd.DataFrame, batch_size: int = 1024) -> pd.DataFrame:
        tokenized_sequences = self._preprocess_sequences(data.sequence.values)

        # prepare masses, charges, sequences
        m = np.expand_dims(data.mz.values, 1)
        charges_one_hot = tf.one_hot(np.array(data.charge.values) - 1, 4)

        ds = tf.data.Dataset.from_tensor_slices(((m, charges_one_hot, tokenized_sequences), np.zeros_like(m))).batch(batch_size)
        ccs, _ = self.model.predict(ds, verbose=self.verbose)

        data[f'mobility_{self.name}'] = np.array([1 / ccs_to_k0(c, m, z) for c, m, z in zip(ccs, m, data.charge.values)])
        data = data[['peptide_id', 'monoisotopic-mass', 'mz', 'charge', 'relative_abundance', f'mobility_{self.name}']]
        return data

    def __repr__(self):
        return f'DeepPeptideIonMobilityApex(name={self.name}, model={self.model})'
