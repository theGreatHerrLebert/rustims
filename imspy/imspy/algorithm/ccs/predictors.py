import numpy as np
import pandas as pd
import tensorflow as tf
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from imspy.chemistry.mobility import ccs_to_one_over_k0
from scipy.optimize import curve_fit
from imspy.utility import tokenize_unimod_sequence
from imspy.algorithm.utility import get_model_path


def load_deep_ccs_predictor() -> tf.keras.models.Model:
    """ Get a pretrained deep predictor model

    Returns:
        The pretrained deep predictor model
    """
    return tf.keras.models.load_model(get_model_path('IonmobPredictor'))


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


def get_sqrt_slopes_and_intercepts(
        mz: np.ndarray,
        charge: np.ndarray,
        ccs: np.ndarray,
        fit_charge_state_one: bool = False
) -> (np.ndarray, np.ndarray):
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


@tf.keras.saving.register_keras_serializable()
class SquareRootProjectionLayer(tf.keras.layers.Layer):
    """
    Simple sqrt regression layer, calculates ccs value as linear mapping from mz, charge -> ccs
    """

    def __init__(self, slopes, intercepts, trainable: bool = True, **kwargs):
        super(SquareRootProjectionLayer, self).__init__(**kwargs)
        self.slopes_init = list(slopes)
        self.intercepts_init = list(intercepts)
        self.trainable = trainable
        self.slopes = None
        self.intercepts = None

    def build(self, input_shape):
        num_charges = input_shape[1][-1]

        self.slopes = self.add_weight(name='sqrt-coefficients',
                                      shape=(num_charges,),
                                      initializer=tf.constant_initializer(self.slopes_init),
                                      trainable=self.trainable)

        self.intercepts = self.add_weight(name='intercepts',
                                          shape=(num_charges,),
                                          initializer=tf.constant_initializer(self.intercepts_init),
                                          trainable=self.trainable)

    def call(self, inputs):
        mz, charge = inputs[0], inputs[1]
        projection = (self.slopes * tf.sqrt(mz) + self.intercepts) * charge
        result = tf.reduce_sum(projection, axis=-1, keepdims=True)

        return result

    def get_config(self):
        config = super(SquareRootProjectionLayer, self).get_config()
        config.update({
            'slopes': self.slopes_init,
            'intercepts': self.intercepts_init,
            'trainable': self.trainable
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        return f"SquareRootProjectionLayer(slopes={self.slopes_init}, intercepts={self.intercepts_init})"


@tf.keras.saving.register_keras_serializable()
class GRUCCSPredictor(tf.keras.models.Model):
    """
    Deep Learning model combining initial linear fit with sequence-based features, both scalar and complex
    """

    def __init__(self, slopes, intercepts, num_tokens,
                 max_peptide_length=50,
                 emb_dim=128,
                 gru_1=128,
                 gru_2=64,
                 rdo=0.0,
                 do=0.2):
        super(GRUCCSPredictor, self).__init__()
        self.max_peptide_length = max_peptide_length

        self.initial = SquareRootProjectionLayer(slopes, intercepts)

        self.emb = tf.keras.layers.Embedding(input_dim=num_tokens + 1, output_dim=emb_dim)

        self.gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_1, return_sequences=True, name='GRU1'))

        self.gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_2, return_sequences=False,
                                                                      name='GRU2',
                                                                      recurrent_dropout=rdo))

        self.dense1 = tf.keras.layers.Dense(128, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))

        self.dropout = tf.keras.layers.Dropout(do)

        self.out = tf.keras.layers.Dense(1, activation=None)

    def build(self, input_shape):
        self.initial.build(input_shape)
        self.emb.build((input_shape[2][0], self.max_peptide_length))
        self.gru1.build((input_shape[2][0], self.max_peptide_length, self.emb.output_dim))
        gru1_output_dim = self.gru1.forward_layer.units + self.gru1.backward_layer.units
        self.gru2.build((input_shape[2][0], self.max_peptide_length, gru1_output_dim))
        gru2_output_dim = self.gru2.forward_layer.units + self.gru2.backward_layer.units

        dense1_input_shape = (input_shape[1][0], input_shape[1][1] + gru2_output_dim)
        self.dense1.build(dense1_input_shape)

        dense2_input_shape = (dense1_input_shape[0], self.dense1.units)
        self.dense2.build(dense2_input_shape)

        self.dropout.build(dense2_input_shape)

        out_input_shape = (dense2_input_shape[0], self.dense2.units)
        self.out.build(out_input_shape)

        super(GRUCCSPredictor, self).build(input_shape)

    def call(self, inputs, training=False):
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
        d1 = self.dropout(self.dense1(concat), training=training)
        d2 = self.dense2(d1)

        # combine simple linear hypotheses with deep part
        initial_output = self.initial([mz, charge])
        out_output = self.out(d2)

        # Only return the primary output during training
        return initial_output + out_output, out_output

    def get_config(self):
        config = super(GRUCCSPredictor, self).get_config()
        config.update({
            'slopes': self.initial.slopes_init,
            'intercepts': self.initial.intercepts_init,
            'num_tokens': self.emb.input_dim - 1,
            'max_peptide_length': self.max_peptide_length,
            'emb_dim': self.emb.output_dim,
            'gru_1': self.gru1.forward_layer.units,
            'gru_2': self.gru2.forward_layer.units,
            'rdo': self.gru2.forward_layer.recurrent_dropout,
            'do': self.dropout.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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

    def simulate_ion_mobilities(
            self,
            sequences: list[str],
            charges: list[int],
            mz: list[float],
            batch_size: int = 1024
    ) -> NDArray:
        tokenized_sequences = self._preprocess_sequences(sequences)

        # prepare masses, charges, sequences
        m = np.expand_dims(mz, 1)
        charges_one_hot = tf.one_hot(np.array(charges) - 1, 4)

        ds = tf.data.Dataset.from_tensor_slices(((m, charges_one_hot, tokenized_sequences), np.zeros_like(mz))).batch(batch_size)
        ccs, _ = self.model.predict(ds, verbose=self.verbose)

        return np.array([ccs_to_one_over_k0(c, m, z) for c, m, z in zip(ccs, mz, charges)])

    def simulate_ion_mobilities_pandas(self, data: pd.DataFrame, batch_size: int = 1024, return_ccs: bool = False) -> pd.DataFrame:
        tokenized_sequences = self._preprocess_sequences(data.sequence.values)

        # prepare masses, charges, sequences
        m = np.expand_dims(data.mz.values, 1)
        charges_one_hot = tf.one_hot(np.array(data.charge.values) - 1, 4)

        ds = tf.data.Dataset.from_tensor_slices(((m, charges_one_hot, tokenized_sequences),
                                                 np.zeros_like(m))).batch(batch_size)

        ccs, _ = self.model.predict(ds, verbose=self.verbose)

        if not return_ccs:
            data[f'inv_mobility_{self.name}'] = np.array([ccs_to_one_over_k0(c, m, z)
                                                          for c, m, z in zip(ccs, m, data.charge.values)])
        else:
            data[f'ccs_{self.name}'] = ccs

        if 'inv_mobility_gru_predictor' in data.columns:
            data = data[['peptide_id', 'sequence', 'charge', 'relative_abundance', f'inv_mobility_{self.name}']]

        else:
            data = data[['peptide_id', 'sequence', 'charge', 'relative_abundance', f'ccs_{self.name}']]

        return data

    def __repr__(self):
        return f'DeepPeptideIonMobilityApex(name={self.name}, model={self.model})'
