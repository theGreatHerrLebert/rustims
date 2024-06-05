import pandas as pd
import tensorflow as tf
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from imspy.algorithm.utility import get_model_path
from imspy.utility import tokenize_unimod_sequence
from imspy.simulation.utility import irt_to_rts_numba

from tensorflow.keras.models import load_model


def load_deep_retention_time_predictor() -> tf.keras.models.Model:

    path = get_model_path('rt/rtpred-26-05-2024.keras')

    # Ensure that the custom objects are registered when loading the model
    custom_objects = {
        'GRURetentionTimePredictor': GRURetentionTimePredictor
    }

    return load_model(path, custom_objects=custom_objects)


class PeptideChromatographyApex(ABC):
    """
    ABSTRACT INTERFACE for a chromatographic separation for peptides
    """

    def __init__(self):
        pass

    @abstractmethod
    def simulate_separation_times(self, sequences: list[str]) -> NDArray:
        pass

    @abstractmethod
    def simulate_separation_times_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


@tf.keras.saving.register_keras_serializable()
class GRURetentionTimePredictor(tf.keras.models.Model):

    def __init__(self, num_tokens, seq_len=50, emb_dim=128, gru_1=128, gru_2=64, rdo=0.0, do=0.2):
        super(GRURetentionTimePredictor, self).__init__()
        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.emb = tf.keras.layers.Embedding(input_dim=num_tokens + 1, output_dim=emb_dim)
        self.gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_1, return_sequences=True, name='GRU1'))
        self.gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_2, return_sequences=False, name='GRU2', recurrent_dropout=rdo))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name='Dense1', kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu', name='Dense2', kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))
        self.dropout = tf.keras.layers.Dropout(do, name='Dropout')
        self.out = tf.keras.layers.Dense(1, activation=None, name='Output')

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
        super(GRURetentionTimePredictor, self).build(input_shape)

    def call(self, inputs, training=False):
        seq = inputs
        x_recurrent = self.gru2(self.gru1(self.emb(seq)))
        d1 = self.dropout(self.dense1(x_recurrent), training=training)
        d2 = self.dense2(d1)
        return self.out(d2)

    def get_config(self):
        config = super(GRURetentionTimePredictor, self).get_config()
        config.update({
            "num_tokens": self.num_tokens,
            "seq_len": self.seq_len,
            "emb_dim": self.emb.output_dim,
            "gru_1": self.gru1.forward_layer.units,
            "gru_2": self.gru2.forward_layer.units,
            "rdo": self.gru2.forward_layer.recurrent_dropout,
            "do": self.dropout.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DeepChromatographyApex(PeptideChromatographyApex):

    def __init__(self, model: GRURetentionTimePredictor, tokenizer: tf.keras.preprocessing.text.Tokenizer,
                 name: str = 'gru_predictor', verbose: bool = False):
        super(DeepChromatographyApex, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.verbose = verbose

    def _preprocess_sequences(self, sequences: list[str], pad_len: int = 50) -> NDArray:
        char_tokens = [tokenize_unimod_sequence(seq) for seq in sequences]
        char_tokens = self.tokenizer.texts_to_sequences(char_tokens)
        char_tokens = tf.keras.preprocessing.sequence.pad_sequences(char_tokens, pad_len, padding='post')
        return char_tokens

    def simulate_separation_times(self, sequences: list[str], batch_size: int = 1024) -> NDArray:
        tokens = self._preprocess_sequences(sequences)
        tf_ds = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)

        return self.model.predict(tf_ds, verbose=self.verbose)

    def fit_model(self, data: pd.DataFrame, epochs: int = 10, batch_size: int = 1024, re_compile=False):
        assert 'sequence' in data.columns, 'Data must contain a column named "sequence"'
        assert 'retention_time_observed' in data.columns, 'Data must contain a column named "retention_time_observed"'
        tokens = self._preprocess_sequences(data.sequence.values)
        rts = data.retention_time_observed.values
        tf_ds = tf.data.Dataset.from_tensor_slices((tokens, rts)).shuffle(len(data)).batch(batch_size)
        if re_compile:
            self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(tf_ds, epochs=epochs, verbose=False)

    def simulate_separation_times_pandas(self, data: pd.DataFrame,
                                         gradient_length: float, batch_size: int = 1024) -> pd.DataFrame:
        tokens = self._preprocess_sequences(data.sequence.values)
        tf_ds = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)

        irts = self.model.predict(tf_ds, verbose=self.verbose)
        rts = irt_to_rts_numba(irts, new_max=gradient_length)
        data[f'retention_time_{self.name}'] = rts
        return data
