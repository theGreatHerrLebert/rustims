import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from imspy.algorithm.utility import get_model_path
from imspy.utility import tokenize_unimod_sequence
from imspy.simulation.utility import irt_to_rts_numba

from tensorflow.keras.models import load_model


def get_rt_train_set(tokenizer, sequence, rt, rt_min=0.0, rt_max=60.0) -> tf.data.Dataset:
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(sequence),
                                                               50, padding='post')
    rt_min = np.expand_dims(np.repeat(rt_min, len(sequence)), 1)
    rt_max = np.expand_dims(np.repeat(rt_max, len(sequence)), 1)

    return tf.data.Dataset.from_tensor_slices(((rt_min, rt_max, seq_padded), rt))


def get_prediction_set(tokenizer, sequence, rt_min=0.0, rt_max=60.0) -> tf.data.Dataset:
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(sequence),
                                                               50, padding='post')
    rt_min = np.expand_dims(np.repeat(rt_min, len(sequence)), 1)
    rt_max = np.expand_dims(np.repeat(rt_max, len(sequence)), 1)

    target = np.squeeze(np.zeros_like(rt_min))

    return tf.data.Dataset.from_tensor_slices(((rt_min, rt_max, seq_padded), target))


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
class LinearProjection(tf.keras.layers.Layer):
    def __init__(self, train_min, train_max, **kwargs):
        super(LinearProjection, self).__init__(**kwargs)
        self.train_min = tf.constant(train_min, dtype=tf.float32)
        self.train_max = tf.constant(train_max, dtype=tf.float32)
        self.train_min_init = train_min
        self.train_max_init = train_max

    def call(self, inputs):
        value, new_min, new_max = inputs
        scale = (new_max - new_min) / (self.train_max - self.train_min)
        offset = new_min - self.train_min * scale
        return value * scale + offset

    def get_config(self):
        config = super(LinearProjection, self).get_config()
        config.update({
            'train_min': self.train_min_init,
            'train_max': self.train_max_init,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.saving.register_keras_serializable()
class GRURetentionTimePredictor(tf.keras.models.Model):
    def __init__(self, num_tokens, train_min, train_max, seq_len=50, emb_dim=128, gru_1=128, gru_2=64, rdo=0.0, do=0.2):
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
        self.linear_projection = LinearProjection(train_min, train_max)
        self.train_min = train_min
        self.train_max = train_max

    def build(self, input_shape):
        self.emb.build((input_shape[2][0], self.seq_len))
        self.gru1.build((input_shape[2][0], self.seq_len, self.emb.output_dim))
        gru1_output_dim = self.gru1.forward_layer.units + self.gru1.backward_layer.units
        self.gru2.build((input_shape[2][0], self.seq_len, gru1_output_dim))
        gru2_output_dim = self.gru2.forward_layer.units + self.gru2.backward_layer.units
        self.dense1.build((input_shape[2][0], gru2_output_dim))
        self.dense2.build((input_shape[2][0], self.dense1.units))
        self.dropout.build((input_shape[2][0], self.dense1.units))
        self.out.build((input_shape[2][0], self.dense2.units))
        self.linear_projection.build((input_shape[2][0], 1))
        super(GRURetentionTimePredictor, self).build(input_shape)

    def call(self, inputs, training=False):
        new_min, new_max, seq = inputs[0], inputs[1], inputs[2]
        x_recurrent = self.gru2(self.gru1(self.emb(seq)))
        d1 = self.dropout(self.dense1(x_recurrent), training=training)
        d2 = self.dense2(d1)
        out = self.out(d2)
        return self.linear_projection([out, new_min, new_max])

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
            "train_min": self.train_min,
            "train_max": self.train_max,
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

    def generate_tf_ds_inference(self, sequences: list[str], rt_min: float, rt_max: float) -> tf.data.Dataset:
        char_tokens = [tokenize_unimod_sequence(seq) for seq in sequences]
        return get_prediction_set(tokenizer=self.tokenizer, sequence=char_tokens, rt_min=rt_min, rt_max=rt_max)

    def generate_tf_ds_train(self, sequences: list[str], rt_target, rt_min: float, rt_max: float) -> tf.data.Dataset:
        char_tokens = [tokenize_unimod_sequence(seq) for seq in sequences]
        return get_rt_train_set(tokenizer=self.tokenizer, sequence=char_tokens, rt=rt_target, rt_min=rt_min, rt_max=rt_max)

    def simulate_separation_times(self,
                                  sequences: list[str],
                                  rt_min: float = 0.0,
                                  rt_max: float = 60.0,
                                  batch_size: int = 1024) -> NDArray:
        tf_ds = self.generate_tf_ds_inference(sequences, rt_min, rt_max).batch(batch_size)

        return self.model.predict(tf_ds, verbose=self.verbose)

    def fine_tune_model(self,
                        data: pd.DataFrame,
                        rt_min: float = 0.0,
                        rt_max: float = 60.0,
                        batch_size: int = 64,
                        re_compile=False,
                        verbose=False
                        ):
        assert 'sequence' in data.columns, 'Data must contain a column named "sequence"'
        assert 'retention_time_observed' in data.columns, 'Data must contain a column named "retention_time_observed"'

        sequences = data.sequence.values
        rts = data.retention_time_observed.values
        ds = self.generate_tf_ds_train(sequences, rt_target=rts, rt_min=rt_min, rt_max=rt_max).shuffle(len(sequences))

        # split data into training and validation
        n = len(sequences)
        n_train = int(0.8 * n)
        n_val = n - n_train

        ds_train = ds.take(n_train).batch(batch_size)
        ds_val = ds.skip(n_train).take(n_val).batch(batch_size)

        if re_compile:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_absolute_error')

        self.model.fit(ds_train, verbose=verbose, epochs=150, validation_data=ds_val,
                       # use early stopping and learning rate reduction where
                       callbacks=[tf.keras.callbacks.EarlyStopping(patience=6),
                                  tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-6, patience=3)])

    def simulate_separation_times_pandas(self,
                                         data: pd.DataFrame,
                                         rt_min: float = 0.0,
                                         rt_max: float = 60.0, batch_size: int = 1024) -> pd.DataFrame:

        assert 'sequence' in data.columns, 'Data must contain a column named "sequence"'
        sequences = data.sequence.values
        tf_ds = self.generate_tf_ds_inference(sequences, rt_min, rt_max).batch(batch_size)

        rts = self.model.predict(tf_ds, verbose=self.verbose)
        data[f'retention_time_{self.name}'] = rts
        return data
