from typing import Union, List

import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from sagepy.core import PeptideSpectrumMatch
from sagepy.utility import peptide_spectrum_match_collection_to_pandas

from imspy.algorithm.utility import get_model_path, InMemoryCheckpoint, load_tokenizer_from_resources
from imspy.timstof.dbsearch.utility import linear_map, generate_balanced_rt_dataset
from imspy.utility import tokenize_unimod_sequence

from tensorflow.keras.models import load_model


def predict_retention_time(
        psm_collection: List[PeptideSpectrumMatch],
        refine_model: bool = True,
        verbose: bool = False) -> None:
    """
    Predict the retention times for a collection of peptide-spectrum matches
    Args:
        psm_collection: a list of peptide-spectrum matches
        refine_model: whether to refine the model
        verbose: whether to print verbose output

    Returns:
        None, retention times are set in the peptide-spectrum matches
    """

    rt_predictor = DeepChromatographyApex(load_deep_retention_time_predictor(),
                                              load_tokenizer_from_resources("tokenizer-ptm"),
                                              verbose=verbose)
    if refine_model:
        rt_predictor.fine_tune_model(
            generate_balanced_rt_dataset(peptide_spectrum_match_collection_to_pandas(psm_collection)),
            batch_size=128,
            re_compile=True,
            verbose=verbose
        )

    # predict retention times
    inv_mob = rt_predictor.simulate_separation_times(
        sequences=[x.sequence for x in psm_collection],
    )

    rt_min = np.min([x.retention_time_observed for x in psm_collection])
    rt_max = np.max([x.retention_time_observed for x in psm_collection])

    # set the predicted retention times
    for rt, ps in zip(inv_mob, psm_collection):
        ps.retention_time_predicted = rt
        # map the retention times to a 0-60 scale, which is the range the model was trained on
        # projected RT can be used if no fine-tuning was performed
        ps.rt_projected = linear_map(rt, old_min=rt_min, old_max=rt_max, new_min=0, new_max=60)


def get_rt_train_set(tokenizer, sequence, rt) -> tf.data.Dataset:
    sequences = tokenizer.texts_to_sequences(sequence)
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, 50, padding='post')
    return tf.data.Dataset.from_tensor_slices((seq_padded, rt))


def get_rt_prediction_set(tokenizer, sequence) -> tf.data.Dataset:
    sequences = tokenizer.texts_to_sequences(sequence)
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, 50, padding='post')
    target = np.squeeze(np.zeros(len(sequence)))

    return tf.data.Dataset.from_tensor_slices((seq_padded, target))


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
        x_recurrent = self.gru2(self.gru1(self.emb(inputs)))
        d1 = self.dropout(self.dense1(x_recurrent), training=training)
        d2 = self.dense2(d1)
        out = self.out(d2)
        return out

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

    def generate_tf_ds_inference(self, sequences: list[str]) -> tf.data.Dataset:
        char_tokens = [tokenize_unimod_sequence(seq) for seq in sequences]
        return get_rt_prediction_set(tokenizer=self.tokenizer, sequence=char_tokens)

    def generate_tf_ds_train(self, sequences: list[str], rt_target) -> tf.data.Dataset:
        char_tokens = [tokenize_unimod_sequence(seq) for seq in sequences]
        return get_rt_train_set(tokenizer=self.tokenizer, sequence=char_tokens, rt=rt_target)

    def simulate_separation_times(self,
                                  sequences: list[str],
                                  batch_size: int = 1024) -> NDArray:
        tf_ds = self.generate_tf_ds_inference(sequences).batch(batch_size)

        return self.model.predict(tf_ds, verbose=self.verbose)

    def fine_tune_model(self,
                        data: pd.DataFrame,
                        batch_size: int = 64,
                        re_compile=False,
                        verbose=False
                        ):
        assert 'sequence' in data.columns, 'Data must contain a column named "sequence"'
        assert 'projected_rt' in data.columns, 'Data must contain a column named "projected_rt"'

        sequences = data.sequence.values
        rts = data.projected_rt.values
        ds = self.generate_tf_ds_train(sequences, rt_target=rts).shuffle(len(sequences))

        # split data into training and validation
        n = len(sequences)
        n_train = int(0.8 * n)
        n_val = n - n_train

        ds_train = ds.take(n_train).batch(batch_size)
        ds_val = ds.skip(n_train).take(n_val).batch(batch_size)

        checkpoint = InMemoryCheckpoint()

        if re_compile:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mean_absolute_error')

        self.model.fit(ds_train, verbose=verbose, epochs=150, validation_data=ds_val,
                       # use early stopping and learning rate reduction where
                       callbacks=[tf.keras.callbacks.EarlyStopping(patience=6), checkpoint,
                                  tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-6, patience=3)])

    def simulate_separation_times_pandas(self,
                                         data: pd.DataFrame,
                                         batch_size: int = 1024,
                                         gradient_length: Union[None, float] = None
                                         ) -> pd.DataFrame:

        assert 'sequence' in data.columns, 'Data must contain a column named "sequence"'
        sequences = data.sequence.values
        tf_ds = self.generate_tf_ds_inference(sequences).batch(batch_size)

        rts = self.model.predict(tf_ds, verbose=self.verbose)
        if gradient_length is not None:
            rts = linear_map(rts, old_min=rts.min(), old_max=rts.max(), new_min=0, new_max=gradient_length)

        data[f'retention_time_{self.name}'] = rts
        return data
