from typing import Union, List
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from imspy_predictors.utility import get_model_path, InMemoryCheckpoint, load_tokenizer_from_resources
from imspy_core.utility import tokenize_unimod_sequence

from tensorflow.keras.models import load_model


# Lazy import for sagepy (optional dependency, requires imspy-search)
def _get_sagepy_utils():
    """Lazy import of sagepy utilities. Requires imspy-search package."""
    try:
        from sagepy.core.scoring import Psm
        from sagepy.utility import psm_collection_to_pandas
        return Psm, psm_collection_to_pandas
    except ImportError:
        raise ImportError(
            "sagepy is required for PSM-based predictions. "
            "Install imspy-search package for this functionality."
        )


# Lazy import for dbsearch utility (optional, requires imspy-search)
def _get_dbsearch_utils():
    """Lazy import of dbsearch utilities. Requires imspy-search package."""
    try:
        from imspy_search.utility import linear_map, generate_balanced_rt_dataset
        return linear_map, generate_balanced_rt_dataset
    except ImportError:
        raise ImportError(
            "dbsearch utilities require imspy-search package."
        )


def linear_map(x, old_min, old_max, new_min, new_max):
    """Map values from one range to another linearly."""
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def predict_retention_time(
        psm_collection: List,
        refine_model: bool = True,
        verbose: bool = False) -> None:
    """
    Predict the retention times for a collection of peptide-spectrum matches.

    Note: This function requires sagepy (via imspy-search package).

    Args:
        psm_collection: a list of peptide-spectrum matches (sagepy Psm objects)
        refine_model: whether to refine the model
        verbose: whether to print verbose output

    Returns:
        None, retention times are set in the peptide-spectrum matches
    """
    Psm, psm_collection_to_pandas = _get_sagepy_utils()
    _linear_map, generate_balanced_rt_dataset = _get_dbsearch_utils()

    rt_predictor = DeepChromatographyApex(load_deep_retention_time_predictor(),
                                              load_tokenizer_from_resources("tokenizer-ptm"),
                                              verbose=verbose)

    rt_min = np.min([x.retention_time for x in psm_collection])
    rt_max = np.max([x.retention_time for x in psm_collection])

    for psm in psm_collection:
        psm.retention_time_projected = _linear_map(psm.retention_time, old_min=rt_min, old_max=rt_max, new_min=0, new_max=60)

    if refine_model:
        rt_predictor.fine_tune_model(
            psm_collection_to_pandas(generate_balanced_rt_dataset(psm_collection)),
            batch_size=128,
            re_compile=True,
            verbose=verbose
        )

    # predict retention times
    rt_predicted = rt_predictor.simulate_separation_times(
        sequences=[x.sequence_modified if x.decoy == False else x.sequence_decoy_modified for x in psm_collection],
    )

    # set the predicted retention times
    for rt, ps in zip(rt_predicted, psm_collection):
        ps.retention_time_predicted = rt


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

    return load_model(path, custom_objects=custom_objects, compile=False)


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


@keras.saving.register_keras_serializable()
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
                        verbose=False,
                        decoys_separate=True,
                        ):
        assert 'sequence' in data.columns, 'Data must contain a column named "sequence"'
        assert 'retention_time_projected' in data.columns, 'Data must contain a column named "retention_time_projected"'

        sequences = []

        if decoys_separate:
            for index, row in data.iterrows():
                if not row.decoy:
                    sequences.append(row.sequence)
                else:
                    sequences.append(row.sequence_decoy_modified)
        else:
            sequences = data.sequence.values

        rts = data.retention_time_projected.values
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
                                         gradient_length: Union[None, float] = None,
                                         decoys_separate=True,
                                         ) -> pd.DataFrame:

        assert 'sequence' in data.columns, 'Data must contain a column named "sequence"'

        sequences = []
        if decoys_separate:
            for index, row in data.iterrows():
                if not row.decoy:
                    try:
                        sequences.append(row.sequence_modified)
                    except AttributeError:
                        sequences.append(row.sequence)
                else:
                    try:
                        sequences.append(row.sequence_decoy_modified)
                    except AttributeError:
                        sequences.append(row.sequence_decoy)
        else:
            sequences = data.sequence.values

        tf_ds = self.generate_tf_ds_inference(sequences).batch(batch_size)

        rts = self.model.predict(tf_ds, verbose=self.verbose)
        if gradient_length is not None:
            rts = linear_map(rts, old_min=rts.min(), old_max=rts.max(), new_min=0, new_max=gradient_length)

        data[f"retention_time_{self.name}"] = rts
        return data


def predict_retention_time_with_koina(
    model_name,
        data,
        seq_col="sequence",
        gradient_length=None,
        verbose=False,
):
    """
    Predict retention times using Koina.
    Args:
        model_name: Name of the model.
        data: DataFrame with peptide sequence.

    Returns:
        DataFrame with predicted retention times.
    """
    from imspy_predictors.koina_models import ModelFromKoina

    rt_model_from_koina = ModelFromKoina(model_name=model_name)
    inputs = data.copy()
    inputs.rename(columns={seq_col: "peptide_sequences"}, inplace=True)
    rts = rt_model_from_koina.predict(inputs[["peptide_sequences"]])

    if verbose:
        print(f"[DEBUG] Koina model {model_name} predicted retention times for {len(rts)} peptides. Columns: {rts.columns}")

    if gradient_length is not None:
        mapped_rt = linear_map(
            rts.iloc[:, 1].values,
            old_min=rts.iloc[:, 1].min(),
            old_max=rts.iloc[:, 1].max(),
            new_min=0,
            new_max=gradient_length,
        )
    data["retention_time_gru_predictor"] = (
        mapped_rt  # FIXME: dirty naming for downstream compatibility
    )
    return data
