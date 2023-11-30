import pandas as pd
import tensorflow as tf
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from imspy.algorithm.utilities import get_model_path
from imspy.simulation.exp import ExperimentDataHandle
from imspy.utility import tokenize_unimod_sequence
from imspy.simulation.utility import irt_to_rts_numba


def load_deep_retention_time() -> tf.keras.models.Model:
    """ Get a pretrained deep predictor model

    Returns:
        The pretrained deep predictor model
    """
    return tf.keras.models.load_model(get_model_path('DeepRetentionTimePredictor'))


class PeptideChromatographyApex(ABC):
    """
    ABSTRACT INTERFACE for a chromatographic separation for peptides
    """

    def __init__(self):
        pass

    @abstractmethod
    def simulate_separation_times(self, sequences: list[str]) -> NDArray:
        pass


class GRURetentionTimePredictor(tf.keras.models.Model):

    def __init__(self,
                 num_tokens,
                 seq_len=50,
                 emb_dim=128,
                 gru_1=128,
                 gru_2=64,
                 rdo=0.0,
                 do=0.2):
        super(GRURetentionTimePredictor, self).__init__()

        self.emb = tf.keras.layers.Embedding(input_dim=num_tokens + 1, output_dim=emb_dim, input_length=seq_len)

        self.gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_1, return_sequences=True,
                                                                      name='GRU1'))

        self.gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_2, return_sequences=False,
                                                                      name='GRU2',
                                                                      recurrent_dropout=rdo))

        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name='Dense1',
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))

        self.dense2 = tf.keras.layers.Dense(64, activation='relu', name='Dense2',
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))

        self.dropout = tf.keras.layers.Dropout(do, name='Dropout')

        self.out = tf.keras.layers.Dense(1, activation=None, name='Output')

    def call(self, inputs):
        """
        :param inputs: should contain: (sequence)
        """
        # get inputs
        seq = inputs
        # sequence learning
        x_recurrent = self.gru2(self.gru1(self.emb(seq)))
        # regularize
        d1 = self.dropout(self.dense1(x_recurrent))
        # output
        return self.out(self.dense2(d1))


class DeepChromatographyApex(PeptideChromatographyApex):

    def __init__(self, model: GRURetentionTimePredictor, tokenizer: tf.keras.preprocessing.text.Tokenizer,
                 name: str = 'gru_predictor'):
        super(DeepChromatographyApex, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.name = name

    def _preprocess_sequences(self, sequences: list[str], pad_len: int = 50) -> NDArray:
        char_tokens = [tokenize_unimod_sequence(seq) for seq in sequences]
        char_tokens = self.tokenizer.texts_to_sequences(char_tokens)
        char_tokens = tf.keras.preprocessing.sequence.pad_sequences(char_tokens, pad_len, padding='post')
        return char_tokens

    def simulate_separation_times(self, sequences: list[str], batch_size: int = 1024, verbose: bool = False) -> NDArray:
        tokens = self._preprocess_sequences(sequences)
        tf_ds = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)

        return self.model.predict(tf_ds, verbose=verbose)

    def simulate_separation_times_pandas(self, data: pd.DataFrame,
                                         gradient_length: float,
                                         batch_size: int = 1024, verbose: bool = False) -> pd.DataFrame:
        tokens = self._preprocess_sequences(data.sequence.values)
        tf_ds = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)

        irts = self.model.predict(tf_ds, verbose=verbose)
        rts = irt_to_rts_numba(irts, new_max=gradient_length)
        data[f'retention_time_{self.name}'] = rts
        return data
