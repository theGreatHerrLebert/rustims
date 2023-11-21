import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from imspy.chemistry import ccs_to_one_over_k0


class PeptideIonMobilityApex(ABC):
    """
    ABSTRACT INTERFACE for simulation of ion-mobility apex value
    """

    def __init__(self):
        pass

    @abstractmethod
    def simulate_ion_mobility(self, sequence: str, charge: int):
        pass

    @abstractmethod
    def simulate_ion_mobilities(self, sequences: list[str], charges: list[int]):
        pass


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


class GRUIonMobilityPredictor(tf.keras.models.Model):
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
        super(GRUIonMobilityPredictor, self).__init__()
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

    def call(self, inputs):
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
    def __init__(self, model: GRUIonMobilityPredictor, tokenizer: tf.keras.preprocessing.text.Tokenizer):
        super(DeepPeptideIonMobilityApex, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

    def _preprocess_sequences(self, sequences: list[str], pad_len: int = 50):
        # TODO: change to UNIMOD annotated sequences
        char_tokens = [s.split(' ') for s in sequences]
        char_tokens = self.tokenizer.texts_to_sequences(char_tokens)
        char_tokens = tf.keras.preprocessing.sequence.pad_sequences(char_tokens, pad_len, padding='post')
        return char_tokens

    def _preprocess_sequence(self, sequence: str, seq_len: int = 50):
        # TODO: change to UNIMOD annotated sequence
        char_tokens = sequence.split(' ')
        char_tokens = self.tokenizer.texts_to_sequences([char_tokens])
        char_tokens = tf.keras.preprocessing.sequence.pad_sequences(char_tokens, seq_len, padding='post')
        return char_tokens

    def simulate_ion_mobility(self, sequence: str, charge: int, mz: float, verbose: bool = False):
        tokenized_sequence = self._preprocess_sequence(sequence)

        # prepare masses, charges, sequences
        m = np.expand_dims(np.array([mz]), 1)
        charges_one_hot = tf.one_hot(np.array([charge]) - 1, 4)

        ds = tf.data.Dataset.from_tensor_slices(((m, charges_one_hot, tokenized_sequence), np.zeros_like(m))).batch(1)

        ccs, _ = self.model.predict(ds, verbose=verbose)

        return ccs_to_one_over_k0(ccs[0], mz, charge)[0]

    def simulate_ion_mobilities(self, sequences: list[str], charges: list[int], mz: list[float], verbose: bool = False):
        tokenized_sequences = self._preprocess_sequences(sequences)

        # prepare masses, charges, sequences
        m = np.expand_dims(mz, 1)
        charges_one_hot = tf.one_hot(np.array(charges) - 1, 4)

        ds = tf.data.Dataset.from_tensor_slices(((m, charges_one_hot, tokenized_sequences), np.zeros_like(mz))).batch(1024)
        ccs, _ = self.model.predict(ds, verbose=verbose)

        return np.array([ccs_to_one_over_k0(c, m, z) for c, m, z in zip(ccs, mz, charges)])