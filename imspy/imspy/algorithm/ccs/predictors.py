import numpy as np
import tensorflow as tf
import importlib.resources as resources

"""
def get_model_path(model_name: str) -> str:
    return resources.files('ionmob.pretrained_models').joinpath(model_name)


def get_gru_predictor(model_name: str = 'GRUPredictor') -> tf.keras.models.Model:
    return tf.keras.models.load_model(get_model_path(model_name))
"""


class ProjectToInitialSqrtCCS(tf.keras.layers.Layer):
    """
    Simple sqrt regression layer, calculates ccs value as linear mapping from mz, charge -> ccs
    """

    def __init__(self, slopes, intercepts):
        super(ProjectToInitialSqrtCCS, self).__init__()
        self.slopes = tf.constant([slopes])
        self.intercepts = tf.constant([intercepts])

    def call(self, inputs, **kwargs):
        mz, charge = inputs[0], inputs[1]
        # since charge is one-hot encoded, can use it to gate linear prediction by charge state
        return tf.expand_dims(tf.reduce_sum((self.slopes * tf.sqrt(mz) + self.intercepts) * tf.squeeze(charge), axis=1),
                              1)


class GRUCCSPredictor(tf.keras.models.Model):
    """
    Deep Learning model combining initial linear fit with sequence based features, both scalar and complex
    Model architecture is inspired by Meier et al.: https://doi.org/10.1038/s41467-021-21352-8
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
