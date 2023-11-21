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
