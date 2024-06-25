import numpy as np
import tensorflow as tf
import importlib.resources as resources
from imspy.utility.utilities import tokenizer_from_json
from importlib.abc import Traversable


def get_model_path(model_name: str) -> Traversable:
    """ Get the path to a pretrained model

    Args:
        model_name: The name of the model to load

    Returns:
        The path to the pretrained model
    """
    return resources.files('imspy.algorithm.pretrained').joinpath(model_name)


def load_tokenizer_from_resources(tokenizer_name: str) -> tf.keras.preprocessing.text.Tokenizer:
    """ Load a tokenizer from resources

    Returns:
        The pretrained tokenizer
    """
    return tokenizer_from_json(resources.files('imspy.algorithm.pretrained').joinpath(f'{tokenizer_name}.json'))


class InMemoryCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, validation_target="val_loss"):
        super(InMemoryCheckpoint, self).__init__()
        self.best_weights = None
        self.best_val_loss = np.Inf
        self.initial_weights = None
        self.validation_target = validation_target

    def on_train_begin(self, logs=None):
        self.initial_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get(self.validation_target)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
