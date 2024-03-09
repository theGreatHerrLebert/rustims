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
