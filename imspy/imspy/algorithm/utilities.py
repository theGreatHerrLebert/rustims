import importlib.resources as resources
from importlib.abc import Traversable


def get_model_path(model_name: str) -> Traversable:
    """ Get the path to a pretrained model

    Args:
        model_name: The name of the model to load

    Returns:
        The path to the pretrained model
    """
    return resources.files('imspy.algorithm.pretrained').joinpath(model_name)
