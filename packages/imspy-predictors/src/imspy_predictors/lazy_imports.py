"""
Lazy imports for optional dependencies.

This module provides lazy import functions for packages that are optional
dependencies (e.g., sagepy from imspy-search). This allows imspy-predictors
to work without these dependencies for basic functionality, while still
providing integration when they are available.
"""


def get_sagepy_psm():
    """
    Lazy import of sagepy Psm class.

    Returns:
        The Psm class from sagepy.core.scoring

    Raises:
        ImportError: If sagepy is not installed
    """
    try:
        from sagepy.core.scoring import Psm
        return Psm
    except ImportError:
        raise ImportError(
            "sagepy is required for PSM-based predictions. "
            "Install imspy-search package for this functionality."
        )


def get_sagepy_psm_utils():
    """
    Lazy import of sagepy Psm class and psm_collection_to_pandas utility.

    Returns:
        Tuple of (Psm, psm_collection_to_pandas)

    Raises:
        ImportError: If sagepy is not installed
    """
    try:
        from sagepy.core.scoring import Psm
        from sagepy.utility import psm_collection_to_pandas
        return Psm, psm_collection_to_pandas
    except ImportError:
        raise ImportError(
            "sagepy is required for PSM-based predictions. "
            "Install imspy-search package for this functionality."
        )


def get_sagepy_fragment_utils():
    """
    Lazy import of sagepy fragment ion utilities.

    Returns:
        Tuple of (associate_fragment_ions_with_prosit_predicted_intensities, Psm)

    Raises:
        ImportError: If sagepy is not installed
    """
    try:
        from sagepy.core.scoring import associate_fragment_ions_with_prosit_predicted_intensities, Psm
        return associate_fragment_ions_with_prosit_predicted_intensities, Psm
    except ImportError:
        raise ImportError(
            "sagepy is required for PSM-based predictions. "
            "Install imspy-search package for this functionality."
        )


def get_search_rt_utils():
    """
    Lazy import of RT dataset utilities from imspy-search.

    Returns:
        generate_balanced_rt_dataset function

    Raises:
        ImportError: If imspy-search is not installed
    """
    try:
        from imspy_search.utility import generate_balanced_rt_dataset
        return generate_balanced_rt_dataset
    except ImportError:
        raise ImportError(
            "generate_balanced_rt_dataset requires imspy-search package."
        )


def get_search_im_utils():
    """
    Lazy import of ion mobility dataset utilities from imspy-search.

    Returns:
        generate_balanced_im_dataset function

    Raises:
        ImportError: If imspy-search is not installed
    """
    try:
        from imspy_search.utility import generate_balanced_im_dataset
        return generate_balanced_im_dataset
    except ImportError:
        raise ImportError(
            "generate_balanced_im_dataset requires imspy-search package."
        )


def get_simulation_flatten_prosit():
    """
    Lazy import of flatten_prosit_array from imspy-simulation.

    Returns:
        flatten_prosit_array function

    Raises:
        ImportError: If imspy-simulation is not installed
    """
    try:
        from imspy_simulation.utility import flatten_prosit_array
        return flatten_prosit_array
    except ImportError:
        raise ImportError(
            "flatten_prosit_array requires imspy-simulation package."
        )
