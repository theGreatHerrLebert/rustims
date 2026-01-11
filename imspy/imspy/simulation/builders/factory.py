"""Factory for creating frame builders.

This module provides a clean factory interface for creating frame builders
based on acquisition mode (DDA/DIA) and loading strategy (standard/lazy).
"""

from enum import Enum, auto
from pathlib import Path
from typing import Union
import logging

from imspy.simulation.core.protocols import FrameBuilder

logger = logging.getLogger(__name__)


class AcquisitionMode(Enum):
    """Acquisition mode for the simulation."""
    DDA = auto()
    DIA = auto()


class LoadingStrategy(Enum):
    """Data loading strategy for frame building.

    STANDARD: Loads all data into memory at construction time.
              Higher memory usage but potentially faster for small datasets.

    LAZY: Loads data on-demand for each batch of frames.
          Lower memory usage, better for large simulations.
          Currently only supported for DIA mode.
    """
    STANDARD = auto()
    LAZY = auto()


def create_frame_builder(
    db_path: Union[str, Path],
    acquisition_mode: AcquisitionMode,
    loading_strategy: LoadingStrategy = LoadingStrategy.STANDARD,
    num_threads: int = -1,
    with_annotations: bool = False,
) -> FrameBuilder:
    """Create a frame builder based on acquisition mode and loading strategy.

    This is the recommended way to create frame builders, as it provides
    a unified interface and handles the complexity of selecting the
    appropriate implementation.

    Args:
        db_path: Path to the synthetic data database (synthetic_data.db).
        acquisition_mode: DDA or DIA acquisition mode.
        loading_strategy: Standard (load all data upfront) or Lazy (load on-demand).
        num_threads: Number of threads for parallel processing. -1 for auto-detect.
        with_annotations: If True, enable annotation support (memory-intensive).
                         Only supported for standard loading strategy.

    Returns:
        A FrameBuilder implementation appropriate for the given parameters.

    Raises:
        ValueError: If lazy loading is requested for DDA mode.
        ValueError: If annotations are requested with lazy loading.

    Example:
        >>> builder = create_frame_builder(
        ...     db_path="/path/to/synthetic_data.db",
        ...     acquisition_mode=AcquisitionMode.DIA,
        ...     loading_strategy=LoadingStrategy.LAZY,
        ...     num_threads=4,
        ... )
        >>> frames = builder.build_frames([1, 2, 3])
    """
    db_path = str(db_path)

    # Validate parameter combinations
    if loading_strategy == LoadingStrategy.LAZY and acquisition_mode == AcquisitionMode.DDA:
        logger.warning("Lazy loading is not yet supported for DDA mode, using standard loading.")
        loading_strategy = LoadingStrategy.STANDARD

    if with_annotations and loading_strategy == LoadingStrategy.LAZY:
        raise ValueError("Annotation support is not available with lazy loading strategy.")

    # Import here to avoid circular imports
    if acquisition_mode == AcquisitionMode.DDA:
        from .dda import DDAFrameBuilder
        logger.info("Creating DDA frame builder with standard loading.")
        return DDAFrameBuilder(
            db_path=db_path,
            num_threads=num_threads,
            with_annotations=with_annotations,
        )

    elif acquisition_mode == AcquisitionMode.DIA:
        if loading_strategy == LoadingStrategy.LAZY:
            from .dia import DIAFrameBuilder
            logger.info("Creating DIA frame builder with lazy loading.")
            return DIAFrameBuilder(
                db_path=db_path,
                num_threads=num_threads,
                lazy=True,
            )
        else:
            from .dia import DIAFrameBuilder
            logger.info("Creating DIA frame builder with standard loading.")
            return DIAFrameBuilder(
                db_path=db_path,
                num_threads=num_threads,
                lazy=False,
                with_annotations=with_annotations,
            )

    raise ValueError(f"Unknown acquisition mode: {acquisition_mode}")
