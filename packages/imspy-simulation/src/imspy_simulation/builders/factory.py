"""Factory for creating frame builders.

This module provides a clean factory interface for creating frame builders
based on acquisition mode (DDA/DIA) and loading strategy (standard/lazy).
"""

from enum import Enum, auto
from pathlib import Path
from typing import Union
import logging

from imspy_simulation.core.protocols import FrameBuilder

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
    quad_isotope_transmission_mode: str = 'none',
    quad_transmission_min_probability: float = 0.5,
    quad_transmission_max_isotopes: int = 10,
    precursor_survival_min: float = 0.0,
    precursor_survival_max: float = 0.0,
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
        quad_isotope_transmission_mode: Mode for quad-selection dependent
            isotope transmission (DDA and DIA). Options:
            - "none": Standard isotope patterns (default)
            - "precursor_scaling": Fast mode - uniform scaling based on precursor transmission
            - "per_fragment": Accurate mode - individual fragment ion recalculation
        quad_transmission_min_probability: Minimum probability threshold for
            isotope transmission (default 0.5).
        quad_transmission_max_isotopes: Maximum number of isotope peaks to
            consider for transmission (default 10).
        precursor_survival_min: Minimum fraction of precursor ions that survive
            fragmentation intact (0.0-1.0, default 0.0).
        precursor_survival_max: Maximum fraction of precursor ions that survive
            fragmentation intact (0.0-1.0, default 0.0).

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
        if quad_isotope_transmission_mode != 'none':
            logger.info(f"Quad-dependent isotope transmission enabled: {quad_isotope_transmission_mode}")
        if precursor_survival_max > 0.0:
            logger.info(f"Precursor survival enabled: {precursor_survival_min:.2f}-{precursor_survival_max:.2f}")
        return DDAFrameBuilder(
            db_path=db_path,
            num_threads=num_threads,
            with_annotations=with_annotations,
            quad_isotope_transmission_mode=quad_isotope_transmission_mode,
            quad_transmission_min_probability=quad_transmission_min_probability,
            quad_transmission_max_isotopes=quad_transmission_max_isotopes,
            precursor_survival_min=precursor_survival_min,
            precursor_survival_max=precursor_survival_max,
        )

    elif acquisition_mode == AcquisitionMode.DIA:
        if loading_strategy == LoadingStrategy.LAZY:
            from .dia import DIAFrameBuilder
            logger.info("Creating DIA frame builder with lazy loading.")
            if quad_isotope_transmission_mode != 'none':
                logger.warning("Quad-dependent isotope transmission is not supported with lazy loading, ignoring.")
            if precursor_survival_max > 0.0:
                logger.warning("Precursor survival is not supported with lazy loading, ignoring.")
            return DIAFrameBuilder(
                db_path=db_path,
                num_threads=num_threads,
                lazy=True,
            )
        else:
            from .dia import DIAFrameBuilder
            logger.info("Creating DIA frame builder with standard loading.")
            if quad_isotope_transmission_mode != 'none':
                logger.info(f"Quad-dependent isotope transmission enabled: {quad_isotope_transmission_mode}")
            if precursor_survival_max > 0.0:
                logger.info(f"Precursor survival enabled: {precursor_survival_min:.2f}-{precursor_survival_max:.2f}")
            return DIAFrameBuilder(
                db_path=db_path,
                num_threads=num_threads,
                lazy=False,
                with_annotations=with_annotations,
                quad_isotope_transmission_mode=quad_isotope_transmission_mode,
                quad_transmission_min_probability=quad_transmission_min_probability,
                quad_transmission_max_isotopes=quad_transmission_max_isotopes,
                precursor_survival_min=precursor_survival_min,
                precursor_survival_max=precursor_survival_max,
            )

    raise ValueError(f"Unknown acquisition mode: {acquisition_mode}")
