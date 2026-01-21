"""
Collision Energy Model for m/z-dependent intensity predictions.

This module provides a linear collision energy model that computes CE based on
precursor m/z, matching typical timsTOF instrument behavior where CE ramps with m/z.

The model follows: CE = intercept + slope * m/z

Typical timsTOF values:
- intercept: ~20 NCE
- slope: ~0.015 NCE/Da

This gives CE(500) ≈ 27.5, CE(1000) ≈ 35.0, CE(1500) ≈ 42.5

Example usage:
    # Extract from TDF dataset
    from imspy.timstof import TimsDatasetDDA
    dataset = TimsDatasetDDA("data.d")
    ce_model = CollisionEnergyModel.from_tdf_dataset(dataset)

    # Predict CE for m/z values
    ce_model.predict(1000.0)  # Single m/z
    ce_model.predict_batch(np.array([500.0, 1000.0, 1500.0]))  # Batch

    # Calibrate intercept offset using PSM hits
    calibrated_model, similarity = calibrate_collision_energy_model(
        psms=good_psms,
        model=prosit_model,
        initial_ce_model=ce_model,
    )
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, List, Optional
import logging

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    from imspy.timstof import TimsDatasetDDA
    from .predictors import Prosit2023TimsTofWrapper
    from sagepy.core.scoring import Psm

logger = logging.getLogger(__name__)


# Default CE model parameters (typical timsTOF ramp)
DEFAULT_CE_INTERCEPT = 20.0
DEFAULT_CE_SLOPE = 0.015  # NCE per Da


@dataclass
class CollisionEnergyModel:
    """Linear collision energy model: CE = intercept + slope * m/z

    This model represents the typical collision energy ramp used in timsTOF
    instruments, where CE increases linearly with precursor m/z.

    Attributes:
        intercept: Base collision energy (NCE) at m/z = 0
        slope: CE increase per Da of m/z (NCE/Da)

    Example:
        >>> model = CollisionEnergyModel(intercept=20.0, slope=0.015)
        >>> model.predict(1000.0)
        35.0
        >>> model.predict_batch(np.array([500.0, 1000.0, 1500.0]))
        array([27.5, 35.0, 42.5])
    """
    intercept: float = DEFAULT_CE_INTERCEPT
    slope: float = DEFAULT_CE_SLOPE

    def predict(self, mz: float) -> float:
        """Compute collision energy for a single m/z value.

        Args:
            mz: Precursor m/z value

        Returns:
            Collision energy in NCE
        """
        return self.intercept + self.slope * mz

    def predict_batch(self, mz_array: NDArray) -> NDArray:
        """Compute collision energy for an array of m/z values.

        Args:
            mz_array: Array of precursor m/z values

        Returns:
            Array of collision energies in NCE
        """
        return self.intercept + self.slope * np.asarray(mz_array)

    def with_intercept_offset(self, offset: float) -> 'CollisionEnergyModel':
        """Return a new model with adjusted intercept.

        This is useful for calibration where only the intercept (not slope)
        is adjusted based on observed data.

        Args:
            offset: Amount to add to the intercept

        Returns:
            New CollisionEnergyModel with adjusted intercept
        """
        return CollisionEnergyModel(
            intercept=self.intercept + offset,
            slope=self.slope
        )

    @classmethod
    def fit(cls, mz: NDArray, ce: NDArray) -> 'CollisionEnergyModel':
        """Fit a linear CE model from observed (m/z, CE) data points.

        Uses simple linear regression to fit CE = intercept + slope * mz.

        Args:
            mz: Array of precursor m/z values
            ce: Array of corresponding collision energies

        Returns:
            Fitted CollisionEnergyModel
        """
        mz = np.asarray(mz)
        ce = np.asarray(ce)

        if len(mz) != len(ce):
            raise ValueError("mz and ce arrays must have the same length")

        if len(mz) < 2:
            raise ValueError("Need at least 2 data points to fit a linear model")

        # Simple linear regression: CE = intercept + slope * mz
        # Using numpy's polyfit with degree 1
        coeffs = np.polyfit(mz, ce, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        return cls(intercept=intercept, slope=slope)

    @classmethod
    def from_tdf_dataset(cls, dataset: 'TimsDatasetDDA') -> 'CollisionEnergyModel':
        """Extract CE model from a TDF dataset's PASEF metadata.

        For DDA data, extracts (isolation_mz, collision_energy) pairs from
        the PasefFrameMsMsInfo table and fits a linear model.

        Args:
            dataset: TimsDatasetDDA instance

        Returns:
            Fitted CollisionEnergyModel

        Raises:
            ValueError: If dataset has no PASEF metadata or too few data points
        """
        meta = dataset.pasef_meta_data

        if meta is None or len(meta) == 0:
            raise ValueError("Dataset has no PASEF metadata")

        # Extract m/z and CE columns
        mz = meta['isolation_mz'].values
        ce = meta['collision_energy'].values

        # Filter out any invalid values
        valid_mask = (mz > 0) & (ce > 0) & np.isfinite(mz) & np.isfinite(ce)
        mz = mz[valid_mask]
        ce = ce[valid_mask]

        if len(mz) < 2:
            raise ValueError(
                f"Too few valid (mz, CE) data points: {len(mz)}. "
                "Need at least 2 to fit a linear model."
            )

        model = cls.fit(mz, ce)
        logger.info(
            f"Fitted CE model from TDF: CE = {model.intercept:.2f} + {model.slope:.5f} * m/z "
            f"(from {len(mz)} data points)"
        )
        return model

    @classmethod
    def default(cls) -> 'CollisionEnergyModel':
        """Return the default CE model (20 + 0.015 * m/z).

        This represents typical timsTOF CE ramp settings.

        Returns:
            Default CollisionEnergyModel instance
        """
        return cls()

    @classmethod
    def constant(cls, ce: float = 35.0) -> 'CollisionEnergyModel':
        """Return a constant CE model (slope = 0).

        This is useful for backward compatibility with fixed CE predictions.

        Args:
            ce: Constant collision energy value (default: 35.0 NCE)

        Returns:
            CollisionEnergyModel with slope=0
        """
        return cls(intercept=ce, slope=0.0)

    def __repr__(self) -> str:
        return f"CollisionEnergyModel(intercept={self.intercept:.2f}, slope={self.slope:.5f})"


def calibrate_collision_energy_model(
    psms: List['Psm'],
    model: 'Prosit2023TimsTofWrapper',
    initial_ce_model: CollisionEnergyModel,
    lower: int = -30,
    upper: int = 30,
    max_psms: int = 2048,
    verbose: bool = False,
) -> Tuple[CollisionEnergyModel, float, List[Tuple[int, float]]]:
    """Calibrate CE model intercept using top PSM hits.

    This function performs a grid search over intercept offsets to find the
    value that maximizes spectral angle similarity between predicted and
    observed intensities. Only the intercept is calibrated (not slope) for
    robustness with limited data.

    Args:
        psms: List of PSMs to use for calibration (targets only, sorted by score)
        model: Prosit intensity predictor wrapper
        initial_ce_model: Initial CE model to calibrate
        lower: Lower bound for intercept offset search (default: -30)
        upper: Upper bound for intercept offset search (default: +30)
        max_psms: Maximum number of PSMs to use (default: 2048)
        verbose: Whether to show progress

    Returns:
        Tuple of (calibrated CollisionEnergyModel, best spectral angle similarity,
                  calibration_curve_data as list of (offset, similarity) tuples)

    Example:
        >>> # Get top PSMs by hyperscore (targets only)
        >>> psms = [p for p in all_psms if not p.decoy]
        >>> psms = sorted(psms, key=lambda x: x.hyperscore, reverse=True)[:1000]
        >>>
        >>> calibrated_model, similarity, curve_data = calibrate_collision_energy_model(
        ...     psms=psms,
        ...     model=prosit_model,
        ...     initial_ce_model=CollisionEnergyModel.from_tdf_dataset(dataset),
        ... )
    """
    from tqdm import tqdm
    from sagepy.core.scoring import associate_fragment_ions_with_prosit_predicted_intensities

    # Filter to targets only and limit sample size
    targets = [p for p in psms if not p.decoy]
    sample = targets[:max_psms]

    if len(sample) == 0:
        logger.warning("No target PSMs provided for CE calibration, returning original model")
        return initial_ce_model, 0.0, []

    if verbose:
        logger.info(f"Calibrating CE model intercept using {len(sample)} PSMs...")
        logger.info(f"Search range: [{lower}, {upper}]")

    similarities = []

    for offset in tqdm(range(lower, upper), disable=not verbose, desc='Calibrating CE', ncols=100):
        # Create test model with offset
        test_model = initial_ce_model.with_intercept_offset(offset)

        # Get CE values for each PSM based on their m/z
        mz_values = np.array([p.mono_mz_calculated for p in sample])
        collision_energies = test_model.predict_batch(mz_values).tolist()

        # Predict intensities
        # Use raw sequence (without UNIMOD) - the Prosit model uses ALPHABET_UNMOD
        # which doesn't support modified tokens like C[UNIMOD:4].
        # p.sequence is the correct sequence for both targets and decoys.
        sequences = [p.sequence for p in sample]
        charges = np.array([p.charge for p in sample])

        intensity_pred = model.predict_intensities(
            sequences=sequences,
            charges=charges,
            collision_energies=collision_energies,
            batch_size=2048,
            flatten=True,
        )

        # Associate with PSMs to compute spectral angle
        psm_with_intensity = associate_fragment_ions_with_prosit_predicted_intensities(
            sample, intensity_pred
        )

        # Compute mean spectral angle similarity for targets
        mean_similarity = np.mean([p.spectral_angle_similarity for p in psm_with_intensity])
        similarities.append((offset, mean_similarity))

    # Find best offset
    best_offset, best_similarity = max(similarities, key=lambda x: x[1])

    if verbose:
        logger.info(f"Best intercept offset: {best_offset}")
        logger.info(f"Best spectral angle similarity: {best_similarity:.4f}")

    # Create calibrated model
    calibrated_model = initial_ce_model.with_intercept_offset(best_offset)

    return calibrated_model, best_similarity, similarities
