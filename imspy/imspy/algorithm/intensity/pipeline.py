"""
End-to-end intensity prediction pipeline for Sage weighted scoring.

This module provides a complete pipeline for:
1. Building/loading a peptide database
2. Predicting fragment intensities with Prosit
3. Creating and storing intensity references (.sagi files)
4. Running weighted scoring in Sage

Example usage:
    from imspy.algorithm.intensity.pipeline import IntensityPredictionPipeline

    # Create pipeline
    pipeline = IntensityPredictionPipeline(
        fasta_path="proteins.fasta",
        output_dir="./intensity_store",
    )

    # Run full prediction pipeline
    pipeline.predict_all(
        charges=[2, 3, 4],
        batch_size=2048,
    )

    # Get path to .sagi file for Sage
    sagi_path = pipeline.get_sagi_path()
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

# Sagepy imports
from sagepy.core import (
    SageSearchConfiguration,
    EnzymeBuilder,
    IndexedDatabase,
    PredictedIntensityStore,
)

if TYPE_CHECKING:
    from .collision_energy import CollisionEnergyModel

# Local imports
from .sage_interface import (
    PredictionRequest,
    PredictionResult,
    predict_intensities_for_sage,
    write_predictions_for_database,
    write_intensity_file,
    read_intensity_file,
    aggregate_predictions_by_peptide,
    remove_unimod_annotation,
    ION_KIND_B,
    ION_KIND_Y,
    DEFAULT_ION_KINDS,
    DEFAULT_COLLISION_ENERGY,
)


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the intensity prediction pipeline.

    Attributes:
        static_mods: Static modifications (applied to all occurrences)
        variable_mods: Variable modifications (optional)
        enzyme: Enzyme name or EnzymeBuilder
        missed_cleavages: Number of missed cleavages allowed
        min_peptide_len: Minimum peptide length
        max_peptide_len: Maximum peptide length
        generate_decoys: Whether to generate decoy sequences
        decoy_tag: Tag for decoy sequences
        charges: Precursor charge states to predict
        max_fragment_charge: Maximum fragment charge for predictions
        ion_kinds: Ion types to predict (default: B and Y)
        collision_energy: Default constant collision energy (NCE), used when
            collision_energy_model is None
        collision_energy_model: Optional m/z-dependent CE model. When set,
            collision_energy is ignored and CE is computed from precursor m/z.
        calibrate_collision_energy: Whether to calibrate CE model intercept
            using top PSM hits after initial search
        aggregation: How to aggregate multi-charge predictions
    """
    static_mods: dict = field(default_factory=lambda: {"C": "[UNIMOD:4]"})
    variable_mods: dict = field(default_factory=dict)
    enzyme: Union[str, EnzymeBuilder] = "trypsin"
    missed_cleavages: int = 2
    min_peptide_len: int = 7
    max_peptide_len: int = 30
    generate_decoys: bool = True
    decoy_tag: str = "DECOY_"
    charges: List[int] = field(default_factory=lambda: [2, 3])
    max_fragment_charge: int = 2
    ion_kinds: List[int] = field(default_factory=lambda: [ION_KIND_B, ION_KIND_Y])
    collision_energy: float = DEFAULT_COLLISION_ENERGY
    collision_energy_model: Optional['CollisionEnergyModel'] = None
    calibrate_collision_energy: bool = False
    aggregation: str = "max_charge"


class IntensityPredictionPipeline:
    """Complete pipeline for intensity prediction and storage.

    This class manages the full workflow from FASTA to .sagi file,
    suitable for integration with Sage's weighted scoring.

    Example:
        >>> pipeline = IntensityPredictionPipeline("proteins.fasta", "./output")
        >>> pipeline.build_database()
        >>> pipeline.predict_intensities(charges=[2, 3])
        >>> pipeline.save_intensity_store()
        >>> store = pipeline.load_intensity_store()  # PredictedIntensityStore
    """

    def __init__(
        self,
        fasta_path: Optional[str] = None,
        output_dir: str = "./intensity_predictions",
        config: Optional[PipelineConfig] = None,
        indexed_database: Optional[IndexedDatabase] = None,
    ):
        """Initialize the pipeline.

        Args:
            fasta_path: Path to FASTA file (required if indexed_database not provided)
            output_dir: Directory for output files
            config: Pipeline configuration
            indexed_database: Pre-built IndexedDatabase (optional, skips FASTA parsing)
        """
        self.fasta_path = fasta_path
        self.output_dir = Path(output_dir)
        self.config = config or PipelineConfig()

        self._indexed_database = indexed_database
        self._peptide_sequences: Optional[List[str]] = None
        self._peptide_lengths: Optional[List[int]] = None
        self._prediction_result: Optional[PredictionResult] = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def indexed_database(self) -> IndexedDatabase:
        """Get or build the indexed database."""
        if self._indexed_database is None:
            self.build_database()
        return self._indexed_database

    @property
    def peptide_sequences(self) -> List[str]:
        """Get peptide sequences from database."""
        if self._peptide_sequences is None:
            self._peptide_sequences = list(self.indexed_database.peptides_as_string())
        return self._peptide_sequences

    @property
    def peptide_lengths(self) -> List[int]:
        """Get peptide lengths."""
        if self._peptide_lengths is None:
            self._peptide_lengths = [len(seq) for seq in self.peptide_sequences]
        return self._peptide_lengths

    @property
    def num_peptides(self) -> int:
        """Total number of peptides in database."""
        return len(self.peptide_sequences)

    def get_sagi_path(self) -> Path:
        """Get the path to the .sagi file."""
        return self.output_dir / "predictions.sagi"

    def build_database(self) -> IndexedDatabase:
        """Build the peptide database from FASTA.

        Returns:
            IndexedDatabase ready for searching and prediction
        """
        if self.fasta_path is None:
            raise ValueError("fasta_path required to build database")

        logger.info(f"Building database from {self.fasta_path}")

        # Read FASTA content
        with open(self.fasta_path, 'r') as f:
            fasta_content = f.read()

        # Build enzyme
        if isinstance(self.config.enzyme, str):
            if self.config.enzyme.lower() == "trypsin":
                enzyme_builder = EnzymeBuilder(
                    missed_cleavages=self.config.missed_cleavages,
                    min_len=self.config.min_peptide_len,
                    max_len=self.config.max_peptide_len,
                    cleave_at="KR",
                    restrict="P",
                    c_terminal=True,
                )
            else:
                raise ValueError(f"Unknown enzyme: {self.config.enzyme}")
        else:
            enzyme_builder = self.config.enzyme

        # Build database
        search_config = SageSearchConfiguration(
            fasta=fasta_content,
            static_mods=self.config.static_mods,
            variable_mods=self.config.variable_mods,
            enzyme_builder=enzyme_builder,
            generate_decoys=self.config.generate_decoys,
            decoy_tag=self.config.decoy_tag,
        )

        self._indexed_database = search_config.generate_indexed_database()
        self._peptide_sequences = None  # Reset cache
        self._peptide_lengths = None

        logger.info(f"Database built with {self.num_peptides} peptides")
        return self._indexed_database

    def get_target_peptides(
        self,
        max_length: int = 30,
        exclude_decoys: bool = True,
    ) -> Tuple[List[int], List[str]]:
        """Get peptides suitable for intensity prediction.

        Args:
            max_length: Maximum peptide length (Prosit limit is 30)
            exclude_decoys: Whether to exclude decoy sequences

        Returns:
            Tuple of (indices, sequences) for target peptides

        Note:
            Decoys are identified using the database's internal peptide objects
            (db._peptides[i].decoy), NOT by checking for a tag in the sequence
            string. The decoy tag only appears in protein headers.
        """
        indices = []
        sequences = []

        # Access the internal peptide list for decoy information
        db_peptides = self.indexed_database._peptides

        for i, seq in enumerate(self.peptide_sequences):
            # Skip decoys if requested - use the peptide's decoy attribute
            if exclude_decoys and db_peptides[i].decoy:
                continue

            # Skip peptides too long for Prosit
            unmod_len = len(remove_unimod_annotation(seq))
            if unmod_len > max_length:
                continue

            indices.append(i)
            sequences.append(seq)

        return indices, sequences

    def predict_intensities(
        self,
        charges: Optional[List[int]] = None,
        collision_energy: Optional[float] = None,
        collision_energy_model: Optional['CollisionEnergyModel'] = None,
        precursor_mz: Optional[List[float]] = None,
        batch_size: int = 2048,
        max_peptide_length: int = 30,
        exclude_decoys: bool = True,
        verbose: bool = True,
    ) -> PredictionResult:
        """Predict fragment intensities using Prosit.

        Collision energy can be specified in three ways (in order of priority):
        1. Explicit `collision_energy` (constant for all peptides)
        2. `collision_energy_model` from config or parameter (m/z-dependent)
        3. Default from config.collision_energy

        Args:
            charges: Precursor charge states to predict (default from config)
            collision_energy: Constant collision energy (overrides config)
            collision_energy_model: m/z-dependent CE model (overrides config)
            precursor_mz: Precursor m/z values for CE model (auto-computed if None)
            batch_size: Batch size for Prosit prediction
            max_peptide_length: Maximum peptide length for prediction
            exclude_decoys: Whether to exclude decoys from prediction
            verbose: Whether to show progress

        Returns:
            PredictionResult with all predictions
        """
        charges = charges or self.config.charges

        # Determine which CE approach to use
        ce_model = collision_energy_model or self.config.collision_energy_model

        # Get target peptides
        indices, sequences = self.get_target_peptides(
            max_length=max_peptide_length,
            exclude_decoys=exclude_decoys,
        )

        logger.info(f"Predicting intensities for {len(sequences)} peptides "
                   f"at charges {charges}")

        # Expand for all charge states
        all_indices = []
        all_sequences = []
        all_charges = []

        for idx, seq in zip(indices, sequences):
            for charge in charges:
                all_indices.append(idx)
                all_sequences.append(seq)
                all_charges.append(charge)

        # Determine collision energies
        if collision_energy is not None:
            # Explicit constant CE provided
            collision_energies = [collision_energy] * len(all_sequences)
            ce_model_for_predict = None
            mz_for_predict = None
        elif ce_model is not None:
            # Use m/z-dependent model
            collision_energies = None
            ce_model_for_predict = ce_model
            # Compute precursor m/z if not provided
            if precursor_mz is None:
                from imspy.data.peptide import PeptideSequence
                mz_for_predict = []
                for seq, charge in zip(all_sequences, all_charges):
                    pep = PeptideSequence(seq)
                    mz_for_predict.append(pep.mono_isotopic_mass / charge + 1.007276)
            else:
                mz_for_predict = precursor_mz
        else:
            # Fall back to config constant
            collision_energies = [self.config.collision_energy] * len(all_sequences)
            ce_model_for_predict = None
            mz_for_predict = None

        self._prediction_result = predict_intensities_for_sage(
            sequences=all_sequences,
            charges=all_charges,
            peptide_indices=all_indices,
            collision_energies=collision_energies,
            collision_energy_model=ce_model_for_predict,
            precursor_mz=mz_for_predict,
            ion_kinds=self.config.ion_kinds,
            max_fragment_charge=self.config.max_fragment_charge,
            batch_size=batch_size,
            verbose=verbose,
        )

        logger.info(f"Predicted {len(self._prediction_result.intensities)} intensity arrays")
        return self._prediction_result

    def save_intensity_store(
        self,
        output_path: Optional[str] = None,
        default_value: float = 1.0,
    ) -> Path:
        """Save predictions to .sagi file.

        Args:
            output_path: Custom output path (default: output_dir/predictions.sagi)
            default_value: Value to use for unpredicted peptides (e.g., decoys)

        Returns:
            Path to the saved .sagi file
        """
        if self._prediction_result is None:
            raise ValueError("No predictions available. Call predict_intensities() first.")

        output_path = Path(output_path) if output_path else self.get_sagi_path()

        logger.info(f"Saving intensity store to {output_path}")

        write_predictions_for_database(
            str(output_path),
            self._prediction_result,
            num_peptides=self.num_peptides,
            peptide_lengths=self.peptide_lengths,
            aggregation=self.config.aggregation,
            default_value=default_value,
        )

        logger.info(f"Saved intensity store with {self.num_peptides} peptides")
        return output_path

    def load_intensity_store(
        self,
        sagi_path: Optional[str] = None,
    ) -> PredictedIntensityStore:
        """Load intensity store from .sagi file.

        Args:
            sagi_path: Path to .sagi file (default: output_dir/predictions.sagi)

        Returns:
            PredictedIntensityStore for use with Sage scoring
        """
        sagi_path = sagi_path or str(self.get_sagi_path())
        logger.info(f"Loading intensity store from {sagi_path}")
        return PredictedIntensityStore(sagi_path)

    def create_uniform_store(self) -> PredictedIntensityStore:
        """Create a uniform intensity store (all values = 1.0).

        This is useful for testing or when predictions are not available.

        Returns:
            PredictedIntensityStore with uniform intensities
        """
        return PredictedIntensityStore.uniform(
            peptide_lengths=self.peptide_lengths,
            max_charge=self.config.max_fragment_charge,
            ion_kinds=self.config.ion_kinds,
        )

    def run(
        self,
        charges: Optional[List[int]] = None,
        batch_size: int = 2048,
        verbose: bool = True,
    ) -> Path:
        """Run the complete pipeline: build -> predict -> save.

        Args:
            charges: Precursor charge states to predict
            batch_size: Batch size for prediction
            verbose: Whether to show progress

        Returns:
            Path to the saved .sagi file
        """
        # Build database if needed
        if self._indexed_database is None:
            self.build_database()

        # Predict intensities
        self.predict_intensities(
            charges=charges,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Save to file
        return self.save_intensity_store()

    def get_summary(self) -> dict:
        """Get a summary of the pipeline state.

        Returns:
            Dictionary with pipeline statistics
        """
        target_indices, target_seqs = self.get_target_peptides()

        summary = {
            "fasta_path": str(self.fasta_path) if self.fasta_path else None,
            "output_dir": str(self.output_dir),
            "total_peptides": self.num_peptides,
            "target_peptides": len(target_seqs),
            "decoy_peptides": self.num_peptides - len(target_seqs),
            "config": {
                "charges": self.config.charges,
                "max_fragment_charge": self.config.max_fragment_charge,
                "ion_kinds": self.config.ion_kinds,
                "collision_energy": self.config.collision_energy,
            },
            "predictions_available": self._prediction_result is not None,
            "sagi_file_exists": self.get_sagi_path().exists(),
        }

        if self._prediction_result is not None:
            summary["num_predictions"] = len(self._prediction_result.intensities)

        return summary


def create_intensity_store_from_database(
    indexed_database: IndexedDatabase,
    output_path: str,
    charges: List[int] = [2, 3],
    max_fragment_charge: int = 2,
    collision_energy: Optional[float] = None,
    collision_energy_model: Optional['CollisionEnergyModel'] = None,
    batch_size: int = 2048,
    verbose: bool = True,
) -> PredictedIntensityStore:
    """Convenience function to create an intensity store from an existing database.

    This is a simplified interface for the common use case of predicting
    intensities for an already-built database.

    Collision energy can be specified in two ways:
    1. `collision_energy`: Constant CE for all peptides (default: 35.0 NCE)
    2. `collision_energy_model`: m/z-dependent CE model

    Args:
        indexed_database: Pre-built IndexedDatabase
        output_path: Path for the output .sagi file
        charges: Precursor charge states to predict
        max_fragment_charge: Maximum fragment charge
        collision_energy: Constant collision energy for prediction (default: 35.0)
        collision_energy_model: Optional m/z-dependent CE model
        batch_size: Batch size for Prosit
        verbose: Whether to show progress

    Returns:
        Loaded PredictedIntensityStore ready for scoring

    Example:
        >>> from sagepy.core import SageSearchConfiguration
        >>> config = SageSearchConfiguration(fasta=fasta_str, ...)
        >>> db = config.generate_indexed_database()
        >>> store = create_intensity_store_from_database(db, "predictions.sagi")
        >>> features = scorer.score(db, spectrum, intensity_store=store)

        >>> # With m/z-dependent CE
        >>> from imspy.algorithm.intensity.collision_energy import CollisionEnergyModel
        >>> ce_model = CollisionEnergyModel.from_tdf_dataset(dataset)
        >>> store = create_intensity_store_from_database(
        ...     db, "predictions.sagi", collision_energy_model=ce_model
        ... )
    """
    config = PipelineConfig(
        charges=charges,
        max_fragment_charge=max_fragment_charge,
        collision_energy=collision_energy or DEFAULT_COLLISION_ENERGY,
        collision_energy_model=collision_energy_model,
    )

    pipeline = IntensityPredictionPipeline(
        output_dir=str(Path(output_path).parent),
        config=config,
        indexed_database=indexed_database,
    )

    pipeline.predict_intensities(
        charges=charges,
        collision_energy_model=collision_energy_model,
        batch_size=batch_size,
        verbose=verbose,
    )

    pipeline.save_intensity_store(output_path)

    return pipeline.load_intensity_store(output_path)
