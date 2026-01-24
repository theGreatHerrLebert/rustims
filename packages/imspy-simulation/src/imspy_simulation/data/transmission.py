"""Ion transmission calculation handle.

This module provides the TransmissionHandle class that wraps the Rust
implementation for calculating which ions are transmitted through the
quadrupole based on their m/z and mobility values.
"""

import os

import pandas as pd

import imspy_connector

ims = imspy_connector.py_simulation


class TransmissionHandle:
    """Handle for calculating ion transmission through the quadrupole.

    This class wraps a Rust implementation that efficiently calculates
    which peptide ions are transmitted based on the quadrupole isolation
    windows and ion mobility filtering settings.

    The transmission calculation considers:
    - m/z ranges of quadrupole isolation windows
    - Ion mobility (1/K0) ranges per scan
    - Frame timing (retention time)

    Attributes:
        path: Path to the synthetic data database.
    """

    def __init__(self, path: str):
        """Initialize the transmission handle.

        Args:
            path: Path to the synthetic_data.db file.
        """
        self.path = path
        self._py_ptr = ims.PyTimsTofSyntheticsDataHandle(path)

    @property
    def _ptr(self):
        """Get the underlying PyO3 pointer."""
        return self._py_ptr

    def get_transmitted_ions(
        self, num_threads: int = -1, dda: bool = False
    ) -> pd.DataFrame:
        """Get all transmitted ions and their collision energies.

        Calculates which ions are transmitted through the quadrupole
        for the entire experiment. This loads all peptides and ions
        into memory.

        Args:
            num_threads: Number of threads. -1 for auto-detect.
            dda: If True, use DDA transmission logic; otherwise DIA.

        Returns:
            DataFrame with columns:
                - peptide_id: Peptide identifier
                - ion_id: Ion identifier (peptide + charge)
                - sequence: Peptide sequence
                - charge: Ion charge state
                - collision_energy: Collision energy for fragmentation
        """
        if num_threads == -1:
            num_threads = os.cpu_count() or 4

        (
            peptide_ids,
            ion_ids,
            sequences,
            charges,
            collision_energies,
        ) = self._py_ptr.get_transmitted_ions(num_threads, dda)

        return pd.DataFrame(
            {
                "peptide_id": peptide_ids,
                "ion_id": ion_ids,
                "sequence": sequences,
                "charge": charges,
                "collision_energy": collision_energies,
            }
        )

    def get_transmitted_ions_for_frame_range(
        self,
        frame_min: int,
        frame_max: int,
        num_threads: int = -1,
        dda: bool = False,
    ) -> pd.DataFrame:
        """Get transmitted ions for a specific frame range (lazy loading).

        This is the memory-efficient version that only loads peptides and
        ions relevant to the specified frame range instead of all data.
        Use this for large simulations to reduce memory usage.

        Args:
            frame_min: Minimum frame ID (inclusive).
            frame_max: Maximum frame ID (inclusive).
            num_threads: Number of threads. -1 for auto-detect.
            dda: If True, use DDA transmission logic; otherwise DIA.

        Returns:
            DataFrame with columns:
                - peptide_id: Peptide identifier
                - ion_id: Ion identifier
                - sequence: Peptide sequence
                - charge: Ion charge state
                - collision_energy: Collision energy for fragmentation
        """
        if num_threads == -1:
            num_threads = os.cpu_count() or 4

        (
            peptide_ids,
            ion_ids,
            sequences,
            charges,
            collision_energies,
        ) = self._py_ptr.get_transmitted_ions_for_frame_range(
            frame_min, frame_max, num_threads, dda
        )

        return pd.DataFrame(
            {
                "peptide_id": peptide_ids,
                "ion_id": ion_ids,
                "sequence": sequences,
                "charge": charges,
                "collision_energy": collision_energies,
            }
        )

    # Legacy compatibility

    def get_py_ptr(self):
        """Get underlying pointer (legacy compatibility)."""
        return self._py_ptr

    def __repr__(self) -> str:
        return f"TransmissionHandle(path={self.path})"
