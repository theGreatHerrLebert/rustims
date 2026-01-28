"""DIA frame builder implementations.

This module provides a unified DIAFrameBuilder that supports both standard
and lazy loading strategies through composition rather than inheritance.
"""

import os
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

import imspy_connector

from imspy_core.data.peptide import PeptideProductIonSeriesCollection
from imspy_core.data.spectrum import MzSpectrum
from imspy_core.timstof.frame import TimsFrame
from imspy_simulation.annotation import TimsFrameAnnotated

ims = imspy_connector.py_simulation


class DIAFrameBuilder:
    """Unified DIA frame builder supporting both standard and lazy loading.

    This class provides a single interface for DIA frame building, using
    composition to delegate to the appropriate Rust implementation based
    on the loading strategy.

    Standard mode:
        Loads all peptides, ions, and fragment data into memory at
        construction time. Higher memory usage but may be faster for
        small datasets due to caching.

    Lazy mode:
        Only stores metadata at construction time. Peptide/ion data is
        loaded on-demand for each batch of frames, then released.
        Significantly lower memory usage for large simulations.

    Attributes:
        path: Path to the synthetic data database.
        lazy: Whether lazy loading is enabled.
    """

    def __init__(
        self,
        db_path: str,
        num_threads: int = -1,
        lazy: bool = False,
        with_annotations: bool = False,
        quad_isotope_transmission_mode: str = 'none',
        quad_transmission_min_probability: float = 0.5,
        quad_transmission_max_isotopes: int = 10,
        precursor_survival_min: float = 0.0,
        precursor_survival_max: float = 0.0,
    ):
        """Initialize the DIA frame builder.

        Args:
            db_path: Path to the synthetic data database.
            num_threads: Number of threads. -1 for auto-detect.
            lazy: If True, use lazy loading strategy.
            with_annotations: If True, enable annotation support.
                            Only available when lazy=False.
            quad_isotope_transmission_mode: Mode for quad-selection dependent
                isotope transmission. Options:
                - "none": Standard isotope patterns (default)
                - "precursor_scaling": Fast mode - uniform scaling based on precursor transmission
                - "per_fragment": Accurate mode - individual fragment ion recalculation
                Note: Not supported with lazy loading.
            quad_transmission_min_probability: Minimum probability threshold for
                isotope transmission (default 0.5).
            quad_transmission_max_isotopes: Maximum number of isotope peaks to
                consider for transmission (default 10).
            precursor_survival_min: Minimum fraction of precursor ions that survive
                fragmentation intact (0.0-1.0, default 0.0).
            precursor_survival_max: Maximum fraction of precursor ions that survive
                fragmentation intact (0.0-1.0, default 0.0).

        Raises:
            ValueError: If annotations requested with lazy loading.
        """
        self.path = db_path
        self.lazy = lazy

        if num_threads == -1:
            num_threads = os.cpu_count() or 4

        if lazy and with_annotations:
            raise ValueError("Annotation support is not available with lazy loading.")

        if lazy:
            if quad_isotope_transmission_mode != 'none':
                import logging
                logging.getLogger(__name__).warning(
                    "Quad-dependent isotope transmission is not supported with lazy loading, ignoring."
                )
            self._py_ptr = ims.PyTimsTofLazyFrameBuilderDIA(db_path, num_threads)
            self._with_annotations = False
        else:
            # Create isotope transmission config
            isotope_config = ims.PyIsotopeTransmissionConfig(
                mode=quad_isotope_transmission_mode,
                min_probability=quad_transmission_min_probability,
                max_isotopes=quad_transmission_max_isotopes,
                precursor_survival_min=precursor_survival_min,
                precursor_survival_max=precursor_survival_max,
            )

            self._py_ptr = ims.PyTimsTofSyntheticsFrameBuilderDIA(
                db_path, with_annotations, num_threads, isotope_config
            )
            self._with_annotations = with_annotations

    @property
    def _ptr(self):
        """Get the underlying PyO3 pointer."""
        return self._py_ptr

    @classmethod
    def _from_ptr(cls, ptr, lazy: bool = False) -> "DIAFrameBuilder":
        """Create a builder from an existing PyO3 pointer.

        Args:
            ptr: The PyO3 object to wrap.
            lazy: Whether this is a lazy builder pointer.

        Returns:
            A new DIAFrameBuilder instance.
        """
        instance = cls.__new__(cls)
        instance._py_ptr = ptr
        instance.lazy = lazy
        instance._with_annotations = False
        instance.path = ""
        return instance

    def build_frames(
        self,
        frame_ids: List[int],
        fragment: bool = True,
        mz_noise_precursor: bool = False,
        mz_noise_uniform: bool = False,
        precursor_noise_ppm: float = 5.0,
        mz_noise_fragment: bool = False,
        fragment_noise_ppm: float = 5.0,
        right_drag: bool = True,
        num_threads: int = 4,
    ) -> List[TimsFrame]:
        """Build frames for the specified frame IDs.

        Args:
            frame_ids: Frame IDs to build.
            fragment: If True, perform synthetic fragmentation.
            mz_noise_precursor: If True, add noise to precursor m/z.
            mz_noise_uniform: If True, use uniform noise distribution.
            precursor_noise_ppm: PPM value for precursor noise.
            mz_noise_fragment: If True, add noise to fragment m/z.
            fragment_noise_ppm: PPM value for fragment noise.
            right_drag: If True, noise is shifted to the right.
            num_threads: Number of threads (ignored for lazy mode).

        Returns:
            List of built TimsFrame objects.
        """
        if self.lazy:
            frames = self._py_ptr.build_frames_lazy(
                frame_ids,
                fragment,
                mz_noise_precursor,
                mz_noise_uniform,
                precursor_noise_ppm,
                mz_noise_fragment,
                fragment_noise_ppm,
                right_drag,
            )
        else:
            frames = self._py_ptr.build_frames(
                frame_ids,
                fragment,
                mz_noise_precursor,
                mz_noise_uniform,
                precursor_noise_ppm,
                mz_noise_fragment,
                fragment_noise_ppm,
                right_drag,
                num_threads,
            )
        return [TimsFrame.from_py_ptr(frame) for frame in frames]

    def build_frame(
        self,
        frame_id: int,
        fragment: bool = True,
        mz_noise_precursor: bool = False,
        mz_noise_uniform: bool = False,
        precursor_noise_ppm: float = 5.0,
        mz_noise_fragment: bool = False,
        fragment_noise_ppm: float = 5.0,
        right_drag: bool = True,
    ) -> TimsFrame:
        """Build a single frame.

        Args:
            frame_id: Frame ID to build.
            fragment: If True, perform synthetic fragmentation.
            mz_noise_precursor: If True, add noise to precursor m/z.
            mz_noise_uniform: If True, use uniform noise distribution.
            precursor_noise_ppm: PPM value for precursor noise.
            mz_noise_fragment: If True, add noise to fragment m/z.
            fragment_noise_ppm: PPM value for fragment noise.
            right_drag: If True, noise is shifted to the right.

        Returns:
            Built TimsFrame object.
        """
        return self.build_frames(
            [frame_id],
            fragment,
            mz_noise_precursor,
            mz_noise_uniform,
            precursor_noise_ppm,
            mz_noise_fragment,
            fragment_noise_ppm,
            right_drag,
        )[0]

    def build_frames_annotated(
        self,
        frame_ids: List[int],
        fragment: bool = True,
        mz_noise_precursor: bool = False,
        mz_noise_uniform: bool = False,
        precursor_noise_ppm: float = 5.0,
        mz_noise_fragment: bool = False,
        fragment_noise_ppm: float = 5.0,
        right_drag: bool = True,
        num_threads: int = 4,
    ) -> List[TimsFrameAnnotated]:
        """Build annotated frames (standard loading only).

        Args:
            frame_ids: Frame IDs to build.
            fragment: If True, perform synthetic fragmentation.
            mz_noise_precursor: If True, add noise to precursor m/z.
            mz_noise_uniform: If True, use uniform noise distribution.
            precursor_noise_ppm: PPM value for precursor noise.
            mz_noise_fragment: If True, add noise to fragment m/z.
            fragment_noise_ppm: PPM value for fragment noise.
            right_drag: If True, noise is shifted to the right.
            num_threads: Number of threads.

        Returns:
            List of TimsFrameAnnotated objects.

        Raises:
            RuntimeError: If called on a lazy builder or annotations not enabled.
        """
        if self.lazy:
            raise RuntimeError("Annotated frame building is not supported in lazy mode.")
        if not self._with_annotations:
            raise RuntimeError(
                "Annotations not enabled. Create builder with with_annotations=True."
            )

        frames = self._py_ptr.build_frames_annotated(
            frame_ids,
            fragment,
            mz_noise_precursor,
            mz_noise_uniform,
            precursor_noise_ppm,
            mz_noise_fragment,
            fragment_noise_ppm,
            right_drag,
            num_threads,
        )
        return [TimsFrameAnnotated.from_py_ptr(frame) for frame in frames]

    def build_frame_annotated(
        self,
        frame_id: int,
        fragment: bool = True,
        mz_noise_precursor: bool = False,
        mz_noise_uniform: bool = False,
        precursor_noise_ppm: float = 5.0,
        mz_noise_fragment: bool = False,
        fragment_noise_ppm: float = 5.0,
        right_drag: bool = True,
    ) -> TimsFrameAnnotated:
        """Build a single annotated frame (standard loading only).

        Args:
            frame_id: Frame ID to build.
            fragment: If True, perform synthetic fragmentation.
            mz_noise_precursor: If True, add noise to precursor m/z.
            mz_noise_uniform: If True, use uniform noise distribution.
            precursor_noise_ppm: PPM value for precursor noise.
            mz_noise_fragment: If True, add noise to fragment m/z.
            fragment_noise_ppm: PPM value for fragment noise.
            right_drag: If True, noise is shifted to the right.

        Returns:
            TimsFrameAnnotated object.
        """
        return self.build_frames_annotated(
            [frame_id],
            fragment,
            mz_noise_precursor,
            mz_noise_uniform,
            precursor_noise_ppm,
            mz_noise_fragment,
            fragment_noise_ppm,
            right_drag,
        )[0]

    def get_collision_energy(self, frame_id: int, scan_id: int) -> float:
        """Get collision energy for a specific frame and scan.

        Args:
            frame_id: Frame ID.
            scan_id: Scan ID.

        Returns:
            Collision energy value.
        """
        return self._py_ptr.get_collision_energy(frame_id, scan_id)

    def get_collision_energies(
        self, frame_ids: List[int], scan_ids: List[int]
    ) -> List[float]:
        """Get collision energies for multiple frame/scan pairs.

        Args:
            frame_ids: List of frame IDs.
            scan_ids: List of scan IDs.

        Returns:
            List of collision energy values.

        Raises:
            RuntimeError: If called on a lazy builder.
        """
        if self.lazy:
            raise RuntimeError(
                "Batch collision energy lookup not supported in lazy mode. "
                "Use get_collision_energy() for individual lookups."
            )
        return self._py_ptr.get_collision_energies(frame_ids, scan_ids)

    def get_fragment_ions_map(self):
        """Get the fragment ions map (standard loading only).

        Returns:
            Dictionary mapping to (PeptideProductIonSeriesCollection, [MzSpectrum]).

        Raises:
            RuntimeError: If called on a lazy builder.
        """
        if self.lazy:
            raise RuntimeError("Fragment ions map not available in lazy mode.")

        ions_map = self._py_ptr.get_fragment_ions_map()
        ret_map = {}
        for key, value in ions_map.items():
            ret_map[key] = (
                PeptideProductIonSeriesCollection.from_py_ptr(value[0]),
                [MzSpectrum.from_py_ptr(spectrum) for spectrum in value[1]],
            )
        return ret_map

    def get_ion_transmission_matrix(
        self, peptide_id: int, charge: int, include_precursor_frames: bool = False
    ) -> NDArray:
        """Get ion transmission matrix (standard loading only).

        Args:
            peptide_id: Peptide ID.
            charge: Charge state.
            include_precursor_frames: If True, include precursor frames.

        Returns:
            NumPy array of transmission values.

        Raises:
            RuntimeError: If called on a lazy builder.
        """
        if self.lazy:
            raise RuntimeError("Ion transmission matrix not available in lazy mode.")
        return np.array(
            self._py_ptr.get_ion_transmission_matrix(
                peptide_id, charge, include_precursor_frames
            )
        )

    def count_number_transmissions(
        self, peptide_id: int, charge: int
    ) -> Tuple[int, int]:
        """Count transmissions for a peptide (standard loading only).

        Args:
            peptide_id: Peptide ID.
            charge: Charge state.

        Returns:
            Tuple of (precursor_count, fragment_count).

        Raises:
            RuntimeError: If called on a lazy builder.
        """
        if self.lazy:
            raise RuntimeError("Transmission counting not available in lazy mode.")
        return self._py_ptr.count_number_transmissions(peptide_id, charge)

    def count_number_transmissions_parallel(
        self,
        peptide_ids: List[int],
        charges: List[int],
        num_threads: int = 4,
    ) -> List[Tuple[int, int]]:
        """Count transmissions for multiple peptides (standard loading only).

        Args:
            peptide_ids: List of peptide IDs.
            charges: List of charge states.
            num_threads: Number of threads.

        Returns:
            List of (precursor_count, fragment_count) tuples.

        Raises:
            RuntimeError: If called on a lazy builder.
        """
        if self.lazy:
            raise RuntimeError("Transmission counting not available in lazy mode.")
        return self._py_ptr.count_number_transmissions_parallel(
            peptide_ids, charges, num_threads
        )

    # Lazy-mode specific methods

    def num_frames(self) -> int:
        """Get total number of frames (lazy mode only).

        Returns:
            Total frame count.

        Raises:
            RuntimeError: If called on a standard builder.
        """
        if not self.lazy:
            raise RuntimeError(
                "num_frames() is only available in lazy mode. "
                "For standard mode, query the database directly."
            )
        return self._py_ptr.num_frames()

    def frame_ids(self) -> List[int]:
        """Get all frame IDs (lazy mode only).

        Returns:
            List of all frame IDs.

        Raises:
            RuntimeError: If called on a standard builder.
        """
        if not self.lazy:
            raise RuntimeError(
                "frame_ids() is only available in lazy mode. "
                "For standard mode, query the database directly."
            )
        return self._py_ptr.frame_ids()

    def precursor_frame_ids(self) -> List[int]:
        """Get precursor frame IDs (lazy mode only).

        Returns:
            List of precursor frame IDs.

        Raises:
            RuntimeError: If called on a standard builder.
        """
        if not self.lazy:
            raise RuntimeError("precursor_frame_ids() is only available in lazy mode.")
        return self._py_ptr.precursor_frame_ids()

    def fragment_frame_ids(self) -> List[int]:
        """Get fragment frame IDs (lazy mode only).

        Returns:
            List of fragment frame IDs.

        Raises:
            RuntimeError: If called on a standard builder.
        """
        if not self.lazy:
            raise RuntimeError("fragment_frame_ids() is only available in lazy mode.")
        return self._py_ptr.fragment_frame_ids()

    # Legacy compatibility

    def get_py_ptr(self):
        """Get underlying pointer (legacy compatibility)."""
        return self._py_ptr

    @classmethod
    def from_py_ptr(cls, ptr, lazy: bool = False) -> "DIAFrameBuilder":
        """Create from pointer (legacy compatibility)."""
        return cls._from_ptr(ptr, lazy)

    def __repr__(self) -> str:
        mode = "lazy" if self.lazy else "standard"
        return f"DIAFrameBuilder(path={self.path}, mode={mode})"
