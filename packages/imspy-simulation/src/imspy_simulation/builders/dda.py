"""DDA frame builder implementation.

This module provides the DDAFrameBuilder class for Data-Dependent Acquisition
mode frame building.
"""

import os
from typing import List

import imspy_connector

from imspy_core.data.peptide import PeptideProductIonSeriesCollection
from imspy_core.data.spectrum import MzSpectrum
from imspy_core.timstof.frame import TimsFrame
from imspy_core.timstof.quadrupole import PasefMeta
from imspy_simulation.annotation import TimsFrameAnnotated

ims = imspy_connector.py_simulation


class DDAFrameBuilder:
    """DDA frame builder for Data-Dependent Acquisition simulations.

    This builder handles frame construction for DDA mode, where precursor
    selection is based on intensity (top-N) rather than systematic windows.

    Attributes:
        path: Path to the synthetic data database.
    """

    def __init__(
        self,
        db_path: str,
        num_threads: int = -1,
        with_annotations: bool = False,
        quad_isotope_transmission_mode: str = 'none',
        quad_transmission_min_probability: float = 0.5,
        quad_transmission_max_isotopes: int = 10,
        precursor_survival_min: float = 0.0,
        precursor_survival_max: float = 0.0,
    ):
        """Initialize the DDA frame builder.

        Args:
            db_path: Path to the synthetic data database.
            num_threads: Number of threads. -1 for auto-detect.
            with_annotations: If True, enable annotation support.
            quad_isotope_transmission_mode: Mode for quad-selection dependent
                isotope transmission. Options:
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
        """
        self.path = db_path

        if num_threads == -1:
            num_threads = os.cpu_count() or 4

        # Create isotope transmission config
        isotope_config = ims.PyIsotopeTransmissionConfig(
            mode=quad_isotope_transmission_mode,
            min_probability=quad_transmission_min_probability,
            max_isotopes=quad_transmission_max_isotopes,
            precursor_survival_min=precursor_survival_min,
            precursor_survival_max=precursor_survival_max,
        )

        self._py_ptr = ims.PyTimsTofSyntheticsFrameBuilderDDA(
            db_path, with_annotations, num_threads, isotope_config
        )
        self._with_annotations = with_annotations

    @property
    def _ptr(self):
        """Get the underlying PyO3 pointer."""
        return self._py_ptr

    @classmethod
    def _from_ptr(cls, ptr) -> "DDAFrameBuilder":
        """Create a builder from an existing PyO3 pointer.

        Args:
            ptr: The PyO3 object to wrap.

        Returns:
            A new DDAFrameBuilder instance.
        """
        instance = cls.__new__(cls)
        instance._py_ptr = ptr
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
            num_threads: Number of threads.

        Returns:
            List of built TimsFrame objects.
        """
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
        frame = self._py_ptr.build_frame(
            frame_id,
            fragment,
            mz_noise_precursor,
            mz_noise_uniform,
            precursor_noise_ppm,
            mz_noise_fragment,
            fragment_noise_ppm,
            right_drag,
        )
        return TimsFrame.from_py_ptr(frame)

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
        """Build annotated frames.

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
            RuntimeError: If annotations not enabled at construction.
        """
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
        """Build a single annotated frame.

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
        if not self._with_annotations:
            raise RuntimeError(
                "Annotations not enabled. Create builder with with_annotations=True."
            )

        frame = self._py_ptr.build_frame_annotated(
            frame_id,
            fragment,
            mz_noise_precursor,
            mz_noise_uniform,
            precursor_noise_ppm,
            mz_noise_fragment,
            fragment_noise_ppm,
            right_drag,
        )
        return TimsFrameAnnotated.from_py_ptr(frame)

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
        """
        return self._py_ptr.get_collision_energies(frame_ids, scan_ids)

    def get_pasef_meta(self) -> List[PasefMeta]:
        """Get PASEF metadata.

        Returns:
            List of PasefMeta objects.
        """
        return [PasefMeta.from_py_ptr(meta) for meta in self._py_ptr.get_pasef_meta()]

    def get_fragment_frames(self) -> List[int]:
        """Get fragment frame IDs.

        Returns:
            List of fragment frame IDs.
        """
        return self._py_ptr.get_fragment_frames()

    def get_fragment_ions_map(self):
        """Get the fragment ions map.

        Returns:
            Dictionary mapping to (PeptideProductIonSeriesCollection, [MzSpectrum]).
        """
        ions_map = self._py_ptr.get_fragment_ions_map()
        ret_map = {}
        for key, value in ions_map.items():
            ret_map[key] = (
                PeptideProductIonSeriesCollection.from_py_ptr(value[0]),
                [MzSpectrum.from_py_ptr(spectrum) for spectrum in value[1]],
            )
        return ret_map

    # Legacy compatibility

    def get_py_ptr(self):
        """Get underlying pointer (legacy compatibility)."""
        return self._py_ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> "DDAFrameBuilder":
        """Create from pointer (legacy compatibility)."""
        return cls._from_ptr(ptr)

    def __repr__(self) -> str:
        return f"DDAFrameBuilder(path={self.path})"
