"""Protocol definitions for simulation frame builders.

This module defines the interface contracts that all frame builders must implement.
Using Protocol (structural subtyping) instead of ABC allows for duck-typing
compatibility while still providing type checking benefits.
"""

from typing import List, Protocol, runtime_checkable

from imspy.timstof.frame import TimsFrame


@runtime_checkable
class FrameBuilder(Protocol):
    """Protocol for frame builders used in timsTOF simulation.

    All frame builders (DDA, DIA, Lazy) implement this interface,
    allowing them to be used interchangeably in the simulation pipeline.

    The `build_frames` method is the core interface that takes frame IDs
    and various noise parameters, returning a list of built TimsFrame objects.
    """

    path: str

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
            frame_ids: List of frame IDs to build.
            fragment: If True, perform synthetic fragmentation on fragment frames.
            mz_noise_precursor: If True, add noise to precursor m/z values.
            mz_noise_uniform: If True, use uniform noise distribution.
            precursor_noise_ppm: PPM value for precursor noise.
            mz_noise_fragment: If True, add noise to fragment m/z values.
            fragment_noise_ppm: PPM value for fragment noise.
            right_drag: If True, noise is shifted to the right.
            num_threads: Number of threads for parallel processing.

        Returns:
            List of built TimsFrame objects.
        """
        ...

    def get_collision_energy(self, frame_id: int, scan_id: int) -> float:
        """Get collision energy for a specific frame and scan.

        Args:
            frame_id: Frame ID.
            scan_id: Scan ID.

        Returns:
            Collision energy value.
        """
        ...


@runtime_checkable
class AnnotatedFrameBuilder(FrameBuilder, Protocol):
    """Protocol for frame builders that support annotation.

    Extends FrameBuilder with methods that return annotated frames
    containing additional metadata about the source of each signal.

    Note: Annotated frame building is memory-intensive and should
    be used sparingly, primarily for debugging or validation.
    """

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
    ):
        """Build an annotated frame for a single frame ID.

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
        ...

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
    ):
        """Build annotated frames for multiple frame IDs.

        Args:
            frame_ids: List of frame IDs to build.
            fragment: If True, perform synthetic fragmentation.
            mz_noise_precursor: If True, add noise to precursor m/z.
            mz_noise_uniform: If True, use uniform noise distribution.
            precursor_noise_ppm: PPM value for precursor noise.
            mz_noise_fragment: If True, add noise to fragment m/z.
            fragment_noise_ppm: PPM value for fragment noise.
            right_drag: If True, noise is shifted to the right.
            num_threads: Number of threads for parallel processing.

        Returns:
            List of TimsFrameAnnotated objects.
        """
        ...
