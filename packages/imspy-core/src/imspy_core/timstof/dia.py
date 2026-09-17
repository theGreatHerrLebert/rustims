import sqlite3
from typing import List, Optional

from imspy_core.core.base import RustWrapperObject
from imspy_core.timstof.data import TimsDataset
import pandas as pd

import imspy_connector

from imspy_core.timstof.frame import TimsFrame

ims = imspy_connector.py_dia


class TimsDatasetDIA(TimsDataset, RustWrapperObject):
    def __init__(self, data_path: str, in_memory: bool = False, use_bruker_sdk: bool = True):
        super().__init__(data_path=data_path, in_memory=in_memory, use_bruker_sdk=use_bruker_sdk)
        self.__dataset = ims.PyTimsDatasetDIA(self.data_path, self.binary_path, in_memory, self.use_bruker_sdk)

    @property
    def dia_ms_ms_windows(self):
        """Get PASEF meta data for DIA.

        Returns:
            pd.DataFrame: PASEF meta data.
        """
        return pd.read_sql_query("SELECT * from DiaFrameMsMsWindows",
                                 sqlite3.connect(self.data_path + "/analysis.tdf"))

    @property
    def dia_ms_ms_info(self):
        """Get DIA MS/MS info.

        Returns:
            pd.DataFrame: DIA MS/MS info.
        """
        return pd.read_sql_query("SELECT * from DiaFrameMsMsInfo",
                                 sqlite3.connect(self.data_path + "/analysis.tdf"))

    def sample_precursor_signal(self, num_frames: int, max_intensity: float = 25.0, take_probability: float = 0.5) -> TimsFrame:
        """Sample precursor signal.

        Args:
            num_frames: Number of frames.
            max_intensity: Maximum intensity.
            take_probability: Probability to take signals from sampled frames.

        Returns:
            TimsFrame: Frame.
        """

        assert num_frames > 0, "Number of frames must be greater than 0."
        assert 0 < take_probability <= 1, " Probability to take signals from sampled frames must be between 0 and 1."

        return TimsFrame.from_py_ptr(self.__dataset.sample_precursor_signal(num_frames, max_intensity, take_probability))

    def sample_fragment_signal(self, num_frames: int, window_group: int, max_intensity: float = 25.0, take_probability: float = 0.5) -> TimsFrame:
        """Sample fragment signal.

        Args:
            num_frames: Number of frames.
            window_group: Window group to take frames from.
            max_intensity: Maximum intensity.
            take_probability: Probability to take signals from sampled frames.

        Returns:
            TimsFrame: Frame.
        """

        assert num_frames > 0, "Number of frames must be greater than 0."
        assert 0 < take_probability <= 1, " Probability to take signals from sampled frames must be between 0 and 1."

        return TimsFrame.from_py_ptr(self.__dataset.sample_fragment_signal(num_frames, window_group, max_intensity, take_probability))

    def overlay_reference_noise(
            self,
            frames: List[TimsFrame],
            window_groups: List[Optional[int]],
            num_precursor_frames: int = 5,
            num_fragment_frames: int = 5,
            max_intensity_precursor: float = 30.0,
            max_intensity_fragment: float = 30.0,
            take_precursor: float = 0.2,
            take_fragment: float = 0.2,
            seed: int = 0,
            num_threads: int = 4,
    ) -> List[TimsFrame]:
        """Add sampled reference-data noise to a batch of frames, in parallel and reproducibly.

        Equivalent to calling ``sample_precursor_signal`` / ``sample_fragment_signal`` per frame and
        adding the result with ``frame + noise``, but done in Rust over a thread pool with a
        per-frame RNG stream derived from ``seed`` and the frame id.

        Args:
            frames: Simulated frames.
            window_groups: Per frame, the DIA window group (fragment frame) or None (precursor frame).
            num_precursor_frames / num_fragment_frames: Reference frames sampled per output frame.
            max_intensity_precursor / max_intensity_fragment: Keep reference peaks up to this intensity.
            take_precursor / take_fragment: Fraction of reference peaks to keep.
            seed: Master seed; identical inputs and seed give identical output.
            num_threads: Rayon threads.

        Returns:
            List[TimsFrame]: frames with noise added, same order as the input.
        """
        assert len(frames) == len(window_groups), "frames and window_groups must have the same length"
        out = self.__dataset.overlay_reference_noise(
            [f.get_py_ptr() for f in frames],
            [None if g is None else int(g) for g in window_groups],
            int(num_precursor_frames), int(num_fragment_frames),
            float(max_intensity_precursor), float(max_intensity_fragment),
            float(take_precursor), float(take_fragment),
            int(seed) & 0xFFFFFFFFFFFFFFFF, int(num_threads),
        )
        return [TimsFrame.from_py_ptr(f) for f in out]

    def read_compressed_data_full(self) -> List[bytes]:
        """Read compressed data.

        Returns:
            List[bytes]: Compressed data.
        """
        return self.__dataset.read_compressed_data_full()

    @classmethod
    def from_py_ptr(cls, obj):
        instance = cls.__new__(cls)
        instance.__dataset = obj
        return instance

    @classmethod
    def with_mz_calibration(cls, data_path: str, in_memory: bool, tof_intercept: float, tof_slope: float):
        """Create a DIA dataset with custom m/z calibration coefficients.

        This method allows providing externally-derived m/z calibration coefficients
        (e.g., from linear regression on SDK data) for accurate m/z conversion without
        requiring the Bruker SDK at runtime.

        The calibration formula is: sqrt(mz) = tof_intercept + tof_slope * tof_index

        Args:
            data_path: Path to the .d folder
            in_memory: Whether to load all data into memory
            tof_intercept: Intercept for sqrt(mz) = intercept + slope * tof
            tof_slope: Slope for sqrt(mz) = intercept + slope * tof

        Returns:
            TimsDatasetDIA with custom m/z calibration
        """
        instance = cls.__new__(cls)
        instance.data_path = data_path
        instance.binary_path = "CALIBRATED"
        instance.use_bruker_sdk = False
        instance._TimsDatasetDIA__dataset = ims.PyTimsDatasetDIA.with_mz_calibration(
            data_path, in_memory, tof_intercept, tof_slope
        )

        # Load metadata (needed for some properties)
        instance.meta_data = pd.read_sql_query(
            "SELECT * from Frames",
            sqlite3.connect(data_path + "/analysis.tdf")
        )
        instance.global_meta_data = dict(zip(
            *pd.read_sql_query(
                "SELECT * from GlobalMetadata",
                sqlite3.connect(data_path + "/analysis.tdf")
            ).values.T
        ))

        return instance

    def get_py_ptr(self):
        return self.__dataset
