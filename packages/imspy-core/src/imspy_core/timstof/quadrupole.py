from typing import Union, List

from imspy_core.data.spectrum import MzSpectrum
from imspy_core.timstof.frame import TimsFrame
from numpy.typing import NDArray

import imspy_connector
ims = imspy_connector.py_quadrupole

class PasefMeta:
    def __init__(self, frame: int, scan_start: int, scan_end: int, isolation_mz: float, isolation_width: float,
                 collision_energy: float, precursor: int):
        self.__py_ptr = ims.PyPasefMeta(frame, scan_start, scan_end, isolation_mz, isolation_width, collision_energy, precursor)

    @property
    def frame(self) -> int:
        return self.__py_ptr.frame

    @property
    def scan_start(self) -> int:
        return self.__py_ptr.scan_start

    @property
    def scan_end(self) -> int:
        return self.__py_ptr.scan_end

    @property
    def isolation_mz(self) -> float:
        return self.__py_ptr.isolation_mz

    @property
    def isolation_width(self) -> float:
        return self.__py_ptr.isolation_width

    @property
    def collision_energy(self) -> float:
        return self.__py_ptr.collision_energy

    @property
    def precursor(self) -> int:
        return self.__py_ptr.precursor

    def __repr__(self):
        return f"PasefMeta(frame={self.frame}, scan_start={self.scan_start}, scan_end={self.scan_end}, " \
               f"isolation_mz={self.isolation_mz}, isolation_width={self.isolation_width}, " \
               f"collision_energy={self.collision_energy}, precursor={self.precursor})"

    @classmethod
    def from_py_ptr(cls, py_ptr):
        self = cls.__new__(cls)
        self.__py_ptr = py_ptr
        return self

    def get_py_ptr(self):
        return self.__py_ptr


class TimsTofQuadrupoleDDA:
    def __init__(self, pasef_meta: List[PasefMeta], k: float | None = None):
        self.__py_ptr = ims.PyTimsTransmissionDDA(
            [pasef_meta[i].get_py_ptr() for i in range(len(pasef_meta))], k
        )
    def apply_transmission(self, frame_id: int, scan_id: int, mz: NDArray) -> NDArray:
        return self.__py_ptr.apply_transmission(frame_id, scan_id, mz)

    def transmit_spectrum(self, frame_id: int, scan_id: int, spectrum: MzSpectrum, min_probability: float | None = None) -> MzSpectrum:
        return MzSpectrum.from_py_ptr(self.__py_ptr.transmit_spectrum(frame_id, scan_id, spectrum.get_py_ptr(), min_probability))

    def transmit_frame(self, frame: TimsFrame, min_probability: float | None = None) -> TimsFrame:
        return TimsFrame.from_py_ptr(self.__py_ptr.transmit_tims_frame(frame.get_py_ptr(), min_probability))

    def transmit_ion(self, frames: NDArray, scans: NDArray, spectrum: MzSpectrum, min_probability: float | None = None) -> List[List[MzSpectrum]]:
        transmission_profile = self.__py_ptr.transmit_ion(frames, scans, spectrum.get_py_ptr(), min_probability)
        result = []
        for i in enumerate(frames):
            scan_list = []
            for j in enumerate(scans):
                scan_list.append(MzSpectrum.from_py_ptr(transmission_profile[i][j]))
            result.append(scan_list)

        return result

    def is_transmitted(self, frame_id: int, scan_id: int, mz: float, min_probability: float | None = None) -> bool:
        return self.__py_ptr.is_transmitted(frame_id, scan_id, mz, min_probability)

    def any_transmitted(self, frame_id: int, scan_id: int, mz: NDArray, min_probability: float | None = None) -> bool:
        return self.__py_ptr.any_transmitted(frame_id, scan_id, mz, min_probability)

    def all_transmitted(self, frame_id: int, scan_id: int, mz: NDArray, min_probability: float | None) -> bool:
        return self.__py_ptr.all_transmitted(frame_id, scan_id, mz, min_probability)

    def get_transmission_set(self, frame_id: int, scan_id: int, mz: NDArray, min_probability: float | None) -> set[int]:
        return self.__py_ptr.get_transmission_set(frame_id, scan_id, mz, min_probability)

    def isotopes_transmitted(self, frame_id: int, scan_id: int, mz_mono: float, mz: NDArray, min_probability: float | None) -> tuple[float, list[tuple[float, float]]]:
        return self.__py_ptr.isotopes_transmitted(frame_id, scan_id, mz_mono, mz, min_probability)

    def __repr__(self):
        return f"TimsTofQuadrupoleDDA()"

    @classmethod
    def from_py_ptr(cls, py_ptr):
        self = cls.__new__(cls)
        self.__py_ptr = py_ptr
        return self

    def get_py_ptr(self):
        return self.__py_ptr

class TimsTofQuadrupoleDIA:
    def __init__(self, frame: NDArray, frame_window_group: NDArray, window_group: NDArray, scan_start: NDArray,
                 scan_end: NDArray, isolation_mz: NDArray, isolation_width: NDArray, k: float | None = None):
        self.handle = ims.PyTimsTransmissionDIA(
            frame, frame_window_group, window_group, scan_start, scan_end, isolation_mz, isolation_width, k
        )

    def apply_transmission(self, frame_id: int, scan_id: int, mz: NDArray) -> NDArray:
        return self.handle.apply_transmission(frame_id, scan_id, mz)

    def transmit_spectrum(self, frame_id: int, scan_id: int, spectrum: MzSpectrum,
                          min_probability: float | None = None) -> MzSpectrum:
        return MzSpectrum.from_py_ptr(self.handle.transmit_spectrum(frame_id, scan_id,
                                                                    spectrum.get_py_ptr(), min_probability))

    def transmit_frame(self, frame: TimsFrame, min_probability: float | None = None) -> TimsFrame:
        return TimsFrame.from_py_ptr(self.handle.transmit_tims_frame(frame.get_py_ptr(), min_probability))

    def frame_to_window_group(self, frame_id: int) -> int:
        return self.handle.frame_to_window_group(frame_id)

    def is_transmitted(self, frame_id: int, scan_id: int, mz: float, min_proba: float | None = None) -> bool:
        return self.handle.is_transmitted(frame_id, scan_id, mz, min_proba)

    def any_transmitted(self, frame_id: int, scan_id: int, mz: NDArray, min_proba: float | None = None) -> bool:
        return self.handle.any_transmitted(frame_id, scan_id, mz, min_proba)

    def all_transmitted(self, frame_id: int, scan_id: int, mz: NDArray, min_proba: float | None = None) -> bool:
        return self.handle.all_transmitted(frame_id, scan_id, mz, min_proba)

    def get_transmission_set(self, frame_id: int, scan_id: int, mz: NDArray, min_proba: float | None = None) -> set[int]:
        return self.handle.get_transmission_set(frame_id, scan_id, mz, min_proba)

    def transmit_ion(self, frame_ids: NDArray, scan_ids: NDArray, spectrum: MzSpectrum, min_probability: Union[float, None]) -> List[List[MzSpectrum]]:
        transmission_profile = self.handle.transmit_ion(frame_ids, scan_ids, spectrum.get_py_ptr(), min_probability)
        result = []
        for i in enumerate(frame_ids):
            scan_list = []
            for j in enumerate(scan_ids):
                scan_list.append(MzSpectrum.from_py_ptr(transmission_profile[i][j]))
            result.append(scan_list)

        return result

    def is_precursor(self, frame_id: int) -> bool:
        return self.handle.is_precursor(frame_id)

    def isotopes_transmitted(
            self,
            frame_id: int,
            scan_id: int,
            mz_mono: float,
            mz: NDArray,
            min_proba: float | None = None
    ) -> tuple[float, list[tuple[float, float]]]:
        """
        Get the transmission probability for a list of isotopes
        Args:
            frame_id:
            scan_id:
            mz_mono:
            mz:
            min_proba:

        Returns:

        """
        return self.handle.isotopes_transmitted(frame_id, scan_id, mz_mono, mz, min_proba)

    def __repr__(self):
        return f"TimsTofQuadrupoleDIA()"

    @classmethod
    def from_py_ptr(cls, py_ptr):
        self = cls.__new__(cls)
        self.handle = py_ptr
        return self

    def get_py_ptr(self):
        return self.handle
