from typing import Union, List

from imspy.data.spectrum import MzSpectrum
from imspy.timstof.frame import TimsFrame
from numpy.typing import NDArray

import imspy_connector
ims = imspy_connector.py_quadrupole


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
        return MzSpectrum.from_py_mz_spectrum(self.handle.transmit_spectrum(frame_id, scan_id,
                                                                            spectrum.get_spec_ptr(), min_probability))

    def transmit_frame(self, frame: TimsFrame, min_probability: float | None = None) -> TimsFrame:
        return TimsFrame.from_py_tims_frame(self.handle.transmit_tims_frame(frame.get_frame_ptr(), min_probability))

    def frame_to_window_group(self, frame_id: int) -> int:
        return self.handle.frame_to_window_group(frame_id)

    def is_transmitted(self, frame_id: int, scan_id: int, mz: float, min_proba: float | None = None) -> bool:
        return self.handle.is_transmitted(frame_id, scan_id, mz, min_proba)

    def any_transmitted(self, frame_id: int, scan_id: int, mz: NDArray, min_proba: float | None = None) -> bool:
        return self.handle.any_transmitted(frame_id, scan_id, mz, min_proba)

    def transmit_ion(self, frame_ids: NDArray, scan_ids: NDArray, spectrum: MzSpectrum, min_probability: Union[float, None]) -> List[List[MzSpectrum]]:
        transmission_profile = self.handle.transmit_ion(frame_ids, scan_ids, spectrum.get_spec_ptr(), min_probability)
        result = []
        for i in enumerate(frame_ids):
            scan_list = []
            for j in enumerate(scan_ids):
                scan_list.append(MzSpectrum.from_py_mz_spectrum(transmission_profile[i][j]))
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
