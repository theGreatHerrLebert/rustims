from abc import abstractmethod
from typing import Callable

from imspy.core import TimsFrame
from numpy.typing import NDArray


class TimsTofQuadrupoleSetting:
    @abstractmethod
    def get_transmission_function(self, frame_id: int) -> Callable[[NDArray], NDArray]:
        pass

    @abstractmethod
    def apply_transmission(self, frame: TimsFrame) -> TimsFrame:
        pass


class TransmissionDDA(TimsTofQuadrupoleSetting):
    def get_transmission_function(self, frame_id: int) -> Callable[[NDArray], NDArray]:
        pass

    def apply_transmission(self, frame: TimsFrame) -> TimsFrame:
        pass


class TransmissionDIA(TimsTofQuadrupoleSetting):
    def get_transmission_function(self, frame_id: int) -> Callable[[NDArray], NDArray]:
        pass

    def apply_transmission(self, frame: TimsFrame) -> TimsFrame:
        pass


class TransmissionMIDIA(TimsTofQuadrupoleSetting):
    def get_transmission_function(self, frame_id: int) -> Callable[[NDArray], NDArray]:
        pass

    def apply_transmission(self, frame: TimsFrame) -> TimsFrame:
        pass
