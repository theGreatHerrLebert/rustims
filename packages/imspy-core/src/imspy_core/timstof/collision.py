from abc import abstractmethod
import imspy_connector
ims = imspy_connector.py_quadrupole


class TimsTofCollisionEnergy:
    def __init__(self):
        pass

    @abstractmethod
    def get_collision_energy(self) -> float:
        pass

    @abstractmethod
    def get_collision_energies(self) -> list[float]:
        pass


class TimsTofCollisionEnergyDIA(TimsTofCollisionEnergy):
    def __init__(self, frame: list[int], frame_window_group: list[int], window_group: list[int], scan_start: list[int],
                 scan_end: list[int], collision_energy: list[float]):

        super().__init__()
        self.__ptr = ims.PyTimsTofCollisionEnergyDIA(frame, frame_window_group, window_group,
                                                     scan_start, scan_end, collision_energy)

    def get_collision_energy(self, frame_id: int, scan_id: int) -> float:
        return self.__ptr.get_collision_energy(frame_id, scan_id)

    def get_collision_energies(self, frame_ids: list[int], scan_ids: list[int]) -> list[float]:
        return self.__ptr.get_collision_energies(frame_ids, scan_ids)
