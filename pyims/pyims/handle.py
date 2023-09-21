from typing import List

import pyims_connector as pims
import opentims_bruker_bridge as obb

from abc import ABC, abstractmethod


class TimsDataHandle(ABC):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.bp: List[str] = obb.get_so_paths()
        self.__handle = None

        appropriate_found = False
        for so_path in self.bp:
            try:
                self.__handle = pims.PyTimsDataHandle(self.data_path, so_path)
                appropriate_found = True
                break
            except Exception as e:
                continue
        assert appropriate_found is True, ("No appropriate bruker binary could be found, please check if your "
                                           "operating system is supported by open-tims-bruker-bridge.")

    def acquisition_mode(self) -> str:
        return self.__handle.get_acquisition_mode_as_string()

    def acquisition_mode_numerical(self) -> int:
        return self.__handle.get_acquisition_mode()


class TimsFrame:
    def __init__(self, frame_id: int, ms_type: int, ):
        pass
