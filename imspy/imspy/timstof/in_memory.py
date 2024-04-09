import opentims_bruker_bridge as obb

import imspy_connector
from numpy.typing import NDArray

from imspy.timstof.frame import TimsFrame
from imspy.timstof.slice import TimsSlice

ims = imspy_connector.py_tdf_inmem


class TimsDatasetInMemory:
    def __init__(self, data_path: str):
        """TimsDataHandle class.
        Args:
            data_path (str): Path to the data.
        """
        self.__dataset = None
        self.binary_path = None

        self.data_path = data_path
        if data_path[-1] == "/":
            data_path = data_path[:-1]

        self.experiment_name = data_path.split("/")[-1]
        # Try to load the data with the first binary found
        appropriate_found = False
        for so_path in obb.get_so_paths():
            try:
                self.__dataset = ims.PyTimsDatasetInMemory(self.data_path, so_path)
                self.binary_path = so_path
                appropriate_found = True
                break
            except Exception:
                continue
        assert appropriate_found is True, ("No appropriate bruker binary could be found, please check if your "
                                           "operating system is supported by open-tims-bruker-bridge.")

    def get_frame(self, frame_id: int) -> TimsFrame:
        """Get a frame.
        Returns:
            TimsFrame: Frame.
        """
        return TimsFrame.from_py_tims_frame(self.__dataset.get_frame(frame_id))

    def get_slice(self, frame_ids: NDArray, num_threads: int = 8) -> TimsSlice:
        """Get a slice.
        Returns:
            TimsSlice: Slice.
        """
        return TimsSlice.from_py_tims_slice(self.__dataset.get_slice(frame_ids, num_threads))
