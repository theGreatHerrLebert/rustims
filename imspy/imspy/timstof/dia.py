import sqlite3
from typing import List

from imspy.timstof.data import TimsDataset
import pandas as pd

import imspy_connector
ims = imspy_connector.py_dia


class TimsDatasetDIA(TimsDataset):
    def __init__(self, data_path: str, in_memory: bool = False):
        super().__init__(data_path=data_path, in_memory=in_memory)
        self.__dataset = ims.PyTimsDatasetDIA(self.data_path, self.binary_path)

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

    def read_compressed_data_full(self) -> List[bytes]:
        """Read compressed data.

        Returns:
            List[bytes]: Compressed data.
        """
        return self.__dataset.read_compressed_data_full()
