from imspy.data import TimsDataset
import pandas as pd

import imspy_connector as pims


class TimsDatasetDIA(TimsDataset):
    def __init__(self, data_path: str):
        super().__init__(data_path=data_path)
        self.__dataset = pims.PyTimsDatasetDIA(self.data_path, self.binary_path)

    @property
    def pasef_meta_data(self):
        """Get PASEF meta data for DIA.

        Returns:
            pd.DataFrame: PASEF meta data.
        """
        return pd.read_sql_query("SELECT * from DiaFrameMsMsWindows",
                                 sqlite3.connect(self.data_path + "/analysis.tdf"))
