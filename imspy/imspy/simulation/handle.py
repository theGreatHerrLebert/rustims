import pandas as pd
import numpy as np

import imspy_connector as ims


class TimsTofSyntheticsDataHandleRust:
    def __init__(self, path: str):
        self.path = path
        self.__handle = ims.PyTimsTofSyntheticsDataHandle(path)

    def get_py_ptr(self):
        return self.__handle

    def __repr__(self):
        return f"TimsTofSyntheticsDataHandleRust(path={self.path})"

    @property
    def peptide_to_ions(self) -> pd.DataFrame:
        peptide_ids, charges = self.__handle.get_peptide_ions()

        return pd.DataFrame({
            'peptide_id': peptide_ids,
            'charge': charges,
        })
