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

    def get_transmitted_ions(self, num_threads: int = 4) -> pd.DataFrame:
        peptide_ids, sequences, charges, collision_energies = self.__handle.get_transmitted_ions(num_threads)
        return pd.DataFrame({
            'peptide_id': peptide_ids,
            'sequence': sequences,
            'charge': charges,
            'collision_energy': collision_energies
        })
