import pandas as pd

import imspy_connector
ims = imspy_connector.py_simulation


class TimsTofSyntheticsDataHandleRust:
    def __init__(self, path: str):
        self.path = path
        self.__handle = ims.PyTimsTofSyntheticsDataHandle(path)

    def get_py_ptr(self):
        return self.__handle

    def __repr__(self):
        return f"TimsTofSyntheticsDataHandleRust(path={self.path})"

    def get_transmitted_ions(self, num_threads: int=4, dda: bool=False) -> pd.DataFrame:
        peptide_ids, ion_ids, sequences, charges, collision_energies = self.__handle.get_transmitted_ions(num_threads, dda)
        return pd.DataFrame({
            'peptide_id': peptide_ids,
            'ion_id': ion_ids,
            'sequence': sequences,
            'charge': charges,
            'collision_energy': collision_energies
        })
