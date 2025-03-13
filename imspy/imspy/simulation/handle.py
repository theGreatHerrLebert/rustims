import os
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

    def get_transmitted_ions(self, num_threads: int=-1, dda: bool=False) -> pd.DataFrame:
        """
        Get transmitted ions, needed to create fragment ion table for simulation.
        Args:
            num_threads: Number of threads to use for the calculation.
            dda: if true, the ions are sampled from a DDA experiment, otherwise from a DIA experiment.

        Returns:
            pd.DataFrame: DataFrame with the following columns:
                - peptide_id: Peptide ID.
                - ion_id: Ion ID.
                - sequence: Peptide sequence.
                - charge: Ion charge.
                - collision_energy: Collision energy.
        """
        if num_threads == -1:
            num_threads = os.cpu_count()

        peptide_ids, ion_ids, sequences, charges, collision_energies = self.__handle.get_transmitted_ions(num_threads, dda)
        return pd.DataFrame({
            'peptide_id': peptide_ids,
            'ion_id': ion_ids,
            'sequence': sequences,
            'charge': charges,
            'collision_energy': collision_energies
        })
