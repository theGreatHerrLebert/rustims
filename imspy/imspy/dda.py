import sqlite3
from imspy.data import TimsDataset
import pandas as pd

import imspy_connector as pims
from imspy.frame import TimsFrame


class TimsDatasetDDA(TimsDataset):

    def __init__(self, data_path: str):
        super().__init__(data_path=data_path)
        self.__dataset = pims.PyTimsDatasetDDA(self.data_path, self.binary_path)
        self.meta_data = self.meta_data.rename(columns={"Id": "frame_id"})
        self.fragmented_precursors = self._load_selected_precursors().rename(
            columns={
                'Id': 'precursor_id',
                'LargestPeakMz': 'largest_peak_mz',
                'AverageMz': 'average_mz',
                'MonoisotopicMz': 'monoisotopic_mz',
                'Charge': 'charge',
                'ScanNumber': 'average_scan',
                'Intensity': 'intensity',
                'Parent': 'parent_id',
            }
        )
        self.pasef_meta_data = self._load_pasef_meta_data().rename(
            columns={
                'Frame': 'frame_id',
                'ScanNumBegin': 'scan_begin',
                'ScanNumEnd': 'scan_end',
                'IsolationMz': 'isolation_mz',
                'IsolationWidth': 'isolation_width',
                'CollisionEnergy': 'collision_energy',
                'Precursor': 'precursor_id'
            }
        )

    def _load_selected_precursors(self):
        """Get precursors selected for fragmentation.

        Returns:
            pd.DataFrame: Precursors selected for fragmentation.
        """
        return pd.read_sql_query("SELECT * from Precursors", sqlite3.connect(self.data_path + "/analysis.tdf"))

    def _load_pasef_meta_data(self):
        """Get PASEF meta data for DDA.

        Returns:
            pd.DataFrame: PASEF meta data.
        """
        return pd.read_sql_query("SELECT * from PasefFrameMsMsInfo",
                                 sqlite3.connect(self.data_path + "/analysis.tdf"))

    def get_pasef_fragments(self) -> pd.DataFrame:
        """Get PASEF fragments.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 4.

        Returns:
            List[FragmentDDA]: List of PASEF fragments.
        """
        pasef_fragments = [FragmentDDA.from_py_tims_fragment_dda(fragment)
                           for fragment in self.__dataset.get_pasef_fragments(1)]

        pasef_fragments = pd.DataFrame({
            'frame_id': [s.frame_id for s in pasef_fragments],
            'precursor_id': [s.precursor_id for s in pasef_fragments],
            'raw_data': [s.selected_fragment for s in pasef_fragments]
        })

        A = pd.merge(
            pasef_fragments, self.pasef_meta_data,
            left_on=['precursor_id', 'frame_id'],
            right_on=['precursor_id', 'frame_id'],
            how='inner',
        )

        B = pd.merge(
            A, self.fragmented_precursors,
            left_on=['precursor_id'],
            right_on=['precursor_id'],
            how='inner'
        )

        time = self.meta_data[['frame_id']]
        time.insert(time.shape[1], "time", self.meta_data['Time'] / 60)
        
        return pd.merge(time, B, left_on=['frame_id'], right_on=['frame_id'], how='inner')


class FragmentDDA:
    def __init__(self, frame_id: int, precursor_id: int, selected_fragment: TimsFrame):
        self._fragment_ptr = pims.PyTimsFragmentDDA(frame_id, precursor_id, selected_fragment.get_fragment_ptr())

    @classmethod
    def from_py_tims_fragment_dda(cls, fragment: pims.PyTimsFragmentDDA):
        instance = cls.__new__(cls)
        instance._fragment_ptr = fragment
        return instance

    @property
    def frame_id(self) -> int:
        return self._fragment_ptr.frame_id

    @property
    def precursor_id(self) -> int:
        return self._fragment_ptr.precursor_id

    @property
    def selected_fragment(self) -> TimsFrame:
        return TimsFrame.from_py_tims_frame(self._fragment_ptr.selected_fragment)

    def __repr__(self):
        return f"FragmentDDA(frame_id={self.frame_id}, precursor_id={self.precursor_id}, " \
               f"selected_fragment={self.selected_fragment})"

    def get_fragment_ptr(self):
        return self._fragment_ptr
