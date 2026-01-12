import sqlite3
from typing import List, Optional

import numpy as np
import pandas as pd

from imspy.core.base import RustWrapperObject
from imspy.timstof.data import TimsDataset
from imspy.timstof.frame import TimsFrame

from sagepy.core import Precursor, Tolerance, ProcessedSpectrum, RawSpectrum, Representation, SpectrumProcessor

import imspy_connector
ims = imspy_connector.py_dda
import warnings



class PrecursorDDA(RustWrapperObject):
    def __init__(self, frame_id: int, precursor_id: int, highest_intensity_mz: float, average_mz: float,
                 inverse_ion_mobility: float, collision_energy: float, precuror_total_intensity: float,
                 isolation_mz: float, isolation_width: float, mono_mz: Optional[float] = None, charge: Optional[int] = None):
        self._precursor_ptr = ims.PyDDAPrecursor(
            frame_id, precursor_id, highest_intensity_mz, average_mz, inverse_ion_mobility, collision_energy,
            precuror_total_intensity, isolation_mz, isolation_width, mono_mz, charge
        )

    @classmethod
    def from_py_ptr(cls, precursor: ims.PyDDAPrecursor):
        instance = cls.__new__(cls)
        instance._precursor_ptr = precursor
        return instance

    @property
    def frame_id(self) -> int:
        return self._precursor_ptr.frame_id

    @property
    def precursor_id(self) -> int:
        return self._precursor_ptr.precursor_id

    @property
    def mono_mz(self) -> Optional[float]:
        return self._precursor_ptr.mono_mz

    @property
    def highest_intensity_mz(self) -> float:
        return self._precursor_ptr.highest_intensity_mz

    @property
    def average_mz(self) -> float:
        return self._precursor_ptr.average_mz

    @property
    def charge(self) -> Optional[int]:
        return self._precursor_ptr.charge

    @property
    def inverse_ion_mobility(self) -> float:
        return self._precursor_ptr.inverse_ion_mobility

    @property
    def collision_energy(self) -> float:
        return self._precursor_ptr.collision_energy

    @property
    def precuror_total_intensity(self) -> float:
        return self._precursor_ptr.precuror_total_intensity

    @property
    def isolation_mz(self) -> float:
        return self._precursor_ptr.isolation_mz

    @property
    def isolation_width(self) -> float:
        return self._precursor_ptr.isolation_width

    def __repr__(self):
        return (f"DDAPrecursor(frame_id={self.frame_id}, precursor_id={self.precursor_id}, "
                f"highest_intensity_mz={self.highest_intensity_mz}, average_mz={self.average_mz}, "
                f"inverse_ion_mobility={self.inverse_ion_mobility}, collision_energy={self.collision_energy}, "
                f"precuror_total_intensity={self.precuror_total_intensity}, isolation_mz={self.isolation_mz}, "
                f"isolation_width={self.isolation_width}, mono_mz={self.mono_mz}, charge={self.charge})")

    def get_py_ptr(self):
        return self._precursor_ptr

    def to_sage_precursor(self) -> Precursor:

        mz = self.mono_mz if self.mono_mz is not None else self.average_mz

        return Precursor(
            mz=mz,
            intensity=self.precuror_total_intensity,
            charge=self.charge,
            spectrum_ref=str(self.frame_id),
            isolation_window=Tolerance(da=(-self.isolation_width / 2, self.isolation_width / 2)),
            inverse_ion_mobility=self.inverse_ion_mobility,
            collision_energy=self.collision_energy,
        )


class TimsDatasetDDA(TimsDataset, RustWrapperObject):

    def __init__(self, data_path: str, in_memory: bool = False, use_bruker_sdk: bool = True, rename_id: bool = True):
        super().__init__(data_path=data_path, in_memory=in_memory, use_bruker_sdk=use_bruker_sdk)
        self.__dataset = ims.PyTimsDatasetDDA(self.data_path, self.binary_path, in_memory, self.use_bruker_sdk)
        if rename_id:
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

    def get_pasef_fragments(self, num_threads: int = 1) -> pd.DataFrame:
        """Get PASEF fragments.

        Args: num_threads (int, optional): Number of threads. Defaults to 1. CAUTION: As long as connection to
        datasets is established via bruker so / dll, using multiple threads is unstable.

        Returns:
            List[FragmentDDA]: List of PASEF fragments.
        """

        if self.use_bruker_sdk:
            warnings.warn("Using multiple threads is currently not supported when using Bruker SDK, "
                            "setting num_threads to 1.")
            num_threads = 1

        pasef_fragments = [FragmentDDA.from_py_ptr(fragment)
                           for fragment in self.__dataset.get_pasef_fragments(num_threads)]

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

    def get_precursor_frames(self, min_intensity: float = 75, max_peaks: int = 500, num_threads: int = 4) -> List[TimsFrame]:
        """
        Get precursor frames.
        Args:
            min_intensity: minimum intensity a peak must have to be considered
            max_peaks: maximum number of peaks to consider, frames will be sorted by intensity and only the top max_peaks will be considered
            num_threads: number of threads to use for processing

        Returns:
            List[TimsFrame]: List of all precursor frames
        """
        precursor_frames = [TimsFrame.from_py_ptr(frame) for frame in self.__dataset.get_precursor_frames(min_intensity, max_peaks, num_threads)]
        return precursor_frames

    def get_sage_processed_precursors(self, min_intensity: float = 75, max_peaks: int = 5000, file_id: int = 0, num_threads: int = 16) -> List[ProcessedSpectrum]:

        if self.use_bruker_sdk:
            warnings.warn("Using multiple threads is currently not supported when using Bruker SDK, "
                            "setting num_threads to 1.")
            num_threads = 1

        # get all selected precursors
        precursor_meta = self.get_selected_precursors()

        # create a dictionary with frame_id as key and a list of precursors as value
        precursor_dict = {}
        for precursor in precursor_meta:
            if precursor.frame_id not in precursor_dict:
                precursor_dict[precursor.frame_id] = []
            precursor_dict[precursor.frame_id].append(precursor)

        # get all precursor frames
        precursor_frames = self.get_precursor_frames(min_intensity, max_peaks, num_threads)

        # process all precursor frames
        processed_spectra = []

        spectrum_processor = SpectrumProcessor(
            take_top_n=max_peaks,
            min_deisotope_mz=0.0,
            deisotope=False,
        )

        # associate precursors with frames, precursor will be associated with the frame with the same frame_id
        for frame in precursor_frames:
            if frame.frame_id in precursor_dict:
                precursors = [p.to_sage_precursor() for p in precursor_dict[frame.frame_id]]
                raw_spectrum = RawSpectrum(
                    file_id=file_id,
                    spec_id=str(frame.frame_id),
                    total_ion_current=np.sum(frame.intensity),
                    precursors=precursors,
                    mz=frame.mz.astype(np.float32),
                    intensity=frame.intensity.astype(np.float32),
                    representation=Representation(representation="centroid"),
                    scan_start_time=frame.retention_time / 60,
                    ion_injection_time=frame.retention_time / 60,
                    ms_level=1,
                )

                # process the spectrum
                processed_spectrum = spectrum_processor.process(raw_spectrum)
                processed_spectra.append(processed_spectrum)

        # delete precursor_frames to free memory
        del precursor_frames

        return processed_spectra


    def get_selected_precursors(self) -> List[PrecursorDDA]:
        """
        Get meta data for all selected precursors
        Returns:
            List[PrecursorDDA]: List of all selected precursors
        """
        return [PrecursorDDA.from_py_ptr(precursor) for precursor in self.__dataset.get_selected_precursors()]

    def sample_pasef_fragments_random(self,
                                      scan_apex_values: List[int],
                                      scan_max_value: int,
                                      ) -> TimsFrame:
        """
        Sample PASEF fragments randomly from the dataset.
        Args:
            scan_apex_values: List of scan apex values to sample from
            scan_max_value: maximum scan value to sample from

        Returns:
            TimsFrame: sampled PASEF fragments
        """

        return TimsFrame.from_py_ptr(
            self.__dataset.sample_pasef_fragments_random(scan_apex_values, scan_max_value)
        )

    def sample_precursor_signal(self, num_frames: int, max_intensity: float, take_probability: float) -> TimsFrame:
        """
        Sample precursor signal from the dataset.
        Args:
            num_frames: number of frames to sample
            max_intensity: maximum intensity of the sampled frames
            take_probability: probability of taking a frame

        Returns:
            TimsFrame: sampled precursor signal
        """

        return TimsFrame.from_py_ptr(
            self.__dataset.sample_precursor_signal(num_frames, max_intensity, take_probability)
        )

    def __repr__(self):
        return (f"TimsDatasetDDA(data_path={self.data_path}, num_frames={self.frame_count}, "
                f"fragmented_precursors={self.fragmented_precursors.shape[0]})")

    def get_py_ptr(self):
        return self.__dataset

    @classmethod
    def from_py_ptr(cls, ptr):
        instance = cls.__new__(cls)
        instance.__dataset = ptr
        return instance


class FragmentDDA(RustWrapperObject):
    def __init__(self, frame_id: int, precursor_id: int, collision_energy: float, selected_fragment: TimsFrame):
        self._fragment_ptr = ims.PyTimsFragmentDDA(frame_id, precursor_id, collision_energy, selected_fragment.get_py_ptr())

    @classmethod
    def from_py_ptr(cls, fragment: ims.PyTimsFragmentDDA):
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
    def collision_energy(self) -> float:
        return self._fragment_ptr.collision_energy

    @property
    def selected_fragment(self) -> TimsFrame:
        return TimsFrame.from_py_ptr(self._fragment_ptr.selected_fragment)

    def __repr__(self):
        return f"FragmentDDA(frame_id={self.frame_id}, precursor_id={self.precursor_id}, " \
               f"collision_energy={self.collision_energy}, " \
               f"selected_fragment={self.selected_fragment})"

    def get_py_ptr(self):
        return self._fragment_ptr
