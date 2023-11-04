import os
import json
from multiprocessing import Pool
import functools
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from pyims.data import MzSpectrum, TimsFrame
from pyims.proteome import PeptideDigest, ProteomicsExperimentSampleSlice, ProteomicsExperimentDatabaseHandle
from pyims.isotopes import AveragineGenerator
import pyims.simulation.hardware_models as hardware

class ProteomicsExperiment(ABC):
    def __init__(self, path: str):

        # path strings to experiment folder, database and output subfolder
        self.path = path
        self.output_path = f"{os.path.dirname(path)}/output"
        self.database_path = f"{self.path}/experiment_database.db"

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if os.path.exists(self.database_path):
            raise FileExistsError("Experiment found in the given path.")

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        # output folder must be empty, otherwise it is
        # assumend that it contains old experiments
        if len(os.listdir(self.output_path)) > 0:
            raise FileExistsError("Experiment found in the given path.")

        # init database and loaded sample
        self.database = ProteomicsExperimentDatabaseHandle(self.database_path)
        self.loaded_sample = None

        # hardware methods
        self._lc_method = None
        self._ionization_method = None
        self._ion_mobility_separation_method = None
        self._mz_separation_method = None

    @property
    def lc_method(self):
        return self._lc_method

    @lc_method.setter
    def lc_method(self, method: hardware.LiquidChromatography):
        self._lc_method = method

    @property
    def ionization_method(self):
        return self._ionization_method

    @ionization_method.setter
    def ionization_method(self, method: hardware.IonSource):
        self._ionization_method = method

    @property
    def ion_mobility_separation_method(self):
        return self._ion_mobility_separation_method

    @ion_mobility_separation_method.setter
    def ion_mobility_separation_method(self, method: hardware.IonMobilitySeparation):
        self._ion_mobility_separation_method = method

    @property
    def mz_separation_method(self):
        return self._mz_separation_method

    @mz_separation_method.setter
    def mz_separation_method(self, method: hardware.MzSeparation):
        self._mz_separation_method = method

    @abstractmethod
    def load_sample(self, sample: PeptideDigest):
        self.database.push("PeptideDigest",sample)

    @abstractmethod
    def run(self):
        pass


class LcImsMsMs(ProteomicsExperiment):
    def __init__(self, path:str):
        super().__init__(path)

    def load_sample(self, sample: PeptideDigest):
        return super().load_sample(sample)

    def run(self, chunk_size: int = 1000, assemble_processes: int = 8, frames_per_assemble_process:int = 20):
        self._simulate_features(chunk_size)
        self._assemble(frames_per_assemble_process, assemble_processes)


    def _simulate_features(self, chunk_size):
        # load bulks of data here as dataframe if necessary
        for data_chunk in self.database.load_chunks(chunk_size):
            self.lc_method.run(data_chunk)
            self.ionization_method.run(data_chunk)
            self.ion_mobility_separation_method.run(data_chunk)
            self.mz_separation_method.run(data_chunk)
            self.database.update(data_chunk)
    
    @staticmethod
    def _assemble_frame_range(frame_range, scan_id_min, scan_id_max, default_abundance, resolution, output_path, database_path):

        frame_range_start = frame_range[0]
        frame_range_end = frame_range[1]
        # generate file_name
        file_name = f"frames_{frame_range_start}_{frame_range_end}.parquet"
        output_file_path = f"{output_path}/{file_name}"

        frame_range = range(frame_range_start,frame_range_end)
        scan_range  = range(scan_id_min, scan_id_max+1)

        thread_db_handle = ProteomicsExperimentDatabaseHandle(database_path)
        ions_in_split = thread_db_handle.load_frames((frame_range_start, frame_range_end), spectra_as_jsons = True)


        # skip if no peptides in split
        if ions_in_split.shape[0] == 0:
            return {}

        # spectra are currently stored in json format (from SQL db)
        ions_in_split.loc[:,"simulated_mz_spectrum"] = ions_in_split["simulated_mz_spectrum"].transform(lambda s: MzSpectrum.from_jsons(jsons=s))

        # construct signal data set
        signal = {f_id:{s_id:[] for s_id in scan_range} for f_id in frame_range }

        for _,row in ions_in_split.iterrows():

            ion_frame_start = max(frame_range_start, row["frame_min"])
            ion_frame_end = min(frame_range_end-1, row["frame_max"]) # -1 here because frame_range_end is covered by next frame range

            ion_scan_start = max(scan_id_min, row["scan_min"])
            ion_scan_end = min(scan_id_max, row["scan_max"])

            ion_frame_profile = row["simulated_frame_profile"]
            ion_scan_profile = row["simulated_scan_profile"]

            ion_charge_abundance = row["abundancy"]*row["relative_abundancy"]

            ion_spectrum = row["simulated_mz_spectrum"]

            # frame start and end inclusive
            for f_id in range(ion_frame_start, ion_frame_end+1):
                # scan start and end inclusive
                for s_id in range(ion_scan_start, ion_scan_end+1):

                    abundance = ion_charge_abundance*ion_frame_profile[f_id]*ion_scan_profile[s_id]
                    rel_to_default_abundance = abundance/default_abundance

                    signal[f_id][s_id].append(ion_spectrum*rel_to_default_abundance)

        output_dict = {"frame_id" : [],
                       "scan_id" : [],
                       "mz" : [],
                       "intensity" : [],
        }
        for (f_id,frame_dict) in signal.items():
            for (s_id,scan_spectra_list) in frame_dict.items():

                if len(scan_spectra_list) > 0:
                    scan_spectrum = MzSpectrum.from_mz_spectra_list(scan_spectra_list,resolution = resolution)
                    output_dict["mz"].append(scan_spectrum.mz.tolist())
                    output_dict["intensity"].append(scan_spectrum.intensity.tolist())
                    output_dict["scan_id"].append(s_id)
                    output_dict["frame_id"].append(f_id)
                    signal[f_id][s_id] = None

        for key, value in output_dict.items():
            output_dict[key] = pa.array(value)

        pa_table = pa.Table.from_pydict(output_dict)

        pq.write_table(pa_table, output_file_path, compression=None)

    def _assemble(self, frames_per_process:int, num_processes:int):

        scan_id_min = self.ion_mobility_separation_method.scan_id_min
        scan_id_max = self.ion_mobility_separation_method.scan_id_max
        default_abundance = self.mz_separation_method.model.default_abundance
        resolution = self.mz_separation_method.resolution

        split_positions = np.arange(0, self.lc_method.num_frames , step=frames_per_process ).astype(int)
        split_start = split_positions[:-1]
        split_end = split_positions[1:]

        assemble_frame_range = functools.partial(self._assemble_frame_range, scan_id_min = scan_id_min, scan_id_max = scan_id_max, default_abundance = default_abundance, resolution = resolution, output_path = self.output_path, database_path= self.database_path)

        if num_processes > 1:
            with Pool(num_processes) as pool:
                list(tqdm(pool.imap(assemble_frame_range, zip(split_start, split_end)),total=len(split_start)))
        else:
            for start,end in tqdm(zip(split_start,split_end), total = len(split_start)):
                assemble_frame_range((start, end))



