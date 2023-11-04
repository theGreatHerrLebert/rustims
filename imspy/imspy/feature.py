import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

import json

from pyims.data import TimsSlice, TimsFrame
from pyims.utility import gaussian, exp_gaussian
from pyims.isotopes import IsotopePatternGenerator, create_initial_feature_distribution
from abc import ABC, abstractmethod
from typing import Optional, Dict
class Profile:

    def __init__(self,positions:Optional[ArrayLike] = None, rel_abundancies:Optional[ArrayLike] = None, model_params: Optional[Dict] = None, jsons:Optional[str] = None):
        if jsons is not None:
            self._jsons = jsons
            self.positions, self.rel_abundancies, self.model_params, self.access_dictionary = self._from_jsons(jsons)
        else:
            self.positions = np.asarray(positions)
            self.rel_abundancies = np.asarray(rel_abundancies)
            self.model_params = model_params
            self.access_dictionary = {p:i for p,i in zip(positions, rel_abundancies)}
            self._jsons = self._to_jsons()

    def __iter__(self):
        self.__size = len(self.positions)
        self.__iterator_pos = 0
        return self

    def __next__(self):
        if self.__iterator_pos < self.__size:
            p = self.positions[self.__iterator_pos]
            ra = self.rel_abundancies[self.__iterator_pos]
            self.__iterator_pos += 1
            return p,ra
        raise StopIteration

    def __getitem__(self, position: int):
        return self.access_dictionary[position]

    def _to_jsons(self):
        mp = {}
        for key,val in self.model_params.items():
            if isinstance(val,np.generic):
                mp[key] = val.item()
            else:
                mp[key] = val

        json_dict = {"positions": self.positions.tolist(),
                     "rel_abundancies": self.rel_abundancies.tolist(),
                     "model_params": mp}
        return json.dumps(json_dict, allow_nan = False)

    def _from_jsons(self, jsons:str):
        json_dict = json.loads(jsons)
        positions = json_dict["positions"]
        rel_abundancies = json_dict["rel_abundancies"]
        access_dictionary = {p:i for p,i in zip(positions, rel_abundancies)}
        return positions,rel_abundancies,json_dict["model_params"], access_dictionary

    @property
    def jsons(self):
        return self._jsons

class RTProfile(Profile):

    def __init__(self,frames:Optional[ArrayLike] = None, rel_abundancies:Optional[ArrayLike] = None, model_params: Optional[Dict] = None, jsons:Optional[str] = None):
        super().__init__(frames, rel_abundancies, model_params, jsons)

    @property
    def frames(self):
        return self.positions

class ScanProfile(Profile):

    def __init__(self,scans:Optional[ArrayLike] = None, rel_abundancies:Optional[ArrayLike] = None, model_params: Optional[Dict] = None, jsons:Optional[str] = None):
        super().__init__(scans, rel_abundancies, model_params,  jsons)

    @property
    def scans(self):
        return self.positions

class ChargeProfile(Profile):
    def __init__(self,charges:Optional[ArrayLike] = None, rel_abundancies:Optional[ArrayLike] = None, model_params: Optional[Dict] = None, jsons:Optional[str] = None):
        abundant_charges = charges[rel_abundancies>0]
        relative_abundancies_over_0 = rel_abundancies[rel_abundancies>0]
        super().__init__(abundant_charges, relative_abundancies_over_0, model_params, jsons)

    @property
    def charges(self):
        return self.positions

class FeatureGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_feature(self, mz: float, charge: int):
        pass


class PrecursorFeatureGenerator(FeatureGenerator):

    def __init__(self):
        super(PrecursorFeatureGenerator).__init__()

    def generate_feature(self, mz: float, charge: int,
                         pattern_generator: IsotopePatternGenerator,
                         num_rt: int = 64,
                         num_im: int = 32,
                         distr_im=gaussian,
                         distr_rt=exp_gaussian,
                         rt_lower: float = -9,
                         rt_upper: float = 18,
                         im_lower: float = -4,
                         im_upper: float = 4,
                         intensity_amplitude: float = 1e3,
                         min_intensity: int = 5) -> TimsSlice:

        I = create_initial_feature_distribution(num_rt, num_im, rt_lower, rt_upper,
                                                im_lower, im_upper, distr_im, distr_rt)

        frame_list = []

        spec = pattern_generator.generate_spectrum(mz, charge, min_intensity=5, centroided=True, k=12)
        mz, intensity = spec.mz, spec.intensity / np.max(spec.intensity) * 100

        for i in range(num_rt):

            scan_arr, mz_arr, intensity_arr, tof_arr, inv_mob_arr = [], [], [], [], []

            for j in range(num_im):
                intensity_scaled = intensity * I[i, j]
                intensity_scaled = intensity_scaled * intensity_amplitude
                int_intensity = np.array([int(x) for x in intensity_scaled])

                bt = [(x, y) for x, y in zip(mz, int_intensity) if y >= min_intensity]

                mz_tmp = np.array([x for x, y in bt])
                intensity_tmp = np.array([y for x, y in bt]).astype(np.int32)

                rts = np.repeat(i, intensity_tmp.shape[0]).astype(np.float32)
                scans = np.repeat(j, intensity_tmp.shape[0]).astype(np.int32)
                tof = np.ones_like(rts).astype(np.int32)
                inv_mob = np.ones_like(scans)

                scan_arr.append(scans)
                mz_arr.append(mz_tmp)
                intensity_arr.append(intensity_tmp)
                tof_arr.append(tof)
                inv_mob_arr.append(inv_mob)

            frame = TimsFrame(None, i, float(i),
                              np.hstack(scan_arr), np.hstack(mz_arr), np.hstack(intensity_arr),
                              np.hstack(tof_arr), np.hstack(inv_mob_arr))

            frame_list.append(frame)

        return TimsSlice(None, frame_list, [])


class FeatureBatch:
    def __init__(self, feature_table: pd.DataFrame, raw_data: TimsSlice):
        """

        :param feature_table:
        :param raw_data:
        """
        self.feature_table = feature_table.sort_values(['rt_length'], ascending=False)
        self.raw_data = raw_data
        self.__feature_counter = 0
        self.current_row = self.feature_table.iloc[0]
        r = self.current_row
        self.current_feature = self.get_feature(r.mz - 0.1, r.mz + (r.num_peaks / r.charge) + 0.1,
                                                r.im_start,
                                                r.im_stop,
                                                r.rt_start,
                                                r.rt_stop)

    def __repr__(self):
        return f'FeatureBatch(features={self.feature_table.shape[0]}, slice={self.raw_data})'

    def get_feature(self, mz_min, mz_max, scan_min, scan_max, rt_min, rt_max, intensity_min=20):
        return self.raw_data.filter_ranged(mz_min=mz_min,
                                           mz_max=mz_max,
                                           scan_min=scan_min,
                                           scan_max=scan_max,
                                           intensity_min=intensity_min,
                                           rt_min=rt_min,
                                           rt_max=rt_max)

    def __iter__(self):
        return self.current_row, self.current_feature

    def __next__(self):
        feature = self.current_row
        data = self.current_feature

        if self.__feature_counter < self.feature_table.shape[0]:
            self.__feature_counter += 1
            self.current_row = self.feature_table.iloc[self.__feature_counter]
            r = self.current_row
            self.current_feature = self.get_feature(r.mz - 0.1,
                                                    r.mz + (r.num_peaks / r.charge) + 0.1,
                                                    r.im_start,
                                                    r.im_stop,
                                                    r.rt_start,
                                                    r.rt_stop)
            return feature, data

        raise StopIteration
