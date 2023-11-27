from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
from multiprocessing import Pool

import tensorflow as tf
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.stats import exponnorm, norm, binom, gamma

from imspy.chemistry.mass import STANDARD_TEMPERATURE, STANDARD_PRESSURE, BufferGas, get_num_protonizable_sites
from imspy.chemistry.mobility import SUMMARY_CONSTANT as CCS_K0_CONVERSION_CONSTANT
from imspy.proteome import ProteomicsExperimentSampleSlice
from imspy.feature import RTProfile, ScanProfile, ChargeProfile
from imspy.isotopes import AveragineGenerator
from imspy.utility.utilities import tokenizer_from_json


class Device(ABC):
    def __init__(self, name: str):
        self.name = name
        self._temperature = STANDARD_TEMPERATURE
        self._pressure = STANDARD_PRESSURE

    @property
    def temperature(self):
        """
        Get device temperature

        :return: Temperature of device in Kelvin.
        :rtype: float
        """
        return self._temperature

    @temperature.setter
    def temperature(self, T:float):
        """
        Set device temperature

        :param T: Temperature in Kelvin.
        :type T: float
        """
        self._temperature = T

    @property
    def pressure(self):
        """
        Get device pressure

        :return: Pressure of device in Pa.
        :rtype: float
        """
        return self._pressure

    @pressure.setter
    def pressure(self, p:float):
        """
        Set device pressure
        :param p: Pressure in Pa.
        :type p: float
        """
        self._pressure = p

    @abstractmethod
    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: Device):
        pass


class Chromatography(Device):
    def __init__(self, name: str="ChromatographyDevice"):
        super().__init__(name)
        self._apex_model = None
        self._profile_model = None
        self._irt_to_rt_converter = None
        self._frame_length = 1200
        self._gradient_length = 120*60*1000 # 120 minutes in miliseconds

    @property
    def frame_length(self):
        return self._frame_length

    @frame_length.setter
    def frame_length(self, milliseconds: int):
        self._frame_length = milliseconds

    @property
    def gradient_length(self):
        return self._gradient_length/(60*1000)

    @gradient_length.setter
    def gradient_length(self, minutes: int):
        self._gradient_length = minutes*60*1000

    @property
    def num_frames(self):
        return self._gradient_length//self._frame_length

    @property
    def apex_model(self):
        return self._apex_model

    @apex_model.setter
    def apex_model(self, model: ChromatographyApexModel):
        self._apex_model = model

    @property
    def profile_model(self):
        return self._profile_model

    @property
    def irt_to_rt_converter(self):
        return self._irt_to_rt_converter

    @irt_to_rt_converter.setter
    def irt_to_rt_converter(self, converter:callable):
        self._irt_to_rt_converter = converter

    @profile_model.setter
    def profile_model(self, model:ChromatographyProfileModel):
        self._profile_model = model

    def irt_to_frame_id(self, irt: float):
        return self.rt_to_frame_id(self.irt_to_rt(irt))

    @abstractmethod
    def rt_to_frame_id(self, rt: float):
        pass

    def irt_to_rt(self, irt):
        return self.irt_to_rt_converter(irt)

    def frame_time_interval(self, frame_id:ArrayLike):
        frame_id = np.atleast_1d(frame_id)
        frame_length_minutes = self.frame_length/(60*1000)
        s = (frame_id-1)*frame_length_minutes
        e = frame_id*frame_length_minutes
        return np.stack([s, e], axis = 1)

    def frame_time_middle(self, frame_id: ArrayLike):
        return np.mean(self.frame_time_interval(frame_id),axis=1)

    def frame_time_end(self, frame_id: ArrayLike):
        return self.frame_time_interval(frame_id)[:,1]

    def frame_time_start(self, frame_id: ArrayLike):
        return self.frame_time_interval(frame_id)[:,0]


class LiquidChromatography(Chromatography):
    def __init__(self, name: str = "LiquidChromatographyDevice"):
        super().__init__(name)

    def run(self, sample: ProteomicsExperimentSampleSlice):
        # retention time apex simulation
        retention_time_apex = self._apex_model.simulate(sample, self)
        # in irt and rt
        sample.add_simulation("simulated_irt_apex", retention_time_apex)
        # in frame id
        sample.add_simulation("simulated_frame_apex", self.irt_to_frame_id(retention_time_apex))

        # profile simulation

        retention_profile = self._profile_model.simulate(sample, self)
        sample.add_simulation("simulated_frame_profile", retention_profile)

    def rt_to_frame_id(self,  rt_minutes: ArrayLike):
        rt_minutes = np.asarray(rt_minutes)
        # first frame is completed not at 0 but at frame_length
        frame_id = (rt_minutes/self.frame_length*1000*60).astype(np.int64)+1
        return frame_id


class ChromatographyApexModel(Model):
    def __init__(self):
        self._device = None

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: Chromatography) -> NDArray[np.float64]:
        pass


class ChromatographyProfileModel(Model):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: Chromatography) -> List[RTProfile]:
        pass


class EMGChromatographyProfileModel(ChromatographyProfileModel):

    def __init__(self):
        super().__init__()

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: Chromatography) -> List[RTProfile]:
        mus = sample.peptides["simulated_irt_apex"].values
        profile_list = []

        for mu in mus:
            σ = 0.1 # must be sampled
            λ = 10 # must be sampled
            K = 1/(σ*λ)
            μ = device.irt_to_rt(mu)
            model_params = { "sigma":σ,
                             "lambda":λ,
                             "mu":μ,
                             "name":"EMG"
                            }

            emg = exponnorm(loc=μ, scale=σ, K = K)
            # start and end value (in retention times)
            s_rt, e_rt = emg.ppf([0.01,0.9])
            # as frames
            s_frame, e_frame = device.rt_to_frame_id(s_rt), device.rt_to_frame_id(e_rt)

            profile_frames = np.arange(s_frame-1,e_frame+1) # starting with s_frame-1 for cdf interval calculation
            profile_rt_ends = device.frame_time_end(profile_frames)
            profile_rt_cdfs = emg.cdf(profile_rt_ends)
            profile_rel_intensities = np.diff(profile_rt_cdfs)

            profile_list.append(RTProfile(profile_frames[1:],profile_rel_intensities,model_params))
        return profile_list


class NormalChromatographyProfileModel(ChromatographyProfileModel):

    def __init__(self):
        super().__init__()

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: Chromatography) -> List[RTProfile]:
        mus = sample.peptides["simulated_irt_apex"].values
        sigmas = gamma(a=4.929,scale=1/18.784).rvs(mus.size)/6
        profile_list = []

        for mu,σ in zip(mus,sigmas):

            μ = device.irt_to_rt(mu)
            model_params = { "sigma":σ,
                             "mu":μ,
                             "name":"Normal"
                            }

            normal = norm(loc=μ, scale=σ)
            # start and end value (in retention times)
            s_rt, e_rt = normal.ppf([0.02,0.98])
            # as frames
            s_frame, e_frame = device.rt_to_frame_id(s_rt), device.rt_to_frame_id(e_rt)

            profile_frames = np.arange(s_frame-1,e_frame+1) # starting with s_frame-1 for cdf interval calculation
            profile_rt_ends = device.frame_time_end(profile_frames)
            profile_rt_cdfs = normal.cdf(profile_rt_ends)
            profile_rel_intensities = np.diff(profile_rt_cdfs)

            profile_list.append(RTProfile(profile_frames[1:],profile_rel_intensities,model_params))
        return profile_list


class NeuralChromatographyApex(ChromatographyApexModel):

    def __init__(self, model_path: str, tokenizer_path: str):
        super().__init__()
        self.model_path = model_path
        self.tokenizer = tokenizer_from_json(tokenizer_path)

    def sequences_to_tokens(self, sequences_tokenized: np.array) -> np.array:
        print('tokenizing sequences...')
        tokens = self.tokenizer.texts_to_sequences(sequences_tokenized)
        tokens_padded = tf.keras.preprocessing.sequence.pad_sequences(tokens, 50, padding='post')
        return tokens_padded

    @staticmethod
    def _worker(model_path: str, tokens_padded: np.array, batched: bool = True, bs: int = 2048):
        pseudo_target = np.expand_dims(np.zeros_like(tokens_padded[:, 0]), axis=1)
        if batched:
            dataset = tf.data.Dataset.from_tensor_slices((tokens_padded, pseudo_target)).batch(bs)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((tokens_padded, pseudo_target))
        model = tf.keras.models.load_model(model_path)
        print('predicting irts...')
        return_val = model.predict(dataset)
        return return_val

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: Chromatography) ->  NDArray[np.float64]:

        data = sample.peptides
        tokens = data["sequence_tokenized"].apply(lambda st: st.sequence_tokenized).to_numpy()
        print('generating tf dataset...')
        tokens_padded = self.sequences_to_tokens(tokens)

        with Pool(1) as pool:
            r = pool.starmap(self._worker, [(self.model_path, tokens_padded)])
        return r[0]


class IonSource(Device):
    def __init__(self, name:str ="IonizationDevice"):
        super().__init__(name)
        self._ionization_model = None

    @property
    def ionization_model(self):
        return self._ionization_model

    @ionization_model.setter
    def ionization_model(self, model: IonizationModel):
        self._ionization_model = model

    @abstractmethod
    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass

class ElectroSpray(IonSource):
    def __init__(self, name:str ="ElectrosprayDevice"):
        super().__init__(name)

    def run(self, sample: ProteomicsExperimentSampleSlice):
        charge_profiles = self.ionization_model.simulate(sample, self)
        sample.add_simulation("simulated_charge_profile", charge_profiles)

class IonizationModel(Model):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, sample:ProteomicsExperimentSampleSlice, device: IonSource) -> NDArray:
        pass

class RandomIonSource(IonizationModel):
    def __init__(self):
        super().__init__()
        self._charge_probabilities =  np.array([0.1, 0.5, 0.3, 0.1])
        self._allowed_charges =  np.array([1, 2, 3, 4], dtype=np.int8)

    @property
    def allowed_charges(self):
        return self._allowed_charges

    @allowed_charges.setter
    def allowed_charges(self, charges: ArrayLike):
        self._allowed_charges = np.asarray(charges, dtype=np.int8)

    @property
    def charge_probabilities(self):
        return self._charge_probabilities

    @charge_probabilities.setter
    def charge_probabilities(self, probabilities: ArrayLike):
        self._charge_probabilities = np.asarray(probabilities)

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: IonSource) -> List[ChargeProfile]:
        if self.charge_probabilities.shape != self.allowed_charges.shape:
            raise ValueError("Number of allowed charges must fit to number of probabilities")

        data = sample.peptides
        charge = np.random.choice(self.allowed_charges, data.shape[0], p=self.charge_probabilities)
        rel_intensity = np.ones_like(charge)
        charge_profiles = []
        for c,i in zip(charge, rel_intensity):
            charge_profiles.append(ChargeProfile([c],[i],model_params={"name":"RandomIonSource"}))
        return charge_profiles

class BinomialIonSource():
    def __init__(self):
        super().__init__()
        self._charged_probability =  0.5
        self._allowed_charges =  np.array([1, 2, 3, 4], dtype=np.int8)

    @property
    def allowed_charges(self):
        return self._allowed_charges

    @allowed_charges.setter
    def allowed_charges(self, charges: ArrayLike):
        self._allowed_charges = np.asarray(charges, dtype=np.int8)

    @property
    def charged_probability(self):
        return self._charged_probability

    @charged_probability.setter
    def charged_probability(self, probability: float):
        self._charged_probability = probability

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: IonSource) -> List[ChargeProfile]:
        sequences = sample.peptides.sequence
        vec_get_num_protonizable_sites = np.vectorize(get_num_protonizable_sites)
        basic_aa_nums = vec_get_num_protonizable_sites(sequences)

        charge_profiles = []
        for baa_num in basic_aa_nums:
            rel_intensities = binom(baa_num,self.charged_probability).pmf(self.allowed_charges)
            charge_profiles.append(ChargeProfile(self.allowed_charges,rel_intensities,model_params={"name":"BinomialIonSource","basic_aa_num":baa_num}))
        return charge_profiles

class IonMobilitySeparation(Device):
    def __init__(self, name:str = "IonMobilityDevice"):
        super().__init__(name)
        # models
        self._apex_model = None
        self._profile_model = None

        # hardware parameter
        self._scan_intervall = 1
        self._scan_time = None
        self._scan_id_min = None
        self._scan_id_max = None
        self._buffer_gas = None

        # converters
        self._reduced_im_to_scan_converter = None
        self._scan_to_reduced_im_interval_converter = None

    @property
    def reduced_im_to_scan_converter(self):
        return self._reduced_im_to_scan_converter

    @reduced_im_to_scan_converter.setter
    def reduced_im_to_scan_converter(self, converter:callable):
        self._reduced_im_to_scan_converter = converter

    @property
    def scan_to_reduced_im_interval_converter(self):
        return self._scan_to_reduced_im_interval_converter

    @scan_to_reduced_im_interval_converter.setter
    def scan_to_reduced_im_interval_converter(self, converter:callable):
        self._scan_to_reduced_im_interval_converter = converter

    @property
    def buffer_gas(self):
        return self._buffer_gas

    @buffer_gas.setter
    def buffer_gas(self, gas: BufferGas):
        self._buffer_gas = gas

    @property
    def scan_intervall(self):
        return self._scan_intervall

    @scan_intervall.setter
    def scan_intervall(self, number:int):
        self._scan_intervall = number

    @property
    def scan_id_min(self):
        return self._scan_id_min

    @scan_id_min.setter
    def scan_id_min(self, number:int):
        self._scan_id_min = number

    @property
    def scan_id_max(self):
        return self._scan_id_max

    @scan_id_max.setter
    def scan_id_max(self, number:int):
        self._scan_id_max = number

    @property
    def scan_time(self):
        return self._scan_time

    @scan_time.setter
    def scan_time(self, microseconds:int):
        self._scan_time = microseconds

    @property
    def apex_model(self):
        return self._apex_model

    @apex_model.setter
    def apex_model(self, model: IonMobilityApexModel):
        self._apex_model = model

    @property
    def profile_model(self):
        return self._profile_model

    @profile_model.setter
    def profile_model(self, model: IonMobilityProfileModel):
        self._profile_model = model

    def scan_to_reduced_im_interval(self, scan_id: ArrayLike):
        return self.scan_to_reduced_im_interval_converter(scan_id)

    def reduced_im_to_scan(self, ion_mobility):
        return self.reduced_im_to_scan_converter(ion_mobility)

    def scan_im_middle(self, scan_id: ArrayLike):
        return np.mean(self.scan_to_reduced_im_interval(scan_id), axis = 1)

    def scan_im_lower(self, scan_id: ArrayLike):
        return self.scan_to_reduced_im_interval(scan_id)[:,0]

    def scan_im_upper(self, scan_id:ArrayLike):
        return self.scan_to_reduced_im_interval(scan_id)[:,1]

    def im_to_reduced_im(self, ion_mobility: float, p_0: float = STANDARD_PRESSURE, T_0: float = STANDARD_TEMPERATURE):
        """
        Calculate reduced ion mobility K_0
        (normalized to standard pressure p_0
        and standard temperature T_0), from
        ion mobility K.

        K_0 = K * p/p_0 * T_0/T

        [1] J. N. Dodds and E. S. Baker,
        “Ion mobility spectrometry: fundamental concepts,
        instrumentation, applications, and the road ahead,”
        Journal of the American Society for Mass Spectrometry,
        vol. 30, no. 11, pp. 2185–2195, 2019,
        doi: 10.1007/s13361-019-02288-2.

        :param ion_mobility: Ion mobility K to
            normalize to standard conditions
        :param p_0: Standard pressure (Pa).
        :param T_0: Standard temperature (K).
        """
        T = self.temperature
        p = self.pressure
        return ion_mobility*p/p_0*T_0/T

    def reduced_im_to_im(self, reduced_ion_mobility: float, p_0: float = STANDARD_PRESSURE, T_0: float = STANDARD_TEMPERATURE):
        """
        Inverse of `.im_to_reduced_im()`
        """
        T = self.temperature
        p = self.pressure
        return reduced_ion_mobility*p_0/p*T/T_0

    def ccs_to_reduced_im(self, ccs:float, mz:float, charge:int):
        # TODO Citation
        """
        Conversion of collision cross-section values (ccs)
        to reduced ion mobility according to
        Mason-Schamp equation.

        :param ccs: collision cross-section (ccs)
        :type ccs: float
        :param mz: Mass (Da) to charge ratio of peptide
        :type mz: float
        :param charge: Charge of peptide
        :type charge: int
        :return: Reduced ion mobility
        :rtype: float
        """

        T = self.temperature
        mass = mz*charge
        μ = self.buffer_gas.mass*mass/(self.buffer_gas.mass+mass)
        z = charge

        K0 = CCS_K0_CONVERSION_CONSTANT*z*1/(np.sqrt(μ*T)*ccs)
        return K0

    def reduced_im_to_ccs(self, reduced_ion_mobility:float, mz:float, charge:int):
        """
        Conversion of reduced ion mobility
        to collision cross-section values (ccs)
        according to Mason-Schamp equation.

        :param reduced_ion_mobility: reduced ion mobility K0
        :type reduced_ion_mobility: float
        :param mz: Mass (Da) to charge ratio of peptide
        :type mz: float
        :param charge: Charge of peptide
        :type charge: int
        :return: Collision cross-section (ccs)
        :rtype: float
        """

        T = self.temperature
        mass = mz*charge
        μ = self.buffer_gas.mass*mass/(self.buffer_gas.mass+mass)
        z = charge
        K0 = reduced_ion_mobility

        ccs = CCS_K0_CONVERSION_CONSTANT*z*1/(np.sqrt(μ*T)*K0)
        return ccs

    @abstractmethod
    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass



class TrappedIon(IonMobilitySeparation):

    def __init__(self, name:str = "TrappedIonMobilitySeparation"):
        super().__init__()
        self._scan_id_min = 1
        self._scan_id_max = 918

    def run(self, sample: ProteomicsExperimentSampleSlice):
        # scan apex simulation
        one_over_k0, scan_apex = self._apex_model.simulate(sample, self)
        # in irt and rt
        sample.add_simulation("simulated_scan_apex", scan_apex)
        sample.add_simulation("simulated_k0", one_over_k0)
        # scan profile simulation
        scan_profile = self._profile_model.simulate(sample, self)
        sample.add_simulation("simulated_scan_profile", scan_profile)


class IonMobilityApexModel(Model):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: IonMobilitySeparation) -> NDArray[np.float64]:
        return super().simulate(sample, device)

class IonMobilityProfileModel(Model):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: IonMobilitySeparation) -> NDArray:
        return super().simulate(sample, device)

class NeuralIonMobilityApex(IonMobilityApexModel):

    def __init__(self, model_path: str, tokenizer_path: str):
        super().__init__()
        self.model_path = model_path
        self.tokenizer = tokenizer_from_json(tokenizer_path)

    def sequences_to_tokens(self, sequences_tokenized: np.array) -> np.array:
        print('tokenizing sequences...')
        tokens = self.tokenizer.texts_to_sequences(sequences_tokenized)
        tokens_padded = tf.keras.preprocessing.sequence.pad_sequences(tokens, 50, padding='post')
        return tokens_padded

    @staticmethod
    def _worker(model_path: str, tokens_padded: np.array, mz: np.array, charges: np.array, batched: bool = True, bs: int = 2048):

        mz = np.expand_dims(mz, 1)
        c = tf.one_hot(charges - 1, depth=4)
        pseudo_target = np.expand_dims(np.zeros_like(tokens_padded[:, 0]), axis=1)
        if batched:
            dataset = tf.data.Dataset.from_tensor_slices(((mz, c, tokens_padded), pseudo_target)).batch(bs)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(((mz, c, tokens_padded), pseudo_target))

        print('predicting mobilities...')
        model = tf.keras.models.load_model(model_path)
        ccs, _ = model.predict(dataset)
        return ccs

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: IonMobilitySeparation) -> Tuple[NDArray]:
        data = sample.ions.merge(sample.peptides.loc[:,["pep_id","sequence_tokenized"]],on="pep_id",validate="many_to_one")
        tokens = data.sequence_tokenized.apply(lambda st: st.sequence_tokenized).to_numpy()
        tokens_padded = self.sequences_to_tokens(tokens)
        mz = data['mz'].values
        charges = data["charge"].values
        with Pool(1) as pool:
            r = pool.starmap(self._worker, [(self.model_path, tokens_padded, mz, charges)])
            ccs = r[0]
        K0s = device.ccs_to_reduced_im(np.squeeze(ccs), mz, data['charge'].values)
        scans = device.reduced_im_to_scan(K0s)
        return K0s,scans

class NormalIonMobilityProfileModel(IonMobilityProfileModel):
    def __init__(self):
        super().__init__()

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: IonMobilitySeparation) -> List[ScanProfile]:
        mus = 1/sample.ions["simulated_k0"].values #1 over k0
        sigmas = gamma(a=3.703,scale=1/79.614).rvs(mus.size)/6
        profile_list = []
        for μ,σ in zip(mus,sigmas):

            model_params = { "sigma":σ,
                             "mu":μ,
                             "name":"NORMAL"
                            }

            normal = norm(loc=μ, scale=σ)
            # start and end value (k0)
            s_one_over_k0, e_one_over_k0 = normal.ppf([0.01,0.99])
            # as scan ids, remember first scans elutes largest ions
            e_scan, s_scan = device.reduced_im_to_scan(1/s_one_over_k0), device.reduced_im_to_scan(1/e_one_over_k0)

            profile_scans = np.arange(s_scan-1,e_scan+1) # starting s_scan-1 is necessary here to include its end value for cdf interval
            profile_end_im = 1/device.scan_im_upper(profile_scans)
            profile_end_cdfs = 1-normal.cdf(profile_end_im)
            profile_rel_intensities = np.diff(profile_end_cdfs)

            profile_list.append(ScanProfile(profile_scans[1:],profile_rel_intensities,model_params))
        return profile_list


class MzSeparation(Device):
    def __init__(self, name:str = "MassSpectrometer"):
        super().__init__(name)
        self._model = None
        self._resolution = 3

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, res: int):
        self._resolution = res

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: MzSeparationModel):
        self._model = model

    @abstractmethod
    def run(self, sample: ProteomicsExperimentSampleSlice):
        pass

class TOF(MzSeparation):
    def __init__(self, name:str = "TimeOfFlightMassSpectrometer"):
        super().__init__(name)

    def run(self, sample: ProteomicsExperimentSampleSlice):
        spectra = self.model.simulate(sample, self)
        sample.add_simulation("simulated_mz_spectrum", spectra)

class MzSeparationModel(Model):
    def __init__(self):
        self._default_abundance = 1e4

    @property
    def default_abundance(self):
        return self._default_abundance

    @default_abundance.setter
    def default_abundance(self, abundance:int):
        self._default_abundance = abundance

    @abstractmethod
    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: MzSeparation):
        return super().simulate(sample, device)

class AveragineModel(MzSeparationModel):
    def __init__(self):
        super().__init__()

    def simulate(self, sample: ProteomicsExperimentSampleSlice, device: MzSeparation):

        avg = AveragineGenerator()
        # right join here to keep order of keys in right df
        joined_df = pd.merge(sample.peptides[["pep_id","mass_theoretical"]],
                        sample.ions[["pep_id","charge"]],
                        how = "right",
                        on = "pep_id")
        masses = joined_df["mass_theoretical"].values
        charges = joined_df["charge"].values
        spectra = []
        for (m, c) in zip(masses, charges):
            spectrum = avg.generate_spectrum(m,c,amp = self.default_abundance, centroided=False)
            spectra.append(spectrum)
        return spectra
