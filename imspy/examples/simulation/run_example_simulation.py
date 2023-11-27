from imspy.simulation.experiment import LcImsMsMs
from imspy.simulation.hardware_models import (NeuralChromatographyApex, 
                                              NormalChromatographyProfileModel, 
                                              LiquidChromatography, 
                                              ElectroSpray,
                                              TrappedIon,
                                              TOF,
                                              NeuralIonMobilityApex, 
                                              NormalIonMobilityProfileModel,
                                              AveragineModel,
                                              BinomialIonSource
                                              )
from imspy.proteome import ProteinSample, Trypsin, ORGANISM
from imspy.chemistry.mass import BufferGas

import pandas as pd
import numpy as np


def irt_to_rt(irt):
    return irt


def scan_im_interval(scan_id):
    intercept = 1451.357
    slope = -877.361
    scan_id = np.atleast_1d(scan_id)
    lower = ( scan_id    - intercept ) / slope
    upper = ((scan_id+1) - intercept ) / slope
    return np.stack([1/lower, 1/upper], axis=1)


def im_to_scan(reduced_ion_mobility):
    intercept = 1451.357
    slope = -877.361
    # TODO more appropriate function here ?
    one_over_k0 = 1/reduced_ion_mobility
    return np.round(one_over_k0 * slope + intercept).astype(np.int16)


def build_experiment():
    t = LcImsMsMs("./timstofexp1_binomial_ion_source_21_7/") # maybe rather call this class LCIMSMSExperiment

    lc = LiquidChromatography()
    lc.frame_length = 1200 #ms
    lc.gradient_length = 120 # min
    esi = ElectroSpray()
    tims = TrappedIon()
    tims.scan_time = 110 # us
    tof_mz = TOF()

    t.lc_method = lc
    t.ionization_method = esi
    t.ion_mobility_separation_method = tims
    t.mz_separation_method = tof_mz

    N2 = BufferGas("N2")

    tokenizer_path = '/home/tim/Workspaces/ionmob/pretrained-models/tokenizers/tokenizer.json'

    rt_model_weights = "/home/tim/Workspaces/Resources/models/DeepChromatograpy/"
    t.lc_method.apex_model = NeuralChromatographyApex(rt_model_weights,tokenizer_path = tokenizer_path)

    t.lc_method.profile_model = NormalChromatographyProfileModel()
    t.lc_method.irt_to_rt_converter = irt_to_rt

    im_model_weights = "/home/tim/Workspaces/ionmob/pretrained-models/GRUPredictor"
    t.ion_mobility_separation_method.apex_model = NeuralIonMobilityApex(im_model_weights, tokenizer_path = tokenizer_path)

    t.ion_mobility_separation_method.profile_model = NormalIonMobilityProfileModel()
    t.ion_mobility_separation_method.buffer_gas = N2
    t.ion_mobility_separation_method.scan_to_reduced_im_interval_converter = scan_im_interval
    t.ion_mobility_separation_method.reduced_im_to_scan_converter = im_to_scan

    t.ionization_method.ionization_model = BinomialIonSource()

    t.mz_separation_method.model = AveragineModel()

    rng = np.random.default_rng(2023)
    # read proteome
    proteome = pd.read_feather('/home/tim/Workspaces/Resources/Homo-sapiens-proteome.feather')
    random_abundances = rng.integers(1e3,1e7,size=proteome.shape[0])
    proteome = proteome.assign(abundancy= random_abundances)
    # create sample and sample digest; TODO: add missed cleavages to ENZYMEs
    sample = ProteinSample(proteome, ORGANISM.HOMO_SAPIENS)
    sample_digest = sample.digest(Trypsin())

    # to reduce computational load in example
    sample_digest.data = sample_digest.data.sample(100, random_state= rng)


    t.load_sample(sample_digest)
    return t


if __name__ == "__main__":

    t = build_experiment()

    #cProfile.run("t.run(10000)", filename="profiler_10000_8_process",sort="cumtime")
    t.run(100, frames_per_assemble_process=10)