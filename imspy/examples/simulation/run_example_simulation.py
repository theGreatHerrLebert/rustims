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
from imspy.simulation.proteome import ProteinSample, Trypsin, ORGANISM
from imspy.chemistry.mass import BufferGas

import pandas as pd
import numpy as np
from sagepy.core import EnzymeBuilder, SageSearchConfiguration, validate_mods, validate_var_mods, SAGE_KNOWN_MODS
from sagepy.core import Precursor, RawSpectrum, ProcessedSpectrum, SpectrumProcessor, Tolerance, Scorer, Representation


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


def build_sage_db(fasta_path):
    # configure a trypsin-like digestor of fasta files
    enzyme_builder = EnzymeBuilder(
        missed_cleavages=2,
        min_len=5,
        max_len=50,
        cleave_at='KR',
        restrict='P',
        c_terminal=True,
    )

    # generate static cysteine modification
    static_mods = {k: v for k, v in [SAGE_KNOWN_MODS.cysteine_static()]}

    # generate variable methionine modification
    variable_mods = {k: v for k, v in [SAGE_KNOWN_MODS.methionine_variable()]}

    # generate SAGE compatible mod representations
    static = validate_mods(static_mods)
    variab = validate_var_mods(variable_mods)

    with open(fasta_path, 'r') as infile:
        fasta = infile.read()

    # set-up a config for a sage-database
    sage_config = SageSearchConfiguration(
        fasta=fasta,
        static_mods=static,
        variable_mods=variab,
        enzyme_builder=enzyme_builder,
        generate_decoys=True,
        bucket_size=int(np.power(2, 14))
    )

    # generate the database for searching against
    indexed_db = sage_config.generate_indexed_database()

    return indexed_db


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

    tokenizer_path = '../../imspy/algorithm/pretrained/tokenizer-ptm.json'

    rt_model_weights = "../../imspy/algorithm/pretrained/DeepRetentionTimePredictor"
    t.lc_method.apex_model = NeuralChromatographyApex(rt_model_weights, tokenizer_path=tokenizer_path)

    t.lc_method.profile_model = NormalChromatographyProfileModel()
    t.lc_method.irt_to_rt_converter = irt_to_rt

    im_model_weights = "../../imspy/algorithm/pretrained/DeepCCSPredictor"
    t.ion_mobility_separation_method.apex_model = NeuralIonMobilityApex(im_model_weights, tokenizer_path=tokenizer_path)

    t.ion_mobility_separation_method.profile_model = NormalIonMobilityProfileModel()
    t.ion_mobility_separation_method.buffer_gas = N2
    t.ion_mobility_separation_method.scan_to_reduced_im_interval_converter = scan_im_interval
    t.ion_mobility_separation_method.reduced_im_to_scan_converter = im_to_scan

    t.ionization_method.ionization_model = BinomialIonSource()

    t.mz_separation_method.model = AveragineModel()

    rng = np.random.default_rng(2023)
    # read proteome
    proteome = pd.read_feather('/home/administrator/Documents/promotion/EXPERIMENT/proteolizard-algorithm/notebook/resources/Homo-sapiens-proteome.feather')
    random_abundances = rng.integers(1e3, 1e7, size=proteome.shape[0])
    proteome = proteome.assign(abundancy=random_abundances)
    # create sample and sample digest; TODO: add missed cleavages to ENZYMEs
    sample = ProteinSample(proteome, ORGANISM.HOMO_SAPIENS)
    sample_digest = sample.digest(Trypsin())

    # to reduce computational load in example
    sample_digest.data = sample_digest.data.sample(100, random_state=rng)

    t.load_sample(sample_digest)
    return t


if __name__ == "__main__":
    path = '/home/administrator/Documents/promotion/ccs-workflow/rescoring/data/human_20365_conts_172_2020-3-2_2-0-11_validated.fasta'

    sage_db = build_sage_db(path)

    # t = build_experiment()

    #cProfile.run("t.run(10000)", filename="profiler_10000_8_process",sort="cumtime")
    # t.run(100, frames_per_assemble_process=10)