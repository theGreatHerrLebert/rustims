import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sagepy.rescore.rescore import rescore_psms
from sagepy.qfdr.tdc import assign_sage_spectrum_q, assign_sage_peptide_q, assign_sage_protein_q

from sagepy.core.scoring import Psm
from sagepy.utility import psm_collection_to_pandas
from typing import List
from imspy.algorithm.utility import load_tokenizer_from_resources
from imspy.algorithm.ccs.predictors import load_deep_ccs_predictor, DeepPeptideIonMobilityApex
from imspy.algorithm.rt.predictors import load_deep_retention_time_predictor, DeepChromatographyApex

from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper, get_collision_energy_calibration_factor
from imspy.algorithm.intensity.predictors import associate_fragment_ions_with_prosit_predicted_intensities
from imspy.algorithm.rt.predictors import linear_map

from imspy.timstof.dbsearch.utility import generate_balanced_rt_dataset
from imspy.timstof.dbsearch.utility import generate_balanced_im_dataset


def re_score_psms(psms: List[Psm], verbose: bool = True, use_logreg: bool = True) -> List[Psm]:
    """ Re-score the PSMs

    Args:
        psms: The PSMs
        verbose: Whether to print information
        use_logreg: Whether to use logistic regression

    Returns:
        The re-scored PSMs
    """
    if use_logreg:
        model = LogisticRegression()
    else:
        model = SVC(probability=True)

    psms_rescored = rescore_psms(
        psm_collection=psms,
        model=model,
        num_splits=3,
        verbose=True
    )

    psms_rescored = list(filter(lambda x: x.rank == 1, psms))

    assign_sage_spectrum_q(psms_rescored, use_hyper_score=False)
    assign_sage_peptide_q(psms_rescored, use_hyper_score=False)
    assign_sage_protein_q(psms_rescored, use_hyper_score=False)

    return psms_rescored


def create_feature_space(
        psms: List[Psm],
        rt_min: float,
        rt_max: float,
        fine_tune_im: bool = True,
        fine_tune_rt: bool = True,
        verbose: bool = False) -> List[Psm]:
    """ Create a feature space for the PSMs

    Args:
        psms: The PSMs
        rt_min: The minimum retention time
        rt_max: The maximum retention time
        fine_tune_im: Whether to fine-tune the ion mobility predictor
        fine_tune_rt: Whether to fine-tune the retention time predictor
        verbose: Whether to print information

    Returns:
        The PSMs with the feature space
    """


    # take the top-n scoring PSMs to calibrate collision energy
    sample = sorted(psms, key=lambda s: s.hyperscore)[-2 ** 8:]

    # load prosit model
    prosit_model = Prosit2023TimsTofWrapper(verbose=verbose)

    # load ionmob model
    im_predictor = DeepPeptideIonMobilityApex(load_deep_ccs_predictor(),
                                              load_tokenizer_from_resources("tokenizer-ptm"))

    # load rt predictor
    rt_predictor = DeepChromatographyApex(load_deep_retention_time_predictor(),
                                          load_tokenizer_from_resources("tokenizer-ptm"), verbose=verbose)

    # calculate the calibration factor
    collision_energy_calibration_factor, angles = get_collision_energy_calibration_factor(
        sample,
        prosit_model,
        verbose=verbose
    )

    # add the calibration factor to the PSMs
    for p in psms:
        p.collision_energy_calibrated = p.collision_energy + collision_energy_calibration_factor

    # predict the intensity values,
    I = prosit_model.predict_intensities(
        [p.sequence_modified if p.decoy == False else p.sequence_decoy_modified for p in psms],
        np.array([p.charge for p in psms]),
        [p.collision_energy_calibrated for p in psms],
        batch_size=2048,
        flatten=True,
    )

    # add intensity values to PSMs
    psms = associate_fragment_ions_with_prosit_predicted_intensities(psms, I, num_threads=16)

    if fine_tune_im:
        # fit ion mobility predictor
        im_predictor.fine_tune_model(
            data=psm_collection_to_pandas(generate_balanced_im_dataset(psms=psms)),
            batch_size=1024,
            re_compile=True,
            verbose=verbose,
        )

    # predict ion mobilities
    inv_mob = im_predictor.simulate_ion_mobilities(
        sequences=[x.sequence_modified if x.decoy == False else x.sequence_decoy_modified for x in psms],
        charges=[x.charge for x in psms],
        mz=[x.mono_mz_calculated for x in psms]
    )

    # set ion mobilities
    for mob, p in zip(inv_mob, psms):
        p.inverse_ion_mobility_predicted = mob

    # map the observed retention time into the domain [0, 60]
    for value in psms:
        value.retention_time_projected = linear_map(value.retention_time, rt_min, rt_max, 0.0, 60.0)

    if fine_tune_rt:
        # fit retention time predictor
        rt_predictor.fine_tune_model(
            data=psm_collection_to_pandas(generate_balanced_rt_dataset(psms=psms)),
            batch_size=1024,
            re_compile=True,
            verbose=verbose,
        )

    # predict retention times
    rt_pred = rt_predictor.simulate_separation_times(
        sequences=[x.sequence_modified if x.decoy == False else x.sequence_decoy_modified for x in psms],
    )

    # set retention times
    for rt, p in zip(rt_pred, psms):
        p.retention_time_predicted = rt

    return psms
