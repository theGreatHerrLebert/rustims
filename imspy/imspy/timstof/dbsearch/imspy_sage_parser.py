# start by importing logging and cmd argument parser
import logging
import argparse
import sys
import os

from imspy.algorithm import DeepPeptideIonMobilityApex, load_deep_ccs_predictor, load_tokenizer_from_resources
from imspy.algorithm.rt.predictors import DeepChromatographyApex, load_deep_retention_time_predictor
from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper
from imspy.chemistry.utility import calculate_mz
from imspy.timstof.dbsearch.utility import linear_map
from sagepy.core.scoring import prosit_intensities_to_fragments_par

from .sage_output_utility import *

# supress pandas warnings about column assignment
pd.options.mode.chained_assignment = None

def main():
    # argument parser
    parser = argparse.ArgumentParser(description='🦀💻 IMSPY - SAGE Parser DDA 🔬🐍 - PROTEOMICS IMS DDA '
                                                 'data re-scoring using imspy and sagepy.')

    # add the arguments
    parser.add_argument("sage_results", help="The path to the SAGE results file")
    parser.add_argument("sage_fragments", help="The path to the SAGE fragments file")
    parser.add_argument("output", help="The path to the output file")

    # add target decoy competition method
    parser.add_argument("--tdc_method", default="peptide_psm_peptide", help="The target decoy competition method, default is peptide_psm_peptide")
    # re-scoring parameters
    parser.add_argument("--num_splits", default=10, type=int, help="The number of splits for the target decoy competition cross-validation, default is 10")
    parser.add_argument("--no_balanced_split",
                        action="store_false", dest="balance",
                        help="Whether to balance the training dataset, sampling same amount of target and decoy examples, default is True")
    parser.set_defaults(balance=True)

    # parse the arguments
    args = parser.parse_args()

    # check if the SAGE results file exists
    if not os.path.exists(args.sage_results):
        logging.error(f"The SAGE results file {args.sage_results} does not exist.")
        sys.exit(1)

    # check if the SAGE fragments file exists
    if not os.path.exists(args.sage_fragments):
        logging.error(f"The SAGE fragments file {args.sage_fragments} does not exist.")
        sys.exit(1)

    # read the SAGE results file, check if .tsv or .parquet
    if args.sage_results.endswith(".tsv"):
        results = pd.read_csv(args.sage_results, sep="\t")
    elif args.sage_results.endswith(".parquet"):
        results = pd.read_parquet(args.sage_results)
    else:
        logging.error(f"Unknown file format for SAGE results file {args.sage_results}.")
        sys.exit(1)

    # read the SAGE fragments file, check if .tsv or .parquet
    if args.sage_fragments.endswith(".tsv"):
        fragments = pd.read_csv(args.sage_fragments, sep="\t")
    elif args.sage_fragments.endswith(".parquet"):
        fragments = pd.read_parquet(args.sage_fragments)
    else:
        logging.error(f"Unknown file format for SAGE fragments file {args.sage_fragments}.")
        sys.exit(1)

    # set up the logging
    logging.basicConfig(level=logging.INFO)

    # the intensity predictor model
    prosit_model = Prosit2023TimsTofWrapper(verbose=False)
    # the ion mobility predictor model
    im_predictor = DeepPeptideIonMobilityApex(load_deep_ccs_predictor(),
                                              load_tokenizer_from_resources("tokenizer-ptm"))
    # the retention time predictor model
    rt_predictor = DeepChromatographyApex(load_deep_retention_time_predictor(),
                                          load_tokenizer_from_resources("tokenizer-ptm"), verbose=True)

    # add sequence length to results table to filter out sequences that are too long for intensity prediction
    results["sequence_length"] = results.apply(lambda s: len(remove_substrings(s.peptide)), axis=1)
    results_filtered = results[results.sequence_length <= 30]

    # add decoy True or False column
    results_filtered["decoy"] = results_filtered.apply(lambda r: r.label == -1, axis=1)

    # create the token replacer
    token_replacer = PatternReplacer(replace_tokens)
    results_filtered["peptide"] = results_filtered.apply(lambda r: token_replacer.apply(r.peptide), axis=1)

    # calculate missing mz value for PSMs
    results_filtered["mz"] = results_filtered.apply(lambda r: calculate_mz(r.calcmass, r.charge), axis=1)
    results_filtered["rt_projected"] = results_filtered.apply(lambda r:
                                                              linear_map(r.rt, old_min=results_filtered.rt.min(),
                                                                         old_max=results_filtered.rt.max()), axis=1)

    if len(results_filtered) < len(results):
        s = len(results) - len(results_filtered)
        logging.info(f"Removed {s} sequences with sequence length > 30.")

    # create filtered set of ids
    S = set(results_filtered.psm_id)

    # filter fragments to remove sequences not in the results
    fragments = fragments[[f in S for f in fragments.psm_id.values]]

    # logg how many psms are going to be processed
    logging.info(f"Processing {len(results_filtered)} PSMs.")

    # group the fragments by PSM id, create a Fragments object for each PSM
    fragments_grouped = fragments.groupby("psm_id").agg({
        "fragment_type": list,
        "fragment_ordinals": list,
        "fragment_charge": list,
        "fragment_mz_calculated": list,
        "fragment_mz_experimental": list,
        "fragment_intensity": list,
    }).reset_index()

    fragments_grouped["fragments_observed"] = fragments_grouped.apply(lambda r: row_to_fragment(r), axis=1)

    # select only the psm_id and fragments_observed columns
    fragments_grouped = fragments_grouped[["psm_id", "fragments_observed"]]

    # log that intensity prediction is starting
    logging.info("Predicting intensities...")

    # use prosit to predict intensities
    intensity_pred = prosit_model.predict_intensities(
        results_filtered.peptide.values,
        results_filtered.charge.values,
        collision_energies=np.zeros_like(results_filtered.charge.values) + 30,
        batch_size=2048,
        flatten=True,
    )

    # log that ion mobility prediction is starting
    logging.info("Predicting peptide retention times...")

    # predict retention times
    rt_pred = rt_predictor.simulate_separation_times(
        sequences=results_filtered.peptide.values,
    )

    # log that ion mobility prediction is starting
    logging.info("Predicting ion mobilities...")

    # predict ion mobilities
    inv_mob = im_predictor.simulate_ion_mobilities(
        sequences=results_filtered.peptide.values,
        charges=results_filtered.charge.values,
        mz=results_filtered.mz.values,
    )

    results_filtered["inv_mob_predicted"] = inv_mob
    results_filtered["rt_predicted"] = rt_pred
    results_filtered["fragments_predicted"] = prosit_intensities_to_fragments_par(intensity_pred)

    # merge the results and fragments
    PSMS = pd.merge(results_filtered, fragments_grouped, on="psm_id")

    # features for re-scoring
    PSMS["observed_dict"] = PSMS.apply(lambda r: fragments_to_dict(r.fragments_observed), axis=1)
    PSMS["predicted_dict"] = PSMS.apply(lambda r: fragments_to_dict(r.fragments_predicted), axis=1)
    PSMS["cosine_similarity"] = PSMS.apply(lambda s: cosim_from_dict(s.observed_dict, s.predicted_dict), axis=1)
    PSMS["delta_rt"] = PSMS.rt_projected - PSMS.rt_predicted
    PSMS["delta_ims"] = PSMS.ion_mobility - PSMS.inv_mob_predicted
    PSMS["intensity_ms1"] = 0.0
    PSMS["collision_energy"] = 0.0
    PSMS = PSMS.rename(columns={"ms2_intensity": "intensity_ms2",
                                "fragment_ppm": "average_ppm", "precursor_ppm": "delta_mass"})

    # IDs for TDC
    PSMS["match_idx"] = PSMS.peptide
    PSMS["spec_idx"] = [str(x) for x in PSMS.psm_id]
    PSMS["score"] = PSMS.hyperscore
    PSMS["q_value"] = None

    # log that re-scoring is starting
    logging.info("Re-scoring PSMs...")

    # run the target decoy competition
    PSMS["re_score"] = re_score_psms(PSMS, num_splits=args.num_splits, balance=args.balance)

    TDC = target_decoy_competition_pandas(PSMS, method=args.tdc_method, score="hyperscore")
    TDC_rescore = target_decoy_competition_pandas(PSMS, method=args.tdc_method, score="re_score")

    before, after = len(TDC[TDC.q_value <= 0.01]), len(TDC_rescore[TDC_rescore.q_value <= 0.01])
    logging.info(f"Before re-scoring: {before} PSMs with q-value <= 0.01")
    logging.info(f"After re-scoring: {after} PSMs with q-value <= 0.01")

    # create output path for both files
    output_path = os.path.dirname(args.output)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_name = os.path.join(output_path, "imspy_sage_hyperscore.tsv")
    file_name_rescore = os.path.join(output_path, "imspy_sage_rescore.tsv")

    # save the results
    TDC.to_csv(file_name, sep="\t", index=False)
    TDC_rescore.to_csv(file_name_rescore, sep="\t", index=False)

    # log the output file
    logging.info(f"Output file {file_name} saved.")
    logging.info(f"Output file {file_name_rescore} saved.")
