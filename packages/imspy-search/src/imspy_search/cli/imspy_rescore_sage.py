"""IMSPY SAGE Rescore CLI - Re-score SAGE search results using deep learning features."""

import logging
import argparse
import sys
import os

import pandas as pd
import numpy as np

from imspy_predictors import (
    DeepPeptideIonMobilityApex, load_deep_ccs_predictor,
    DeepChromatographyApex, load_deep_retention_time_predictor,
    Prosit2023TimsTofWrapper, load_tokenizer_from_resources,
)
from imspy_core.chemistry.utility import calculate_mz
from imspy_search.utility import linear_map
from sagepy.core.scoring import prosit_intensities_to_fragments_par
from sagepy.qfdr.tdc import target_decoy_competition_pandas

from imspy_search.sage_output_utility import (
    re_score_psms, row_to_fragment, remove_substrings,
    PatternReplacer, replace_tokens, cosim_from_dict,
    fragments_to_dict, plot_summary
)

# Suppress pandas warnings about column assignment
pd.options.mode.chained_assignment = None


def main():
    """Main entry point for imspy-rescore-sage CLI."""
    parser = argparse.ArgumentParser(
        description='IMSPY - SAGE Parser DDA - Re-score SAGE search results using imspy and sagepy.'
    )

    parser.add_argument("sage_results", help="The path to the SAGE results file")
    parser.add_argument("sage_fragments", help="The path to the SAGE fragments file")
    parser.add_argument("output", help="The path to where the output files should be created")

    parser.add_argument(
        "--tdc_method",
        default="peptide_psm_peptide",
        help="The target decoy competition method",
        choices=["psm", "peptide_psm_only", "peptide_peptide_only", "peptide_psm_peptide"]
    )

    parser.add_argument(
        "--num_splits",
        default=5,
        type=int,
        help="The number of splits for cross-validation"
    )

    parser.add_argument(
        "--no_balanced_split",
        action="store_false",
        dest="balance",
        help="Do not balance the training dataset"
    )
    parser.set_defaults(balance=True)

    parser.add_argument(
        "--no_store_hyperscore",
        action="store_false",
        dest="store_hyperscore",
        help="Do not store the results with the hyperscore"
    )
    parser.set_defaults(store_hyperscore=True)

    parser.add_argument(
        "--fine_tune_predictors",
        action="store_true",
        help="Fine tune the rt and inv-mob predictors"
    )
    parser.set_defaults(fine_tune_predictors=False)

    parser.add_argument(
        "--positive_example_q_max",
        default=0.01,
        type=float,
        help="Maximum q-value allowed for positive examples"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "--no_summary_plot",
        action="store_false",
        dest="summary_plot",
        help="Do not create a summary plot"
    )
    parser.set_defaults(summary_plot=True)

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.sage_results):
        logging.error(f"The SAGE results file {args.sage_results} does not exist.")
        sys.exit(1)

    if not os.path.exists(args.sage_fragments):
        logging.error(f"The SAGE fragments file {args.sage_fragments} does not exist.")
        sys.exit(1)

    # Read SAGE results
    if args.sage_results.endswith(".tsv"):
        results = pd.read_csv(args.sage_results, sep="\t")
    elif args.sage_results.endswith(".parquet"):
        results = pd.read_parquet(args.sage_results)
    else:
        logging.error(f"Unknown file format for SAGE results file {args.sage_results}.")
        sys.exit(1)

    # Read SAGE fragments
    if args.sage_fragments.endswith(".tsv"):
        fragments = pd.read_csv(args.sage_fragments, sep="\t")
    elif args.sage_fragments.endswith(".parquet"):
        fragments = pd.read_parquet(args.sage_fragments)
    else:
        logging.error(f"Unknown file format for SAGE fragments file {args.sage_fragments}.")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    # Load models
    prosit_model = Prosit2023TimsTofWrapper(verbose=False)
    im_predictor = DeepPeptideIonMobilityApex(
        load_deep_ccs_predictor(),
        load_tokenizer_from_resources("tokenizer-ptm")
    )
    rt_predictor = DeepChromatographyApex(
        load_deep_retention_time_predictor(),
        load_tokenizer_from_resources("tokenizer-ptm"),
        verbose=True
    )

    # Filter sequences by length
    results["sequence_length"] = results.apply(lambda s: len(remove_substrings(s.peptide)), axis=1)
    results_filtered = results[results.sequence_length <= 30]

    results_filtered["decoy"] = results_filtered.apply(lambda r: r.label == -1, axis=1)

    token_replacer = PatternReplacer(replace_tokens)
    results_filtered["sequence"] = results_filtered.apply(lambda r: token_replacer.apply(r.peptide), axis=1)

    results_filtered["mono_mz_calculated"] = results_filtered.apply(
        lambda r: calculate_mz(r.calcmass, r.charge), axis=1
    )
    results_filtered["inverse_mobility_observed"] = results.ion_mobility
    results_filtered["projected_rt"] = results_filtered.apply(
        lambda r: linear_map(r.rt, old_min=results_filtered.rt.min(), old_max=results_filtered.rt.max()),
        axis=1
    )

    results_filtered["match_idx"] = results_filtered.sequence
    results_filtered["spec_idx"] = [str(x) for x in results_filtered.psm_id]
    results_filtered["score"] = results_filtered.hyperscore
    results_filtered["q_value"] = None

    if len(results_filtered) < len(results):
        s = len(results) - len(results_filtered)
        logging.info(f"Removed {s} sequences with sequence length > 30.")

    S = set(results_filtered.psm_id)

    # Fine-tuning data
    if args.fine_tune_predictors:
        TDC_train = target_decoy_competition_pandas(results_filtered, method="psm")
        TDC_train_f = TDC_train[(TDC_train.decoy == False) & (TDC_train.q_value <= 0.01)]
        TDC_train_f["spec_idxi"] = [int(x) for x in TDC_train_f.spec_idx]
        FT_data = pd.merge(TDC_train_f, results_filtered, left_on=["spec_idxi"], right_on=["psm_id"])

    fragments = fragments[[f in S for f in fragments.psm_id.values]]

    logging.info(f"Processing {len(results_filtered)} PSMs.")

    # Group fragments by PSM
    fragments_grouped = fragments.groupby("psm_id").agg({
        "fragment_type": list,
        "fragment_ordinals": list,
        "fragment_charge": list,
        "fragment_mz_calculated": list,
        "fragment_mz_experimental": list,
        "fragment_intensity": list,
    }).reset_index()

    fragments_grouped["fragments_observed"] = fragments_grouped.apply(lambda r: row_to_fragment(r), axis=1)
    fragments_grouped = fragments_grouped[["psm_id", "fragments_observed"]]

    logging.info("Predicting intensities...")

    intensity_pred = prosit_model.predict_intensities(
        results_filtered.sequence.values,
        results_filtered.charge.values,
        collision_energies=np.zeros_like(results_filtered.charge.values) + 30,
        batch_size=2048,
        flatten=True,
    )

    logging.info("Predicting peptide retention times...")

    if args.fine_tune_predictors:
        rt_predictor.fine_tune_model(data=FT_data, verbose=args.verbose)

    rt_pred = rt_predictor.simulate_separation_times(sequences=results_filtered.sequence.values)

    logging.info("Predicting ion mobilities...")

    if args.fine_tune_predictors:
        im_predictor.fine_tune_model(data=FT_data, verbose=args.verbose)

    inv_mob = im_predictor.simulate_ion_mobilities(
        sequences=results_filtered.sequence.values,
        charges=results_filtered.charge.values,
        mz=results_filtered.mono_mz_calculated.values,
    )

    results_filtered["inv_mob_predicted"] = inv_mob
    results_filtered["rt_predicted"] = rt_pred
    results_filtered["fragments_predicted"] = prosit_intensities_to_fragments_par(intensity_pred)

    PSMS = pd.merge(results_filtered, fragments_grouped, on="psm_id")

    PSMS["observed_dict"] = PSMS.apply(lambda r: fragments_to_dict(r.fragments_observed), axis=1)
    PSMS["predicted_dict"] = PSMS.apply(lambda r: fragments_to_dict(r.fragments_predicted), axis=1)
    PSMS["cosine_similarity"] = PSMS.apply(lambda s: cosim_from_dict(s.observed_dict, s.predicted_dict), axis=1)
    PSMS["delta_rt"] = PSMS.projected_rt - PSMS.rt_predicted
    PSMS["delta_ims"] = PSMS.ion_mobility - PSMS.inv_mob_predicted
    PSMS["intensity_ms1"] = 0.0
    PSMS["collision_energy"] = 0.0
    PSMS = PSMS.rename(columns={
        "ms2_intensity": "intensity_ms2",
        "fragment_ppm": "average_ppm",
        "precursor_ppm": "delta_mass"
    })

    logging.info("Re-scoring PSMs...")

    RE_SCORE = re_score_psms(
        PSMS,
        num_splits=args.num_splits,
        balance=args.balance,
        positive_example_q_max=args.positive_example_q_max
    )

    PSMS = pd.merge(PSMS, RE_SCORE, on=["spec_idx", "rank"])

    TDC = target_decoy_competition_pandas(PSMS, method=args.tdc_method, score="hyperscore")
    TDC_rescore = target_decoy_competition_pandas(PSMS, method=args.tdc_method, score="re_score")

    TDC = TDC.rename(columns={"match_idx": "peptide", "spec_idx": "psm_id"})
    TDC_rescore = TDC_rescore.rename(columns={"match_idx": "peptide", "spec_idx": "psm_id"})

    before = len(TDC[TDC.q_value <= 0.01])
    after = len(TDC_rescore[TDC_rescore.q_value <= 0.01])
    logging.info(f"Before re-scoring: {before} PSMs with q-value <= 0.01")
    logging.info(f"After re-scoring: {after} PSMs with q-value <= 0.01")

    if args.summary_plot:
        TARGET = PSMS[PSMS.decoy == False]
        DECOY = PSMS[PSMS.decoy]

        logging.info("Creating summary plot...")
        output_path = os.path.join(args.output, "summary_plot.png")
        plot_summary(TARGET, DECOY, output_path, dpi=300)

    output_path = os.path.dirname(args.output)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_name = os.path.join(output_path, "imspy_sage_hyperscore.tsv")
    file_name_rescore = os.path.join(output_path, "imspy_sage_rescore.tsv")

    if args.store_hyperscore:
        TDC.to_csv(file_name, sep="\t", index=False)
        logging.info(f"Output file {file_name} saved.")

    TDC_rescore.to_csv(file_name_rescore, sep="\t", index=False)
    logging.info(f"Output file {file_name_rescore} saved.")


if __name__ == "__main__":
    main()
