import argparse
import logging
import os
import sys
import time

import pandas as pd
import numpy as np

from sagepy.core import (Precursor, Tolerance, SpectrumProcessor, Scorer, EnzymeBuilder,
                         SAGE_KNOWN_MODS, validate_mods, validate_var_mods, SageSearchConfiguration)

from sagepy.core.scoring import associate_fragment_ions_with_prosit_predicted_intensities, json_bin_to_psms, ScoreType

from sagepy.qfdr.tdc import target_decoy_competition_pandas

from imspy.algorithm import DeepPeptideIonMobilityApex, load_deep_ccs_predictor, load_tokenizer_from_resources
from imspy.algorithm.rt.predictors import DeepChromatographyApex, load_deep_retention_time_predictor
from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper

from imspy.timstof import TimsDatasetDDA

from imspy.timstof.dbsearch.utility import sanitize_mz, sanitize_charge, get_searchable_spec, split_fasta, \
    get_collision_energy_calibration_factor, write_psms_binary, re_score_psms, \
    merge_dicts_with_merge_dict, generate_balanced_rt_dataset, generate_balanced_im_dataset, linear_map, beta_score

from sagepy.core.scoring import psms_to_json_bin
from sagepy.utility import peptide_spectrum_match_list_to_pandas
from sagepy.utility import apply_mz_calibration

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# don't use all the memory for the GPU (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for i, _ in enumerate(gpus):
            tf.config.experimental.set_virtual_device_configuration(
                gpus[i],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)]
            )
            print(f"GPU: {i} memory restricted to 4GB.")

    except RuntimeError as e:
        print(e)


def create_database(fasta, static, variab, enzyme_builder, generate_decoys, fragment_max_mz, bucket_size,
                    shuffle_decoys=True, keep_ends=True):
    sage_config = SageSearchConfiguration(
        fasta=fasta,
        static_mods=static,
        variable_mods=variab,
        enzyme_builder=enzyme_builder,
        generate_decoys=generate_decoys,
        fragment_max_mz=fragment_max_mz,
        bucket_size=bucket_size,
        shuffle_decoys=shuffle_decoys,
        keep_ends=keep_ends,
    )

    return sage_config.generate_indexed_database()


def main():
    # use argparse to parse command line arguments
    parser = argparse.ArgumentParser(description='🦀💻 IMSPY - timsTOF DDA 🔬🐍 - PROTEOMICS IMS DDA data analysis '
                                                 'using imspy and sagepy.')

    # Required string argument for path of bruker raw data
    parser.add_argument(
        "path",
        type=str,
        help="Path to bruker raw folders (.d) containing RAW files"
    )

    # Required string argument for path of fasta file
    parser.add_argument(
        "fasta",
        type=str,
        help="Path to the fasta file of proteins to be digested"
    )

    # Optional verbosity flag
    parser.add_argument(
        "-nv",
        "--no_verbose",
        dest="verbose",
        action="store_false",
        help="Decrease output verbosity"
    )
    parser.set_defaults(verbose=True)

    # Optional flag for fasta batch size, defaults to 1
    parser.add_argument(
        "-fbs",
        "--fasta_batch_size",
        type=int,
        default=1,
        help="Batch size for fasta file (default: 1)"
    )

    # SAGE isolation window settings
    parser.add_argument("--isolation_window_lower", type=float, default=-3.0, help="Isolation window (default: -3.0)")
    parser.add_argument("--isolation_window_upper", type=float, default=3.0, help="Isolation window (default: 3.0)")

    # decide whether precursor tolerance should be in ppm or dalton
    parser.add_argument(
        "--precursor_tolerance_da",
        dest="precursor_tolerance_da",
        action="store_true",
        help="Precursor tolerance in Dalton (default: False)"
    )
    parser.set_defaults(precursor_tolerance_da=False)

    # decide whether fragment tolerance should be in ppm or dalton
    parser.add_argument(
        "--fragment_tolerance_da",
        dest="fragment_tolerance_da",
        action="store_true",
        help="Fragment tolerance in Dalton (default: False)"
    )
    parser.set_defaults(fragment_tolerance_da=False)

    # SAGE Scoring settings
    # precursor tolerance lower and upper
    parser.add_argument("--precursor_tolerance_lower", type=float, default=-15.0,
                        help="Precursor tolerance lower (default: -15.0)")
    parser.add_argument("--precursor_tolerance_upper", type=float, default=15.0,
                        help="Precursor tolerance upper (default: 15.0)")

    # fragment tolerance lower and upper
    parser.add_argument("--fragment_tolerance_lower", type=float, default=-25.0,
                        help="Fragment tolerance lower (default: -25.0)")
    parser.add_argument("--fragment_tolerance_upper", type=float, default=25.0,
                        help="Fragment tolerance upper (default: 25.0)")

    # number of psms to report
    parser.add_argument("--report_psms", type=int, default=5, help="Number of PSMs to report (default: 5)")
    # minimum number of matched peaks
    parser.add_argument("--min_matched_peaks", type=int, default=5, help="Minimum number of matched peaks (default: 5)")
    # annotate matches

    parser.add_argument(
        "--no_match_annotation",
        dest="annotate_matches",
        action="store_false",
        help="Annotate matches (default: True)")
    parser.set_defaults(annotate_matches=True)

    parser.add_argument("--score_type", type=str, default="openms", help="Score type (default: openms)")

    # SAGE Preprocessing settings
    parser.add_argument("--take_top_n", type=int, default=150, help="Take top n peaks (default: 150)")

    # SAGE settings for digest of fasta file
    parser.add_argument("--missed_cleavages", type=int, default=2, help="Number of missed cleavages (default: 2)")
    parser.add_argument("--min_len", type=int, default=8, help="Minimum peptide length (default: 8)")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum peptide length (default: 30)")
    parser.add_argument("--cleave_at", type=str, default='KR', help="Cleave at (default: KR)")
    parser.add_argument("--restrict", type=str, default='P', help="Restrict (default: P)")
    parser.add_argument("--calibrate_mz", dest="calibrate_mz", action="store_true", help="Calibrate mz (default: False)")
    parser.set_defaults(calibrate_mz=False)
    parser.add_argument("--no_cysteine_static", dest="cysteine_static", action="store_false",
                        help="Cysteine static (default: True)")
    parser.set_defaults(cysteine_static=True)

    parser.add_argument(
        "--no_decoys",
        dest="decoys",
        action="store_false",
        help="Generate decoys (default: True)"
    )
    parser.set_defaults(decoys=True)

    parser.add_argument("--shuffle_decoys", dest="shuffle_decoys", action="store_true",
                        help="Shuffle decoys (default: False)")
    parser.set_defaults(shuffle_decoys=False)

    parser.add_argument("--include_peptide_ends", dest="keep_ends", action="store_false",
                        help="Keep decoy generated decoy start / end amino acids the same (default: True)")
    parser.set_defaults(keep_ends=True)

    parser.add_argument(
        "--not_c_terminal",
        dest="c_terminal",
        action="store_false",
        help="C terminal (default: True)"
    )
    parser.set_defaults(c_terminal=True)

    # sage search configuration
    parser.add_argument("--fragment_max_mz", type=float, default=1700.0, help="Fragment max mz (default: 1700.0)")
    parser.add_argument("--bucket_size", type=int, default=16384, help="Bucket size (default: 16384)")

    # score configuration
    parser.add_argument("--min_fragment_mz", type=float, default=150.0, help="Minimum fragment mz (default: 150.0)")
    parser.add_argument("--max_fragment_mz", type=float, default=1700.0, help="Maximum fragment mz (default: 1700.0)")
    parser.add_argument("--max_fragment_charge", type=int, default=2, help="Maximum fragment charge (default: 2)")

    # randomize fasta
    parser.add_argument(
        "--randomize_fasta_split",
        dest="randomize_fasta_split",
        action="store_true",
        help="Randomize fasta split (default: False)"
    )
    parser.set_defaults(randomize_fasta_split=False)

    # re-scoring settings
    parser.add_argument("--re_score_num_splits", type=int, default=10, help="Number of splits (default: 10)")

    # fdr threshold
    parser.add_argument("--fdr_threshold", type=float, default=0.01, help="FDR threshold (default: 0.01)")

    # number of threads
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads (default: 16)")

    # if train splits should be balanced
    parser.add_argument(
        "--no_balanced_re_score",
        dest="balanced_re_score",
        action="store_false",
        help="Balanced train splits (default: True)"
    )
    parser.set_defaults(balanced_re_score=True)

    # TDC method
    parser.add_argument(
        "--tdc_method",
        type=str,
        default="peptide_psm_peptide",
        help="TDC method (default: peptide_psm_peptide aka double competition)"
    )

    # load dataset in memory
    parser.add_argument(
        "--in_memory",
        dest="in_memory",
        action="store_true",
        help="Load dataset in memory"
    )
    parser.set_defaults(in_memory=False)

    # rt refinement settings
    parser.add_argument("--refine_rt", dest="refine_rt", action="store_true", help="Refine retention time")
    parser.set_defaults(refine_rt=False)

    # inverse_mobility refinement settings
    parser.add_argument("--refine_im", dest="refine_im", action="store_true", help="Refine inverse mobility")
    parser.set_defaults(refine_im=False)

    parser.add_argument("--refinement_verbose", dest="refinement_verbose", action="store_true", help="Refinement verbose")
    parser.set_defaults(refinement_verbose=False)

    parser.add_argument("--rescore_score", type=str, default="hyper_score", help="Score type to be used for re-scoring (default: hyper_score)")

    args = parser.parse_args()

    paths = []

    # Check if path exists
    if not os.path.exists(args.path):
        print(f"Path {args.path} does not exist. Exiting.")
        sys.exit(1)

    # Check if path is a directory
    if not os.path.isdir(args.path):
        print(f"Path {args.path} is not a directory. Exiting.")
        sys.exit(1)

    # check for fasta file
    if not os.path.exists(args.fasta):
        print(f"Path {args.fasta} does not exist. Exiting.")
        sys.exit(1)

    for root, dirs, _ in os.walk(args.path):
        for file in dirs:
            if file.endswith(".d"):
                paths.append(os.path.join(root, file))

    # get the write folder path
    write_folder_path = "/".join(args.path.split("/")[:-1])

    # create imspy folder if it does not exist
    if not os.path.exists(write_folder_path + "/imspy"):
        os.makedirs(write_folder_path + "/imspy")

    current_time = time.strftime("%Y%m%d-%H%M%S")

    # Set up logging
    logging.basicConfig(filename=f"{write_folder_path}/imspy/imspy-{current_time}.log",
                        level=logging.INFO, format='%(asctime)s %(message)s')

    logging.info("Arguments settings:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    # get time
    start_time = time.time()

    scores = ["hyper_score", "beta_score"]
    assert args.rescore_score in scores, f"Score type {args.rescore_score} not supported. Supported score types are: {scores}"

    if args.verbose:
        print(f"found {len(paths)} RAW data folders in {args.path} ...")

    if args.precursor_tolerance_da:
        if args.verbose:
            print("using precursor tolerance in Da ...")
            print(f"precursor tolerance: {args.precursor_tolerance_lower} Da to {args.precursor_tolerance_upper} Da ...")
        prec_tol = Tolerance(da=(args.precursor_tolerance_lower, args.precursor_tolerance_upper))
    else:
        if args.verbose:
            print("using precursor tolerance in ppm ...")
            print(f"precursor tolerance: {args.precursor_tolerance_lower} ppm to {args.precursor_tolerance_upper} ppm ...")
        prec_tol = Tolerance(ppm=(args.precursor_tolerance_lower, args.precursor_tolerance_upper))

    if args.fragment_tolerance_da:
        if args.verbose:
            print("using fragment tolerance in Da ...")
            print(f"fragment tolerance: {args.fragment_tolerance_lower} Da to {args.fragment_tolerance_upper} Da ...")
        frag_tol = Tolerance(da=(args.fragment_tolerance_lower, args.fragment_tolerance_upper))
    else:
        if args.verbose:
            print("using fragment tolerance in ppm ...")
            print(f"fragment tolerance: {args.fragment_tolerance_lower} ppm to {args.fragment_tolerance_upper} ppm ...")
        frag_tol = Tolerance(ppm=(args.fragment_tolerance_lower, args.fragment_tolerance_upper))

    score_type = ScoreType(args.score_type)

    if args.verbose:
        print(f"using {args.score_type} as score type ...")

    scorer = Scorer(
        precursor_tolerance=prec_tol,
        fragment_tolerance=frag_tol,
        report_psms=args.report_psms,
        min_matched_peaks=args.min_matched_peaks,
        annotate_matches=args.annotate_matches,
        min_fragment_mass=args.min_fragment_mz,
        max_fragment_mass=args.max_fragment_mz,
        max_fragment_charge=args.max_fragment_charge,
        score_type=score_type,
    )

    if args.verbose:
        print("generating fasta digest ...")

    # configure a trypsin-like digestor of fasta files
    enzyme_builder = EnzymeBuilder(
        missed_cleavages=args.missed_cleavages,
        min_len=args.min_len,
        max_len=args.max_len,
        cleave_at=args.cleave_at,
        restrict=args.restrict,
        c_terminal=args.c_terminal,
    )

    # generate static cysteine modification TODO: make configurable
    if args.cysteine_static:
        static_mods = {k: v for k, v in [SAGE_KNOWN_MODS.cysteine_static()]}
    else:
        static_mods = {}

    # generate variable methionine modification TODO: make configurable
    variable_mods = {k: v for k, v in [SAGE_KNOWN_MODS.methionine_variable(),
                                       SAGE_KNOWN_MODS.protein_n_terminus_variable()]}

    # cysteinylation should be set as variable modification if cysteine_static is False (MHC search)
    if args.cysteine_static is False:
        variable_mods["C"] = [119.001]

    # generate SAGE compatible mod representations
    static = validate_mods(static_mods)
    variab = validate_var_mods(variable_mods)

    # check if fasta is a path or a file, if it is a path, read all files ending with fasta in that path
    if os.path.isdir(args.fasta):
        fasta_files = [os.path.join(args.fasta, f) for f in os.listdir(args.fasta) if f.endswith(".fasta")]
        fasta = ""
        for fasta_file in fasta_files:
            with open(fasta_file, 'r') as infile:
                fasta += infile.read()

    # read fasta file
    else:
        with open(args.fasta, 'r') as infile:
            fasta = infile.read()

    fastas = split_fasta(fasta, args.fasta_batch_size, randomize=args.randomize_fasta_split)

    # create indexed database reference
    indexed_db = None

    # if only one fasta file, use the same configuration for all RAW files (removes need to re-create db for each file)
    if len(fastas) == 1:
        indexed_db = create_database(fastas[0], static, variab, enzyme_builder, args.decoys, args.fragment_max_mz,
                                     args.bucket_size)

    if args.verbose:
        print("loading deep learning models for intensity, retention time and ion mobility prediction ...")

    # the intensity predictor model
    prosit_model = Prosit2023TimsTofWrapper(verbose=False)
    # the ion mobility predictor model
    im_predictor = DeepPeptideIonMobilityApex(load_deep_ccs_predictor(),
                                              load_tokenizer_from_resources("tokenizer-ptm"))
    # the retention time predictor model
    rt_predictor = DeepChromatographyApex(load_deep_retention_time_predictor(),
                                          load_tokenizer_from_resources("tokenizer-ptm"), verbose=True)

    # go over RAW data one file at a time
    for p, path in enumerate(paths):
        if args.verbose:
            print(f"processing {path} ...")
            print(f"processing {p + 1} of {len(paths)} ...")

        ds_name = os.path.basename(path).split(".")[0]
        dataset = TimsDatasetDDA(str(path), in_memory=args.in_memory)

        rt_min = dataset.meta_data.Time.min() / 60.0
        rt_max = dataset.meta_data.Time.max() / 60.0

        if args.verbose:
            print("loading PASEF fragments ...")

        fragments = dataset.get_pasef_fragments(num_threads=1)

        if args.verbose:
            print("aggregating re-fragmented PASEF frames ...")

        fragments = fragments.groupby('precursor_id').agg({
            'frame_id': 'first',
            'time': 'first',
            'precursor_id': 'first',
            'raw_data': 'sum',
            'scan_begin': 'first',
            'scan_end': 'first',
            'isolation_mz': 'first',
            'isolation_width': 'first',
            'collision_energy': 'first',
            'largest_peak_mz': 'first',
            'average_mz': 'first',
            'monoisotopic_mz': 'first',
            'charge': 'first',
            'average_scan': 'first',
            'intensity': 'first',
            'parent_id': 'first',
        })

        mobility = fragments.apply(lambda r: r.raw_data.get_inverse_mobility_along_scan_marginal(), axis=1)
        fragments['mobility'] = mobility

        # generate random string for for spec_id
        spec_id = fragments.apply(lambda r: str(np.random.randint(int(1e6))) + '-' + str(r['frame_id']) + '-' + str(r['precursor_id']) + '-' + ds_name, axis=1)
        fragments['spec_id'] = spec_id

        if args.verbose:
            print("loading precursor data ...")

        sage_precursor = fragments.apply(lambda r: Precursor(
            mz=sanitize_mz(r['monoisotopic_mz'], r['largest_peak_mz']),
            intensity=r['intensity'],
            charge=sanitize_charge(r['charge']),
            isolation_window=Tolerance(da=(args.isolation_window_lower, args.isolation_window_upper)),
            collision_energy=r.collision_energy,
            inverse_ion_mobility=r.mobility,
        ), axis=1)

        fragments['sage_precursor'] = sage_precursor

        if args.verbose:
            print("preprocessing spectra ...")

        processed_spec = fragments.apply(
            lambda r: get_searchable_spec(
                precursor=r.sage_precursor,
                raw_fragment_data=r.raw_data,
                spec_processor=SpectrumProcessor(take_top_n=args.take_top_n),
                spec_id=r.spec_id,
                time=r['time'],
            ),
            axis=1
        )

        fragments['processed_spec'] = processed_spec

        if args.verbose:
            print(f"generated: {len(fragments)} spectra to be scored ...")
            print("creating search configuration ...")

        psm_dicts = []

        logging.info(f"Processing {ds_name} ...")

        for j, fasta in enumerate(fastas):

            if len(fastas) > 1:

                if args.verbose:
                    print(f"generating indexed database for fasta split {j + 1} of {len(fastas)} ...")

                indexed_db = create_database(fasta, static, variab, enzyme_builder, args.decoys, args.fragment_max_mz,
                                             args.bucket_size, shuffle_decoys=args.shuffle_decoys,
                                             keep_ends=args.keep_ends)

            if args.verbose:
                print("searching database ...")

            psm_dict = scorer.score_collection_psm(
                db=indexed_db,
                spectrum_collection=fragments['processed_spec'].values,
                num_threads=args.num_threads,
            )

            if args.calibrate_mz:

                if args.verbose:
                    print("calibrating mz ...")

                ppm_error = apply_mz_calibration(psm_dict, fragments)

                if args.verbose:
                    print(f"calibrated mz with error: {np.round(ppm_error, 2)} ppm ...")

                if args.verbose:
                    print("re-scoring PSMs after mz calibration ...")

                psm_dict = scorer.score_collection_psm(
                    db=indexed_db,
                    spectrum_collection=fragments['processed_spec'].values,
                    num_threads=16,
                )

                for _, values in psm_dict.items():
                    for value in values:
                        value.file_name = ds_name
                        if args.calibrate_mz:
                            value.mz_calibration_ppm = ppm_error

            counter = 0

            for _, values in psm_dict.items():
                counter += len(values)

            psm_dicts.append(psm_dict)

        if args.verbose:
            print("merging PSMs ...")

        if len(psm_dicts) > 1:
            merged_dict = merge_dicts_with_merge_dict(psm_dicts)
        else:
            merged_dict = psm_dicts[0]

        psm = []

        for _, values in merged_dict.items():
            psm.extend(values)

        # map PSMs rt domain to [0, 60]
        for p in psm:
            p.projected_rt = linear_map(p.retention_time_observed, rt_min, rt_max, 0.0, 60.0)

        if args.verbose:
            print(f"generated {len(psm)} PSMs ...")

        sample = list(sorted(psm, key=lambda x: x.hyper_score, reverse=True))[:int(2 ** 11)]

        collision_energy_calibration_factor, _ = get_collision_energy_calibration_factor(
            list(filter(lambda match: match.decoy is not True, sample)),
            prosit_model,
            verbose=args.verbose,
        )

        for ps in psm:
            ps.collision_energy_calibrated = ps.collision_energy + collision_energy_calibration_factor

        if args.verbose:
            print("predicting ion intensities ...")

        intensity_pred = prosit_model.predict_intensities(
            [p.sequence for p in psm],
            np.array([p.charge for p in psm]),
            [p.collision_energy_calibrated for p in psm],
            batch_size=2048,
            flatten=True,
        )

        psm = associate_fragment_ions_with_prosit_predicted_intensities(psm, intensity_pred,
                                                                        num_threads=args.num_threads)

        if args.verbose:
            print("calculating beta score ...")

        for ps in psm:
            ps.beta_score = beta_score(ps.fragments_observed, ps.fragments_predicted)

        if args.verbose:
            print("predicting ion mobilities ...")

        if args.refine_im:
            # re-load ion mobility predictor, make sure to not re-fit on already refined ion mobilities
            im_predictor = DeepPeptideIonMobilityApex(load_deep_ccs_predictor(),
                                                      load_tokenizer_from_resources("tokenizer-ptm"))

            if args.verbose:
                print("refining ion mobility predictions ...")
            # fit ion mobility predictor
            im_predictor.fine_tune_model(
                data=peptide_spectrum_match_list_to_pandas(generate_balanced_im_dataset(psms=psm)),
                batch_size=1024,
                re_compile=True,
                verbose=args.refinement_verbose,
            )

        # predict ion mobilities
        inv_mob = im_predictor.simulate_ion_mobilities(
            sequences=[x.sequence for x in psm],
            charges=[x.charge for x in psm],
            mz=[x.mono_mz_calculated for x in psm]
        )

        # set ion mobilities
        for mob, ps in zip(inv_mob, psm):
            ps.inverse_mobility_predicted = mob

        if not args.refine_im:
            # calculate calibration factor
            inv_mob_calibration_factor = np.mean(
                [x.inverse_mobility_observed - x.inverse_mobility_predicted for x in psm])

            # set calibrated ion mobilities
            for p in psm:
                p.inverse_mobility_predicted += inv_mob_calibration_factor

        if args.verbose:
            print("predicting retention times ...")

        if args.refine_rt:
            # re-load retention time predictor, make sure to not re-fit on already refined retention times
            rt_predictor = DeepChromatographyApex(load_deep_retention_time_predictor(),
                                                  load_tokenizer_from_resources("tokenizer-ptm"), verbose=args.verbose)

            if args.verbose:
                print("refining retention time predictions ...")

            ds = peptide_spectrum_match_list_to_pandas(
                generate_balanced_rt_dataset(psms=psm)
            )

            # fit retention time predictor
            rt_predictor.fine_tune_model(
                data=ds,
                batch_size=1024,
                re_compile=True,
                verbose=args.refinement_verbose,
            )

        # predict retention times
        rt_pred = rt_predictor.simulate_separation_times(
            sequences=[x.sequence for x in psm],
        )

        # set retention times
        for rt, p in zip(rt_pred, psm):
            p.retention_time_predicted = rt

        # serialize PSMs to JSON binary
        bts = psms_to_json_bin(psm)

        if args.verbose:
            print("writing PSMs to temp file ...")

        # write PSMs to binary file
        write_psms_binary(byte_array=bts, folder_path=write_folder_path, file_name=ds_name)

        logging.info(f"Processed {ds_name} ...")

        if args.verbose:
            time_end_tmp = time.time()
            minutes, seconds = divmod(time_end_tmp - start_time, 60)
            print(f"file {ds_name} processed after {minutes} minutes and {seconds:.2f} seconds.")


    psms = []

    # read PSMs from binary files
    for file in os.listdir(write_folder_path + "/imspy/psm/"):
        if file.endswith(".bin"):
            f = open(os.path.join(write_folder_path + "/imspy/psm/", file), 'rb')
            data = f.read()
            f.close()
            psms.extend(json_bin_to_psms(data))

    # sort PSMs to avoid leaking information into predictions during re-scoring
    psms = list(sorted(psms, key=lambda psm: (psm.spec_idx, psm.peptide_idx)))

    psms = re_score_psms(psms=psms, verbose=args.verbose, num_splits=args.re_score_num_splits,
                         balance=args.balanced_re_score, score=args.rescore_score)

    # serialize all PSMs to JSON binary
    bts = psms_to_json_bin(psms)

    # write all PSMs to binary file
    write_psms_binary(byte_array=bts, folder_path=write_folder_path, file_name="total_psms", total=True)

    PSM_pandas = peptide_spectrum_match_list_to_pandas(psms)
    PSM_pandas = PSM_pandas.drop(columns=["q_value", "score"])

    if args.verbose:
        print(f"FDR calculation, using target decoy competition: {args.tdc_method} ...")

    psms_rescored = target_decoy_competition_pandas(peptide_spectrum_match_list_to_pandas(psms, re_score=True),
                                                    method=args.tdc_method)

    psms_rescored = psms_rescored[(psms_rescored.q_value <= 0.01) & (psms_rescored.decoy == False)]

    TDC = pd.merge(psms_rescored, PSM_pandas, left_on=["spec_idx", "match_idx", "decoy"],
                   right_on=["spec_idx", "match_idx", "decoy"]).sort_values(by="score", ascending=False)

    TDC.to_csv(f"{write_folder_path}" + "/imspy/Peptides.csv", index=False)

    end_time = time.time()

    logging.info("Done processing all RAW files.")
    minutes, seconds = divmod(end_time - start_time, 60)

    if args.verbose:
        print("Done processing all RAW files.")
        print(f"Processed {len(paths)} RAW files in {minutes} minutes and {seconds:.2f} seconds.")

    # add time to log
    logging.info(f"Processed {len(paths)} RAW files in {minutes} minutes and {seconds:.2f} seconds.")


if __name__ == "__main__":
    main()
