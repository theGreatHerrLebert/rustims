import argparse
import os
import sys
import time

import pandas as pd
import numpy as np
from sagepy.core import Precursor, Tolerance, SpectrumProcessor, Scorer, EnzymeBuilder, SAGE_KNOWN_MODS, validate_mods, \
    validate_var_mods, SageSearchConfiguration
from sagepy.core.scoring import associate_fragment_ions_with_prosit_predicted_intensities, \
    peptide_spectrum_match_list_to_pandas, json_bin_to_psms
from sagepy.qfdr.tdc import target_decoy_competition_pandas

from imspy.algorithm import DeepPeptideIonMobilityApex, DeepChromatographyApex, load_deep_ccs_predictor, \
    load_tokenizer_from_resources, load_deep_retention_time
from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper
from imspy.timstof import TimsDatasetDDA
from imspy.timstof.dbsearch.utility import sanitize_mz, sanitize_charge, get_searchable_spec, split_fasta, \
    get_collision_energy_calibration_factor, write_psms_binary, re_score_psms

from sagepy.core.scoring import psms_to_json_bin


def main():
    # use argparse to parse command line arguments
    parser = argparse.ArgumentParser(description='🦀💻 IMSPY - timsTOF DDA 🔬🐍 - PROTEOMICS IMS DDA data analysis '
                                                 'using imspy and sagepy.')

    # Required string argument for path and fasta file
    parser.add_argument(
        "path",
        type=str,
        help="Path to bruker raw folders (.d) containing RAW files"
    )
    parser.add_argument(
        "fasta",
        type=str,
        help="Path to the fasta file of proteins to be digested"
    )

    # Optional verbosity flag
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        default=True,
        help="Increase output verbosity"
    )
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

    # SAGE Scoring settings
    # precursor tolerance lower and upper
    parser.add_argument("--precursor_tolerance_lower", type=float, default=-5.0,
                        help="Precursor tolerance lower (default: -5.0)")
    parser.add_argument("--precursor_tolerance_upper", type=float, default=5.0,
                        help="Precursor tolerance upper (default: 5.0)")

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
    parser.add_argument("--annotate_matches", type=bool, default=True, help="Annotate matches (default: True)")
    # SAGE Preprocessing settings
    parser.add_argument("--take_top_n", type=int, default=100, help="Take top n peaks (default: 100)")

    # SAGE settings for digest of fasta file
    parser.add_argument("--missed_cleavages", type=int, default=2, help="Number of missed cleavages (default: 2)")
    parser.add_argument("--min_len", type=int, default=7, help="Minimum peptide length (default: 7)")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum peptide length (default: 30)")
    parser.add_argument("--cleave_at", type=str, default='KR', help="Cleave at (default: KR)")
    parser.add_argument("--restrict", type=str, default='P', help="Restrict (default: P)")
    parser.add_argument("--decoys", type=bool, default=True, help="Generate decoys (default: True)")
    parser.add_argument("--c_terminal", type=bool, default=True, help="C terminal (default: True)")

    # sage search configuration
    parser.add_argument("--fragment_max_mz", type=float, default=4000, help="Fragment max mz (default: 4000)")
    parser.add_argument("--bucket_size", type=int, default=16384, help="Bucket size (default: 16384)")

    # randomize fasta
    parser.add_argument("--randomize_fasta_split", type=bool, default=False,
                        help="Randomize fasta split (default: False)")

    # re-scoring settings
    parser.add_argument("--num_splits", type=int, default=10, help="Number of splits (default: 10)")

    # fdr threshold
    parser.add_argument("--fdr_threshold", type=float, default=0.01, help="FDR threshold (default: 0.01)")

    # number of threads
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads (default: 16)")
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

    # check for fasta file
    if not os.path.isfile(args.fasta):
        print(f"Path {args.fasta} is not a file. Exiting.")
        sys.exit(1)

    for root, dirs, _ in os.walk(args.path):
        for file in dirs:
            if file.endswith(".d"):
                paths.append(os.path.join(root, file))

    # get the write folder path
    write_folder_path = "/".join(args.path.split("/")[:-1])

    # get time
    start_time = time.time()

    if args.verbose:
        print(f"Found {len(paths)} RAW data folders in {args.path} ...")

    # go over RAW data one file at a time
    for path in paths:
        if args.verbose:
            print(f"Processing {path} ...")

        ds_name = os.path.basename(path).split(".")[0]
        dataset = TimsDatasetDDA(str(path), in_memory=False)

        if args.verbose:
            print("loading PASEF fragments ...")

        fragments = dataset.get_pasef_fragments()

        mobility = fragments.apply(lambda r: np.mean(r.raw_data.mobility), axis=1)
        fragments['mobility'] = mobility

        # generate random string for for spec_id
        spec_id = fragments.apply(lambda r: str(np.random.randint(int(1e9))) + '-' + str(r['frame_id']) + '-' + str(r['precursor_id']) + '-' + ds_name, axis=1)
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

        scorer = Scorer(
            precursor_tolerance=Tolerance(da=(args.precursor_tolerance_lower, args.precursor_tolerance_upper)),
            fragment_tolerance=Tolerance(ppm=(args.fragment_tolerance_lower, args.fragment_tolerance_upper)),
            report_psms=args.report_psms,
            min_matched_peaks=args.min_matched_peaks,
            annotate_matches=args.annotate_matches
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
        static_mods = {k: v for k, v in [SAGE_KNOWN_MODS.cysteine_static()]}

        # generate variable methionine modification TODO: make configurable
        variable_mods = {k: v for k, v in [SAGE_KNOWN_MODS.methionine_variable()]}

        # generate SAGE compatible mod representations
        static = validate_mods(static_mods)
        variab = validate_var_mods(variable_mods)

        with open(args.fasta, 'r') as infile:
            fasta = infile.read()

        fasta_list = split_fasta(fasta, args.fasta_batch_size, randomize=args.randomize_fasta_split)

        if args.verbose:
            print("loading deep learning models for intensity, retention time and ion mobility prediction ...")

        # the intensity predictor model
        prosit_model = Prosit2023TimsTofWrapper(verbose=False)
        # the ion mobility predictor model
        im_predictor = DeepPeptideIonMobilityApex(load_deep_ccs_predictor(),
                                                  load_tokenizer_from_resources("tokenizer_ionmob"))
        # the retention time predictor model
        rt_predictor = DeepChromatographyApex(load_deep_retention_time(),
                                              load_tokenizer_from_resources("tokenizer-ptm"), verbose=True)

        psm_list = []

        if args.verbose:
            print("generating search configuration ...")

        for i, fasta in enumerate(fasta_list):
            sage_config = SageSearchConfiguration(
                fasta=fasta,
                static_mods=static,
                variable_mods=variab,
                enzyme_builder=enzyme_builder,
                generate_decoys=args.decoys,
                fragment_max_mz=args.fragment_max_mz,
                bucket_size=int(args.bucket_size),
            )

            if args.verbose:
                print(f"generating indexed database for fasta {i + 1} of {len(fasta_list)} ...")

            # generate the database for searching against
            indexed_db = sage_config.generate_indexed_database()

            if args.verbose:
                print("searching database ...")

            psm = scorer.score_collection_psm(
                db=indexed_db,
                spectrum_collection=fragments['processed_spec'].values,
                num_threads=args.num_threads,
            )

            for p in psm:
                p.file_name = ds_name

            psm_list.extend(psm)

        psm = psm_list

        sample = np.random.choice(psm, 4096)
        collision_energy_calibration_factor, _ = get_collision_energy_calibration_factor(
            sample,
            prosit_model,
            verbose=args.verbose,
        )

        for p in psm:
            p.collision_energy_calibrated = p.collision_energy + collision_energy_calibration_factor

        if args.verbose:
            print("predicting ion intensities ...")

        intensity_pred = prosit_model.predict_intensities(
            [p.sequence for p in psm],
            np.array([p.charge for p in psm]),
            [p.collision_energy_calibrated for p in psm],
            batch_size=2048,
            flatten=True,
        )

        psm = associate_fragment_ions_with_prosit_predicted_intensities(psm, intensity_pred, num_threads=args.num_threads)

        if args.verbose:
            print("predicting ion mobilities ...")

        # predict ion mobilities
        inv_mob = im_predictor.simulate_ion_mobilities(
            sequences=[x.sequence for x in psm],
            charges=[x.charge for x in psm],
            mz=[x.mono_mz_calculated for x in psm]
        )

        # set ion mobilities
        for mob, p in zip(inv_mob, psm):
            p.inverse_mobility_predicted = mob

        # calculate calibration factor
        inv_mob_calibration_factor = np.mean(
            [x.inverse_mobility_observed - x.inverse_mobility_predicted for x in psm])

        # set calibrated ion mobilities
        for p in psm:
            p.inverse_mobility_predicted += inv_mob_calibration_factor

        PSM_pandas = peptide_spectrum_match_list_to_pandas(psm)
        PSM_q = target_decoy_competition_pandas(PSM_pandas)

        PSM_pandas = PSM_pandas.drop(columns=["q_value", "score"])

        # use good psm hits to fine tune RT predictor for dataset
        TDC = pd.merge(PSM_q, PSM_pandas, left_on=["spec_idx", "match_idx", "decoy"],
                       right_on=["spec_idx", "match_idx", "decoy"])

        TDC_filtered = TDC[TDC.q_value <= 0.001]

        if args.verbose:
            print("fine tuning retention time predictor ...")

        rt_predictor.fit_model(TDC_filtered[TDC_filtered.decoy == False], epochs=4, batch_size=2048)

        if args.verbose:
            print("predicting retention times ...")

        # predict retention times
        rt_pred = rt_predictor.simulate_separation_times(
            sequences=[x.sequence for x in psm]
        )

        # set retention times
        for rt, p in zip(rt_pred, psm):
            p.retention_time_predicted = rt

        # serialize PSMs to JSON binary
        bts = psms_to_json_bin(psm)

        # write PSMs to binary file
        write_psms_binary(byte_array=bts, folder_path=write_folder_path, file_name=ds_name)

    psms = []

    # read PSMs from binary files
    for file in os.listdir(write_folder_path):
        if file.endswith(".bin"):
            f = open(os.path.join(write_folder_path, file), 'rb')
            data = f.read()
            f.close()
            psms.extend(json_bin_to_psms(data))

    psms = list(sorted(psms, key=lambda psm: (psm.spec_idx, psm.peptide_idx)))

    psms = re_score_psms(psms, verbose=args.verbose, num_splits=args.num_splits)
    PSM_pandas = peptide_spectrum_match_list_to_pandas(psms)
    PSM_pandas = PSM_pandas.drop(columns=["q_value", "score"])

    psms_rescored = target_decoy_competition_pandas(peptide_spectrum_match_list_to_pandas(psms, re_score=True))
    psms_rescored = psms_rescored[(psms_rescored.q_value <= 0.01) & (psms_rescored.decoy == False)]

    TDC = pd.merge(psms_rescored, PSM_pandas, left_on=["spec_idx", "match_idx", "decoy"],
                   right_on=["spec_idx", "match_idx", "decoy"])

    TDC.to_csv(f"{write_folder_path}/Peptides.csv", index=False)

    end_time = time.time()

    if args.verbose:
        print("Done processing all RAW files.")
        minutes, seconds = divmod(end_time - start_time, 60)
        print(f"Processed {len(paths)} RAW files in {minutes} minutes and {seconds:.2f} seconds.")


if __name__ == "__main__":
    main()
