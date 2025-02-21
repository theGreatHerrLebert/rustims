import argparse
import logging
import os
import sys
import time
import toml

import mokapot

import pandas as pd
import numpy as np

from pathlib import Path

from imspy.timstof.dbsearch.mgf import mgf_to_sagepy_query
from sagepy.core import Precursor, Tolerance, SpectrumProcessor, Scorer, EnzymeBuilder, SageSearchConfiguration
from sagepy.core.scoring import associate_fragment_ions_with_prosit_predicted_intensities, ScoreType
from sagepy.qfdr.tdc import target_decoy_competition_pandas, assign_sage_spectrum_q, assign_sage_peptide_q, \
    assign_sage_protein_q

from imspy.algorithm.ccs.predictors import DeepPeptideIonMobilityApex, load_deep_ccs_predictor
from imspy.algorithm.utility import load_tokenizer_from_resources
from imspy.algorithm.rt.predictors import DeepChromatographyApex, load_deep_retention_time_predictor
from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper

from imspy.timstof import TimsDatasetDDA

from sklearn.svm import SVC
from sagepy.rescore.rescore import rescore_psms
from sagepy.core.fdr import sage_fdr_psm

from imspy.timstof.dbsearch.utility import sanitize_mz, sanitize_charge, get_searchable_spec, split_fasta, \
    write_psms_binary, \
    merge_dicts_with_merge_dict, generate_balanced_rt_dataset, generate_balanced_im_dataset, linear_map

from sagepy.rescore.utility import transform_psm_to_mokapot_pin

from imspy.algorithm.intensity.predictors import get_collision_energy_calibration_factor

from sagepy.utility import psm_collection_to_pandas
from sagepy.utility import apply_mz_calibration
from sagepy.utility import decompress_psms, compress_psms

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

def create_database(fasta, static, variab, enzyme_builder, generate_decoys, bucket_size,
                    shuffle_decoys=True, keep_ends=True):
    sage_config = SageSearchConfiguration(
        fasta=fasta,
        static_mods=static,
        variable_mods=variab,
        enzyme_builder=enzyme_builder,
        generate_decoys=generate_decoys,
        bucket_size=bucket_size,
        shuffle_decoys=shuffle_decoys,
        keep_ends=keep_ends,
    )

    return sage_config.generate_indexed_database()

# helper function to load configuration of modifications from a TOML file
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = toml.load(config_file)
    return config

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

    # Path to the script directory
    script_dir = Path(__file__).parent

    # Default configs modification configs path
    default_config_path = script_dir / "configs" / "config_tryptic.toml"

    # Optional argument for path to the configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help="Path to the configuration file (TOML format). Default: configs/config_tryptic.toml"
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
        default=None,
        help="Batch size for fasta file (default: 1)"
    )

    parser.add_argument(
        "--no_re_score_mokapot",
        dest="re_score_mokapot",
        action="store_false",
        help="Do not re-score PSMs using mokapot"
    )

    parser.set_defaults(re_score_mokapot=None)

    # SAGE isolation window settings
    parser.add_argument("--isolation_window_lower", type=float, default=None, help="Isolation window lower offset (default: -3.0)")
    parser.add_argument("--isolation_window_upper", type=float, default=None, help="Isolation window upper offset (default: 3.0)")

    # decide whether precursor tolerance should be in ppm or dalton
    parser.add_argument(
        "--precursor_tolerance_da",
        dest="precursor_tolerance_da",
        action="store_true",
        help="Precursor tolerance in Dalton (default: False)"
    )
    parser.set_defaults(precursor_tolerance_da=None)

    # decide whether fragment tolerance should be in ppm or dalton
    parser.add_argument(
        "--fragment_tolerance_da",
        dest="fragment_tolerance_da",
        action="store_true",
        help="Fragment tolerance in Dalton (default: False)"
    )
    parser.set_defaults(fragment_tolerance_da=None)

    # SAGE Scoring settings
    # precursor tolerance lower and upper
    parser.add_argument("--precursor_tolerance_lower", type=float, default=None,
                        help="Precursor tolerance lower (default: -15.0)")
    parser.add_argument("--precursor_tolerance_upper", type=float, default=None,
                        help="Precursor tolerance upper (default: 15.0)")

    # fragment tolerance lower and upper
    parser.add_argument("--fragment_tolerance_lower", type=float, default=None,
                        help="Fragment tolerance lower (default: -20.0)")
    parser.add_argument("--fragment_tolerance_upper", type=float, default=None,
                        help="Fragment tolerance upper (default: 20.0)")

    # number of psms to report
    parser.add_argument("--report_psms", type=int, default=None, help="Number of PSMs to report (default: 5)")
    # minimum number of matched peaks
    parser.add_argument("--min_matched_peaks", type=int, default=None, help="Minimum number of matched peaks (default: 5)")
    # annotate matches

    parser.add_argument(
        "--no_match_annotation",
        dest="annotate_matches",
        action="store_false",
        help="Do not annotate matches (default: True)")
    parser.set_defaults(annotate_matches=None)

    parser.add_argument("--score_type", type=str, default=None, help="Score type (default: openmshyperscore)")

    # SAGE Preprocessing settings
    parser.add_argument("--take_top_n", type=int, default=None, help="Take top n peaks (default: 150)")

    # SAGE settings for digest of fasta file
    parser.add_argument("--missed_cleavages", type=int, default=None, help="Number of missed cleavages (default: 2)")
    parser.add_argument("--min_len", type=int, default=None, help="Minimum peptide length (default: 7)")
    parser.add_argument("--max_len", type=int, default=None, help="Maximum peptide length (default: 30)")
    parser.add_argument("--cleave_at", type=str, default=None, help="Cleave at (default: KR)")
    parser.add_argument("--restrict", type=str, default=None, help="Restrict (default: P)")
    parser.add_argument(
        "--not_c_terminal",
        dest="c_terminal",
        action="store_false",
        help="Not C terminal (default: True)"
    )
    parser.set_defaults(c_terminal=None)

    parser.add_argument("--calibrate_mz", dest="calibrate_mz", action="store_true", help="Calibrate mz (default: False)")
    parser.set_defaults(calibrate_mz=None)

    parser.add_argument(
        "--no_decoys",
        dest="decoys",
        action="store_false",
        help="Do not generate decoys (default: True)"
    )
    parser.set_defaults(decoys=None)

    parser.add_argument("--shuffle_decoys", dest="shuffle_decoys", action="store_true",
                        help="Shuffle decoys (default: False)")
    parser.set_defaults(shuffle_decoys=None)

    parser.add_argument("--include_peptide_ends", dest="keep_ends", action="store_false",
                        help="Do not keep decoy generated decoy start/end amino acids the same (default: True)")
    parser.set_defaults(keep_ends=None)

    # sage search configuration
    parser.add_argument("--fragment_max_mz", type=float, default=None, help="Fragment max mz (default: 1700.0)")
    parser.add_argument("--bucket_size", type=int, default=None, help="Bucket size (default: 16384)")

    # score configuration
    parser.add_argument("--max_fragment_charge", type=int, default=None, help="Maximum fragment charge (default: 2)")

    # randomize fasta
    parser.add_argument(
        "--randomize_fasta_split",
        dest="randomize_fasta_split",
        action="store_true",
        help="Randomize fasta split (default: False)"
    )
    parser.set_defaults(randomize_fasta_split=None)

    # re-scoring settings
    parser.add_argument("--re_score_num_splits", type=int, default=None, help="Number of splits (default: 5)")
    parser.add_argument("--re_score_metric", type=str, default=None, help="Score metric to use in re-scoring (default: hyperscore)")

    # fdr threshold, aka q-value to filter PSMs
    parser.add_argument("--fdr_threshold", type=float, default=None, help="FDR threshold (default: 0.01)")
    parser.add_argument("--fdr_psm_method", type=str, default=None, help="FDR calculation method for PSMs (default: psm)")
    parser.add_argument("--fdr_peptide_method", type=str, default=None, help="FDR calculation method for peptides (default: peptide_psm_peptide)")
    parser.add_argument("--fdr_score", type=str, default=None, help="Score to use for FDR calculation (default: re_score)")

    # number of threads
    parser.add_argument("--num_threads", type=int, default=None, help="Number of threads (default: -1)")

    # do not remove decoys
    parser.add_argument(
        "--no_remove_decoys",
        dest="remove_decoys",
        action="store_false",
        help="Do not remove decoys (default: True)"
    )
    # remove decoys
    parser.set_defaults(remove_decoys=None)

    # if train splits should be balanced
    parser.add_argument(
        "--no_balanced_re_score",
        dest="balanced_re_score",
        action="store_false",
        help="Do not balance train splits (default: True)"
    )
    parser.set_defaults(balanced_re_score=None)

    # load dataset in memory
    parser.add_argument(
        "--in_memory",
        dest="in_memory",
        action="store_true",
        help="Load dataset in memory"
    )
    parser.set_defaults(in_memory=None)

    # dont use bruker sdk
    parser.add_argument(
        "--no_bruker_sdk",
        dest="bruker_sdk",
        action="store_false",
        help="Do not use bruker sdk"
    )
    parser.set_defaults(bruker_sdk=None)

    # rt refinement settings
    parser.add_argument("--refine_rt", dest="refine_rt", action="store_true", help="Refine retention time")
    parser.set_defaults(refine_rt=None)

    # inverse_mobility refinement settings
    parser.add_argument("--refine_im", dest="refine_im", action="store_true", help="Refine inverse mobility")
    parser.set_defaults(refine_im=None)

    parser.add_argument("--refinement_verbose", dest="refinement_verbose", action="store_true", help="Refinement verbose")
    parser.set_defaults(refinement_verbose=None)

    # Batch sizes and sample size for collision energy calibration
    parser.add_argument("--intensity_prediction_batch_size", type=int, default=None,
                        help="Batch size for intensity prediction (default: 2048)")
    parser.add_argument("--model_fine_tune_batch_size", type=int, default=None,
                        help="Batch size for model fine-tuning (default: 1024)")
    parser.add_argument("--sample_size_collision_energy_calibration", type=int, default=None,
                        help="Sample size for collision energy calibration (default: 256)")
    parser.add_argument("--tims2rescore_table", dest="tims2rescore_table", action="store_true", help="Write PSM table that can be passed to tims2rescore")
    parser.set_defaults(tims2rescore_table=None)
    parser.add_argument("--use_mgf", action="store_true", help="Use Bruker DataAnalysis parsed MGF files stored in the .d folders instead of raw data.")
    parser.set_defaults(use_mgf=None)

    args = parser.parse_args()

    # Load the configuration from the specified file
    config = load_config(args.config)

    # Initialize parameters with defaults from the config file
    params = {
        'variable_modifications': config.get('variable_modifications', {}),
        'static_modifications': config.get('static_modifications', {}),
        'score_type': config.get('scoring', {}).get('score_type', 'openmshyperscore'),
        'report_psms': config.get('scoring', {}).get('report_psms', 5),
        'min_matched_peaks': config.get('scoring', {}).get('min_matched_peaks', 5),
        'annotate_matches': config.get('scoring', {}).get('annotate_matches', True),
        'max_fragment_charge': config.get('scoring', {}).get('max_fragment_charge', 2),
        'precursor_tolerance_da': config.get('precursor_tolerance', {}).get('use_da', False),
        'precursor_tolerance_lower': config.get('precursor_tolerance', {}).get('lower', -15.0),
        'precursor_tolerance_upper': config.get('precursor_tolerance', {}).get('upper', 15.0),
        'fragment_tolerance_da': config.get('fragment_tolerance', {}).get('use_da', False),
        'fragment_tolerance_lower': config.get('fragment_tolerance', {}).get('lower', -20.0),
        'fragment_tolerance_upper': config.get('fragment_tolerance', {}).get('upper', 20.0),
        'isolation_window_lower': config.get('isolation_window', {}).get('lower', -3.0),
        'isolation_window_upper': config.get('isolation_window', {}).get('upper', 3.0),
        'take_top_n': config.get('preprocessing', {}).get('take_top_n', 150),
        'missed_cleavages': config.get('enzyme', {}).get('missed_cleavages', 2),
        'min_len': config.get('enzyme', {}).get('min_len', 7),
        'max_len': config.get('enzyme', {}).get('max_len', 30),
        'cleave_at': config.get('enzyme', {}).get('cleave_at', 'KR'),
        'restrict': config.get('enzyme', {}).get('restrict', 'P'),
        'c_terminal': config.get('enzyme', {}).get('c_terminal', True),
        'decoys': config.get('database', {}).get('generate_decoys', True),
        'shuffle_decoys': config.get('database', {}).get('shuffle_decoys', False),
        'keep_ends': config.get('database', {}).get('keep_ends', True),
        'bucket_size': config.get('database', {}).get('bucket_size', 16384),
        'fragment_max_mz': config.get('search', {}).get('fragment_max_mz', 1700.0),
        'randomize_fasta_split': config.get('other', {}).get('randomize_fasta_split', False),
        're_score_num_splits': config.get('re_scoring', {}).get('re_score_num_splits', 5),
        're_score_metric': config.get('re_scoring', {}).get('re_score_metric', 'hyperscore'),
        'fdr_threshold': config.get('fdr', {}).get('fdr_threshold', 0.01),
        'fdr_psm_method': config.get('fdr', {}).get('fdr_psm_method', 'psm'),
        'fdr_peptide_method': config.get('fdr', {}).get('fdr_peptide_method', 'peptide_psm_peptide'),
        'fdr_score': config.get('fdr', {}).get('fdr_score', 're_score'),
        'num_threads': config.get('parallelization', {}).get('num_threads', -1),
        'remove_decoys': config.get('fdr', {}).get('remove_decoys', True),
        'balanced_re_score': config.get('re_scoring', {}).get('balanced_re_score', True),
        'calibrate_mz': config.get('other', {}).get('calibrate_mz', False),
        'in_memory': config.get('other', {}).get('in_memory', False),
        'bruker_sdk': config.get('other', {}).get('bruker_sdk', True),
        'refine_rt': config.get('refinement', {}).get('refine_rt', False),
        'refine_im': config.get('refinement', {}).get('refine_im', False),
        'refinement_verbose': config.get('refinement', {}).get('refinement_verbose', False),
        'intensity_prediction_batch_size': config.get('batch_sizes', {}).get('intensity_prediction_batch_size', 2048),
        'model_fine_tune_batch_size': config.get('batch_sizes', {}).get('model_fine_tune_batch_size', 1024),
        'sample_size_collision_energy_calibration': config.get('batch_sizes', {}).get('sample_size_collision_energy_calibration', 256),
        'verbose': config.get('other', {}).get('verbose', True),
        'fasta_batch_size': config.get('other', {}).get('fasta_batch_size', 1),
        're_score_mokapot': config.get('re_scoring', {}).get('re_score_mokapot', True),
        'tims2rescore_table': config.get('other', {}).get('tims2rescore_table', False),
        'use_mgf': config.get('other', {}).get('use_mgf', False)
    }

    # Override parameters with command-line arguments if provided
    for key in vars(args):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)

    variable_modifications = params['variable_modifications']
    static_modifications = params['static_modifications']

    args.verbose = params['verbose']

    if args.verbose:
        print(f"Variable modifications to be applied: {variable_modifications}")
        print(f"Static modifications to be applied: {static_modifications}")

    paths = []
    mgfs = []

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
                path = os.path.join(root, file)
                if params['use_mgf']:
                    if args.verbose:
                        print(f"Looking for mgf in folder `{path}` ...")
                    mgf_path = None
                    mgf_path_cnt = 0
                    for potential_mgf_path in Path(path).iterdir():
                        if potential_mgf_path.suffix == ".mgf":
                            mgf_path = str(potential_mgf_path)
                            mgf_path_cnt += 1
                    assert mgf_path_cnt == 1, f"Found {mgf_path_cnt} mgfs in folder `{path}`. We need exactly one. From Bruker DataAnalysis."
                    mgfs.append(mgf_path)
                paths.append(path)

    # Get the write folder path
    write_folder_path = str(Path(args.path))

    # create imspy folder if it does not exist
    if not os.path.exists(write_folder_path + "/imspy"):
        os.makedirs(write_folder_path + "/imspy")

    current_time = time.strftime("%Y%m%d-%H%M%S")

    # Set up logging
    logging.basicConfig(filename=f"{write_folder_path}/imspy/imspy-{current_time}.log",
                        level=logging.INFO, format='%(asctime)s %(message)s')

    if params['num_threads'] == -1 and args.verbose:
        print(f"Using all available CPU threads: {os.cpu_count()} ...")
        params['num_threads'] = os.cpu_count()

    logging.info("Arguments settings:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    # get time
    start_time = time.time()

    if args.verbose:
        print(f"found {len(paths)} RAW data folders in {args.path} ...")

    if params['precursor_tolerance_da']:
        if args.verbose:
            print("using precursor tolerance in Da ...")
            print(f"precursor tolerance: {params['precursor_tolerance_lower']} Da to {params['precursor_tolerance_upper']} Da ...")
        prec_tol = Tolerance(da=(params['precursor_tolerance_lower'], params['precursor_tolerance_upper']))
    else:
        if args.verbose:
            print("using precursor tolerance in ppm ...")
            print(f"precursor tolerance: {params['precursor_tolerance_lower']} ppm to {params['precursor_tolerance_upper']} ppm ...")
        prec_tol = Tolerance(ppm=(params['precursor_tolerance_lower'], params['precursor_tolerance_upper']))

    if params['fragment_tolerance_da']:
        if args.verbose:
            print("using fragment tolerance in Da ...")
            print(f"fragment tolerance: {params['fragment_tolerance_lower']} Da to {params['fragment_tolerance_upper']} Da ...")
        frag_tol = Tolerance(da=(params['fragment_tolerance_lower'], params['fragment_tolerance_upper']))
    else:
        if args.verbose:
            print("using fragment tolerance in ppm ...")
            print(f"fragment tolerance: {params['fragment_tolerance_lower']} ppm to {params['fragment_tolerance_upper']} ppm ...")
        frag_tol = Tolerance(ppm=(params['fragment_tolerance_lower'], params['fragment_tolerance_upper']))

    score_type = ScoreType(params['score_type'])

    if args.verbose:
        print(f"using {params['score_type']} as score type ...")

    scorer = Scorer(
        precursor_tolerance=prec_tol,
        fragment_tolerance=frag_tol,
        report_psms=params['report_psms'],
        min_matched_peaks=params['min_matched_peaks'],
        annotate_matches=params['annotate_matches'],
        max_fragment_charge=params['max_fragment_charge'],
        score_type=score_type,
        variable_mods=variable_modifications,
        static_mods=static_modifications,
    )

    if args.verbose:
        print("generating fasta digest ...")

    # configure a trypsin-like digestor of fasta files
    enzyme_builder = EnzymeBuilder(
        missed_cleavages=params['missed_cleavages'],
        min_len=params['min_len'],
        max_len=params['max_len'],
        cleave_at=params['cleave_at'],
        restrict=params['restrict'],
        c_terminal=params['c_terminal'],
    )

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

    fastas = split_fasta(fasta, params['fasta_batch_size'], randomize=params['randomize_fasta_split'])

    # create indexed database reference
    indexed_db = None

    # if only one fasta file, use the same configuration for all RAW files (removes need to re-create db for each file)
    if len(fastas) == 1:
        indexed_db = create_database(
            fastas[0],
            static_modifications,
            variable_modifications,
            enzyme_builder,
            params['decoys'],
            params['bucket_size'],
            shuffle_decoys=params['shuffle_decoys'],
            keep_ends=params['keep_ends']
        )

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
    for file_id, path in enumerate(paths):
        if args.verbose:
            print(f"processing {path} ...")
            print(f"processing {file_id + 1} of {len(paths)} ...")

        ds_name = os.path.basename(path).split(".")[0]
        dataset = TimsDatasetDDA(str(path), in_memory=params['in_memory'], use_bruker_sdk=params['bruker_sdk'])

        rt_min = dataset.meta_data.Time.min() / 60.0
        rt_max = dataset.meta_data.Time.max() / 60.0

        if args.verbose:
            print("loading PASEF fragments ...")

        fragments = None

        if params['use_mgf']:
            mgf_path = mgfs[file_id]
            fragments = mgf_to_sagepy_query(mgf_path, top_n=params['take_top_n'])
        else:
            fragments = dataset.get_pasef_fragments(num_threads=params['num_threads'] if not params['bruker_sdk'] else 1)

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
                isolation_window=Tolerance(da=(params['isolation_window_lower'], params['isolation_window_upper'])),
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
                    spec_processor=SpectrumProcessor(take_top_n=params['take_top_n'], deisotope=True),
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

                indexed_db = create_database(
                    fasta,
                    static_modifications,
                    variable_modifications,
                    enzyme_builder,
                    params['decoys'],
                    params['bucket_size'],
                    shuffle_decoys=params['shuffle_decoys'],
                    keep_ends=params['keep_ends']
                )

            if args.verbose:
                print("searching database ...")

            psm_dict = scorer.score_collection_psm(
                db=indexed_db,
                spectrum_collection=fragments if params['use_mgf'] else fragments['processed_spec'].values,
                num_threads=params['num_threads'],
            )

            if params['calibrate_mz']:

                if params['use_mgf']:
                    raise NotImplementedError("Mass calibration is not yet supported in --use_mgf mode.")

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
                    num_threads=params['num_threads'],
                )

                for _, values in psm_dict.items():
                    for value in values:
                        value.file_name = ds_name
                        if params['calibrate_mz']:
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
            psm.extend(list(filter(lambda p: p.sage_feature.rank <= 5, values)))

        # map PSMs rt domain to [0, 60]
        for p in psm:
            p.retention_time_projected = linear_map(p.retention_time, rt_min, rt_max, 0.0, 60.0)

        if args.verbose:
            print(f"generated {len(psm)} PSMs ...")

        sample_size = params['sample_size_collision_energy_calibration']
        sample = list(sorted(psm, key=lambda x: x.hyperscore, reverse=True))[:sample_size]

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
            [p.sequence_modified if p.decoy == False else p.sequence_decoy_modified for p in psm],
            np.array([p.charge for p in psm]),
            [p.collision_energy_calibrated for p in psm],
            batch_size=params['intensity_prediction_batch_size'],
            flatten=True,
        )

        psm = associate_fragment_ions_with_prosit_predicted_intensities(psm, intensity_pred,
                                                                        num_threads=params['num_threads'])

        if args.verbose:
            print("predicting ion mobilities ...")

        if params['refine_im']:
            # re-load ion mobility predictor, make sure to not re-fit on already refined ion mobilities
            im_predictor = DeepPeptideIonMobilityApex(load_deep_ccs_predictor(),
                                                      load_tokenizer_from_resources("tokenizer-ptm"))

            if args.verbose:
                print("refining ion mobility predictions ...")
            # fit ion mobility predictor
            im_predictor.fine_tune_model(
                data=psm_collection_to_pandas(generate_balanced_im_dataset(psms=psm)),
                batch_size=params['model_fine_tune_batch_size'],
                re_compile=True,
                verbose=params['refinement_verbose'],
            )

        # predict ion mobilities
        inv_mob = im_predictor.simulate_ion_mobilities(
            sequences=[x.sequence_modified if x.decoy == False else x.sequence_decoy_modified for x in psm],
            charges=[x.charge for x in psm],
            mz=[x.mono_mz_calculated for x in psm]
        )

        # set ion mobilities
        for mob, ps in zip(inv_mob, psm):
            ps.inverse_ion_mobility_predicted = mob

        if not params['refine_im']:
            # calculate calibration factor
            inv_mob_calibration_factor = np.mean(
                [x.inverse_ion_mobility - x.inverse_ion_mobility_predicted for x in psm]
            )

            if args.verbose:
                print(f"calibrated ion mobilities with factor: {np.round(inv_mob_calibration_factor, 4)} ...")

            # set calibrated ion mobilities
            for p in psm:
                p.inverse_ion_mobility_predicted += inv_mob_calibration_factor

        if args.verbose:
            print("predicting retention times ...")

        if params['refine_rt']:
            # re-load retention time predictor, make sure to not re-fit on already refined retention times
            rt_predictor = DeepChromatographyApex(load_deep_retention_time_predictor(),
                                                  load_tokenizer_from_resources("tokenizer-ptm"), verbose=args.verbose)

            if args.verbose:
                print("refining retention time predictions ...")

            ds = psm_collection_to_pandas(
                generate_balanced_rt_dataset(psms=psm)
            )

            # fit retention time predictor
            rt_predictor.fine_tune_model(
                data=ds,
                batch_size=params['model_fine_tune_batch_size'],
                re_compile=True,
                verbose=params['refinement_verbose'],
            )

        # predict retention times
        rt_pred = rt_predictor.simulate_separation_times(
            sequences=[x.sequence_modified if x.decoy == False else x.sequence_decoy_modified for x in psm],
        )

        # set retention times
        for rt, p in zip(rt_pred, psm):
            p.retention_time_predicted = rt

        # add file id to PSMs
        for p in psm:
            p.sage_feature_file_id = file_id

        # serialize PSMs to bincode binary
        bts = compress_psms(psm)

        if args.verbose:
            print("writing PSMs to temp file ...")

        # write PSMs to binary file
        write_psms_binary(byte_array=bts, folder_path=write_folder_path, file_name=ds_name)

        logging.info(f"Processed {ds_name} ...")

        if args.verbose:
            time_end_tmp = time.time()
            minutes, seconds = divmod(time_end_tmp - start_time, 60)
            print(f"file {ds_name} processed after {minutes} minutes and {seconds:.2f} seconds.")


    if args.verbose:
        print("loading all PSMs ...")

    psms = []

    # read PSMs from binary files
    for file in os.listdir(write_folder_path + "/imspy/psm/"):
        if file.endswith(".bin"):
            f = open(os.path.join(write_folder_path + "/imspy/psm/", file), 'rb')
            data = f.read()
            f.close()
            psms.extend(decompress_psms(data))

    # sort PSMs to avoid leaking information into predictions during re-scoring
    psms = list(sorted(psms, key=lambda psm: (psm.spec_idx, psm.peptide_idx)))

    psms = rescore_psms(
        psm_collection=psms,
        verbose=args.verbose,
        model=SVC(probability=True),
        num_splits=params['re_score_num_splits'],
        balance=params['balanced_re_score'],
        score=params['re_score_metric'],
        num_threads=params['num_threads'],
    )

    # if we have only one fasta file, we can use sage core fdr calculation
    if len(fastas) == 1:
        if args.verbose:
            print("calculating q-values using SAGE internal functions...")
        sage_fdr_psm(psms, indexed_db, use_hyper_score=False)

    # if we have multiple fasta files, q-values need to be calculated with database independent functions
    else:
        if args.verbose:
            print("calculating q-values using SAGE-style re-implemented functions...")
        assign_sage_spectrum_q(psms, use_hyper_score=True)
        assign_sage_peptide_q(psms, use_hyper_score=True)
        assign_sage_protein_q(psms, use_hyper_score=True)

    # serialize all PSMs to JSON binary
    bts = compress_psms(psms)

    if args.verbose:
        print("writing all re-scored PSMs to temp file ...")

    # write all PSMs to binary file
    write_psms_binary(byte_array=bts, folder_path=write_folder_path, file_name="total_psms", total=True)

    PSM_pandas = psm_collection_to_pandas(psms)

    if params['re_score_mokapot']:
        if args.verbose:
            print("re-scoring PSMs using mokapot ...")

        # create a mokapot folder in the imspy folder
        if not os.path.exists(write_folder_path + "/imspy/mokapot"):
            os.makedirs(write_folder_path + "/imspy/mokapot")

        # create a PIN file from the PSMs
        PSM_pin = transform_psm_to_mokapot_pin(PSM_pandas)
        PSM_pin.to_csv(f"{write_folder_path}" + "/imspy/mokapot/PSMs.pin", index=False, sep="\t")

        psms_moka = mokapot.read_pin(f"{write_folder_path}" + "/imspy/mokapot/PSMs.pin")
        results, models = mokapot.brew(psms_moka, max_workers=params['num_threads'])

        results.to_txt(dest_dir=f"{write_folder_path}" + "/imspy/mokapot/")

    PSM_pandas = PSM_pandas.drop(columns=["re_score"])

    if args.verbose:
        print(f"FDR calculation ...")

    psms_rescored = target_decoy_competition_pandas(psm_collection_to_pandas(psms),
                                                    method=params['fdr_psm_method'], score=params['fdr_score'])

    # remove decoys if specified
    if params['remove_decoys']:
        psms_rescored = psms_rescored[psms_rescored.decoy == False]

    psms_rescored = psms_rescored[(psms_rescored.q_value <= params['fdr_threshold'])]

    TDC = pd.merge(psms_rescored, PSM_pandas, left_on=["spec_idx", "match_idx", "decoy"],
                   right_on=["spec_idx", "match_idx", "decoy"]).sort_values(by="re_score", ascending=False)

    TDC.to_csv(f"{write_folder_path}" + "/imspy/PSMs.csv", index=False)

    if params['tims2rescore_table']:
        if args.verbose:
            print(f"Writing PSM table that can be passed to tims2rescore ...")

        if params['use_mgf']:
            print("Warning: tims2rescore does not support parsed MGF files due to duplicate precursor IDs.")

        from imspy.timstof.dbsearch.utility import parse_to_tims2rescore
        TDC_tims2rescore = parse_to_tims2rescore(TDC, from_mgf=params['use_mgf'], file_name=ds_name + ".d")
        TDC_tims2rescore.to_csv(f"{write_folder_path}" + "/imspy/results.sagepy.tsv", sep="\t", index=False)

    peptides_rescored = target_decoy_competition_pandas(psm_collection_to_pandas(psms),
                                                        method=params['fdr_peptide_method'], score=params['fdr_score'])

    # remove decoys if specified
    if params['remove_decoys']:
        peptides_rescored = peptides_rescored[peptides_rescored.decoy == False]

    peptides_rescored = peptides_rescored[(peptides_rescored.q_value <= params['fdr_threshold'])]

    TDC = pd.merge(peptides_rescored, PSM_pandas, left_on=["spec_idx", "match_idx", "decoy"],
                   right_on=["spec_idx", "match_idx", "decoy"]).sort_values(by="re_score", ascending=False)

    TDC.to_csv(f"{write_folder_path}" + "/imspy/Peptides.csv", index=False, encoding='utf-8')

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