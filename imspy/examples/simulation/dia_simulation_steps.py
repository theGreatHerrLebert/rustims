import os
import argparse
import pandas as pd
from tqdm import tqdm
from imspy.chemistry import calculate_mz

from imspy.simulation.proteome import PeptideDigest
from imspy.simulation.aquisition import TimsTofAcquisitionBuilderDIA
from imspy.algorithm import DeepPeptideIonMobilityApex, DeepChromatographyApex
from imspy.algorithm import (load_tokenizer_from_resources, load_deep_retention_time, load_deep_ccs_predictor)

from imspy.simulation.utility import generate_events, python_list_to_json_string, sequence_to_all_ions
from imspy.simulation.isotopes import generate_isotope_patterns_rust
from imspy.simulation.utility import (get_z_score_for_percentile, get_frames_numba, get_scans_numba,
                                      accumulated_intensity_cdf_numba)

from imspy.simulation.exp import TimsTofSyntheticFrameBuilderDIA
from imspy.algorithm.ionization.predictors import BinomialChargeStateDistributionModel

from pathlib import Path

os.environ["WANDB_SILENT"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    # use argparse to parse command line arguments
    parser = argparse.ArgumentParser(description='Run a proteomics experiment simulation '
                                                 'with DIA acquisition on a BRUKER TimsTOF.')

    # check if the path exists
    def check_path(p):
        if not os.path.exists(p):
            raise argparse.ArgumentTypeError(f"Invalid path: {p}")
        return p

    # Required string argument for path
    parser.add_argument("path", type=str, help="Path to save the experiment to")
    parser.add_argument("name", type=str, help="Name of the experiment")
    parser.add_argument("dia_ms_ms_windows", type=str, help="Path to the DIA window groups file")
    parser.add_argument("fasta", type=str, help="Path to the fasta file")

    # Optional verbosity flag
    parser.add_argument("-v", "--verbose", type=bool, default=True, help="Increase output verbosity")

    # Other arguments with default values
    parser.add_argument("--gradient_length", type=int, default=60 * 60, help="Gradient length (default: 7200)")
    parser.add_argument("--mz_lower", type=int, default=100, help="Lower bound for mz (default: 100)")
    parser.add_argument("--mz_upper", type=int, default=1700, help="Upper bound for mz (default: 1700)")
    parser.add_argument("--im_lower", type=float, default=0.6, help="Lower bound for IM (default: 0.6)")
    parser.add_argument("--im_upper", type=float, default=1.6, help="Upper bound for IM (default: 1.6)")
    parser.add_argument("--num_scans", type=int, default=927, help="Number of scans to simulate (default: 927)")

    # Peptide digestion arguments
    parser.add_argument("--sample-fraction", type=float, default=0.1, help="Sample fraction (default: 0.1)")
    parser.add_argument("--missed_cleavages", type=int, default=2, help="Number of missed cleavages (default: 2)")
    parser.add_argument("--min_len", type=int, default=9, help="Minimum peptide length (default: 7)")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum peptide length (default: 30)")
    parser.add_argument("--cleave_at", type=str, default='KR', help="Cleave at (default: KR)")
    parser.add_argument("--restrict", type=str, default='P', help="Restrict (default: P)")
    parser.add_argument("--decoys", type=bool, default=False, help="Generate decoys (default: False)")

    # Peptide intensities
    parser.add_argument("--intensity_mean", type=float, default=1e6, help="Mean peptide intensity (default: 1e6)")
    parser.add_argument("--intensity_min", type=float, default=1e5, help="Std peptide intensity (default: 1e5)")
    parser.add_argument("--intensity_max", type=float, default=1e9, help="Min peptide intensity (default: 1e9)")

    # Precursor isotopic pattern settings
    parser.add_argument("--isotope_k", type=int, default=8, help="Number of isotopes to simulate (default: 8)")
    parser.add_argument("--isotope_min_intensity", type=int, default=1, help="Min intensity for isotopes (default: 1)")
    parser.add_argument("--isotope_centroid", type=bool, default=True, help="Centroid isotopes (default: True)")

    # Distribution parameters
    parser.add_argument("--z_score", type=float, default=.99,
                        help="Z-score for frame and scan distributions (default: .99)")
    parser.add_argument("--std_rt", type=float, default=3.3,
                        help="Standard deviation for retention time distribution (default: 1.6)")
    parser.add_argument("--std_im", type=float, default=0.008,
                        help="Standard deviation for mobility distribution (default: 0.008)")

    # Number of cores to use
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads to use (default: 16)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256)")

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments in your program
    path = check_path(args.path)
    name = args.name
    dia_window_groups = check_path(args.dia_ms_ms_windows)
    fasta = check_path(args.fasta)
    verbose = args.verbose

    gradient_length = args.gradient_length
    assert 0 < gradient_length < 240 * 60, f"Gradient length must be between 0 and 240 minutes, was {gradient_length}"

    mz_lower = args.mz_lower
    mz_upper = args.mz_upper
    assert 50 < mz_lower < mz_upper < 2000, f"mz bounds must be between 50 and 2000, were {mz_lower} and {mz_upper}"

    im_lower = args.im_lower
    im_upper = args.im_upper
    assert 0.5 < im_lower < im_upper < 2, f"IM bounds must be between 0.5 and 2, were {im_lower} and {im_upper}"

    assert 0.0 < args.z_score < 1.0, f"Z-score must be between 0 and 1, was {args.z_score}"

    num_scans = args.num_scans

    print(f"Gradient Length: {args.gradient_length} seconds.")
    print(f"mz Lower Bound: {args.mz_lower}.")
    print(f"mz Upper Bound: {args.mz_upper}.")
    print(f"IM Lower Bound: {args.im_lower}.")
    print(f"IM Upper Bound: {args.im_upper}.")
    print(f"Number of Scans: {args.num_scans}.")