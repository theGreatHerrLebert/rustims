import os
os.environ["WANDB_SILENT"] = "true"
import argparse
import sqlite3

import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from tqdm import tqdm
from imspy.simulation.proteome import PeptideDigest
from imspy.simulation.aquisition import TimsTofAcquisitionBuilderDIA
from imspy.simulation.exp import SyntheticExperimentDataHandleDIA
from imspy.algorithm import DeepPeptideIonMobilityApex, DeepChromatographyApex, DeepChargeStateDistribution
from imspy.algorithm import (load_tokenizer_from_resources, load_deep_retention_time,
                             load_deep_charge_state_predictor, load_deep_ccs_predictor)

from imspy.simulation.utility import irt_to_rts_numba, generate_events


def main():
    # use argparse to parse command line arguments
    parser = argparse.ArgumentParser(description='Run a proteomics experiment simulation '
                                                 'with DIA acquisition on a BRUKER TimsTOF.')

    # check if the path exists
    def check_path(path):
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"Invalid path: {path}")
        return path

    # Required string argument for path
    parser.add_argument("path", type=str, help="Path to save the experiment to")
    parser.add_argument("name", type=str, help="Name of the experiment")
    parser.add_argument("dia_window_groups", type=str, help="Path to the DIA window groups file")
    parser.add_argument("dia_frame_to_window_group", type=str, help="Path to the DIA frame to window group file")
    parser.add_argument("fasta", type=str, help="Path to the fasta file")

    # Optional verbosity flag
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")

    # Other arguments with default values
    parser.add_argument("--gradient_length", type=int, default=60*120, help="Gradient length (default: 7200)")
    parser.add_argument("--mz_lower", type=int, default=100, help="Lower bound for mz (default: 100)")
    parser.add_argument("--mz_upper", type=int, default=1700, help="Upper bound for mz (default: 1700)")
    parser.add_argument("--im_lower", type=float, default=0.6, help="Lower bound for IM (default: 0.6)")
    parser.add_argument("--im_upper", type=float, default=1.6, help="Upper bound for IM (default: 1.6)")

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments in your program
    path = check_path(args.path)
    name = args.name
    dia_window_groups = check_path(args.dia_window_groups)
    dia_frame_to_window_group = check_path(args.dia_frame_to_window_group)
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

    print(f"Gradient Length: {args.gradient_length}")
    print(f"mz Lower Bound: {args.mz_lower}")
    print(f"mz Upper Bound: {args.mz_upper}")
    print(f"IM Lower Bound: {args.im_lower}")
    print(f"IM Upper Bound: {args.im_upper}")


if __name__ == '__main__':
    main()
