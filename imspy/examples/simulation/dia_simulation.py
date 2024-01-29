import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from imspy.core import MzSpectrum
from imspy.chemistry import calculate_mz

from imspy.simulation.proteome import PeptideDigest
from imspy.simulation.aquisition import TimsTofAcquisitionBuilderDIA
from imspy.algorithm import DeepPeptideIonMobilityApex, DeepChromatographyApex, DeepChargeStateDistribution
from imspy.algorithm import (load_tokenizer_from_resources, load_deep_retention_time,
                             load_deep_charge_state_predictor, load_deep_ccs_predictor)

from imspy.simulation.utility import generate_events
from imspy.simulation.isotopes import generate_isotope_patterns_rust
from imspy.simulation.utility import (get_z_score_for_percentile, get_frames_numba, get_scans_numba, irt_to_rts_numba,
                                      accumulated_intensity_cdf_numba)

from pathlib import Path
import json

os.environ["WANDB_SILENT"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Function to convert a list (or a pandas series) to a JSON string
def list_to_json_string(lst, as_float=True):
    if as_float:
        return json.dumps([float(np.round(x, 4)) for x in lst])
    return json.dumps([int(x) for x in lst])


# load peptides and ions
def json_string_to_list(json_string):
    return json.loads(json_string)


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
    parser.add_argument("dia_ms_ms_windows", type=str, help="Path to the DIA window groups file")
    parser.add_argument("fasta", type=str, help="Path to the fasta file")

    # Optional verbosity flag
    parser.add_argument("-v", "--verbose", type=bool, default=True, help="Increase output verbosity")

    # Other arguments with default values
    parser.add_argument("--gradient_length", type=int, default=60 * 120, help="Gradient length (default: 7200)")
    parser.add_argument("--mz_lower", type=int, default=100, help="Lower bound for mz (default: 100)")
    parser.add_argument("--mz_upper", type=int, default=1700, help="Upper bound for mz (default: 1700)")
    parser.add_argument("--im_lower", type=float, default=0.6, help="Lower bound for IM (default: 0.6)")
    parser.add_argument("--im_upper", type=float, default=1.6, help="Upper bound for IM (default: 1.6)")
    parser.add_argument("--num_scans", type=int, default=927, help="Number of scans to simulate (default: 927)")

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

    num_scans = args.num_scans

    print(f"Gradient Length: {args.gradient_length} seconds.")
    print(f"mz Lower Bound: {args.mz_lower}.")
    print(f"mz Upper Bound: {args.mz_upper}.")
    print(f"IM Lower Bound: {args.im_lower}.")
    print(f"IM Upper Bound: {args.im_upper}.")
    print(f"Number of Scans: {args.num_scans}.")

    acquisition_builder = TimsTofAcquisitionBuilderDIA(
        Path(path) / name,
        exp_name=name + ".d",
        window_group_file=dia_window_groups,
        gradient_length=gradient_length,
        im_lower=im_lower,
        im_upper=im_upper,
        mz_lower=mz_lower,
        mz_upper=mz_upper,
        num_scans=num_scans,
        verbose=verbose,
    )

    if verbose:
        print("Digesting peptides...")
    digest = PeptideDigest(
        fasta,
        missed_cleavages=2,
        min_len=7,
        max_len=50,
        cleave_at='KR',
        restrict='P',
        generate_decoys=False,
        verbose=verbose,
    )

    acquisition_builder.synthetics_handle.create_table(
        table_name='peptides',
        table=digest.peptides.sample(frac=.02),
    )

    if verbose:
        print("Simulating retention times...")

    # create RTColumn instance
    RTColumn = DeepChromatographyApex(
        model=load_deep_retention_time(),
        tokenizer=load_tokenizer_from_resources(),
        verbose=True
    )

    # predict rts
    peptide_rt = RTColumn.simulate_separation_times_pandas(
        data=acquisition_builder.synthetics_handle.get_table('peptides'),
        gradient_length=acquisition_builder.gradient_length,
    )

    # call this peptide counts instead
    events = generate_events(n=peptide_rt.shape[0], mean=1_000_000, min_val=50_000, max_val=1e9)
    peptide_rt['events'] = events

    # update peptides table in database
    acquisition_builder.synthetics_handle.create_table(
        table_name='peptides',
        table=peptide_rt,
    )

    if verbose:
        print("Simulating charge states...")

    # load charge state predictor
    IonSource = DeepChargeStateDistribution(
        model=load_deep_charge_state_predictor(),
        tokenizer=load_tokenizer_from_resources(),
        verbose=True
    )

    # predict charge states
    # TODO: configure
    peptide_ions = IonSource.simulate_charge_state_distribution_pandas(peptide_rt)

    # merge tables to have sequences with ions, remove mz values outside scope
    ions = pd.merge(left=peptide_ions, right=peptide_rt, left_on=['peptide_id'], right_on=['peptide_id'])

    # TODO: CHECK IF THIS TAKES MODIFICATIONS INTO ACCOUNT
    ions['mz'] = ions.apply(lambda r: calculate_mz(r['monoisotopic-mass'], r['charge']), axis=1)
    ions = ions[(ions.mz >= 100) & (ions.mz <= 1700)]

    if verbose:
        print("Simulating ion mobilities...")

    # load IM predictor
    IMS = DeepPeptideIonMobilityApex(
        model=load_deep_ccs_predictor(),
        tokenizer=load_tokenizer_from_resources(),
        verbose=True
    )

    # simulate mobilities
    dp = IMS.simulate_ion_mobilities_pandas(
        ions
    )

    # filter by mobility range
    ions = dp[(dp.mobility_gru_predictor >= acquisition_builder.im_lower) & (
                ions.mobility_gru_predictor <= acquisition_builder.im_upper)]

    if verbose:
        print("Simulating precursor spectra...")

    specs = generate_isotope_patterns_rust(ions.mz, ions.charge, min_intensity=5, k=8, centroid=True, num_threads=16)
    specs = [spec.to_jsons() for spec in specs]
    ions.insert(6, 'simulated_spectrum', specs)

    acquisition_builder.synthetics_handle.create_table(
        table_name='ions',
        table=ions,
    )

    frames = acquisition_builder.frame_table
    scans = acquisition_builder.scan_table

    z_score = get_z_score_for_percentile(target_score=.99)
    std_rt = 1.6
    std_im = 0.008

    frames_np = frames.time.values
    mobility_np = scans.mobility.values
    scans_np = scans.scan.values

    time_dict = dict(zip(frames.frame_id.values, frames.time.values))
    im_dict = dict(zip(scans.scan, scans.mobility))

    peptide_rt = acquisition_builder.synthetics_handle.get_table('peptides')

    total_list_frames = []
    total_list_frame_contributions = []

    if verbose:
        print("Calculating frame and scan distributions...")

    # generate frame_occurrence and frame_abundance columns
    for _, row in tqdm(peptide_rt.iterrows(), total=peptide_rt.shape[0], desc='frame distribution'):
        frame_occurrence, frame_abundance = [], []

        rt_value = row.retention_time_gru_predictor
        contributing_frames = get_frames_numba(rt_value, frames_np, std_rt, z_score)

        for frame in contributing_frames:
            time = time_dict[frame]
            start = time - acquisition_builder.rt_cycle_length
            i = accumulated_intensity_cdf_numba(start, time, rt_value, std_rt)
            
            # TODO: ADD NOISE HERE?

            frame_occurrence.append(frame)
            frame_abundance.append(i)

        total_list_frames.append(frame_occurrence)
        total_list_frame_contributions.append(frame_abundance)

    if verbose:
        print("Saving frame distributions...")

    peptide_rt['frame_occurrence'] = [list(x) for x in total_list_frames]
    peptide_rt['frame_abundance'] = [list(x) for x in total_list_frame_contributions]

    peptide_rt['frame_occurrence'] = peptide_rt['frame_occurrence'].apply(
        lambda x: list_to_json_string(x, as_float=False))
    peptide_rt['frame_abundance'] = peptide_rt['frame_abundance'].apply(list_to_json_string)

    # save peptide_rt to database
    acquisition_builder.synthetics_handle.create_table(
        table_name='peptides',
        table=peptide_rt
    )

    ions = acquisition_builder.synthetics_handle.get_table('ions')

    im_scans = []
    im_contributions = []

    for _, row in tqdm(ions.iterrows(), total=ions.shape[0], desc='scan distribution'):
        scan_occurrence, scan_abundance = [], []

        im_value = row.mobility_gru_predictor
        contributing_scans = get_scans_numba(im_value, mobility_np, scans_np, std_im, z_score)

        for scan in contributing_scans:
            im = im_dict[scan]
            start = im - acquisition_builder.im_cycle_length
            i = accumulated_intensity_cdf_numba(start, im, im_value, std_im)

            scan_occurrence.append(scan)
            scan_abundance.append(i)

        im_scans.append(scan_occurrence)
        im_contributions.append(scan_abundance)

    if verbose:
        print("Saving scan distributions...")

    ions['scan_occurrence'] = [list(x) for x in im_scans]
    ions['scan_abundance'] = [list(x) for x in im_contributions]

    ions['scan_occurrence'] = ions['scan_occurrence'].apply(lambda x: list_to_json_string(x, as_float=False))
    ions['scan_abundance'] = ions['scan_abundance'].apply(list_to_json_string)

    acquisition_builder.synthetics_handle.create_table(
        table_name='ions',
        table=ions
    )

    print("Starting frame assembly...")


if __name__ == '__main__':
    main()
