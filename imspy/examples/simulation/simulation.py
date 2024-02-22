import os
import argparse

import imspy_connector as ims

import numpy as np
import pandas as pd
from tqdm import tqdm
from imspy.chemistry import calculate_mz

from imspy.simulation.handle import TimsTofSyntheticsDataHandleRust

from imspy.simulation.proteome import PeptideDigest
from imspy.simulation.acquisition import TimsTofAcquisitionBuilderDIA
from imspy.algorithm import DeepPeptideIonMobilityApex, DeepChromatographyApex
from imspy.algorithm import (load_tokenizer_from_resources, load_deep_retention_time, load_deep_ccs_predictor)

from imspy.simulation.utility import generate_events, python_list_to_json_string, \
    read_acquisition_config, flatten_prosit_array
from imspy.simulation.isotopes import generate_isotope_patterns_rust
from imspy.simulation.utility import (get_z_score_for_percentile, get_frames_numba, get_scans_numba,
                                      accumulated_intensity_cdf_numba)

from imspy.simulation.exp import TimsTofSyntheticFrameBuilderDIA
from imspy.algorithm.ionization.predictors import BinomialChargeStateDistributionModel
from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper

from pathlib import Path

# silence warnings, will spam the console otherwise
os.environ["WANDB_SILENT"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# don't use all the memory for the GPU (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for i, _ in enumerate(gpus):
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            tf.config.experimental.set_virtual_device_configuration(
                gpus[i],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)]
            )
            print(f"GPU: {i} memory restricted to 4GB.")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


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
    parser.add_argument("acquisition_type", type=str, help="Type of acquisition to simulate")
    parser.add_argument("fasta", type=str, help="Path to the fasta file")

    # Optional verbosity flag
    parser.add_argument("-v", "--verbose", type=bool, default=True, help="Increase output verbosity")

    # Peptide digestion arguments
    parser.add_argument("--sample_fraction", type=float, default=0.1, help="Sample fraction (default: 0.1)")
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

    # charge state probabilities
    parser.add_argument("--p_charge", type=float, default=0.5, help="Probability of being charged (default: 0.5)")

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    path = check_path(args.path)
    name = args.name
    fasta = check_path(args.fasta)
    verbose = args.verbose

    acquisition_type = args.acquisition_type.lower()

    assert acquisition_type in ['dia', 'midia', 'slice', 'synchro'], \
        f"Acquisition type must be 'dia', 'midia', 'slice' or 'synchro', was {args.acquisition_type}"

    assert 0.0 < args.z_score < 1.0, f"Z-score must be between 0 and 1, was {args.z_score}"

    p_charge = args.p_charge
    assert 0.0 < p_charge < 1.0, f"Probability of being charged must be between 0 and 1, was {p_charge}"

    config = read_acquisition_config(acquisition_name=acquisition_type)['acquisition']

    if verbose:
        print(f"Using acquisition type: {acquisition_type}")
        print(config)

    acquisition_builder = TimsTofAcquisitionBuilderDIA.from_config(
        path=path,
        exp_name=name,
        config=config,
    )

    if verbose:
        print("Digesting peptides...")

    digest = PeptideDigest(
        fasta,
        missed_cleavages=args.missed_cleavages,
        min_len=args.min_len,
        max_len=args.max_len,
        cleave_at=args.cleave_at,
        restrict=args.restrict,
        generate_decoys=args.decoys,
        verbose=verbose,
    )

    acquisition_builder.synthetics_handle.create_table(
        table_name='peptides',
        table=digest.peptides.sample(frac=args.sample_fraction),
    )

    if verbose:
        print("Simulating retention times...")

    # create RTColumn instance
    RTColumn = DeepChromatographyApex(
        model=load_deep_retention_time(),
        tokenizer=load_tokenizer_from_resources(),
        verbose=verbose
    )

    if verbose:
        print("Simulating retention times...")

    # predict rts
    peptide_rt = RTColumn.simulate_separation_times_pandas(
        data=acquisition_builder.synthetics_handle.get_table('peptides'),
        gradient_length=acquisition_builder.gradient_length,
    )

    if verbose:
        print("Sampling peptide intensities...")

    # call this peptide counts instead
    events = generate_events(
        n=peptide_rt.shape[0],
        mean=args.intensity_mean,
        min_val=args.intensity_min,
        max_val=args.intensity_max
    )

    peptide_rt['events'] = events

    # update peptides table in database
    acquisition_builder.synthetics_handle.create_table(
        table_name='peptides',
        table=peptide_rt,
    )

    if verbose:
        print("Simulating charge states...")

    IonSource = BinomialChargeStateDistributionModel(charged_probability=p_charge)
    peptide_ions = IonSource.simulate_charge_state_distribution_pandas(peptide_rt)

    # merge tables to have sequences with ions, remove mz values outside scope
    ions = pd.merge(left=peptide_ions, right=peptide_rt, left_on=['peptide_id'], right_on=['peptide_id'])

    # TODO: CHECK IF THIS TAKES MODIFICATIONS INTO ACCOUNT
    ions['mz'] = ions.apply(lambda r: calculate_mz(r['monoisotopic-mass'], r['charge']), axis=1)
    ions = ions[(ions.mz >= acquisition_builder.mz_lower) & (ions.mz <= acquisition_builder.mz_upper)]

    if verbose:
        print("Simulating ion mobilities...")

    # load IM predictor
    IMS = DeepPeptideIonMobilityApex(
        model=load_deep_ccs_predictor(),
        tokenizer=load_tokenizer_from_resources(),
        verbose=verbose
    )

    # simulate mobilities
    dp = IMS.simulate_ion_mobilities_pandas(
        ions
    )

    # filter by mobility range
    ions = dp[(dp.mobility_gru_predictor >= acquisition_builder.im_lower) & (
            ions.mobility_gru_predictor <= acquisition_builder.im_upper)]

    if verbose:
        print("Simulating precursor isotopic distributions...")

    specs = generate_isotope_patterns_rust(
        ions['monoisotopic-mass'], ions.charge,
        min_intensity=args.isotope_min_intensity,
        k=args.isotope_k,
        centroid=True,
        num_threads=args.num_threads,
    )

    specs = [spec.to_jsons() for spec in specs]
    ions.insert(6, 'simulated_spectrum', specs)

    acquisition_builder.synthetics_handle.create_table(
        table_name='ions',
        table=ions,
    )

    frames = acquisition_builder.frame_table
    scans = acquisition_builder.scan_table

    # distribution parameters
    z_score = get_z_score_for_percentile(target_score=args.z_score)
    std_rt = args.std_rt
    std_im = args.std_im

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
    for _, row in tqdm(peptide_rt.iterrows(), total=peptide_rt.shape[0], desc='frame distribution', ncols=100):
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
        lambda x: python_list_to_json_string(x, as_float=False))
    peptide_rt['frame_abundance'] = peptide_rt['frame_abundance'].apply(python_list_to_json_string)

    # save peptide_rt to database
    acquisition_builder.synthetics_handle.create_table(
        table_name='peptides',
        table=peptide_rt
    )

    ions = acquisition_builder.synthetics_handle.get_table('ions')

    im_scans = []
    im_contributions = []

    for _, row in tqdm(ions.iterrows(), total=ions.shape[0], desc='scan distribution', ncols=100):
        scan_occurrence, scan_abundance = [], []

        im_value = row.mobility_gru_predictor
        contributing_scans = get_scans_numba(im_value, mobility_np, scans_np, std_im, z_score)

        for scan in contributing_scans:
            im = im_dict[scan]
            start = im - acquisition_builder.im_cycle_length
            i = accumulated_intensity_cdf_numba(start, im, im_value, std_im)

            # TODO: ADD NOISE HERE AS WELL?

            scan_occurrence.append(scan)
            scan_abundance.append(i)

        im_scans.append(scan_occurrence)
        im_contributions.append(scan_abundance)

    if verbose:
        print("Saving scan distributions...")

    ions['scan_occurrence'] = [list(x) for x in im_scans]
    ions['scan_abundance'] = [list(x) for x in im_contributions]

    ions['scan_occurrence'] = ions['scan_occurrence'].apply(lambda x: python_list_to_json_string(x, as_float=False))
    ions['scan_abundance'] = ions['scan_abundance'].apply(python_list_to_json_string)

    acquisition_builder.synthetics_handle.create_table(
        table_name='ions',
        table=ions
    )

    if verbose:
        print("Simulating fragment ion intensity distributions...")

    native_path = Path(path) / name / 'synthetic_data.db'

    native_handle = TimsTofSyntheticsDataHandleRust(str(native_path))

    if verbose:
        print("Calculating precursor ion transmissions and collision energies...")

    transmitted_fragment_ions = native_handle.get_transmitted_ions(num_threads=args.num_threads)

    IntensityPredictor = Prosit2023TimsTofWrapper()

    i_pred = IntensityPredictor.simulate_ion_intensities_pandas_batched(transmitted_fragment_ions,
                                                                        batch_size_tf_ds=args.batch_size)

    if verbose:
        print("Mapping fragment ion intensity distributions to b and y ions...")

    N = int(5e4)
    batch_list = []

    for batch_indices in tqdm(
            np.array_split(i_pred.index, np.ceil(len(i_pred)/N)),
            total=int(np.ceil(len(i_pred)/N)),
            desc='matching b/y ions with intensities',
            ncols=100,
            disable=(not verbose)
    ):

        batch = i_pred.loc[batch_indices].reset_index(drop=True)

        batch['intensity_flat'] = batch.apply(lambda r: flatten_prosit_array(r.intensity), axis=1)

        all_ions = ims.sequence_to_all_ions_par(
            batch.sequence, batch.charge, batch.intensity_flat, True, True, args.num_threads
        )

        batch['fragment_intensities'] = all_ions
        batch = batch[['peptide_id', 'collision_energy', 'charge', 'fragment_intensities']]
        batch_list.append(batch)

    fragment_spectra = pd.concat(batch_list).sort_values(by=['peptide_id', 'charge'])

    if verbose:
        print("Saving fragment ion intensity distributions...")

    acquisition_builder.synthetics_handle.create_table(
        table=fragment_spectra,
        table_name='fragment_ions'
    )

    if verbose:
        print("Starting frame assembly...")

    batch_size = args.batch_size
    num_batches = len(frames) // batch_size + 1
    frame_ids = frames.frame_id.values

    frame_builder = TimsTofSyntheticFrameBuilderDIA(
        db_path=str(Path(acquisition_builder.path) / 'synthetic_data.db'),
    )

    # go over all frames in batches
    for b in tqdm(range(num_batches), total=num_batches, desc='frame assembly', ncols=100):
        start_index = b * batch_size
        stop_index = (b + 1) * batch_size
        ids = frame_ids[start_index:stop_index]

        built_frames = frame_builder.build_frames(ids, num_threads=args.num_threads)
        acquisition_builder.tdf_writer.write_frames(built_frames, scan_mode=9, num_threads=args.num_threads)

    if verbose:
        print("Writing frame meta data to database...")

    # write frame meta data to database
    acquisition_builder.tdf_writer.write_frame_meta_data()
    # write frame ms/ms info to database
    acquisition_builder.tdf_writer.write_dia_ms_ms_info(
        acquisition_builder.synthetics_handle.get_table('dia_ms_ms_info'))
    # write frame ms/ms windows to database
    acquisition_builder.tdf_writer.write_dia_ms_ms_windows(
        acquisition_builder.synthetics_handle.get_table('dia_ms_ms_windows'))


if __name__ == '__main__':
    main()
