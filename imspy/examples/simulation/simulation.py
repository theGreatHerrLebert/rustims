import os
import argparse
import time

from examples.simulation.jobs.assemble_frames import assemble_frames
from examples.simulation.jobs.build_acquisition import build_acquisition
from examples.simulation.jobs.digest_fasta import digest_fasta
from examples.simulation.jobs.simulate_charge_states import simulate_charge_states
from examples.simulation.jobs.simulate_fragment_intensities import simulate_fragment_intensities
from examples.simulation.jobs.simulate_frame_distributions import simulate_frame_distributions
from examples.simulation.jobs.simulate_ion_mobilities import simulate_ion_mobilities
from examples.simulation.jobs.simulate_precursor_spectra import simulate_precursor_spectra_sequence
from examples.simulation.jobs.simulate_retention_time import simulate_retention_times
from examples.simulation.jobs.simulate_scan_distributions import simulate_scan_distributions
from examples.simulation.utility import check_path

from imspy.simulation.utility import generate_events

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

    # Required string argument for path
    parser.add_argument("path", type=str, help="Path to save the experiment to")
    parser.add_argument("fasta", type=str, help="Path to the fasta file")

    # Optional verbosity flag
    parser.add_argument("-v", "--verbose", type=bool, default=True, help="Increase output verbosity")
    parser.add_argument("--acquisition_type", type=str, help="Type of acquisition to simulate", default='dia')
    parser.add_argument("--name", type=str, help="Name of the experiment", default=f'imsym-PLACEHOLDER-{int(time.time())}')

    # Peptide digestion arguments
    parser.add_argument("--sample_fraction", type=float, default=0.005, help="Sample fraction (default: 0.005)")
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
    parser.add_argument("--gradient_length", type=float, default=60 * 60, help="Length of the gradient (default: 3600)")
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
    name = args.name.replace('PLACEHOLDER', f'{args.acquisition_type}')
    fasta = check_path(args.fasta)
    verbose = args.verbose

    assert 0.0 < args.z_score < 1.0, f"Z-score must be between 0 and 1, was {args.z_score}"

    p_charge = args.p_charge
    assert 0.0 < p_charge < 1.0, f"Probability of being charged must be between 0 and 1, was {p_charge}"

    # create acquisition
    acquisition_builder = build_acquisition(
        path=path,
        exp_name=name,
        acquisition_type=args.acquisition_type,
        verbose=verbose,
        gradient_length=args.gradient_length,
    )

    # JOB 1: Digest the fasta file
    peptides = digest_fasta(
        fasta_file_path=fasta,
        missed_cleavages=args.missed_cleavages,
        min_len=args.min_len,
        max_len=args.max_len,
        cleave_at=args.cleave_at,
        restrict=args.restrict,
        decoys=args.decoys,
        verbose=verbose,
    ).peptides

    if args.sample_fraction < 1.0:
        peptides = peptides.sample(frac=args.sample_fraction)
        peptides.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"Simulating {peptides.shape[0]} peptides...")

    # JOB 2: Simulate retention times
    peptides = simulate_retention_times(
        peptides=peptides,
        verbose=verbose,
        gradient_length=acquisition_builder.gradient_length
    )

    if verbose:
        print("Sampling peptide intensities...")

    # JOB 3: Simulate peptide intensities
    peptides['events'] = generate_events(
        n=peptides.shape[0],
        mean=args.intensity_mean,
        min_val=args.intensity_min,
        max_val=args.intensity_max
    )

    # JOB 4: Simulate frame distributions
    peptides = simulate_frame_distributions(
        peptides=peptides,
        frames=acquisition_builder.frame_table,
        z_score=args.z_score,
        std_rt=args.std_rt,
        rt_cycle_length=acquisition_builder.rt_cycle_length,
        verbose=verbose
    )

    # save peptides to database
    acquisition_builder.synthetics_handle.create_table(
        table_name='peptides',
        table=peptides,
    )

    # JOB 5: Simulate charge states
    ions = simulate_charge_states(
        peptides=peptides,
        mz_lower=acquisition_builder.mz_lower,
        mz_upper=acquisition_builder.mz_upper,
        p_charge=p_charge
    )

    # JOB 6: Simulate ion mobilities
    ions = simulate_ion_mobilities(
        ions=ions,
        im_lower=acquisition_builder.im_lower,
        im_upper=acquisition_builder.im_upper,
        verbose=verbose
    )

    # JOB 7: Simulate precursor isotopic distributions
    ions = simulate_precursor_spectra_sequence(
        ions=ions,
        num_threads=args.num_threads,
        verbose=verbose
    )

    # JOB 8: Simulate scan distributions
    ions = simulate_scan_distributions(
        ions=ions,
        scans=acquisition_builder.scan_table,
        z_score=args.z_score,
        std_im=args.std_im,
        im_cycle_length=acquisition_builder.im_cycle_length,
        verbose=verbose
    )

    acquisition_builder.synthetics_handle.create_table(
        table_name='ions',
        table=ions
    )

    # JOB 9: Simulate fragment ion intensities
    simulate_fragment_intensities(
        path=path,
        name=name,
        acquisition_builder=acquisition_builder,
        batch_size=args.batch_size,
        verbose=verbose,
        num_threads=args.num_threads,
    )

    # JOB 10: Assemble frames
    assemble_frames(
        acquisition_builder=acquisition_builder,
        frames=acquisition_builder.frame_table,
        batch_size=args.batch_size,
        verbose=verbose,
        num_threads=args.num_threads,
    )


if __name__ == '__main__':
    main()
