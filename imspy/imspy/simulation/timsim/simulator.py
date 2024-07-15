import os
import argparse
import time
import pandas as pd

from imspy.simulation.utility import get_fasta_file_paths, get_dilution_factors
from .jobs.assemble_frames import assemble_frames
from .jobs.build_acquisition import build_acquisition
from .jobs.digest_fasta import digest_fasta
from .jobs.simulate_charge_states import simulate_charge_states
from .jobs.simulate_fragment_intensities import simulate_fragment_intensities
from .jobs.simulate_frame_distributions_emg import simulate_frame_distributions_emg
from .jobs.simulate_ion_mobilities import simulate_ion_mobilities
from .jobs.simulate_precursor_spectra import simulate_precursor_spectra_sequence
from .jobs.simulate_retention_time import simulate_retention_times
from .jobs.simulate_scan_distributions import simulate_scan_distributions
from .jobs.simulate_occurrences import simulate_peptide_occurrences
from imspy.simulation.timsim.jobs.utility import check_path

from tabulate import tabulate

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
    parser = argparse.ArgumentParser(description='🦀💻 TIMSIM 🔬🐍 - Run a proteomics experiment simulation '
                                                 'with diaPASEF-like acquisition on a BRUKER TimsTOF.')

    # Required string argument for path
    parser.add_argument("path", type=str, help="Path to save the experiment to")
    parser.add_argument("reference_path", type=str, help="Path to a real TDF reference dataset")
    parser.add_argument("fasta", type=str, help="Path to the fasta file of proteins to be digested")

    parser.add_argument("--reference_in_memory", dest="reference_in_memory", action="store_true",
                        help="Whether to load the reference dataset into memory (default: False)")
    parser.set_defaults(reference_in_memory=False)

    # Optional verbosity flag
    parser.add_argument("-s", "--silent", dest="verbose", action="store_false",
                        help="Silence output (default: False)")
    parser.set_defaults(verbose=True)

    parser.add_argument("-acq", "--acquisition_type",
                        type=str,
                        help="Type of acquisition to simulate, choose between: [DIA, SYNCHRO, SLICE, MIDIA], default: DIA",
                        default='DIA')

    parser.add_argument("-n", "--name", type=str, help="Name of the experiment",
                        default=f'TIMSIM-[PLACEHOLDER]-{int(time.time())}')

    parser.add_argument("--no_reference_layout", dest="use_reference_layout", action="store_false",
                        help="Use the layout of the reference dataset for the acquisition (default: True)")
    parser.set_defaults(use_reference_layout=True)

    parser.add_argument("--no_sample_peptides", dest="sample_peptides", action="store_false",
                        help="Sample peptides from the digested fasta (default: True)")
    parser.set_defaults(sample_peptides=True)

    # Peptide digestion arguments
    parser.add_argument(
        "--num_sample_peptides",
        type=int,
        default=25_000,
        help="Sample fraction, fraction of peptides to be sampled at random from digested fasta (default: 0.005)")

    parser.add_argument("--missed_cleavages", type=int, default=2, help="Number of missed cleavages (default: 2)")
    parser.add_argument("--min_len", type=int, default=8, help="Minimum peptide length (default: 7)")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum peptide length (default: 30)")
    parser.add_argument("--cleave_at", type=str, default='KR', help="Cleave at (default: KR)")
    parser.add_argument("--restrict", type=str, default='P', help="Restrict (default: P)")
    parser.add_argument("--add_decoys", dest="decoys", action="store_true",
                        help="Generate decoys (default: False)")
    parser.set_defaults(decoys=False)

    # Peptide intensities
    parser.add_argument("--intensity_mean", type=float, default=1e7, help="Mean peptide intensity (default: 1e6)")
    parser.add_argument("--intensity_min", type=float, default=1e5, help="Std peptide intensity (default: 1e5)")
    parser.add_argument("--intensity_max", type=float, default=1e9, help="Min peptide intensity (default: 1e9)")

    # Precursor isotopic pattern settings
    parser.add_argument("--isotope_k", type=int, default=8, help="Number of isotopes to simulate (default: 8)")
    parser.add_argument("--isotope_min_intensity", type=int, default=1, help="Min intensity for isotopes (default: 1)")
    parser.add_argument("--no_isotope_centroid", dest="isotope_centroid", action="store_false",
                        help="Centroid isotopes (default: True)")
    parser.set_defaults(isotope_centroid=True)

    # Sample occurrences parameters
    parser.add_argument("--no_sample_occurrences",
                        dest="sample_occurrences", action="store_false",
                        help="Whether or not sample peptide occurrences should be assigned randomly (default: True)")
    parser.set_defaults(sample_occurrences=True)
    parser.add_argument(
        "--intensity_value",
        type=float, default=1e6,
        help="Intensity value of all peptides if sample occurrence sampling is deactivated (default: 1e6)")

    # Distribution parameters
    parser.add_argument(
        "--gradient_length",
        type=float,
        default=60 * 60,
        help="Length of the gradient in seconds (default: 3600)")
    parser.add_argument("--z_score", type=float, default=.99,
                        help="Z-score for frame and scan distributions (default: .99)")
    parser.add_argument("--mean_std_rt", type=float, default=1.5,
                        help="Mean standard deviation for retention time distribution (default: 1.5)")
    parser.add_argument("--variance_std_rt", type=float, default=0.3,
                        help="Variance standard deviation for retention time distribution (default: 0.3)")
    parser.add_argument("--mean_scewness", type=float, default=0.3,
                        help="Mean scewness for retention time distribution (default: 0.3)")
    parser.add_argument("--variance_scewness", type=float, default=0.1,
                        help="Variance scewness for retention time distribution (default: 0.1)")
    parser.add_argument("--std_im", type=float, default=0.01,
                        help="Standard deviation for mobility distribution (default: 0.01)")
    parser.add_argument("--target_p", type=float, default=0.999,
                        help="Target percentile for frame distributions (default: 0.999)")
    parser.add_argument("--sampling_step_size", type=float, default=0.001,
                        help="Sampling step size for frame distributions (default: 0.001)")

    # Number of cores to use
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads to use (default: 16)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256)")

    # charge state probabilities
    parser.add_argument("--p_charge", type=float, default=0.5, help="Probability of being charged (default: 0.5)")

    # Noise settings
    # -- 1. RT and IM noise
    parser.add_argument(
        "--add_noise_to_signals", dest="add_noise_to_signals", action="store_true",
        help="Add noise to ion distributions in retention time and ion mobility (default: False)")
    parser.set_defaults(add_noise_to_signals=False)

    # -- 2. MZ noise precursor
    parser.add_argument(
        "--mz_noise_precursor", dest="mz_noise_precursor", action="store_true",
        help="Add noise to precursor m/z (default: False)"
    )
    parser.set_defaults(mz_noise_precursor=False)

    parser.add_argument(
        "--precursor_noise_ppm",
        type=float,
        default=5.0,
        help="Precursor noise in ppm (default: 5.0)"
    )

    # -- 3. MZ noise fragment
    parser.add_argument(
        "--mz_noise_fragment", dest="mz_noise_fragment", action="store_true",
        help="Add noise to fragment m/z (default: False)"
    )
    parser.set_defaults(mz_noise_fragment=False)

    parser.add_argument(
        "--fragment_noise_ppm",
        type=float,
        default=5.0,
        help="Fragment noise in ppm (default: 5.0)"
    )
    parser.add_argument(
        "--mz_noise_uniform", dest="mz_noise_uniform", action="store_true",
        help="Use uniform distribution for m/z noise (default: False), otherwise normal distribution"
    )
    parser.set_defaults(mz_noise_uniform=False)

    parser.add_argument(
        "--add_real_data_noise", dest="add_real_data_noise", action="store_true",
        help="Use given reference data to add noise to the simulated data (default: False)"
    )
    parser.set_defaults(add_real_data_noise=False)

    parser.add_argument(
        "--reference_noise_intensity_max",
        type=float,
        default=30,
        help="Maximum intensity for noise reference data (default: 30)"
    )

    parser.add_argument(
        "--down_sample_factor",
        type=float,
        default=0.5,
        help="Down sample fragment peaks generated, sampling probability "
             "is inverse proportional to intensity (default: 0.5)"
    )

    # Proteome mixture settings
    parser.add_argument(
        "--proteome_mix",
        action="store_true",
        dest="proteome_mix",
    )
    parser.set_defaults(proteome_mixture=False)

    # Parse the arguments
    args = parser.parse_args()

    # Convert arguments to a dictionary
    args_dict = vars(args)

    # Convert dictionary to a list of lists for tabulate
    table = [[key, value] for key, value in args_dict.items()]

    # Print table
    print(tabulate(table, headers=["Argument", "Value"], tablefmt="grid"))

    # Save the arguments to a file
    with open(os.path.join(args.path, 'arguments.txt'), 'w') as f:
        f.write(tabulate(table, headers=["Argument", "Value"], tablefmt="grid"))

    # Use the arguments
    path = check_path(args.path)
    reference_path = check_path(args.reference_path)
    name = args.name.replace('[PLACEHOLDER]', f'{args.acquisition_type}').replace("'", "")

    if args.proteome_mix:
        factors = get_dilution_factors()

    fastas = get_fasta_file_paths(args.fasta)

    verbose = args.verbose

    assert 0.0 < args.z_score < 1.0, f"Z-score must be between 0 and 1, was {args.z_score}"

    p_charge = args.p_charge
    assert 0.0 < p_charge < 1.0, f"Probability of being charged must be between 0 and 1, was {p_charge}"

    if verbose:
        print(f"Simulating experiment {name} at {path}...")

    # create acquisition
    acquisition_builder = build_acquisition(
        path=path,
        reference_path=reference_path,
        exp_name=name,
        acquisition_type=args.acquisition_type,
        verbose=verbose,
        gradient_length=args.gradient_length,
        use_reference_ds_layout=args.use_reference_layout,
        reference_in_memory=args.reference_in_memory,
    )

    if verbose:
        print(acquisition_builder)

    peptide_list = []

    for fasta_name, fasta in fastas.items():
        if verbose:
            print(f"Digesting fasta file: {fasta_name}...")

        mixture_factor = 1.0

        if args.proteome_mix:
            try:
                mixture_factor = factors[fasta_name]

                if verbose:
                    print(f"Using mixture factor {mixture_factor} for {fasta_name}")

            except KeyError:
                # print warning and set mixture factor to 1.0
                print(f"Warning: No mixture factor found for {fasta_name}, setting to 1.0")

        # JOB 1: Digest the fasta file(s)
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

        # JOB 2: Simulate peptide occurrences
        peptides = simulate_peptide_occurrences(
            peptides=peptides,
            intensity_mean=args.intensity_mean,
            intensity_min=args.intensity_min,
            intensity_max=args.intensity_max,
            verbose=verbose,
            sample_occurrences=args.sample_occurrences,
            intensity_value=args.intensity_value,
            mixture_contribution=mixture_factor,
        )

        peptide_list.append(peptides)

    peptides = pd.concat(peptide_list)

    if args.sample_peptides:
        peptides = peptides.sample(n=args.num_sample_peptides, random_state=41)
        peptides.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"Simulating {peptides.shape[0]} peptides...")

    # JOB 3: Simulate retention times
    peptides = simulate_retention_times(
        peptides=peptides,
        verbose=verbose,
        gradient_length=acquisition_builder.gradient_length
    )

    # TODO: Need to fix this on the backend,
    #  the columns need to be in the correct order but generation of occurrences and retention times was swapped
    #  to account for multiple fasta files and potential mixture factors of the peptides
    columns = list(peptides.columns)
    columns[-2], columns[-1] = columns[-1], columns[-2]
    peptides = peptides[columns]

    # JOB 4: Simulate frame distributions emg
    peptides = simulate_frame_distributions_emg(
        peptides=peptides,
        frames=acquisition_builder.frame_table,
        mean_std_rt=args.mean_std_rt,
        variance_std_rt=args.variance_std_rt,
        mean_scewness=args.mean_scewness,
        variance_scewness=args.variance_scewness,
        rt_cycle_length=acquisition_builder.rt_cycle_length,
        target_p=args.target_p,
        step_size=args.sampling_step_size,
        verbose=verbose,
        add_noise=args.add_noise_to_signals,
        num_threads=args.num_threads
    )

    # save peptides to database
    acquisition_builder.synthetics_handle.create_table(
        table_name='peptides',
        table=peptides,
    )

    # JOB 5: Simulate charge states
    ions = simulate_charge_states(
        peptides=peptides,
        mz_lower=acquisition_builder.tdf_writer.helper_handle.mz_lower,
        mz_upper=acquisition_builder.tdf_writer.helper_handle.mz_upper,
        p_charge=p_charge
    )

    # JOB 6: Simulate ion mobilities
    ions = simulate_ion_mobilities(
        ions=ions,
        im_lower=acquisition_builder.tdf_writer.helper_handle.im_lower,
        im_upper=acquisition_builder.tdf_writer.helper_handle.im_upper,
        verbose=verbose
    )

    # JOB 7: Simulate precursor isotopic distributions
    ions = simulate_precursor_spectra_sequence(
        ions=ions,
        num_threads=args.num_threads,
        verbose=verbose
    )

    # JOB 8: Simulate scan distributions
    # TODO: sample standard deviation of ion mobility from a distribution (e.g., normal?)
    ions = simulate_scan_distributions(
        ions=ions,
        scans=acquisition_builder.scan_table,
        z_score=args.z_score,
        std_im=args.std_im,
        verbose=verbose,
        add_noise=args.add_noise_to_signals
    )

    ion_id = ions.index
    ions.insert(0, 'ion_id', ion_id)

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
        down_sample_factor=args.down_sample_factor,
    )

    # JOB 10: Assemble frames
    assemble_frames(
        acquisition_builder=acquisition_builder,
        frames=acquisition_builder.frame_table,
        batch_size=args.batch_size,
        verbose=verbose,
        mz_noise_precursor=args.mz_noise_precursor,
        mz_noise_uniform=args.mz_noise_uniform,
        precursor_noise_ppm=args.precursor_noise_ppm,
        mz_noise_fragment=args.mz_noise_fragment,
        fragment_noise_ppm=args.fragment_noise_ppm,
        num_threads=args.num_threads,
        add_real_data_noise=args.add_real_data_noise,
        reference_noise_intensity_max=args.reference_noise_intensity_max,
    )


if __name__ == '__main__':
    main()
