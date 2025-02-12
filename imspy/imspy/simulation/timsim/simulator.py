import os
import argparse
import time
import pandas as pd

import toml
from pathlib import Path

from imspy.simulation.experiment import SyntheticExperimentDataHandleDIA
from imspy.simulation.timsim.jobs.simulate_ion_mobilities_and_variance import simulate_ion_mobilities_and_variance
from imspy.simulation.timsim.jobs.simulate_peptides import simulate_peptides
from imspy.simulation.timsim.jobs.simulate_phosphorylation import simulate_phosphorylation
from imspy.simulation.timsim.jobs.simulate_proteins import simulate_proteins
from imspy.simulation.timsim.jobs.simulate_scan_distributions_with_variance import \
    simulate_scan_distributions_with_variance
from imspy.simulation.utility import get_fasta_file_paths, get_dilution_factors
from .jobs.assemble_frames import assemble_frames
from .jobs.build_acquisition import build_acquisition

from .jobs.simulate_charge_states import simulate_charge_states
from .jobs.simulate_fragment_intensities import simulate_fragment_intensities
from .jobs.simulate_frame_distributions_emg import simulate_frame_distributions_emg
from .jobs.simulate_ion_mobilities import simulate_ion_mobilities
from .jobs.simulate_precursor_spectra import simulate_precursor_spectra_sequence
from .jobs.simulate_retention_time import simulate_retention_times
from .jobs.simulate_scan_distributions import simulate_scan_distributions

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

# helper function to load configuration of modifications from a TOML file
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = toml.load(config_file)
    return config

def main():
    # Use ArgumentDefaultsHelpFormatter to automatically show default values
    parser = argparse.ArgumentParser(
        description="🦀💻 TIMSIM 🔬🐍 - Run a proteomics experiment simulation with diaPASEF-like acquisition on a BRUKER TimsTOF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --------------------------
    # Required arguments
    # --------------------------
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("save_path", type=str, nargs='?', default=None,
                          help="Path to save the experiment to")
    required.add_argument("reference_path", type=str, nargs='?', default=None,
                          help="Path to a real TDF reference dataset")
    required.add_argument("fasta_path", type=str, nargs='?', default=None,
                          help="Path to the fasta file of proteins to be digested")

    # --------------------------
    # Configuration files
    # --------------------------
    config = parser.add_argument_group("Configuration")
    config.add_argument("--config", type=str, default=None,
                        help="Path to configuration file (TOML format)")
    script_dir = Path(__file__).parent
    default_mods_config_path = script_dir / "configs" / "modifications.toml"
    config.add_argument("--modifications", type=str, default=str(default_mods_config_path),
                        help="Path to modifications configuration file (TOML format)")

    # --------------------------
    # General simulation options
    # --------------------------
    general = parser.add_argument_group("General Simulation Options")
    general.add_argument("--reference_in_memory", action="store_true",
                         help="Load the reference dataset into memory")
    general.add_argument("-s", "--silent_mode", action="store_true",
                         help="Silence output")
    general.add_argument("--acquisition_type", type=str, default="DIA",
                         choices=["DIA", "SYNCHRO", "SLICE", "MIDIA"],
                         help="Type of acquisition to simulate")
    general.add_argument("-n", "--experiment_name", type=str, default="",
                         help="Name of the experiment")
    general.add_argument("--no_reference_layout", dest="use_reference_layout", action="store_false",
                         help="Do not use the reference dataset’s layout for acquisition")
    general.add_argument("--no_peptide_sampling", dest="sample_peptides", action="store_false",
                         help="Do not sample peptides from the digested fasta")
    general.add_argument("--sample_seed", type=int, default=41,
                         help="Seed for peptide sampling")
    general.add_argument("--apply_fragmentation", action="store_true",
                         help="Apply fragmentation (default: off)")

    # --------------------------
    # Peptide digestion options
    # --------------------------
    digestion = parser.add_argument_group("Peptide Digestion Options")
    digestion.add_argument("--num_sample_peptides", type=int, default=25000,
                           help="Number of peptides to sample from the digested fasta")
    digestion.add_argument("--missed_cleavages", type=int, default=2,
                           help="Number of missed cleavages")
    digestion.add_argument("--min_len", type=int, default=7,
                           help="Minimum peptide length")
    digestion.add_argument("--max_len", type=int, default=30,
                           help="Maximum peptide length")
    digestion.add_argument("--cleave_at", type=str, default="KR",
                           help="Residues to cleave at")
    digestion.add_argument("--restrict", type=str, default="P",
                           help="Residue to restrict cleavage")
    digestion.add_argument("--decoys", action="store_true",
                           help="Generate decoy peptides")

    # --------------------------
    # Peptide intensity settings
    # --------------------------
    intensity = parser.add_argument_group("Peptide Intensity Settings")
    intensity.add_argument("--intensity_mean", type=float, default=1e7,
                           help="Mean peptide intensity")
    intensity.add_argument("--intensity_min", type=float, default=1e5,
                           help="Minimum peptide intensity")
    intensity.add_argument("--intensity_max", type=float, default=1e9,
                           help="Maximum peptide intensity")

    # --------------------------
    # Precursor isotopic pattern settings
    # --------------------------
    isotopes = parser.add_argument_group("Precursor Isotopic Pattern Settings")
    isotopes.add_argument("--isotope_k", type=int, default=8,
                          help="Number of isotopes to simulate")
    isotopes.add_argument("--isotope_min_intensity", type=int, default=1,
                          help="Minimum intensity for isotopes")
    isotopes.add_argument("--no_isotope_centroid", dest="isotope_centroid", action="store_false",
                          help="Do not centroid isotopes (default: on)")

    # --------------------------
    # Peptide occurrence sampling
    # --------------------------
    sampling = parser.add_argument_group("Peptide Occurrence Sampling")
    sampling.add_argument("--no_sample_occurrences", dest="sample_occurrences", action="store_false",
                          help="Do not randomly assign peptide occurrences")
    sampling.add_argument("--intensity_value", type=float, default=1e6,
                          help="Uniform intensity value if occurrence sampling is disabled")

    # --------------------------
    # Distribution parameters
    # --------------------------
    distribution = parser.add_argument_group("Distribution Parameters")
    distribution.add_argument("--gradient_length", type=float, default=3600,
                              help="Length of the gradient in seconds")
    distribution.add_argument("--z_score", type=float, default=0.99,
                              help="Z-score for frame and scan distributions")
    distribution.add_argument("--mean_std_rt", type=float, default=1.5,
                              help="Mean standard deviation for retention time distribution")
    distribution.add_argument("--variance_std_rt", type=float, default=0.3,
                              help="Variance of the retention time standard deviation")
    distribution.add_argument("--mean_skewness", type=float, default=0.3,
                              help="Mean skewness for retention time distribution")
    distribution.add_argument("--variance_skewness", type=float, default=0.1,
                              help="Variance of the retention time skewness")
    distribution.add_argument("--target_p", type=float, default=0.999,
                              help="Target percentile for frame and scan distributions")
    distribution.add_argument("--sampling_step_size", type=float, default=0.001,
                              help="Sampling step size for frame distributions")

    # --------------------------
    # Threading and batch options
    # --------------------------
    threading = parser.add_argument_group("Threading and Batch Options")
    threading.add_argument("--num_threads", type=int, default=-1,
                           help="Number of threads to use (-1 means all available)")
    threading.add_argument("--batch_size", type=int, default=256,
                           help="Batch size")

    # --------------------------
    # Charge state settings
    # --------------------------
    charge = parser.add_argument_group("Charge State Settings")
    charge.add_argument("--p_charge", type=float, default=0.5,
                        help="Probability that a peptide carries a charge (for binomial modeling)")
    charge.add_argument("--min_charge_contrib", type=float, default=0.25,
                        help="Minimum charge contribution")

    # --------------------------
    # Noise settings
    # --------------------------
    noise = parser.add_argument_group("Noise Settings")
    noise.add_argument("--add_noise_to_signals", action="store_true",
                       help="Add noise to ion distributions in retention time and ion mobility")
    noise.add_argument("--mz_noise_precursor", action="store_true",
                       help="Add noise to precursor m/z values")
    noise.add_argument("--precursor_noise_ppm", type=float, default=5.0,
                       help="Precursor noise (ppm)")
    noise.add_argument("--mz_noise_fragment", action="store_true",
                       help="Add noise to fragment m/z values")
    noise.add_argument("--fragment_noise_ppm", type=float, default=5.0,
                       help="Fragment noise (ppm)")
    noise.add_argument("--mz_noise_uniform", action="store_true",
                       help="Use a uniform distribution for m/z noise (instead of normal)")
    noise.add_argument("--add_real_data_noise", action="store_true",
                       help="Add noise to simulated data based on reference data")
    noise.add_argument("--reference_noise_intensity_max", type=float, default=30,
                       help="Maximum intensity for noise reference data")
    noise.add_argument("--down_sample_factor", type=float, default=0.5,
                       help="Down-sample fragment peaks (sampling probability inversely proportional to intensity)")

    # --------------------------
    # Proteome mixture options
    # --------------------------
    proteome = parser.add_argument_group("Proteome Mixture Options")
    proteome.add_argument("--proteome_mix", action="store_true",
                          help="Enable proteome mixture simulation")
    proteome.add_argument("--multi_fasta_dilution", type=str,
                          help="Path to CSV file with dilution factors for the proteome mixture")

    # --------------------------
    # Debug and additional options
    # --------------------------
    debug = parser.add_argument_group("Debug and Additional Options")
    debug.add_argument("--debug_mode", action="store_true",
                       help="Enable debug mode")
    debug.add_argument("--from_existing", action="store_true",
                       help="Use an existing simulation to derive frame distributions")
    debug.add_argument("--existing_path", type=str,
                       help="Path to an existing simulation")
    debug.add_argument("--no_bruker_sdk", dest="use_bruker_sdk", action="store_false",
                       help="Do not use the Bruker SDK")
    debug.add_argument("--phospho_mode", action="store_true",
                       help="Enable phospho mode to generate a phospho-enriched dataset for testing")

    # Default configuration values
    defaults = {
        'reference_in_memory': False,
        'silent_mode': False,
        'acquisition_type': 'DIA',
        'experiment_name': f'TIMSIM-[PLACEHOLDER]-{int(time.time())}',
        'use_reference_layout': True,
        'sample_peptides': True,
        'sample_seed': 41,
        'apply_fragmentation': True,
        'num_sample_peptides': 25000,
        'missed_cleavages': 2,
        'min_len': 7,
        'max_len': 30,
        'cleave_at': 'KR',
        'restrict': 'P',
        'decoys': False,
        'modifications': None,
        'intensity_mean': 1e7,
        'intensity_min': 1e5,
        'intensity_max': 1e9,
        'isotope_k': 8,
        'isotope_min_intensity': 1,
        'isotope_centroid': True,
        'sample_occurrences': True,
        'intensity_value': 1e6,
        'gradient_length': 3600,
        'z_score': 0.99,
        'mean_std_rt': 1.5,
        'variance_std_rt': 0.3,
        'mean_skewness': 0.3,
        'variance_skewness': 0.1,
        'target_p': 0.999,
        'sampling_step_size': 0.001,
        'num_threads': -1,
        'batch_size': 256,
        'p_charge': 0.5,
        'min_charge_contrib': 0.25,
        'add_noise_to_signals': False,
        'mz_noise_precursor': False,
        'precursor_noise_ppm': 5.0,
        'mz_noise_fragment': False,
        'fragment_noise_ppm': 5.0,
        'mz_noise_uniform': False,
        'add_real_data_noise': False,
        'reference_noise_intensity_max': 30,
        'down_sample_factor': 0.5,
        'proteome_mix': False,
        'multi_fasta_dilution': None,
        'debug_mode': False,
        'from_existing': False,
        'existing_path': None,
        'use_bruker_sdk': True,
        'phospho_mode': False,
    }

    # Parse known arguments to get config file path
    args, remaining_args = parser.parse_known_args()

    # Load configuration from file if provided
    if args.config:
        config = load_config(args.config)
        for section in config:
            params = config[section]
            defaults.update(params)
        # Update defaults with positional arguments if they exist in config
        defaults['save_path'] = config.get('save_path', defaults.get('save_path'))
        defaults['reference_path'] = config.get('reference_path', defaults.get('reference_path'))
        defaults['fasta_path'] = config.get('fasta_path', defaults.get('fasta_path'))

    # Set defaults in parser before parsing arguments
    parser.set_defaults(**defaults)

    # Now parse all arguments with updated defaults
    args = parser.parse_args()

    # Ensure required arguments are present
    if args.save_path is None:
        parser.error("the following argument is required: save_path")
    if args.reference_path is None:
        parser.error("the following argument is required: reference_path")
    if args.fasta_path is None:
        parser.error("the following argument is required: fasta_path")

    # Load the modifications configuration
    if not args.modifications or args.modifications == "":
        # Path to the script directory
        script_dir = Path(__file__).parent
        # Default configs modification configs path
        args.modifications = script_dir / "configs" / "modifications.toml"

    mod_config = load_config(args.modifications)

    # Access the modifications from the configs
    variable_modifications = mod_config.get('variable_modifications', {})
    static_modifications = mod_config.get('static_modifications', {})

    if not args.silent_mode:
        print(f"Using variable modifications: {variable_modifications}")
        print(f"Using static modifications: {static_modifications}")
        if args.apply_fragmentation:
            print("Fragmentation is enabled.")
        else:
            print("Fragmentation is disabled.")

    # Check if current system the simulator runs on is MacOS
    if os.uname().sysname == 'Darwin':
        print("Warning: Using Bruker SDK on MacOS is not supported, setting use_bruker_sdk to False.")
        args.use_bruker_sdk = False

    # Convert arguments to a dictionary
    args_dict = vars(args)

    rt_sigma = None
    rt_lambda = None

    if args.from_existing:
        existing_sim_handle = SyntheticExperimentDataHandleDIA(database_path=args.existing_path)
        peptides = existing_sim_handle.get_table('peptides')
        proteins = existing_sim_handle.get_table('proteins')
        ions = existing_sim_handle.get_table('ions')
        rt_sigma = peptides['rt_sigma'].values
        rt_lambda = peptides['rt_lambda'].values

        # Warn if the absolute difference between the gradient length of the existing simulation and the new one is off by more than 5 percent
        rt_max = peptides['retention_time_gru_predictor'].max()

        if abs(rt_max - args.gradient_length) / args.gradient_length > 0.05:
            print(f"Warning: The gradient length of the existing simulation is {rt_max} seconds, "
                  f"which is off by more than 5 percent of the new gradient length of {args.gradient_length} seconds. "
                  f"This might result in distorted frame distributions and missing ions.")

        if not args.silent_mode:
            print(f"Using existing simulation from {args.existing_path}")
            print(f"Peptides: {peptides.shape[0]}")
            print(f"Ions: {ions.shape[0]}")

    # Convert dictionary to a list of lists for tabulate
    table = [[key, value] for key, value in args_dict.items()]

    # Print table
    if args.debug_mode:
        print(tabulate(table, headers=["Argument", "Value"], tablefmt="grid"))

    # Use the arguments
    path = check_path(args.save_path)
    reference_path = check_path(args.reference_path)
    name = args.experiment_name.replace('[PLACEHOLDER]', f'{args.acquisition_type}').replace("'", "")

    # Save the arguments to a file, should go into the database folder
    with open(os.path.join(path, f'arguments-{name}.txt'), 'w') as f:
        f.write(tabulate(table, headers=["Argument", "Value"], tablefmt="grid"))

    if args.proteome_mix:
        factors = get_dilution_factors(args.multi_fasta_dilution)

    fastas = get_fasta_file_paths(args.fasta_path)

    assert 0.0 < args.z_score < 1.0, f"Z-score must be between 0 and 1, was {args.z_score}"

    p_charge = args.p_charge
    assert 0.0 < p_charge < 1.0, f"Probability of being charged must be between 0 and 1, was {p_charge}"

    if not args.silent_mode:
        print(f"Simulating experiment {name} at {path}...")

    # Create acquisition
    acquisition_builder = build_acquisition(
        path=path,
        reference_path=reference_path,
        exp_name=name,
        acquisition_type=args.acquisition_type,
        verbose=not args.silent_mode,
        gradient_length=args.gradient_length,
        use_reference_ds_layout=args.use_reference_layout,
        reference_in_memory=args.reference_in_memory,
    )

    if not args.silent_mode:
        print(acquisition_builder)

    protein_list, peptide_list = [], []

    for fasta_name, fasta in fastas.items():
        if not args.silent_mode and not args.from_existing:
            print(f"Digesting fasta file: {fasta_name}...")

        mixture_factor = 1.0

        if args.proteome_mix:
            try:
                mixture_factor = factors[fasta_name]

                if not args.silent_mode and not args.from_existing:
                    print(f"Using mixture factor {mixture_factor} for {fasta_name}")

            except KeyError:
                # Print warning and set mixture factor to 1.0
                print(f"Warning: No mixture factor found for {fasta_name}, setting to 1.0")

        if not args.from_existing:
            # JOB 0: Generate Protein Data
            proteins = simulate_proteins(
                fasta_file_path=fasta,
                n_proteins=20_000,
                cleave_at=args.cleave_at,
                restrict=args.restrict,
                missed_cleavages=args.missed_cleavages,
                min_len=args.min_len,
                max_len=args.max_len,
                generate_decoys=args.decoys,
                variable_mods=variable_modifications,
                static_mods=static_modifications,
                verbose=not args.silent_mode,
            )

            if not args.silent_mode:
                print("Creating Peptides from Proteins...")

            # JOB 1: Simulate peptides
            peptides = simulate_peptides(
                protein_table=proteins,
                num_peptides_total=200_000,
                verbose=not args.silent_mode,
                exclude_accumulated_gradient_start=True,
                min_rt_percent=2.0,
                gradient_length=acquisition_builder.gradient_length,
                down_sample=args.sample_peptides,
            )

            # If the proteome is mixed, scale the number of peptides
            if args.proteome_mix:
                peptides['events'] = peptides['events'] * mixture_factor

            protein_list.append(proteins)
            peptide_list.append(peptides)

        if not args.from_existing:
            proteins = pd.concat(protein_list)
            proteins = proteins[["protein_id", "protein", "sequence", "events"]]
            peptides = pd.concat(peptide_list)

        if args.phospho_mode:

            if not args.silent_mode:
                print("Simulating phosphorylation...")

            peptides = simulate_phosphorylation(peptides=peptides,
                                                pick_phospho_sites=2,
                                                template=not args.from_existing, verbose=not args.silent_mode)

    if args.sample_peptides and not args.from_existing:
        try:
            peptides = peptides.sample(n=args.num_sample_peptides, random_state=args.sample_seed)
            peptides.reset_index(drop=True, inplace=True)
        except ValueError:
            print(f"Warning: Not enough peptides to sample {args.num_sample_peptides}, "
                  f"using all {peptides.shape[0]} peptides.")

    if not args.silent_mode and not args.from_existing:
        print(f"Simulating {peptides.shape[0]} peptides...")

    if not args.from_existing:
        # JOB 3: Simulate retention times
        peptides = simulate_retention_times(
            peptides=peptides,
            verbose=not args.silent_mode,
            gradient_length=acquisition_builder.gradient_length,
        )

    if not args.from_existing:
        # TODO: Need to fix this on the backend,
        #  the columns need to be in the correct order but generation of occurrences and retention times was swapped
        #  to account for multiple fasta files and potential mixture factors of the peptides
        columns = list(peptides.columns)
        columns[-2], columns[-1] = columns[-1], columns[-2]
        peptides = peptides[columns]

    # Get the number of available threads of the system if not specified
    if args.num_threads == -1:
        args.num_threads = os.cpu_count()

    # JOB 4: Simulate frame distributions emg
    peptides = simulate_frame_distributions_emg(
        peptides=peptides,
        frames=acquisition_builder.frame_table,
        mean_std_rt=args.mean_std_rt,
        variance_std_rt=args.variance_std_rt,
        mean_scewness=args.mean_skewness,
        variance_scewness=args.variance_skewness,
        rt_cycle_length=acquisition_builder.rt_cycle_length,
        target_p=args.target_p,
        step_size=args.sampling_step_size,
        verbose=not args.silent_mode,
        add_noise=args.add_noise_to_signals,
        num_threads=args.num_threads,
        from_existing=args.from_existing,
        sigmas=rt_sigma,
        lambdas=rt_lambda,
    )

    # Save proteins to database
    acquisition_builder.synthetics_handle.create_table(
        table_name='proteins',
        table=proteins,
    )

    # if phospho mode is enabled, we need to re-order the columns to avoid issues with the rust read-out of the database
    # TODO: this is a temporary fix, the rust code should be able to handle the columns in any order
    if args.phospho_mode:
        columns_order = [
            'protein_id', 'peptide_id', 'sequence', 'protein', 'decoy',
            'missed_cleavages', 'n_term', 'c_term', 'monoisotopic-mass',
            'retention_time_gru_predictor', 'events', 'rt_sigma', 'rt_lambda',
            'frame_occurrence_start', 'frame_occurrence_end', 'frame_occurrence',
            'frame_abundance', 'phospho_site_a', 'phospho_site_b', 'sequence_original'
        ]
        peptides = peptides[columns_order]

    # Save peptides to database
    acquisition_builder.synthetics_handle.create_table(
        table_name='peptides',
        table=peptides,
    )

    if not args.from_existing:
        # JOB 5: Simulate charge states
        ions = simulate_charge_states(
            peptides=peptides,
            mz_lower=acquisition_builder.tdf_writer.helper_handle.mz_lower,
            mz_upper=acquisition_builder.tdf_writer.helper_handle.mz_upper,
            p_charge=p_charge,
            min_charge_contrib=args.min_charge_contrib,
        )

        # JOB 6: Simulate ion mobilities
        ions = simulate_ion_mobilities_and_variance(
            ions=ions,
            im_lower=acquisition_builder.tdf_writer.helper_handle.im_lower,
            im_upper=acquisition_builder.tdf_writer.helper_handle.im_upper,
            verbose=not args.silent_mode,
        )

        # JOB 7: Simulate precursor isotopic distributions
        ions = simulate_precursor_spectra_sequence(
            ions=ions,
            num_threads=args.num_threads,
            verbose=not args.silent_mode,
        )

    # JOB 8: Simulate scan distributions
    # TODO: sample standard deviation of ion mobility from a distribution (e.g., normal?)
    ions = simulate_scan_distributions_with_variance(
        ions=ions,
        scans=acquisition_builder.scan_table,
        verbose=not args.silent_mode,
        p_target=args.target_p,
        num_threads=args.num_threads,
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
        verbose=not args.silent_mode,
        num_threads=args.num_threads,
        down_sample_factor=args.down_sample_factor,
    )

    # JOB 10: Assemble frames
    assemble_frames(
        acquisition_builder=acquisition_builder,
        frames=acquisition_builder.frame_table,
        batch_size=args.batch_size,
        verbose=not args.silent_mode,
        mz_noise_precursor=args.mz_noise_precursor,
        mz_noise_uniform=args.mz_noise_uniform,
        precursor_noise_ppm=args.precursor_noise_ppm,
        mz_noise_fragment=args.mz_noise_fragment,
        fragment_noise_ppm=args.fragment_noise_ppm,
        num_threads=args.num_threads,
        add_real_data_noise=args.add_real_data_noise,
        intensity_max_precursor=args.reference_noise_intensity_max,
        intensity_max_fragment=args.reference_noise_intensity_max,
        precursor_sample_fraction=0.2,
        fragment_sample_fraction=0.2,
        num_precursor_frames=5,
        num_fragment_frames=5,
        fragment=args.apply_fragmentation,
    )


if __name__ == '__main__':
    main()
