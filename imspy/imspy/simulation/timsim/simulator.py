import os
import argparse
import time
from pathlib import Path

import toml
import pandas as pd
import tensorflow as tf
from tabulate import tabulate

# imspy imports
from imspy.simulation.experiment import SyntheticExperimentDataHandleDIA
from imspy.simulation.experiment import TimsTofSyntheticPrecursorFrameBuilder
from imspy.simulation.timsim.jobs.simulate_ion_mobilities_and_variance import simulate_ion_mobilities_and_variance
from imspy.simulation.timsim.jobs.simulate_peptides import simulate_peptides
from imspy.simulation.timsim.jobs.simulate_phosphorylation import simulate_phosphorylation
from imspy.simulation.timsim.jobs.simulate_proteins import simulate_proteins
from imspy.simulation.timsim.jobs.simulate_scan_distributions_with_variance import (
    simulate_scan_distributions_with_variance
)
from imspy.simulation.utility import get_fasta_file_paths, get_dilution_factors
from imspy.simulation.timsim.jobs.utility import check_path

# Local imports
from .jobs.assemble_frames import assemble_frames
from .jobs.build_acquisition import build_acquisition
from .jobs.simulate_charge_states import simulate_charge_states
from .jobs.simulate_fragment_intensities import simulate_fragment_intensities
from .jobs.simulate_frame_distributions_emg import simulate_frame_distributions_emg
from .jobs.simulate_precursor_spectra import simulate_precursor_spectra_sequence
from .jobs.simulate_retention_time import simulate_retention_times
from .jobs.dda_selection_scheme import simulate_dda_pasef_selection_scheme

# ----------------------------------------------------------------------
# Environment setup and GPU configuration
# ----------------------------------------------------------------------

# Silence console spam
os.environ["WANDB_SILENT"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def configure_gpu_memory(memory_limit_gb=4):
    """
    Restricts TensorFlow GPU memory usage to the specified limit per GPU.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for i, _ in enumerate(gpus):
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[i],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * memory_limit_gb)]
                )
                print(f"GPU: {i} memory restricted to {memory_limit_gb}GB.")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


# ----------------------------------------------------------------------
# Configuration / Argument Parsing
# ----------------------------------------------------------------------

def load_toml_config(config_path: str) -> dict:
    """
    Loads a TOML configuration file into a dictionary.
    """
    with open(config_path, 'r') as config_file:
        return toml.load(config_file)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Builds and returns the command line argument parser.
    """
    parser = argparse.ArgumentParser(
        description='🦀💻 TIMSIM 🔬🐍 - Run a proteomics experiment simulation with diaPASEF-like acquisition '
                    'on a BRUKER TimsTOF.'
    )

    # Configuration file argument
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file (TOML format)")

    # Required positional arguments
    parser.add_argument("save_path", type=str, nargs='?', default=None, help="Path to save the experiment to")
    parser.add_argument("reference_path", type=str, nargs='?', default=None,
                        help="Path to a real TDF reference dataset")
    parser.add_argument("fasta_path", type=str, nargs='?', default=None,
                        help="Path to the fasta file of proteins to be digested")

    # Optional flags
    parser.add_argument("--reference_in_memory", dest="reference_in_memory", action="store_true",
                        help="Whether to load the reference dataset into memory (default: False)")
    parser.set_defaults(reference_in_memory=False)

    parser.add_argument("-s", "--silent_mode,", dest="silent_mode", action="store_true",
                        help="Silence output (default: False)")
    parser.set_defaults(silent_mode=False)

    parser.add_argument("-acq", "--acquisition_type",
                        type=str,
                        help="Type of acquisition to simulate, choose between: [DDA, DIA, SYNCHRO, SLICE, MIDIA], default: DIA")

    parser.add_argument("-n", "--experiment_name", type=str, help="Name of the experiment")

    parser.add_argument("--no_reference_layout", dest="use_reference_layout", action="store_false",
                        help="Use the layout of the reference dataset for the acquisition (default: True)")
    parser.set_defaults(use_reference_layout=True)

    parser.add_argument("--no_peptide_sampling", dest="sample_peptides", action="store_false",
                        help="Sample peptides from the digested fasta (default: True)")
    parser.set_defaults(sample_peptides=True)

    parser.add_argument("--sample_seed", type=int, help="Seed for peptide sampling (default: 41)")
    parser.add_argument("--apply_fragmentation", dest="apply_fragmentation", action="store_true",
                        help="Perform fragmentation (default: False)")
    parser.set_defaults(apply_fragmentation=False)

    # Peptide digestion arguments
    parser.add_argument("--num_sample_peptides", type=int, default=25_000,
                        help="Number of peptides to sample from the digested fasta (default: 25_000)")
    parser.add_argument("--num_peptides_total", type=int, help="Total number of peptides to simulate initially before downsampling (default: 250_000)",
                        default=250_000)
    parser.add_argument("--n_proteins", type=int, help="Number of proteins to sample from the fasta (default: 20_000)",
                        default=20_000)
    parser.add_argument("--missed_cleavages", type=int, help="Number of missed cleavages (default: 2)")
    parser.add_argument("--min_len", type=int, help="Minimum peptide length (default: 7)")
    parser.add_argument("--max_len", type=int, help="Maximum peptide length (default: 30)")
    parser.add_argument("--cleave_at", type=str, help="Cleave at (default: KR)")
    parser.add_argument("--restrict", type=str, help="Restrict (default: P)")
    parser.add_argument("--decoys", dest="decoys", action="store_true",
                        help="Generate decoys (default: False)")
    parser.set_defaults(decoys=False)

    # Modifications config
    parser.add_argument("--modifications", type=str, default=None,
                        help="Path to a modifications TOML file. Default: configs/modifications.toml")

    # Precursor isotopic pattern settings
    parser.add_argument("--isotope_k", type=int, help="Number of isotopes to simulate (default: 8)")
    parser.add_argument("--isotope_min_intensity", type=int, help="Min intensity for isotopes (default: 1)")
    parser.add_argument("--no_isotope_centroid", dest="isotope_centroid", action="store_false",
                        help="Centroid isotopes (default: True)")
    parser.set_defaults(isotope_centroid=True)

    # Occurrences
    parser.add_argument("--no_sample_occurrences", dest="sample_occurrences", action="store_false",
                        help="Whether or not sample peptide occurrences should be assigned randomly (default: True)")
    parser.set_defaults(sample_occurrences=True)

    parser.add_argument("--intensity_value", type=float,
                        help="Intensity value of all peptides if sample occurrence sampling is deactivated "
                             "(default: 1e6)")

    # Distribution parameters
    parser.add_argument("--gradient_length", type=float, help="Length of the gradient in seconds (default: 3600)")
    parser.add_argument("--z_score", type=float, help="Z-score for frame and scan distributions (default: .999)")
    parser.add_argument("--mean_std_rt", type=float, help="Mean standard deviation for RT distribution (default: 1.5)")
    parser.add_argument("--variance_std_rt", type=float, help="Variance std for RT distribution (default: 0.3)")
    parser.add_argument("--mean_skewness", type=float, help="Mean skewness for RT distribution (default: 0.3)")
    parser.add_argument("--variance_skewness", type=float, help="Variance skewness (default: 0.1)")
    parser.add_argument("--target_p", type=float, help="Target percentile for frame distributions (default: 0.999)")
    parser.add_argument("--sampling_step_size", type=float, help="Step size for frame distributions (default: 0.001)")
    parser.add_argument("--no_use_inverse_mobility_std_mean", dest="use_inverse_mobility_std_mean", action="store_false",
                        help="Don't use inverse mobility std mean (default: True)")
    parser.set_defaults(use_inverse_mobility_std_mean=True)
    parser.add_argument("--inverse_mobility_std_mean", type=float,
                        help="Inverse mobility std mean (default: 0.009)")

    # Cores, batch, etc.
    parser.add_argument("--num_threads", type=int, help="Number of threads to use (default: -1 for all available)")
    parser.add_argument("--batch_size", type=int, help="Batch size (default: 256)")

    # Charge state probabilities
    parser.add_argument("--p_charge", type=float, help="Probability of being charged (default: 0.5)")
    parser.add_argument("--min_charge_contrib", type=float, help="Minimum charge contribution (default: 0.25)")
    parser.add_argument("--max_charge", type=int, help="Maximum charge state (default: 4)")
    parser.add_argument("--binomial_charge_model", dest="binomial_charge_model", action="store_true",
                        help="Use binomial charge state model (default: False)")
    parser.set_defaults(binomial_charge_model=False)

    parser.add_argument("--noise_frame_abundance", dest="noise_frame_abundance", action="store_true",
                        help="Add noise to frame abundance (default: False)")
    parser.set_defaults(noise_frame_abundance=False)
    parser.add_argument("--noise_scan_abundance", dest="noise_scan_abundance", action="store_true",
                        help="Add noise to scan abundance (default: False)")
    parser.set_defaults(noise_scan_abundance=False)


    parser.add_argument("--mz_noise_precursor", dest="mz_noise_precursor", action="store_true",
                        help="Add noise to precursor m/z (default: False)")
    parser.set_defaults(mz_noise_precursor=False)

    parser.add_argument("--precursor_noise_ppm", type=float, help="Precursor noise in ppm (default: 5.0)")

    parser.add_argument("--mz_noise_fragment", dest="mz_noise_fragment", action="store_true",
                        help="Add noise to fragment m/z (default: False)")
    parser.set_defaults(mz_noise_fragment=False)

    parser.add_argument("--fragment_noise_ppm", type=float, help="Fragment noise in ppm (default: 5.0)")

    parser.add_argument("--mz_noise_uniform", dest="mz_noise_uniform", action="store_true",
                        help="Use uniform distribution for m/z noise (default: False). "
                             "If false, use normal distribution.")
    parser.set_defaults(mz_noise_uniform=False)

    parser.add_argument("--add_real_data_noise", dest="add_real_data_noise", action="store_true",
                        help="Use the reference data to add noise to the simulated data (default: False)")
    parser.set_defaults(add_real_data_noise=False)

    parser.add_argument("--reference_noise_intensity_max", type=float,
                        help="Max intensity for noise reference data (default: 30)")

    parser.add_argument("--down_sample_factor", type=float,
                        help="Probability factor to downsample fragment peaks (default: 0.5)")

    # Proteome mixture
    parser.add_argument("--proteome_mix", action="store_true", dest="proteome_mix",
                        help="Enable proteome mixture mode (default: False)")
    parser.set_defaults(proteome_mix=False)

    parser.add_argument("--multi_fasta_dilution", type=str,
                        help="Path to a CSV with dilution factors if proteome_mix is True")

    # Debug
    parser.add_argument("--debug_mode", action="store_true", dest="debug_mode",
                        help="Enable debug mode for extra printing (default: False)")
    parser.set_defaults(debug_mode=False)

    # Use from existing simulation
    parser.add_argument("--from_existing", action="store_true", dest="from_existing",
                        help="Use existing simulated data from --existing_path (default: False)")
    parser.set_defaults(from_existing=False)

    parser.add_argument("--existing_path", type=str,
                        help="Path to existing simulation (if from_existing is True)")

    # Bruker SDK usage
    parser.add_argument("--no_bruker_sdk", action="store_false", dest="use_bruker_sdk",
                        help="Disable Bruker SDK usage (default: True)")
    parser.set_defaults(use_bruker_sdk=True)

    # Phospho mode
    parser.add_argument("--phospho_mode", action="store_true", dest="phospho_mode",
                        help="Enable phospho-enriched dataset generation (default: False)")
    parser.set_defaults(phospho_mode=False)

    return parser


def get_default_settings() -> dict:
    """
    Returns a dictionary of default values to merge with user args and potential config files.
    """
    return {
        'reference_in_memory': False,
        'silent_mode': False,
        'acquisition_type': 'DIA',
        'experiment_name': f'TIMSIM-[PLACEHOLDER]-{int(time.time())}',
        'use_reference_layout': True,
        'sample_peptides': True,
        'sample_seed': 41,
        'apply_fragmentation': False,
        'num_sample_peptides': 25_000,
        'num_peptides_total': 250_000,
        'n_proteins': 20_000,
        'missed_cleavages': 2,
        'min_len': 7,
        'max_len': 30,
        'cleave_at': 'KR',
        'restrict': 'P',
        'decoys': False,
        'modifications': None,
        'isotope_k': 8,
        'isotope_min_intensity': 1,
        'isotope_centroid': True,
        'sample_occurrences': True,
        'intensity_value': 1e6,
        'gradient_length': 3600,
        'z_score': 0.999,
        'mean_std_rt': 1.5,
        'variance_std_rt': 0.3,
        'mean_skewness': 0.3,
        'variance_skewness': 0.1,
        'target_p': 0.999,
        'sampling_step_size': 0.001,
        'use_inverse_mobility_std_mean': True,
        'inverse_mobility_std_mean': 0.009,
        'num_threads': -1,
        'batch_size': 256,
        'p_charge': 0.5,
        'min_charge_contrib': 0.25,
        'noise_frame_abundance': False,
        'noise_scan_abundance': False,
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
        'max_charge': 4,
        'binomial_charge_model': False,
    }


def check_required_args(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """
    Ensures required arguments are not None; raises errors otherwise.
    """
    if args.save_path is None:
        parser.error("the following argument is required: save_path")
    if args.reference_path is None:
        parser.error("the following argument is required: reference_path")
    if args.fasta_path is None:
        parser.error("the following argument is required: fasta_path")


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------

def main():
    """
    Entry point for the proteomics simulation script.
    """

    # Configure GPU memory (optional/adjustable)
    configure_gpu_memory(memory_limit_gb=4)

    # Create and parse arguments
    parser = build_arg_parser()
    args, remaining_args = parser.parse_known_args()

    # Load defaults and override with config file if provided
    defaults = get_default_settings()
    if args.config:
        config = load_toml_config(args.config)
        for section in config:
            defaults.update(config[section])
        # If config includes path info, handle them
        defaults['save_path'] = config.get('save_path', defaults.get('save_path'))
        defaults['reference_path'] = config.get('reference_path', defaults.get('reference_path'))
        defaults['fasta_path'] = config.get('fasta_path', defaults.get('fasta_path'))

    # Update parser defaults with new merged defaults
    parser.set_defaults(**defaults)

    # Final parse
    args = parser.parse_args()

    # Ensure required arguments are set
    check_required_args(args, parser)

    # MacOS check for Bruker SDK
    if os.uname().sysname == 'Darwin':
        print("Warning: Using Bruker SDK on MacOS is not supported, setting use_bruker_sdk to False.")
        args.use_bruker_sdk = False

    # Load modifications config
    script_dir = Path(__file__).parent
    if not args.modifications or args.modifications == "":
        args.modifications = script_dir / "configs" / "modifications.toml"
    mod_config = load_toml_config(args.modifications)
    variable_modifications = mod_config.get('variable_modifications', {})
    static_modifications = mod_config.get('static_modifications', {})

    # Debug: Show argument table if debug_mode
    args_dict = vars(args)
    if args.debug_mode:
        arg_table = [[key, value] for key, value in args_dict.items()]
        print(tabulate(arg_table, headers=["Argument", "Value"], tablefmt="grid"))

    # Print info about modifications and fragmentation
    if not args.silent_mode:
        print(f"Using variable modifications: {variable_modifications}")
        print(f"Using static modifications: {static_modifications}")
        if args.apply_fragmentation:
            print("Fragmentation is enabled.")
        else:
            print("Fragmentation is disabled.")

    # Basic validations
    assert 0.0 < args.z_score < 1.0, f"Z-score must be between 0 and 1, was {args.z_score}"
    p_charge = args.p_charge
    assert 0.0 < p_charge < 1.0, f"Probability of being charged must be between 0 and 1, was {p_charge}"

    # Prepare final naming / paths
    save_path = check_path(args.save_path)
    reference_path = check_path(args.reference_path)
    name = args.experiment_name.replace('[PLACEHOLDER]', f'{args.acquisition_type}').replace("'", "")

    # Log arguments for future reference
    table_data = [[key, value] for key, value in args_dict.items()]
    with open(os.path.join(save_path, f'arguments-{name}.txt'), 'w') as f:
        f.write(tabulate(table_data, headers=["Argument", "Value"], tablefmt="grid"))

    # Build the acquisition (frames, scans, etc.)
    if not args.silent_mode:
        print(f"Simulating experiment {name} at {save_path}...")

    acquisition_builder = build_acquisition(
        path=save_path,
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

    # Possibly re-used from existing
    rt_sigma = None
    rt_lambda = None
    peptides, proteins, ions = None, None, None

    # Proteome mix setup (if needed)
    if args.proteome_mix and args.multi_fasta_dilution:
        factors = get_dilution_factors(args.multi_fasta_dilution)
    else:
        factors = {}

    if args.from_existing:
        # Load existing simulation data
        existing_sim_handle = SyntheticExperimentDataHandleDIA(database_path=args.existing_path)
        peptides = existing_sim_handle.get_table('peptides')
        proteins = existing_sim_handle.get_table('proteins')
        ions = existing_sim_handle.get_table('ions')
        rt_sigma = peptides['rt_sigma'].values
        rt_lambda = peptides['rt_lambda'].values

        # If mixing is enabled, apply dilution factors by FASTA
        if args.proteome_mix:
            for fasta, dilution_factor in factors.items():
                if not args.silent_mode:
                    print(f"Applying dilution factor {dilution_factor} to {fasta}")
                mask = peptides['fasta'] == fasta
                peptides.loc[mask, 'events'] *= dilution_factor
                peptides.loc[mask, 'fasta'] = fasta

        # Warn if gradient length mismatch is large
        rt_max = peptides['retention_time_gru_predictor'].max()
        if abs(rt_max - args.gradient_length) / args.gradient_length > 0.05:
            print(
                f"Warning: The gradient length of the existing simulation is {rt_max} seconds, "
                f"which is off by more than 5% of the new gradient length {args.gradient_length} seconds."
            )

        if not args.silent_mode:
            print(f"Using existing simulation from {args.existing_path}")
            print(f"Peptides: {peptides.shape[0]}")
            print(f"Ions: {ions.shape[0]}")

    # ----------------------------------------
    # FASTA processing if not from existing
    # ----------------------------------------
    fastas = get_fasta_file_paths(args.fasta_path)
    protein_list, peptide_list = [], []

    if not args.from_existing:
        for fasta_name, fasta_path in fastas.items():
            if not args.silent_mode:
                print(f"Digesting fasta file: {fasta_name}...")

            mixture_factor = 1.0
            if args.proteome_mix:
                mixture_factor = factors.get(fasta_name, 1.0)
                if not args.silent_mode:
                    print(f"Using mixture factor {mixture_factor} for {fasta_name}")

            # JOB 0: Generate Protein Data
            proteins_tmp = simulate_proteins(
                fasta_file_path=fasta_path,
                n_proteins=args.n_proteins,
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

            # JOB 1: Simulate peptides
            if not args.silent_mode:
                print("Creating Peptides from Proteins...")

            peptides_tmp = simulate_peptides(
                protein_table=proteins_tmp,
                num_peptides_total=args.num_peptides_total,
                verbose=not args.silent_mode,
                exclude_accumulated_gradient_start=True,
                min_rt_percent=2.0,
                gradient_length=acquisition_builder.gradient_length,
                down_sample=args.sample_peptides,
                proteome_mix=args.proteome_mix,
            )

            if args.proteome_mix:
                # Scale by mixture factor
                peptides_tmp['events'] *= mixture_factor
                peptides_tmp['fasta'] = fasta_name

            protein_list.append(proteins_tmp)
            peptide_list.append(peptides_tmp)

        # Concatenate across all FASTA inputs
        proteins = pd.concat(protein_list)[["protein_id", "protein", "sequence", "events"]]
        peptides = pd.concat(peptide_list)

        # Phospho mode
        if args.phospho_mode:
            if not args.silent_mode:
                print("Simulating phosphorylation...")
            peptides = simulate_phosphorylation(
                peptides=peptides,
                pick_phospho_sites=2,
                template=True,
                verbose=not args.silent_mode
            )

        # Subsample peptides if needed
        if args.sample_peptides:
            try:
                peptides = peptides.sample(n=args.num_sample_peptides, random_state=args.sample_seed).reset_index(
                    drop=True)
            except ValueError:
                print(
                    f"Warning: Not enough peptides to sample {args.num_sample_peptides}. "
                    f"Using all {peptides.shape[0]} peptides."
                )

        if not args.silent_mode:
            print(f"Simulating {peptides.shape[0]} peptides...")

        # JOB 3: Simulate retention times
        peptides = simulate_retention_times(
            peptides=peptides,
            verbose=not args.silent_mode,
            gradient_length=acquisition_builder.gradient_length,
        )

        # Workaround for the correct column ordering
        columns = list(peptides.columns)
        # The last two might be 'events' and 'retention_time_gru_predictor' or some similar swap:
        columns[-2], columns[-1] = columns[-1], columns[-2]
        peptides = peptides[columns]

    # If multi-threading is not specified, use all cores
    if args.num_threads == -1:
        args.num_threads = os.cpu_count()

    # JOB 4: Frame distributions
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
        add_noise=args.noise_frame_abundance,
        num_threads=args.num_threads,
        from_existing=args.from_existing,
        sigmas=rt_sigma,
        lambdas=rt_lambda,
    )

    # Save proteins
    acquisition_builder.synthetics_handle.create_table(table_name='proteins', table=proteins)

    # Handle final column ordering for phospho or proteome mixes
    if args.phospho_mode:
        # Columns for phospho
        columns_phospho = [
            'protein_id', 'peptide_id', 'sequence', 'protein', 'decoy',
            'missed_cleavages', 'n_term', 'c_term', 'monoisotopic-mass',
            'retention_time_gru_predictor', 'events', 'rt_sigma', 'rt_lambda',
            'frame_occurrence_start', 'frame_occurrence_end', 'frame_occurrence',
            'frame_abundance',
            'phospho_site_a', 'phospho_site_b', 'sequence_original'
        ]
        peptides = peptides[columns_phospho]
    elif args.proteome_mix:
        columns_mixed = [
            'protein_id', 'peptide_id', 'sequence', 'protein', 'decoy',
            'missed_cleavages', 'n_term', 'c_term', 'monoisotopic-mass',
            'retention_time_gru_predictor', 'events', 'rt_sigma', 'rt_lambda',
            'frame_occurrence_start', 'frame_occurrence_end', 'frame_occurrence',
            'frame_abundance',
            'total_events', 'fasta'
        ]
        # Ensure columns exist in the DataFrame
        peptides = peptides[[col for col in columns_mixed if col in peptides.columns]]

    # Save peptides
    acquisition_builder.synthetics_handle.create_table(table_name='peptides', table=peptides)

    # ------------------------------------------------------------------
    # Further steps if not from existing simulation
    # ------------------------------------------------------------------
    if not args.from_existing:
        # JOB 5: Charge states
        ions = simulate_charge_states(
            peptides=peptides,
            mz_lower=acquisition_builder.tdf_writer.helper_handle.mz_lower,
            mz_upper=acquisition_builder.tdf_writer.helper_handle.mz_upper,
            p_charge=args.p_charge,
            max_charge=args.max_charge,
            use_binomial=args.binomial_charge_model,
            min_charge_contrib=args.min_charge_contrib,
        )

        # JOB 6: Ion mobilities
        ions = simulate_ion_mobilities_and_variance(
            ions=ions,
            im_lower=acquisition_builder.tdf_writer.helper_handle.im_lower,
            im_upper=acquisition_builder.tdf_writer.helper_handle.im_upper,
            verbose=not args.silent_mode,
            remove_mods=True,
            use_target_mean_std=args.use_inverse_mobility_std_mean,
            target_std_mean=args.inverse_mobility_std_mean,
        )

        # JOB 7: Precursor isotopic distributions
        ions = simulate_precursor_spectra_sequence(
            ions=ions,
            num_threads=args.num_threads,
            verbose=not args.silent_mode,
        )

    # JOB 8: Scan distributions
    ions = simulate_scan_distributions_with_variance(
        ions=ions,
        scans=acquisition_builder.scan_table,
        verbose=not args.silent_mode,
        p_target=args.target_p,
        add_noise=args.noise_scan_abundance,
        num_threads=args.num_threads,
    )

    # Save ions
    acquisition_builder.synthetics_handle.create_table(table_name='ions', table=ions)

    if args.acquisition_type == 'DDA':
        pasef_meta, precursors = simulate_dda_pasef_selection_scheme(
            acquisition_builder=acquisition_builder,
            verbose=not args.silent_mode,
        )
        acquisition_builder.synthetics_handle.create_table(table_name='pasef_meta', table=pasef_meta)
        acquisition_builder.synthetics_handle.create_table(table_name='precursors', table=precursors)

    # JOB 9: Simulate fragment intensities
    simulate_fragment_intensities(
        path=save_path,
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