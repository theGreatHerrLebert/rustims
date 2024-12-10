import argparse
import os
import toml
import numpy as np

import mokapot
import pandas as pd

from imspy.timstof.dda import TimsDatasetDDA
from imspy.chemistry.mobility import one_over_k0_to_ccs
from sagepy.utility import create_sage_database, compress_psms
from sagepy.rescore.utility import transform_psm_to_mokapot_pin
from sagepy.core import Precursor, Tolerance, Scorer, SpectrumProcessor
from imspy.timstof.dbsearch.utility import sanitize_mz, get_searchable_spec, write_psms_binary
from imspy.algorithm.rescoring import create_feature_space, re_score_psms
from sagepy.utility import psm_collection_to_pandas, apply_mz_calibration


def sanitize_charge(charge):
    """Sanitize charge"""
    try:
        return int(charge)
    except:
        return 2

def group_by_mobility(mobility, intensity):
    """Group by mobility"""
    r_dict = {}
    for mob, i in zip(mobility, intensity):
        r_dict[mob] = r_dict.get(mob, 0) + i
    return np.array(list(r_dict.keys())), np.array(list(r_dict.values()))


def main():
    # Argument parsing at the top of main
    parser = argparse.ArgumentParser(
        description="🚀 Extract CCS from TIMS-TOF DDA data to create training examples for machine learning 🧬📊"
    )
    parser.add_argument("dataset_path", type=str, nargs="?", help="Path to the dataset.")
    parser.add_argument("fasta_path", type=str, nargs="?", help="Path to the FASTA file.")
    parser.add_argument("--config", type=str, help="Path to a TOML configuration file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs.")
    parser.add_argument("--num_threads", type=int, help="Number of threads for processing.")
    parser.add_argument("--cleave_at", type=str, help="Residue to cleave at.")
    parser.add_argument("--restrict", type=str, help="Restriction residues.")
    parser.add_argument("--n_terminal", action="store_true", help="If provided, then c_terminal = False.")
    parser.add_argument("--static_modifications", type=str, help="Static modifications in TOML-compatible string form.")
    parser.add_argument("--variable_modifications", type=str, help="Variable modifications in TOML-compatible string form.")
    parser.add_argument("--silent", action="store_true", help="Silent mode.")

    # Initially parse just to get config
    temp_args, _ = parser.parse_known_args()

    # Load config from TOML if provided
    config = {}
    if temp_args.config and os.path.exists(temp_args.config):
        with open(temp_args.config, "r") as f:
            config = toml.load(f)

    # Set defaults, using config where available
    defaults = {
        "dataset_path": config.get("dataset_path", None),
        "fasta_path": config.get("fasta_path", None),
        "output_dir": config.get("output_dir", "."),
        "num_threads": config.get("num_threads", -1),
        "cleave_at": config.get("cleave_at", "KR"),
        "restrict": config.get("restrict", "P"),
        "c_terminal": config.get("c_terminal", True),
        "n_terminal": config.get("n_terminal", False),
        "static_modifications": config.get("static_modifications", {"C": "[UNIMOD:4]"}),
        "variable_modifications": config.get("variable_modifications", {"M": ["[UNIMOD:35]"], "[": ["[UNIMOD:1]"]}),
        "verbose": config.get("verbose", True)
    }

    # Re-parse arguments with defaults applied where command line not provided
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    # Handle silent mode
    # If --silent is passed, we set verbose to False.
    if args.silent:
        args.verbose = False

    if args.n_terminal:
        args.c_terminal = False

    # If static_modifications or variable_modifications are provided as strings, try to parse them as TOML
    # This allows passing e.g. --static_modifications='{"C":"[UNIMOD:4]"}'
    # safely from the command line.
    if isinstance(args.static_modifications, str):
        args.static_modifications = toml.loads(f"data = {args.static_modifications}")["data"]
    if isinstance(args.variable_modifications, str):
        args.variable_modifications = toml.loads(f"data = {args.variable_modifications}")["data"]

    # Check required arguments
    if args.dataset_path is None:
        parser.error("dataset_path is required (either via command line or config)")
    if args.fasta_path is None:
        parser.error("fasta_path is required (either via command line or config)")

    # check for number of threads
    if args.num_threads == -1:
        args.num_threads = os.cpu_count()

    if args.dataset_path.endswith("/"):
        dataset_name = args.dataset_path.split("/")[-2]
    else:
        dataset_name = args.dataset_path.split("/")[-1]

    if args.verbose:
        print("Processing dataset:", dataset_name)

    dataset = TimsDatasetDDA(args.dataset_path)
    fragments = dataset.get_pasef_fragments(num_threads=args.num_threads)

    # Group by mobility
    if args.verbose:
        print("Grouping by mobility ...")

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

    fragments['spec_id'] = fragments.apply(
        lambda r: f"{r['frame_id']}-{r['precursor_id']}-{dataset_name}", axis=1
    )

    fragments['gaussian_fit'] = fragments.raw_data.apply(lambda r: r.get_mobility_mean_and_variance())

    fragments["ccs_mean"] = fragments.apply(
        lambda r: one_over_k0_to_ccs(r.gaussian_fit[0], r.monoisotopic_mz, sanitize_charge(r.charge)), axis=1
    )
    fragments["ccs_std"] = fragments.apply(
        lambda r: one_over_k0_to_ccs(r.gaussian_fit[1], r.monoisotopic_mz, sanitize_charge(r.charge)), axis=1
    )

    if args.verbose:
        print("Extracting precursors ...")

    fragments['sage_precursor'] = fragments.apply(lambda r: Precursor(
        mz=sanitize_mz(r['monoisotopic_mz'], r['largest_peak_mz']),
        intensity=r['intensity'],
        charge=sanitize_charge(r['charge']),
        isolation_window=Tolerance(da=(-3, 3)),
        collision_energy=r.collision_energy,
        inverse_ion_mobility=r.mobility,
    ), axis=1)

    if args.verbose:
        print("Extracting fragment spectra ...")

    fragments['processed_spec'] = fragments.apply(
        lambda r: get_searchable_spec(
            precursor=r.sage_precursor,
            raw_fragment_data=r.raw_data,
            spec_processor=SpectrumProcessor(take_top_n=150),
            spec_id=r.spec_id,
            time=r['time'],
        ), axis=1
    )

    if args.verbose:
        print("Creating SAGE database ...")

    indexed_db = create_sage_database(
        fasta_path=args.fasta_path,
        cleave_at=args.cleave_at,
        restrict=args.restrict,
        static_mods=args.static_modifications,
        variable_mods=args.variable_modifications,
        c_terminal=args.c_terminal
    )

    scorer = Scorer(
        precursor_tolerance=Tolerance(ppm=(-25.0, 25.0)),
        fragment_tolerance=Tolerance(ppm=(-20.0, 20.0)),
        report_psms=5,
        min_matched_peaks=5,
        annotate_matches=True,
        static_mods=args.static_modifications,
        variable_mods=args.variable_modifications,
    )

    if args.verbose:
        print("Scoring precursors ...")

    psm_collection = scorer.score_collection_psm(
        db=indexed_db,
        spectrum_collection=fragments['processed_spec'].values,
        num_threads=args.num_threads
    )

    # mz calibration
    ppm_error = apply_mz_calibration(psm_collection, fragments)

    for _, values in psm_collection.items():
        for value in values:
            value.file_name = dataset_name
            value.mz_calibration_ppm = ppm_error

    psm_list = [psm for values in psm_collection.values() for psm in values]

    if args.verbose:
        print("Creating re-scoring feature space ...")

    psm_list = create_feature_space(psms=psm_list)

    if args.verbose:
        print("Rescoring PSMs ...")

    psm_list_rescored = re_score_psms(psm_list)

    # serialize PSMs to binary
    bts = compress_psms(psm_list_rescored)

    if args.verbose:
        print("writing PSMs to temp file ...")

    PSM_pandas = psm_collection_to_pandas(psm_list_rescored)

    # If output_dir is not specified, save into dataset directory in "imspy_ccs"
    if args.output_dir == ".":
        args.output_dir = f"{args.dataset_path}/imspy_ccs"

    # create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.verbose:
        print("Saving results ...")

    # write PSMs to binary file
    write_psms_binary(
        byte_array=bts,
        folder_path=args.output_dir,
        file_name=f"{dataset_name}"
    )

    if args.verbose:
        print("Creating mokapot pin ...")

    PSM_pin = transform_psm_to_mokapot_pin(PSM_pandas)
    PSM_pin.to_csv(f"{args.output_dir}/PSMs.pin", index=False, sep="\t")

    psms_moka = mokapot.read_pin(f"{args.output_dir}/PSMs.pin")
    results, _ = mokapot.brew(psms_moka, max_workers=args.num_threads)
    results.to_txt(dest_dir=args.output_dir)

    # read mokapot psms
    moka_peptides = pd.read_table(f"{args.output_dir}/mokapot.psms.txt")
    PSM_pandas = psm_collection_to_pandas(psm_list_rescored)
    results_filtered = pd.merge(PSM_pandas, moka_peptides, right_on=["SpecId"], left_on=["spec_idx"])

    B = pd.merge(fragments, results_filtered, left_on="spec_id", right_on="spec_idx")
    I = B.apply(lambda r: group_by_mobility(r.raw_data.mobility, r.raw_data.intensity), axis=1)

    inv_mob, intensity = [x[0] for x in I], [x[1] for x in I]

    B["inverse_ion_mobility"] = inv_mob
    B["intensity"] = intensity

    IONS_OUT = B[["frame_id", "time", "ims", "expmass", "calcmass", "collision_energy_x",
                  "collision_energy_calibrated", "spectral_angle_similarity", "scan_begin",
                  "scan_end", "largest_peak_mz", "average_mz", "monoisotopic_mz", "sequence",
                  "sequence_modified", "ccs_mean", "ccs_std", "charge_x", "hyperscore", "re_score", "mokapot score",
                  "mokapot q-value", "mokapot PEP", "spectrum_q", "peptide_q", "protein_q",
                  "inverse_ion_mobility", "intensity"
                  ]].rename(columns={"charge_x": "charge", "collision_energy_x": "collision_energy"})

    IONS_OUT.to_parquet(f"{args.output_dir}/{dataset_name}.parquet", index=False)

    if args.verbose:
        print("Saving configuration ...")

    with open(f"{args.output_dir}/config.txt", "w") as f:
        f.write(f"dataset_path: {args.dataset_path}\n")
        f.write(f"fasta_path: {args.fasta_path}\n")
        f.write(f"output_dir: {args.output_dir}\n")
        f.write(f"num_threads: {args.num_threads}\n")
        f.write(f"cleave_at: {args.cleave_at}\n")
        f.write(f"restrict: {args.restrict}\n")
        f.write(f"c_terminal: {args.c_terminal}\n")
        f.write(f"static_modifications: {args.static_modifications}\n")
        f.write(f"variable_modifications: {args.variable_modifications}\n")
        f.write(f"verbose: {args.verbose}\n")

    if args.verbose:
        print("Done!")


if __name__ == "__main__":
    main()