import argparse
import os
import toml
import numpy as np

import mokapot

from imspy.timstof.dda import TimsDatasetDDA
from sagepy.utility import create_sage_database, compress_psms, decompress_psms
from sagepy.rescore.utility import transform_psm_to_mokapot_pin
from sagepy.core import Precursor, Tolerance, Scorer, SpectrumProcessor
from imspy.timstof.dbsearch.utility import sanitize_mz, get_searchable_spec, write_psms_binary
from imspy.algorithm.rescoring import create_feature_space
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
    parser.add_argument("--raw_data_path", type=str, help="Path to the dataset.")
    parser.add_argument("--fasta_path", type=str, help="Path to the FASTA file.")
    parser.add_argument("--config", type=str, help="Path to a TOML configuration file.")
    parser.add_argument("--num_threads", type=int, help="Number of threads for processing.")
    parser.add_argument("--cleave_at", type=str, help="Residue to cleave at.")
    parser.add_argument("--restrict", type=str, help="Restriction residues.")
    parser.add_argument("--n_terminal", action="store_true", help="If provided, then c_terminal = False.")
    parser.add_argument("--static_modifications", type=str, help="Static modifications in TOML-compatible string form.")
    parser.add_argument("--variable_modifications", type=str, help="Variable modifications in TOML-compatible string form.")
    parser.add_argument("--silent", action="store_true", help="Silent mode.")
    parser.add_argument("--no_bruker_sdk", action="store_true", help="Do not use Bruker SDK.")

    # Initially parse just to get config
    temp_args, _ = parser.parse_known_args()

    # Load config from TOML if provided
    config = {}
    if temp_args.config and os.path.exists(temp_args.config):
        with open(temp_args.config, "r") as f:
            config = toml.load(f)

    # Set defaults, using config where available
    defaults = {
        "raw_data_path": config.get("raw_data_path", None),
        "fasta_path": config.get("fasta_path", None),
        "num_threads": config.get("num_threads", -1),
        "cleave_at": config.get("cleave_at", "KR"),
        "restrict": config.get("restrict", "P"),
        "c_terminal": config.get("c_terminal", True),
        "n_terminal": config.get("n_terminal", False),
        "static_modifications": config.get("static_modifications", {"C": "[UNIMOD:4]"}),
        "variable_modifications": config.get("variable_modifications", {"M": ["[UNIMOD:35]"], "[": ["[UNIMOD:1]"]}),
        "verbose": config.get("verbose", True),
        "no_bruker_sdk": config.get("no_bruker_sdk", False)
    }

    # Re-parse arguments with defaults applied where command line not provided
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    # Handle silent mode
    if args.silent:
        args.verbose = False

    if args.n_terminal:
        args.c_terminal = False

    # If static_modifications or variable_modifications are provided as strings, try to parse them as TOML
    if isinstance(args.static_modifications, str):
        args.static_modifications = toml.loads(f"data = {args.static_modifications}")["data"]
    if isinstance(args.variable_modifications, str):
        args.variable_modifications = toml.loads(f"data = {args.variable_modifications}")["data"]

    # Check required arguments
    if args.raw_data_path is None:
        parser.error("raw_data_path is required (either via command line or config)")
    if args.fasta_path is None:
        parser.error("fasta_path is required (either via command line or config)")

    # check for number of threads
    if args.num_threads == -1:
        args.num_threads = os.cpu_count()

    if args.verbose:
        # Print the arguments
        print("Arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

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
        report_psms=3,
        min_matched_peaks=5,
        annotate_matches=True,
        static_mods=args.static_modifications,
        variable_mods=args.variable_modifications,
    )

    # get the count of files in the directory
    count = 0
    for file in os.listdir(args.raw_data_path):
        if file.endswith(".d"):
            count += 1

    if count == 0:
        raise ValueError("No .d files found in the directory.")

    current_count = 0

    for file in os.listdir(args.raw_data_path):
        if file.endswith(".d"):
            current_count += 1

            if args.verbose:
                print(f"Processing {file} ({current_count}/{count}) ...")

            dataset_name = file.split(".")[0]
            ds_path = os.path.join(args.raw_data_path, file)

            # Load the dataset
            dataset = TimsDatasetDDA(ds_path, use_bruker_sdk=not args.no_bruker_sdk)
            fragments = dataset.get_pasef_fragments(args.num_threads)

            # Group by mobility
            if args.verbose:
                print("Assembling re-fragmented precursors ...")

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
                print("Scoring spectra ...")

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
            bts = compress_psms(psm_list)

            write_psms_binary(
                byte_array=bts,
                folder_path=ds_path,
                file_name=f"{dataset_name}"
            )

            I = fragments.apply(lambda r: group_by_mobility(r.raw_data.mobility, r.raw_data.intensity), axis=1)
            inv_mob, intensity = [x[0] for x in I], [x[1] for x in I]

            fragments["inverse_ion_mobility"] = inv_mob
            fragments["intensity"] = intensity

            F = fragments[["spec_id", "monoisotopic_mz", "charge", "inverse_ion_mobility", "intensity"]]

            F.to_parquet(f"{ds_path}/imspy/{dataset_name}.parquet", index=False)

    total_psms = []

    if args.verbose:
        print("Loading PSMs ...")

    # now, go over all created binary PSMs and load them, re-score them and save them

    tmp_count = 0

    for file in os.listdir(args.raw_data_path):
        if file.endswith(".d"):
            tmp_count += 1
            dataset_name = file.split(".")[0]
            psm_bin_path = os.path.join(args.raw_data_path, file, "imspy", "psm", f"{dataset_name}.bin")

            # load binary array and decompress
            bts = np.fromfile(psm_bin_path, dtype=np.uint8)
            psm_list = decompress_psms(bts)
            total_psms.extend(psm_list)

            if args.verbose:
                # print the progress
                print(f"Loaded {dataset_name} ({tmp_count}/{count})")

    PSM_pandas = psm_collection_to_pandas(total_psms)

    if args.verbose:
        print("Creating mokapot pin ...")

    PSM_pin = transform_psm_to_mokapot_pin(PSM_pandas)
    PSM_pin.to_csv(f"{args.raw_data_path}/PSMs.pin", index=False, sep="\t")

    psms_moka = mokapot.read_pin(f"{args.raw_data_path}/PSMs.pin")
    results, _ = mokapot.brew(psms_moka, max_workers=args.num_threads)
    results.to_txt(dest_dir=args.raw_data_path)

    if args.verbose:
        print("Finished.")


if __name__ == "__main__":
    main()