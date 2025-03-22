import os
from pathlib import Path
from typing import Tuple, Any, Optional, List
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from imspy.simulation.acquisition import TimsTofAcquisitionBuilder
from imspy.simulation.experiment import TimsTofSyntheticPrecursorFrameBuilder

Logger = logging.getLogger(__name__)

# Downstream code expects these names:
PASEF_META_COLUMNS_MAPPING = {
    "frame": "Frame",  # fragment frame
    "scan_start": "ScanNumBegin",
    "scan_end": "ScanNumEnd",
    "isolation_mz": "IsolationMz",
    "isolation_width": "IsolationWidth",
    "collision_energy": "CollisionEnergy",
    "precursor": "Precursor",
}

PRECURSOR_MAPPING = {
    "id": "Id",
    "largest_peak_mz": "LargestPeakMz",
    "average_mz": "AverageMz",
    "monoisotopic_mz": "MonoisotopicMz",
    "charge": "Charge",
    "scan_number": "ScanNumber",
    "intensity": "Intensity",
    "parent": "Parent",
}


def simulate_dda_pasef_selection_scheme(
        acquisition_builder: TimsTofAcquisitionBuilder,
        verbose: bool,
        precursors_every: int,
        batch_size: int,
        intensity_threshold: float,
        max_precursors: int,
        exclusion_width: int,
        **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate DDA selection scheme.

    This function selects precursor ions and then distributes them across fragment frames.
    The returned PASEF meta table is later reformatted using the mapping in PASEF_META_COLUMNS_MAPPING,
    and the precursor table is reformatted according to PRECURSOR_MAPPING.
    """
    if verbose:
        print("Simulating dda-PASEF selection scheme...")
        print(f"precursors_every: {precursors_every}")
        print(f"intensity_threshold: {intensity_threshold}")
        print(f"max_precursors: {max_precursors}")
        print(f"exclusion_width: {exclusion_width}")

    # Determine frame types: default 8 for fragmentation; every precursors_every-th frame is MS1 (type 0)
    frames = acquisition_builder.frame_table.frame_id.values
    frame_types = np.full(len(frames), 8, dtype=int)
    for idx in range(len(frames)):
        if idx % precursors_every == 0:
            frame_types[idx] = 0
    acquisition_builder.calculate_frame_types(frame_types=frame_types)
    max_frame = int(np.max(frames))

    # Build precursor frame builder
    synthetic_db_path = str(Path(acquisition_builder.path) / "synthetic_data.db")
    precursor_frame_builder = TimsTofSyntheticPrecursorFrameBuilder(synthetic_db_path)

    # Extract MS1 frame IDs (precursor frames) and skip the last one to avoid index issues.
    ms1_frame_ids = acquisition_builder.frame_table.loc[
        acquisition_builder.frame_table.ms_type == 0, "frame_id"
    ].values

    ms1_frame_ids = np.sort(ms1_frame_ids)

    if ms1_frame_ids.size > 1:
        ms1_frame_ids = ms1_frame_ids[:-1]

    # Build the annotated precursor selection table.
    # Internally, the selected DataFrame will contain at least:
    #  - peptide_id, charge_state, mz_min, mz_max, ScanNumApex, intensity,
    #  - Parent: the original precursor (MS1) frame,
    #  - Frame: the fragment frame to which the precursor is assigned.
    selected_p = select_precursors_from_frames(
        precursor_frame_builder=precursor_frame_builder,
        frame_ids=ms1_frame_ids,
        precursor_every=precursors_every,
        exclusion_width=exclusion_width,
        excluded_precursors_df=None,
        max_precursors=max_precursors,
        intensity_threshold=intensity_threshold,
        num_threads=-1,
        batch_size=batch_size,
        max_frame=max_frame,
        **kwargs,
    )

    # Transform the selected precursors into PASEF meta data.
    # This function distributes the selected ions (in the same MS1 frame, identified by Parent)
    # over the available fragment frames (stored in Frame) in a round-robin fashion.
    pasef_meta = transform_selected_precursor_to_pasefmeta(selected_p, precursors_every)

    # Reformat the meta table using the mapping:
    # First, select the columns in the order expected (the values of the mapping)
    pasef_meta_names = PASEF_META_COLUMNS_MAPPING.values()
    pasef_meta = pasef_meta[list(pasef_meta_names)]
    # Then, rename them using the inverse mapping.
    inverse_mapping = {v: k for k, v in PASEF_META_COLUMNS_MAPPING.items()}
    pasef_meta = pasef_meta.rename(columns=inverse_mapping)
    # Cast specific columns to int32 as expected downstream.
    pasef_meta = pasef_meta.assign(
        scan_start=lambda df: df["scan_start"].astype(np.int32),
        scan_end=lambda df: df["scan_end"].astype(np.int32),
        precursor=lambda df: df["precursor"].astype(np.int32),
    )

    # Annotate the precursor table.
    # Calculate a unique id from peptide_id and charge_state.
    selected_p["peptide_id"] = selected_p["peptide_id"] * 10 + selected_p["charge_state"]
    selected_p_return = (
        selected_p[["peptide_id", "mz_min", "mz_max", "charge_state", "ScanNumApex", "intensity", "Parent"]]
        .assign(
            average_mz=lambda df: (df["mz_min"] + df["mz_max"]) / 2,
            largest_peak_mz=lambda df: df["mz_min"],
            monoisotopic_mz=lambda df: df["mz_min"],
            peptide_id=lambda df: df["peptide_id"].astype(np.int32),
            intensity=lambda df: df["intensity"].round(),
        )
        .rename(columns={
            "peptide_id": "id",
            "Parent": "parent",  # Parent holds the original precursor frame
            "ScanNumApex": "scan_number",
            "charge_state": "charge",
        })
        [['id', 'largest_peak_mz', 'average_mz', 'monoisotopic_mz', 'charge', 'scan_number', 'intensity', 'parent']]
    )

    return pasef_meta, selected_p_return


def get_precursor_isolation_window_from_frame(
        frame: pd.DataFrame,
        ce_bias: float = 54.1984,
        ce_slope: float = -0.0345,
        scan_window: int = 13,
) -> pd.DataFrame:
    """
    Compute the isolation window for precursors in a frame.

    Returns a DataFrame with:
      mz_min, mz_max, intensity, ScanNumApex, ScanNumBegin, ScanNumEnd,
      IsolationWidth, IsolationMz, CollisionEnergy.
    """
    agg_funcs = {"mz": ["min", "max"], "intensity": ["sum", "idxmax"]}
    grouped = frame.groupby(["peptide_id", "charge_state"]).agg(agg_funcs)
    grouped.columns = ["mz_min", "mz_max", "intensity", "idxmax"]
    grouped.reset_index(inplace=True)
    grouped["ScanNumApex"] = frame.loc[grouped["idxmax"], "scan"].values
    grouped.drop(columns=["idxmax"], inplace=True)
    grouped["IsolationWidth"] = np.where((grouped["mz_max"] - grouped["mz_min"]) < 2, 2, 3)
    grouped["IsolationMz"] = (grouped["mz_min"] + grouped["mz_max"]) / 2
    grouped["ScanNumBegin"] = grouped["ScanNumApex"] - scan_window
    grouped["ScanNumEnd"] = grouped["ScanNumApex"] + scan_window
    grouped["CollisionEnergy"] = ce_bias + ce_slope * grouped["ScanNumApex"]
    return grouped


def select_precursors_pasef(
        precursors: pd.DataFrame,
        ms1_frame_id: int,
        precursor_every: int,
        excluded_precursors: Optional[pd.DataFrame] = None,
        max_precursors: int = 25,
        intensity_threshold: float = 1200,
        max_frame: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select precursors from one MS1 frame and assign them to fragment frames.

    Adds two columns:
      - Parent: the MS1 frame id (original precursor frame)
      - Frame: the fragment frame id (ms1_frame_id + ramp)

    Returns a tuple:
      (DataFrame of selected precursors, Updated exclusion DataFrame)
    """
    if excluded_precursors is None:
        excluded_precursors = pd.DataFrame(columns=["peptide_id", "Frame"])

    n_before_int_thres = precursors.shape[0]
    precursors = precursors[precursors["intensity"] > intensity_threshold]
    Logger.debug("Removed %d precursors below intensity threshold", n_before_int_thres - precursors.shape[0])

    n_before = precursors.shape[0]
    precursors = precursors[~precursors["peptide_id"].isin(excluded_precursors["peptide_id"])]
    Logger.debug("Removed %d precursors due to exclusion", n_before - precursors.shape[0])

    precursors = precursors.sort_values("intensity", ascending=False)

    duty_cycle_exclusions = {"peptide_id": [], "Parent": []}
    selected_all = pd.DataFrame()

    # Loop over available fragment windows (ramps)
    for i in range(1, precursor_every + 1):
        if max_frame is not None and (ms1_frame_id + i >= max_frame):
            Logger.debug("Skipping Ramp %d for MS1 frame %d: target frame (%d) exceeds max_frame (%d)",
                         i, ms1_frame_id, ms1_frame_id + i, max_frame)
            break

        Logger.debug("Selecting precursors for Ramp %d (ms1_frame_id %d + %d)", i, ms1_frame_id, i)
        precursors = precursors[~precursors["peptide_id"].isin(duty_cycle_exclusions["peptide_id"])]
        Logger.debug("After duty-cycle exclusion, %d precursors remain", precursors.shape[0])
        selected_precursors_one_ramp = []
        all_precursors_one_ramp = precursors.copy(deep=True)
        while (len(selected_precursors_one_ramp) < max_precursors) and (not all_precursors_one_ramp.empty):
            selected_precursor = all_precursors_one_ramp.iloc[0]
            Logger.debug("Selected peptide id %d for Ramp %d", selected_precursor["peptide_id"], i)
            selected_precursors_one_ramp.append(selected_precursor.to_dict())
            duty_cycle_exclusions["peptide_id"].append(selected_precursor["peptide_id"])
            duty_cycle_exclusions["Parent"].append(ms1_frame_id)
            all_precursors_one_ramp = all_precursors_one_ramp[
                ~((all_precursors_one_ramp["ScanNumApex"] <= selected_precursor["ScanNumEnd"]) &
                  (all_precursors_one_ramp["ScanNumApex"] >= selected_precursor["ScanNumBegin"]))
            ]
            all_precursors_one_ramp = all_precursors_one_ramp[
                all_precursors_one_ramp["peptide_id"] != selected_precursor["peptide_id"]
                ]
        ramp_df = pd.DataFrame(selected_precursors_one_ramp)
        if not ramp_df.empty:
            ramp_df["Parent"] = ms1_frame_id
            ramp_df["Frame"] = ms1_frame_id + i
        selected_all = pd.concat([selected_all, ramp_df], ignore_index=True)

    duty_cycle_exclusions_df = pd.DataFrame(duty_cycle_exclusions)
    if not duty_cycle_exclusions_df.empty:
        excluded_precursors = pd.concat([excluded_precursors, duty_cycle_exclusions_df], ignore_index=True)

    return selected_all, excluded_precursors


def select_precursors_from_frames(
        precursor_frame_builder: Any,
        frame_ids: List[int],
        precursor_every: int,
        exclusion_width: int = 25,
        excluded_precursors_df: Optional[pd.DataFrame] = None,
        max_precursors: int = 25,
        intensity_threshold: float = 1200,
        num_threads: int = -1,
        batch_size: int = 256,
        max_frame: int = None,
        scan_window: int = 13
) -> pd.DataFrame:
    """
    Process a list of MS1 frames and return a DataFrame of selected precursors.
    """
    if num_threads == -1:
        num_threads = os.cpu_count()

    if excluded_precursors_df is None:
        excluded_precursors_df = pd.DataFrame(columns=["peptide_id", "Frame"])

    num_batches = max(1, len(frame_ids) // batch_size)
    frame_batches = np.array_split(frame_ids, num_batches)
    all_selected: List[pd.DataFrame] = []

    for batch in tqdm(frame_batches, total=len(frame_batches), desc="Selecting Precursors", ncols=80):
        precursor_frames = precursor_frame_builder.build_precursor_frames_annotated(batch, num_threads=num_threads)
        batch_selected = pd.DataFrame()
        for frame in precursor_frames:
            Logger.debug("Selecting precursors for frame %d", frame.frame_id)
            excluded_precursors_df = excluded_precursors_df.loc[
                (excluded_precursors_df["Frame"] - frame.frame_id) <= exclusion_width
                ]
            precursors = get_precursor_isolation_window_from_frame(frame.df, scan_window=scan_window)
            selected, excluded_precursors_df = select_precursors_pasef(
                precursors,
                ms1_frame_id=frame.frame_id,
                precursor_every=precursor_every,
                excluded_precursors=excluded_precursors_df,
                max_precursors=max_precursors,
                intensity_threshold=intensity_threshold,
                max_frame=max_frame
            )
            batch_selected = pd.concat([batch_selected, selected], ignore_index=True)
        all_selected.append(batch_selected)

    return pd.concat(all_selected)


def transform_selected_precursor_to_pasefmeta(
        selected_precursors: pd.DataFrame, precursors_every: int
) -> pd.DataFrame:
    """
    Transform the selected precursor DataFrame into PASEF meta data.

    The transformation redistributes the selected precursors (grouped by the original MS1 frame in "Parent")
    over the available fragment frames (in "Frame") in a round-robin fashion.
    The resulting DataFrame contains only the columns required for downstream processing:
      Frame, ScanNumBegin, ScanNumEnd, IsolationMz, IsolationWidth, CollisionEnergy, Precursor
    """
    selected_precursors["Precursor"] = selected_precursors["peptide_id"] * 10 + selected_precursors["charge_state"]

    def assign_fragment_frame(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        num_fragment_frames = precursors_every - 1
        if num_fragment_frames <= 0:
            return group
        group = group.sort_values("Frame")
        offsets = np.arange(len(group)) % num_fragment_frames
        group["Frame"] = group["Frame"].iloc[0] + 1 + offsets
        return group

    distributed = selected_precursors.groupby("Parent", group_keys=False).apply(assign_fragment_frame)
    pasef_meta = distributed[[
        "Frame", "ScanNumBegin", "ScanNumEnd", "IsolationMz",
        "IsolationWidth", "CollisionEnergy", "Precursor"
    ]]
    pasef_meta = pasef_meta.drop_duplicates(subset=["Frame", "ScanNumBegin", "ScanNumEnd"])
    return pasef_meta