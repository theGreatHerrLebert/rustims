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

PASEF_META_COLUMNS_MAPPING = {
    "frame": "Frame",
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

    This function selects precursor ions as before, but then reassigns their Frame
    key to the corresponding fragment frames. Specifically, for each precursor frame,
    the selected ions are distributed over the next k fragment frames, where k = precursors_every - 1.
    (If there are not enough frames left, no fragment frames are generated.)

    Args:
        acquisition_builder: Acquisition builder object.
        verbose: Verbosity flag.
        precursors_every: Number of frames between precursor acquisitions.
        batch_size: Batch size for parallel processing.
        intensity_threshold: Intensity threshold for precursors.
        max_precursors: Maximum number of precursors to select per duty cycle.
        exclusion_width: Number of MS1 frames (duty cycles) to use for dynamic exclusion.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple of two pandas DataFrames:
          - PASEF meta information with updated fragment frame keys.
          - Annotated precursor selection table.
    """

    if verbose:
        print("Simulating dda-PASEF selection scheme...")
        print(f"precursors_every: {precursors_every}...")
        print(f"intensity_threshold: {intensity_threshold}...")
        print(f"max_precursors: {max_precursors}...")
        print(f"exclusion_width: {exclusion_width}...")

    # retrieve all frame IDs and initialize frame types (default: 8 for fragmentation)
    frames = acquisition_builder.frame_table.frame_id.values
    frame_types = np.full(len(frames), 8, dtype=int)

    # set frame types for precursor frames (default: 0) to every precursors_every-th frame
    for idx in range(len(frames)):
        if idx % precursors_every == 0:
            frame_types[idx] = 0

    acquisition_builder.calculate_frame_types(frame_types=frame_types)

    # compute overall maximum frame id (to avoid generating fragment frames beyond available indices)
    max_frame = int(np.max(frames))

    # precursor frame builder things
    synthetic_db_path = str(Path(acquisition_builder.path) / "synthetic_data.db")
    precursor_frame_builder = TimsTofSyntheticPrecursorFrameBuilder(synthetic_db_path)

    # extract MS1 frame IDs (precursor frames)
    ms1_frame_ids = acquisition_builder.frame_table[
        acquisition_builder.frame_table.ms_type == 0
    ].frame_id.values

    # skip the last precursor frame to avoid index out-of-bounds errors ? What is the reason for this?
    if ms1_frame_ids.size > 1:
        ms1_frame_ids = ms1_frame_ids[:-1]

    # create precursor table
    selected_p = select_precursors_from_frames(
        precursor_frame_builder=precursor_frame_builder,
        frame_ids=ms1_frame_ids,  # TODO: refine frame selection if needed.
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

    # transform selected precursors into PASEF meta data and reassign their keys
    pasef_meta = transform_selected_precursor_to_pasefmeta(selected_p, precursors_every)

    # Reformat meta table using the mapping (and cast key columns to int32)
    pasef_meta_names = PASEF_META_COLUMNS_MAPPING.values()
    pasef_meta = pasef_meta[list(pasef_meta_names)]
    inverse_mapping = {v: k for k, v in PASEF_META_COLUMNS_MAPPING.items()}
    pasef_meta = pasef_meta.rename(columns=inverse_mapping)
    pasef_meta = pasef_meta.assign(
        scan_start=lambda df: df["scan_start"].astype(np.int32),
        scan_end=lambda df: df["scan_end"].astype(np.int32),
        precursor=lambda df: df["precursor"].astype(np.int32),
    )

    # annotate the precursor table (calculate ion id)
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
            "Parent": "parent",  # now correctly uses the original precursor frame
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
    Get precursor isolation window from a frame.

    Args:
        frame: DataFrame containing frame data.
        ce_bias: Bias for collision energy calculation.
        ce_slope: Slope for collision energy calculation.
        scan_window: Number of scans to subtract/add from the apex to set the isolation window.

    Returns:
        DataFrame with computed precursor isolation windows.
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
    Select precursors from an MS1 frame and distribute them over fragment frames.

    This function applies a duty cycle: for each MS1 frame (ms1_frame_id), it loops
    over available fragment windows (from ms1_frame_id+1 to ms1_frame_id+precursor_every).
    If a target fragment frame would exceed the overall max_frame, that ramp is skipped.

    Args:
        precursors: DataFrame with precursor information.
        ms1_frame_id: The frame ID of the current MS1 frame.
        precursor_every: Total frames between precursor frames.
        excluded_precursors: DataFrame with already excluded precursor IDs.
        max_precursors: Maximum number of precursors to select per fragment window.
        intensity_threshold: Minimum intensity for precursor selection.
        max_frame: Overall maximum frame ID available.

    Returns:
        Tuple:
            - DataFrame with selected precursors (each assigned a new 'Frame' for the fragment window).
            - Updated DataFrame with exclusion information.
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

    # loop over fragment windows (Ramps)
    for i in range(1, precursor_every + 1):
        # corner case: if target frame exceeds max_frame, break the loop (or fear index out of bounds during frame read-in)
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
            all_precursors_one_ramp = all_precursors_one_ramp[all_precursors_one_ramp["peptide_id"] != selected_precursor["peptide_id"]]
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
    Select precursors for a list of frames.

    Args:
        precursor_frame_builder: Precursor builder object.
        frame_ids: List of MS1 frame IDs.
        precursor_every: Number of frames between precursor acquisitions.
        exclusion_width: Number of frames for dynamic exclusion.
        excluded_precursors_df: DataFrame with precursors to exclude.
        max_precursors: Maximum number of precursors to select per frame.
        intensity_threshold: Minimum intensity for a precursor.
        num_threads: Number of threads to use (-1 uses all cores).
        batch_size: Batch size for processing frames.
        max_frame: Overall maximum frame ID available.
        scan_window: Number of scans to set as the precursor selection window.

    Returns:
        DataFrame with selected precursors from all provided frames.
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
            # Pass scan_window to the isolation window function
            precursors = get_precursor_isolation_window_from_frame(frame.df, scan_window=scan_window)
            selected, excluded_precursors_df = select_precursors_pasef(
                precursors,
                ms1_frame_id=frame.frame_id,
                precursor_every=precursor_every,
                excluded_precursors=excluded_precursors_df,
                max_precursors=max_precursors,
                intensity_threshold=intensity_threshold,
                max_frame=max_frame  # pass the max_frame
            )
            batch_selected = pd.concat([batch_selected, selected], ignore_index=True)
        all_selected.append(batch_selected)

    return pd.concat(all_selected)


def transform_selected_precursor_to_pasefmeta(
        selected_precursors: pd.DataFrame, precursors_every: int
) -> pd.DataFrame:
    """
    Transform selected precursor DataFrame to PASEF meta DataFrame and distribute
    the selections across fragment frames, while preserving the original precursor (MS1) frame.
    """
    # Calculate a unique precursor identifier
    selected_precursors["Precursor"] = selected_precursors["peptide_id"] * 10 + selected_precursors["charge_state"]

    # Instead of overwriting the 'Frame' column, we create a new column for the fragment frame.
    def assign_fragment_frame(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        num_fragment_frames = precursors_every - 1
        if num_fragment_frames <= 0:
            # No fragment frames to assign, so return as is.
            return group
        group = group.sort_index()
        offsets = np.arange(len(group)) % num_fragment_frames
        # Create a new column 'fragment_frame' for the new frame id for fragmentation.
        group["fragment_frame"] = group["Frame"].iloc[0] + 1 + offsets
        return group

    # IMPORTANT: Group by the original precursor frame ('Parent') rather than 'Frame'
    distributed = selected_precursors.groupby("Parent", group_keys=False).apply(assign_fragment_frame)

    # Build PASEF meta data using the new 'fragment_frame' for the fragmentation frame, and keep the original 'Parent'
    pasef_meta = distributed[
        [
            "fragment_frame",
            "ScanNumBegin",
            "ScanNumEnd",
            "IsolationMz",
            "IsolationWidth",
            "CollisionEnergy",
            "Precursor",
            "Parent"  # preserved original precursor frame
        ]
    ]
    pasef_meta = pasef_meta.drop_duplicates(subset=["fragment_frame", "ScanNumBegin", "ScanNumEnd"])
    # Rename for consistency if needed
    pasef_meta = pasef_meta.rename(columns={"fragment_frame": "Frame"})
    return pasef_meta