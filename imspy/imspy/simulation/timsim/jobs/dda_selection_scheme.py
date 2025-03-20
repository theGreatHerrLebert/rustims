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

# Mapping of column names for PASEF meta data
PASEF_META_COLUMNS_MAPPING = {
    "frame": "Frame",
    "scan_start": "ScanNumBegin",
    "scan_end": "ScanNumEnd",
    "isolation_mz": "IsolationMz",
    "isolation_width": "IsolationWidth",
    "collision_energy": "CollisionEnergy",
    "precursor": "Precursor",
}

# Mapping of precursor column names
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

    This function now selects precursor ions as before, but then reassigns their
    Frame key to the corresponding fragment frames. Specifically, for each precursor
    frame, the selected ions are distributed over the next k fragment frames,
    where k = precursors_every - 1 (if you wish to include the precursor frame in the duty cycle,
    you can adjust the range below).

    Args:
        acquisition_builder: Acquisition builder object.
        verbose: Verbosity flag.
        precursors_every: Number of frames between precursor acquisitions.
                          (For example, precursors_every=4 means every 4th frame is an MS1 frame.)
        batch_size: Batch size for parallel processing.
        intensity_threshold: Intensity threshold for precursors.
        max_precursors: Maximum number of precursors to select per duty cycle (per fragment window).
        exclusion_width: Number of MS1 frames (duty cycles) to use for dynamic exclusion.
        **kwargs: Additional keyword arguments for the selection scheme.

    Returns:
        Tuple of two pandas DataFrames:
            - PASEF meta information with updated fragment frame keys.
            - Selected precursor information (annotated precursor table).
    """
    # Retrieve all frame IDs and initialize frame types (default to dda-fragmentation type: 8)
    frames = acquisition_builder.frame_table.frame_id.values
    frame_types = np.full(len(frames), 8, dtype=int)

    # Mark every precursors_every frame as a precursor frame (type 0)
    for idx in range(len(frames)):
        if idx % precursors_every == 0:
            frame_types[idx] = 0

    # Update the acquisition builder with the new frame types
    acquisition_builder.calculate_frame_types(frame_types=frame_types)

    # Build the precursor frame builder (database for synthetic data)
    synthetic_db_path = str(Path(acquisition_builder.path) / "synthetic_data.db")
    precursor_frame_builder = TimsTofSyntheticPrecursorFrameBuilder(synthetic_db_path)

    # Extract frame IDs corresponding to MS1 (precursor) frames
    ms1_frame_ids = acquisition_builder.frame_table[
        acquisition_builder.frame_table.ms_type == 0
    ].frame_id.values

    # Select precursors from frames (note that we now pass precursors_every)
    selected_p = select_precursors_from_frames(
        precursor_frame_builder=precursor_frame_builder,
        frame_ids=ms1_frame_ids,  # TODO: refine frame ID selection if needed.
        precursor_every=precursors_every,
        exclusion_width=exclusion_width,
        excluded_precursors_df=None,
        max_precursors=max_precursors,
        intensity_threshold=intensity_threshold,
        num_threads=-1,
        batch_size=batch_size,
        **kwargs,
    )

    # Transform the selected precursors into PASEF meta data and reassign their keys
    pasef_meta = transform_selected_precursor_to_pasefmeta(selected_p, precursors_every)

    # Reformat the pasef_meta table as before (mapping keys to lowercase names)
    pasef_meta_names = PASEF_META_COLUMNS_MAPPING.values()
    pasef_meta = pasef_meta[list(pasef_meta_names)]
    inverse_mapping = {v: k for k, v in PASEF_META_COLUMNS_MAPPING.items()}
    pasef_meta = pasef_meta.rename(columns=inverse_mapping)

    pasef_meta = pasef_meta.assign(
        scan_start=lambda df: df["scan_start"].astype(np.int32),
        scan_end=lambda df: df["scan_end"].astype(np.int32),
        precursor=lambda df: df["precursor"].astype(np.int32),
    )

    # Annotate the selected precursor table (this part is left mostly unchanged)
    selected_p["peptide_id"] = selected_p["peptide_id"] * 10 + selected_p["charge_state"]

    selected_p_return = (
        selected_p[["peptide_id", "mz_min", "mz_max", "charge_state", "ScanNumApex", "intensity", "Frame"]]
        .assign(
            average_mz=lambda df: (df["mz_min"] + df["mz_max"]) / 2,
            largest_peak_mz=lambda df: df["mz_min"],
            monoisotopic_mz=lambda df: df["mz_min"],
            peptide_id=lambda df: df["peptide_id"].astype(np.int32),
            intensity=lambda df: df["intensity"].round(),
        )
        .rename(columns={
            "peptide_id": "id",
            "Frame": "parent",
            "ScanNumApex": "scan_number",
            "charge_state": "charge",
        })
        [['id', 'largest_peak_mz', 'average_mz', 'monoisotopic_mz',
          'charge', 'scan_number', 'intensity', 'parent']]
    )

    return pasef_meta, selected_p_return

def get_precursor_isolation_window_from_frame(
    frame: pd.DataFrame,
    ce_bias: float = 54.1984,
    ce_slope: float = -0.0345,
) -> pd.DataFrame:
    """
    Get precursor isolation window from a frame.

    Args:
        frame: DataFrame containing frame data.
        ce_bias: Bias for collision energy calculation.
        ce_slope: Slope for collision energy calculation.

    Returns:
        DataFrame with computed precursor isolation windows.
    """
    # Aggregate required statistics in one step to minimize groupby operations
    agg_funcs = {"mz": ["min", "max"], "intensity": ["sum", "idxmax"]}
    grouped = frame.groupby(["peptide_id", "charge_state"]).agg(agg_funcs)
    grouped.columns = ["mz_min", "mz_max", "intensity", "idxmax"]
    grouped.reset_index(inplace=True)

    # Extract ScanNumApex using the stored index
    grouped["ScanNumApex"] = frame.loc[grouped["idxmax"], "scan"].values
    grouped.drop(columns=["idxmax"], inplace=True)

    # Compute derived values in a vectorized way
    grouped["IsolationWidth"] = np.where((grouped["mz_max"] - grouped["mz_min"]) < 2, 2, 3)
    grouped["IsolationMz"] = (grouped["mz_min"] + grouped["mz_max"]) / 2
    grouped["ScanNumBegin"] = grouped["ScanNumApex"] - 9
    grouped["ScanNumEnd"] = grouped["ScanNumApex"] + 9
    grouped["CollisionEnergy"] = ce_bias + ce_slope * grouped["ScanNumApex"]

    return grouped

def select_precursors_pasef(
    precursors: pd.DataFrame,
    ms1_frame_id: int,
    precursor_every: int,
    excluded_precursors: Optional[pd.DataFrame] = None,
    max_precursors: int = 25,
    intensity_threshold: float = 1200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select precursors from an MS1 frame and distribute them over fragment frames.

    This function applies a duty cycle: for each MS1 frame (ms1_frame_id), it loops
    over available fragment windows (from ms1_frame_id+1 to ms1_frame_id+precursor_every)
    and selects up to max_precursors precursors per window, based on intensity and dynamic exclusion.

    Args:
        precursors: DataFrame with precursor information.
        ms1_frame_id: The frame ID of the current MS1 frame.
        precursor_every: Total number of frames between precursor frames.
                         (The duty cycle will be performed over precursor_every windows.)
        excluded_precursors: DataFrame with already excluded precursor ion IDs.
        max_precursors: Maximum number of precursors to select per fragment window.
        intensity_threshold: Minimum intensity for precursor selection.

    Returns:
        Tuple:
            - DataFrame with selected precursors (each assigned a new 'Frame' value for the fragment window).
            - Updated DataFrame with exclusion information.
    """
    if excluded_precursors is None:
        excluded_precursors = pd.DataFrame(columns=["peptide_id", "Frame"])

    # Filter out precursors below intensity threshold
    n_before_int_thres = precursors.shape[0]
    precursors = precursors[precursors["intensity"] > intensity_threshold]
    Logger.debug("Removed %d precursors below intensity threshold", n_before_int_thres - precursors.shape[0])

    # Exclude precursors already flagged in previous duty cycles
    n_before = precursors.shape[0]
    precursors = precursors[~precursors["peptide_id"].isin(excluded_precursors["peptide_id"])]
    Logger.debug("Removed %d precursors due to exclusion", n_before - precursors.shape[0])

    # Sort by intensity (descending)
    precursors = precursors.sort_values("intensity", ascending=False)

    # New logic: loop over duty cycles (fragment windows)
    duty_cycle_exclusions = {"peptide_id": [], "Parent": []}
    selected_all = pd.DataFrame()

    # Loop over fragment windows (Ramps). The college solution uses range(1, precursor_every+1)
    for i in range(1, precursor_every + 1):
        Logger.debug("Selecting precursors for Ramp %d (ms1_frame_id %d + %d)", i, ms1_frame_id, i)
        # Exclude precursors already selected in this duty cycle
        n_before = precursors.shape[0]
        precursors = precursors[~precursors["peptide_id"].isin(duty_cycle_exclusions["peptide_id"])]
        Logger.debug("Removed %d precursors already sampled, %d left", n_before - precursors.shape[0], precursors.shape[0])
        selected_precursors_one_ramp = []
        all_precursors_one_ramp = precursors.copy(deep=True)
        while (len(selected_precursors_one_ramp) < max_precursors) and (not all_precursors_one_ramp.empty):
            selected_precursor = all_precursors_one_ramp.iloc[0]
            Logger.debug("Selected peptide id %d for Ramp %d", selected_precursor["peptide_id"], i)
            selected_precursors_one_ramp.append(selected_precursor.to_dict())
            duty_cycle_exclusions["peptide_id"].append(selected_precursor["peptide_id"])
            duty_cycle_exclusions["Parent"].append(ms1_frame_id)
            # Remove precursors within the same scan window (dynamic exclusion)
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
) -> pd.DataFrame:
    """
    Select precursors for a list of frames.

    Args:
        precursor_frame_builder: Precursor builder object.
        frame_ids: List of frame IDs.
        precursor_every: Number of frames between precursor acquisitions.
        exclusion_width: Number of frames for dynamic exclusion.
        excluded_precursors_df: DataFrame of precursors to exclude.
        max_precursors: Maximum number of precursors to select per frame.
        intensity_threshold: Minimum intensity for a precursor.
        num_threads: Number of threads to use (-1 uses all cores).
        batch_size: Batch size for parallel processing.

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
            precursors = get_precursor_isolation_window_from_frame(frame.df)
            selected, excluded_precursors_df = select_precursors_pasef(
                precursors,
                ms1_frame_id=frame.frame_id,
                precursor_every=precursor_every,
                excluded_precursors=excluded_precursors_df,
                max_precursors=max_precursors,
                intensity_threshold=intensity_threshold,
            )
            batch_selected = pd.concat([batch_selected, selected], ignore_index=True)
        all_selected.append(batch_selected)

    return pd.concat(all_selected)


def transform_selected_precursor_to_pasefmeta(
    selected_precursors: pd.DataFrame, precursors_every: int
) -> pd.DataFrame:
    """
    Transform selected precursor DataFrame to PASEF meta DataFrame and distribute
    the selections across fragment frames.

    For each group of selected precursors (grouped by the original precursor frame),
    this function reassigns the "Frame" key to one of the fragment frames. Specifically,
    if precursors_every is N, then the fragment frames available are:
        precursor_frame + 1, precursor_frame + 2, ..., precursor_frame + (N - 1)
    and the selected ions are assigned in a round-robin fashion over these frames.

    Args:
        selected_precursors: DataFrame with selected precursors.
        precursors_every: Number of frames between precursor acquisitions.
                         (The fragment frames available will be precursors_every - 1.)

    Returns:
        DataFrame with PASEF meta information, keyed by fragment frame IDs.
    """
    # Create ion id: id = 10 * peptide_id + charge_state
    selected_precursors["Precursor"] = selected_precursors["peptide_id"] * 10 + selected_precursors["charge_state"]

    def assign_fragment_frame(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        num_fragment_frames = precursors_every - 1
        # If no fragment frames are available, leave the Frame unchanged.
        if num_fragment_frames <= 0:
            return group
        # Sort for reproducibility and assign offsets in a round-robin fashion.
        group = group.sort_index()
        offsets = np.arange(len(group)) % num_fragment_frames
        # Reassign the Frame: the first fragment frame is precursor_frame + 1
        group["Frame"] = group["Frame"].iloc[0] + 1 + offsets
        return group

    # Group by the original precursor frame and reassign frame IDs
    distributed = selected_precursors.groupby("Frame", group_keys=False).apply(assign_fragment_frame)

    pasef_meta = distributed[
        [
            "Frame",
            "ScanNumBegin",
            "ScanNumEnd",
            "IsolationMz",
            "IsolationWidth",
            "CollisionEnergy",
            "Precursor",
        ]
    ]
    # Remove duplicates, if any
    pasef_meta = pasef_meta.drop_duplicates(subset=["Frame", "ScanNumBegin", "ScanNumEnd"])
    return pasef_meta