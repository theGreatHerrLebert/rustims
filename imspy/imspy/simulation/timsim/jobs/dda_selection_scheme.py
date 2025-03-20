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
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate DDA selection scheme.

    Args:
        acquisition_builder: Acquisition builder object.
        verbose: Verbosity flag.
        precursors_every: Number of frames between precursors.
        batch_size: Batch size for parallel processing.
        intensity_threshold: Intensity threshold for precursors.
        max_precursors: Maximum number of precursors to select.
        **kwargs: Additional keyword arguments for the selection scheme.

    Returns:
        Tuple of two pandas DataFrames:
            - PASEF meta information.
            - Selected precursor information.
    """
    # Retrieve frame IDs and initialize frame types (default to dda-fragmentation: 8)
    frame_ids = acquisition_builder.frame_table.frame_id.values
    frame_types = np.full(len(frame_ids), 8, dtype=int)

    # Mark every `precursors_every` frame as dda-precursor (0)
    for idx, _ in enumerate(frame_ids):
        if idx % precursors_every == 0:
            frame_types[idx] = 0

    # Update the acquisition builder with the new frame types
    acquisition_builder.calculate_frame_types(frame_types=frame_types)

    # Build precursor frame builder with the synthetic database path
    synthetic_db_path = str(Path(acquisition_builder.path) / "synthetic_data.db")
    precursor_frame_builder = TimsTofSyntheticPrecursorFrameBuilder(synthetic_db_path)

    # Extract frame IDs for MS1 frames (ms_type == 0)
    ms1_frame_ids = acquisition_builder.frame_table[
        acquisition_builder.frame_table.ms_type == 0
    ].frame_id.values

    # Select precursors from frames
    selected_precursors = select_precursors_from_frames(
        precursor_frame_builder=precursor_frame_builder,
        frame_ids=ms1_frame_ids,  # TODO: placeholder; refine frame ID selection if needed.
        excluded_precursors_df=None,
        max_precursors=max_precursors,
        intensity_threshold=intensity_threshold,
        num_threads=-1,
        batch_size=batch_size,
        **kwargs,
    )

    # Transform selected precursor data into PASEF meta data format
    pasef_meta_df = transform_selected_precursor_to_pasefmeta(selected_precursors)

    # Rearrange and reformat PASEF meta columns
    pasef_meta_df = pasef_meta_df[list(PASEF_META_COLUMNS_MAPPING.values())]
    # Rename columns to use the inverse mapping
    inverse_mapping = {v: k for k, v in PASEF_META_COLUMNS_MAPPING.items()}
    pasef_meta_df = pasef_meta_df.rename(columns=inverse_mapping)

    pasef_meta_df = pasef_meta_df.assign(
        scan_start=lambda df: df["scan_start"].astype(np.int32),
        scan_end=lambda df: df["scan_end"].astype(np.int32),
        precursor=lambda df: df["precursor"].astype(np.int32),
    )

    # TODO: Consider returning ion id instead of peptide id in the annotated frame.
    # Update peptide_id to encode ion information.
    selected_precursors["peptide_id"] = selected_precursors["peptide_id"] * 10 + selected_precursors["charge_state"]

    # Create a refined DataFrame for selected precursors
    selected_precursors_df = (
        selected_precursors[
            ["peptide_id", "mz_min", "mz_max", "charge_state", "ScanNumApex", "intensity", "Frame"]
        ]
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
        [["id", "largest_peak_mz", "average_mz", "monoisotopic_mz",
          "charge", "scan_number", "intensity", "parent"]]
    )

    return pasef_meta_df, selected_precursors_df


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
    # Aggregate required statistics to minimize groupby operations
    agg_funcs = {"mz": ["min", "max"], "intensity": ["sum", "idxmax"]}
    grouped = frame.groupby(["peptide_id", "charge_state"]).agg(agg_funcs)

    # Flatten the multi-index columns
    grouped.columns = ["mz_min", "mz_max", "intensity", "idxmax"]
    grouped.reset_index(inplace=True)

    # Retrieve the apex scan number using the stored index
    grouped["ScanNumApex"] = frame.loc[grouped["idxmax"], "scan"].values
    grouped.drop(columns=["idxmax"], inplace=True)

    # Compute additional derived columns in a vectorized way
    grouped["IsolationWidth"] = np.where((grouped["mz_max"] - grouped["mz_min"]) < 2, 2, 3)
    grouped["IsolationMz"] = (grouped["mz_min"] + grouped["mz_max"]) / 2
    grouped["ScanNumBegin"] = grouped["ScanNumApex"] - 9
    grouped["ScanNumEnd"] = grouped["ScanNumApex"] + 9
    grouped["CollisionEnergy"] = ce_bias + ce_slope * grouped["ScanNumApex"]

    return grouped


def select_precursors_pasef(
    precursors: pd.DataFrame,
    frame_id: int,
    excluded_precursors: Optional[pd.DataFrame] = None,
    max_precursors: int = 25,
    intensity_threshold: float = 1200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select precursors for a PASEF frame.

    This function selects the top `max_precursors` precursors based on intensity,
    while excluding precursors below the given intensity threshold and those that fall
    within the same scan window as a previously selected precursor.

    Parameters
    ----------
    precursors : pd.DataFrame
        DataFrame with precursor information.
    frame_id : int
        Frame ID.
    excluded_precursors : pd.DataFrame, optional
        DataFrame with precursors to exclude, by default None.
    max_precursors : int, optional
        Maximum number of precursors to select, by default 25.
    intensity_threshold : float, optional
        Intensity threshold to filter out precursors, by default 1200.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - DataFrame with selected precursors.
        - Updated DataFrame with excluded precursors.
    """
    if excluded_precursors is None:
        excluded_precursors = pd.DataFrame(columns=["peptide_id", "Frame"])

    # Filter out low-intensity precursors
    initial_count = precursors.shape[0]
    precursors = precursors[precursors["intensity"] > intensity_threshold]
    Logger.debug("Removed %d precursors below intensity threshold",
                 initial_count - precursors.shape[0])

    # Exclude precursors that are already marked for exclusion
    count_before = precursors.shape[0]
    precursors = precursors[~precursors["peptide_id"].isin(excluded_precursors["peptide_id"])]
    Logger.debug("Removed %d precursors due to exclusion", count_before - precursors.shape[0])

    # Sort precursors in descending order by intensity
    precursors = precursors.sort_values("intensity", ascending=False)

    selected: List[dict] = []
    new_exclusions: List[dict] = []

    # Iteratively select the highest-intensity precursor and update exclusions
    while len(selected) < max_precursors and not precursors.empty:
        Logger.debug("Selecting precursor %d", len(selected) + 1)
        current = precursors.iloc[0]
        selected.append(current.to_dict())
        new_exclusions.append({"peptide_id": current["peptide_id"], "Frame": frame_id})

        # Remove precursors that fall within the scan window of the selected precursor
        precursors = precursors[
            ~((precursors["ScanNumApex"] <= current["ScanNumEnd"]) &
              (precursors["ScanNumApex"] >= current["ScanNumBegin"]))
        ]

    selected_df = pd.DataFrame(selected)
    if not selected_df.empty:
        selected_df["Frame"] = frame_id

    exclusions_df = pd.DataFrame(new_exclusions)
    excluded_precursors = pd.concat([excluded_precursors, exclusions_df])
    return selected_df, excluded_precursors


def select_precursors_from_frames(
    precursor_frame_builder: Any,
    frame_ids: List[int],
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
        exclusion_width: Number of frames to consider for dynamic exclusion.
        excluded_precursors_df: DataFrame of precursors to exclude.
        max_precursors: Maximum number of precursors to select per frame.
        intensity_threshold: Minimum intensity required for a precursor.
        num_threads: Number of threads to use (-1 uses all available cores).
        batch_size: Batch size for parallel processing.

    Returns:
        DataFrame with selected precursors from all provided frames.
    """
    if num_threads == -1:
        num_threads = os.cpu_count()

    if excluded_precursors_df is None:
        excluded_precursors_df = pd.DataFrame(columns=["peptide_id", "Frame"])

    # Split frame IDs into batches
    num_batches = max(1, len(frame_ids) // batch_size)
    frame_batches = np.array_split(frame_ids, num_batches)

    all_selected_precursors: List[pd.DataFrame] = []

    for batch in tqdm(frame_batches, total=len(frame_batches), desc="Selecting Precursors", ncols=80):
        # Build precursor frames for the current batch
        precursor_frames = precursor_frame_builder.build_precursor_frames_annotated(batch, num_threads=num_threads)
        batch_selected = pd.DataFrame()

        for frame in precursor_frames:
            Logger.debug("Selecting precursors for frame %d", frame.frame_id)

            # Update exclusion: keep only those within the specified exclusion width
            excluded_precursors_df = excluded_precursors_df.loc[
                (excluded_precursors_df["Frame"] - frame.frame_id) <= exclusion_width
            ]

            # Compute precursor isolation window for the current frame
            precursors = get_precursor_isolation_window_from_frame(frame.df)

            # Select precursors for the current frame
            selected, excluded_precursors_df = select_precursors_pasef(
                precursors,
                frame.frame_id,
                excluded_precursors=excluded_precursors_df,
                max_precursors=max_precursors,
                intensity_threshold=intensity_threshold,
            )
            batch_selected = pd.concat([batch_selected, selected])

        all_selected_precursors.append(batch_selected)

    return pd.concat(all_selected_precursors)


def transform_selected_precursor_to_pasefmeta(selected_precursors: pd.DataFrame) -> pd.DataFrame:
    """
    Transform selected precursor DataFrame to PASEF meta DataFrame.

    Args:
        selected_precursors: DataFrame with selected precursors.

    Returns:
        DataFrame with PASEF meta information.
    """
    # Create ion id: id = 10 * peptide_id + charge_state
    selected_precursors["Precursor"] = selected_precursors["peptide_id"] * 10 + selected_precursors["charge_state"]

    pasef_meta = selected_precursors[
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
    # Remove duplicate entries if any exist
    pasef_meta = pasef_meta.drop_duplicates(subset=["Frame", "ScanNumBegin", "ScanNumEnd"])
    return pasef_meta