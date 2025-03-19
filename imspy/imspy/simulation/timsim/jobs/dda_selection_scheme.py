import os
from pathlib import Path
from typing import Tuple
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd

from imspy.simulation.acquisition import TimsTofAcquisitionBuilder
from imspy.simulation.experiment import TimsTofSyntheticPrecursorFrameBuilder

Logger = logging.getLogger(__name__)

pasef_meta_columns_mapping = {
    'frame': 'Frame',
    'scan_start': 'ScanNumBegin',
    'scan_end': 'ScanNumEnd',
    'isolation_mz': 'IsolationMz',
    'isolation_width': 'IsolationWidth',
    'collision_energy': 'CollisionEnergy',
    'precursor': 'Precursor',
}

precursor_mapping = {
    'id': 'Id',
    'largest_peak_mz': 'LargestPeakMz',
    'average_mz': 'AverageMz',
    'monoisotopic_mz': 'MonoisotopicMz',
    'charge': 'Charge',
    'scan_number': 'ScanNumber',
    'intensity': 'Intensity',
    'parent': 'Parent',
}


def simulate_dda_pasef_selection_scheme(
    acquisition_builder: TimsTofAcquisitionBuilder,
    verbose: bool,
    precursors_every,
    batch_size,
    intensity_threshold,
    max_precursors,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate DDA selection scheme.

    Args:
        acquisition_builder: Acquisition builder object.
        verbose: Verbosity flag.
        precursors_every: Number of frames between precursors.
        batch_size: Batch size for parallel processing.
        intensity_threshold: Intensity threshold for precursors.
        max_precursors: Maximum number of precursors to select.
        **kwargs: Additional keyword arguments for the selection scheme.

    Returns:
        Tuple of two pandas DataFrames, one holding the DDA PASEF selection scheme and one holding selected precursor information.
    """


    frames = acquisition_builder.frame_table.frame_id.values

    # set all frames types to dda-fragmentation by default (8)
    frame_types = np.repeat(8, len(frames))

    # go over all frames and set every precursors_every frame to dda-precursor
    for i, frame_id in enumerate(frames):
        if i % precursors_every == 0:
            frame_types[i] = 0

    # sets the frame types and saves the updated frame table to the blueprint
    acquisition_builder.calculate_frame_types(frame_types=frame_types)

    precursor_frame_builder = TimsTofSyntheticPrecursorFrameBuilder(
        str(Path(acquisition_builder.path) / "synthetic_data.db")
    )

    precursor_ids = acquisition_builder.frame_table[
        acquisition_builder.frame_table.ms_type == 0
    ].frame_id.values

    selected_p = select_precursors_from_frames(
        precursor_frame_builder=precursor_frame_builder,
        frame_ids=precursor_ids,  # TODO: placeholder, need to get the frame IDs of the MS1 frames
        excluded_precursors_df=None,
        max_precursors=max_precursors,
        intensity_threshold=intensity_threshold,
        num_threads=-1,
        batch_size=batch_size,
        **kwargs,
    )
    pasef_meta = transform_selected_precursor_to_pasefmeta(selected_p)
    r_copy = pasef_meta.copy()

    # TODO: After the precursor table and pasef_meta table are created, the frame_types need to be set to 0 for MS1 frames and 8 for MS2 frames

    # select columns from pasef meta keys
    pasef_meta_names = pasef_meta_columns_mapping.values()
    pasef_meta = pasef_meta[list(pasef_meta_names)]
    # set to the inverse of the columns mapping
    pasef_meta = pasef_meta.rename(columns={v: k for k, v in pasef_meta_columns_mapping.items()})

    pasef_meta = pasef_meta.assign(
        scan_start=lambda df: df["scan_start"].astype(np.int32),
        scan_end=lambda df: df["scan_end"].astype(np.int32),
        precursor=lambda df: df["precursor"].astype(np.int32)
    )

    # TODO: maybe return ion id instead of peptide id in the annotated frame?
    selected_p["peptide_id"] = selected_p["peptide_id"] * 10 + selected_p["charge_state"]

    selected_p_return = (
        selected_p[["peptide_id", "mz_min", "mz_max", "charge_state", "ScanNumApex", "intensity", "Frame"]]
        .assign(
            # compute the average m/z without applying row-wise lambda
            average_mz=lambda df: (df["mz_min"] + df["mz_max"]) / 2,
            # copy mz_min to new columns
            largest_peak_mz=lambda df: df["mz_min"],
            monoisotopic_mz=lambda df: df["mz_min"],
            # convert peptide_id to int32
            peptide_id=lambda df: df["peptide_id"].astype(np.int32),
            # round the intensity values
            intensity=lambda df: df["intensity"].round()
        )
        .rename(columns={
            "peptide_id": "id",
            "Frame": "parent",
            "ScanNumApex": "scan_number",
            "charge_state": "charge"
        })
            # order
        [['id', 'largest_peak_mz', 'average_mz', 'monoisotopic_mz',
          'charge', 'scan_number', 'intensity', 'parent']]
    )

    return pasef_meta, selected_p_return

def get_precursor_isolation_window_from_frame(frame, ce_bias=54.1984, ce_slope=-0.0345):
    """Get precursor isolation window from a frame

    Args:
        frame: pd.DataFrame
        ce_bias: float, the bias of the linear regression fit for CE = slope * ScanNumApex + bias
        ce_slope: float, the slope of the linear regression fit for CE = slope * ScanNumApex + bias

    Returns:
        pd.DataFrame of precursor isolation windows for all available precursors
    """
    # Aggregate required stats in one step to minimize groupby operations
    agg_funcs = {
        "mz": ["min", "max"],
        "intensity": ["sum", "idxmax"],
    }

    grouped = frame.groupby(["peptide_id", "charge_state"]).agg(agg_funcs)
    grouped.columns = [
        "mz_min",
        "mz_max",
        "intensity",
        "idxmax",
    ]
    grouped.reset_index(inplace=True)

    # Extract ScanNumApex using the stored index
    grouped["ScanNumApex"] = frame.loc[grouped["idxmax"], "scan"].values

    # Drop unnecessary column
    grouped.drop(columns=["idxmax"], inplace=True)

    # Compute derived values in a vectorized way
    grouped["IsolationWidth"] = np.where(
        (grouped["mz_max"] - grouped["mz_min"]) < 2, 2, 3
    )
    grouped["IsolationMz"] = (grouped["mz_min"] + grouped["mz_max"]) / 2
    grouped["ScanNumBegin"] = grouped["ScanNumApex"] - 9
    grouped["ScanNumEnd"] = grouped["ScanNumApex"] + 9
    grouped["CollisionEnergy"] = ce_bias + ce_slope * grouped["ScanNumApex"]

    return grouped


def select_precursors_pasef(
    precursors: pd.DataFrame,
    frame_id: int,
    excluded_precursors: pd.DataFrame = None,
    max_precursors: int = 25,
    intensity_threshold: float = 1200,
):
    """
    Select precursors for a PASEF frame
    It selects the top `max_precursors` precursors based on intensity, excluding precursors
    below `intensity_threshold` and precursors that are within the same scan window as the
    selected precursor.

    Parameters
    ----------
    precursors : pd.DataFrame
        DataFrame with precursor information
    frame_id : int
        Frame ID
    excluded_precursors : pd.DataFrame, optional
        DataFrame with precursors to exclude, by default None
    max_precursors : int, optional
        Maximum number of precursors to select, it should be a function of tims ramp time,
        by default 25 (set empirically based on the real data)
    intensity_threshold : float, optional
        Intensity threshold to exclude precursors, by default 1200

    Returns
    -------
    pd.DataFrame
        DataFrame with selected precursors
    pd.DataFrame
        DataFrame with excluded precursors
    """
    if excluded_precursors is None:
        excluded_precursors = pd.DataFrame(columns=["peptide_id", "Frame"])

    # Exclude precursors below intensity threshold
    n_before_int_thres = precursors.shape[0]
    precursors = precursors[precursors["intensity"] > intensity_threshold]
    Logger.debug(
        "Removed %d precursors below intensity threshold",
        n_before_int_thres - precursors.shape[0],
    )

    # Exclude already excluded precursors
    n_before = precursors.shape[0]
    precursors = precursors[
        ~precursors["peptide_id"].isin(excluded_precursors["peptide_id"])
    ]
    Logger.debug(
        "Removed %d precursors due to exclusion", n_before - precursors.shape[0]
    )

    # Sort by intensity (descending)
    precursors = precursors.sort_values("intensity", ascending=False)
    new_exclusion = []
    selected_precursors = []
    while len(selected_precursors) < max_precursors and not precursors.empty:
        Logger.debug("Selecting precursor %d", len(selected_precursors) + 1)

        selected_precursor = precursors.iloc[0]  # Pick the highest-intensity precursor
        selected_precursors.append(selected_precursor.to_dict())  # Store as dict

        # Add to exclusion list
        new_exclusion.append(
            {"peptide_id": selected_precursor["peptide_id"], "Frame": frame_id}
        )

        # Remove precursors within the same scan window
        precursors = precursors[
            ~(
                (precursors["ScanNumApex"] <= selected_precursor["ScanNumEnd"])
                & (precursors["ScanNumApex"] >= selected_precursor["ScanNumBegin"])
            )
        ]

    # Convert selected precursors to a DataFrame
    selected_precursors = pd.DataFrame(selected_precursors)

    if not selected_precursors.empty:
        selected_precursors["Frame"] = frame_id
    new_exclusion = pd.DataFrame(new_exclusion)
    excluded_precursors = pd.concat([excluded_precursors, new_exclusion])
    return selected_precursors, excluded_precursors


def select_precursors_from_frames(
    precursor_frame_builder,
    frame_ids,
    exclusion_width: int = 25,
    excluded_precursors_df: pd.DataFrame = None,
    max_precursors=25,
    intensity_threshold=1200,
    num_threads=-1,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Select precursors for a list of frames

    Args:
        precursor_builder: Precursor builder object
        frame_ids: List of frame IDs
        exclusion_width: Width, i.e. number of frames, for dynamic exclusion of precursors
        excluded_precursors_df: DataFrame with excluded precursors because of dynamic exclusion
        max_precursors: Maximum number of precursors to select
        intensity_threshold: Intensity threshold for precursors
        num_threads: Number of threads to use
        batch_size: Batch size for parallel processing

    Returns:
        pd.DataFrame: DataFrame with selected precursors from all frame_ids given
    """

    if num_threads == -1:
        num_threads = os.cpu_count()

    select_precursors_all_frames = pd.DataFrame()
    if excluded_precursors_df is None:
        excluded_precursors_df = pd.DataFrame(columns=["peptide_id", "Frame"])

        # create batched frame_ids
        num_splits = len(frame_ids) // batch_size
        frame_ids = np.array_split(frame_ids, num_splits)

        select_precursors_all_frames_list = []

        # parallel processing
        for i, batch in tqdm(enumerate(frame_ids), total=len(frame_ids), desc="Selecting Precursors", ncols=80):

            # build precursor frames for the batch
            build_batch = precursor_frame_builder.build_precursor_frames_annotated(batch, num_threads=num_threads)

            # go over all frames in the batch
            for frame in build_batch:

                Logger.debug("Selecting precursors for frame %d", frame.frame_id)

                # exclude precursors based on the exclusion width
                excluded_precursors_df = excluded_precursors_df.loc[
                    (excluded_precursors_df["Frame"] - frame.frame_id) <= exclusion_width
                    ]

                precursors = get_precursor_isolation_window_from_frame(frame.df)

                selected_precursors, excluded_precursors_df = select_precursors_pasef(
                    precursors,
                    frame.frame_id,
                    excluded_precursors=excluded_precursors_df,
                    max_precursors=max_precursors,
                    intensity_threshold=intensity_threshold,
                )  # TODO: can support other selection methods here

                select_precursors_all_frames = pd.concat(
                    [select_precursors_all_frames, selected_precursors]
                )

            select_precursors_all_frames_list.append(select_precursors_all_frames)

        return pd.concat(select_precursors_all_frames_list)


def transform_selected_precursor_to_pasefmeta(selected_precursors):
    """Transform selected precursor DataFrame to PASEF meta DataFrame

    Args:
        selected_precursors: DataFrame with selected precursors

    Returns:
        DataFrame: DataFrame with PASEF meta information
    """
    # TODO: maybe make this a little bit more robust...
    # need to create ion id like id = 10 * peptide_id + charge_state
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

    # TODO: find why there are duplicates
    pasef_meta = pasef_meta.drop_duplicates(subset=["Frame", "ScanNumBegin", "ScanNumEnd"])
    return pasef_meta
