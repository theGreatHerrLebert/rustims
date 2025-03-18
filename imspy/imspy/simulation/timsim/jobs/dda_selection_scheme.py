from pathlib import Path
from typing import Tuple
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd

from imspy.simulation.acquisition import TimsTofAcquisitionBuilder
from imspy.simulation.experiment import TimsTofSyntheticPrecursorFrameBuilder

Logger = logging.getLogger(__name__)


def simulate_dda_pasef_selection_scheme(
    acquisition_builder: TimsTofAcquisitionBuilder,
    verbose: bool,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate DDA selection scheme.

    Args:
        acquisition_builder: Acquisition builder object.
        verbose: Verbosity flag.
        **kwargs: Additional keyword arguments for the selection scheme.

    Returns:
        Tuple of two pandas DataFrames, one holding the DDA PASEF selection scheme and one holding selected precursor information.
    """

    # set frame types to unknown
    frame_types = np.array(
        acquisition_builder.frame_table.frame_id.apply(lambda fid: -1).values
    )

    # sets the frame types and saves the updated frame table to the blueprint
    acquisition_builder.calculate_frame_types(frame_types=frame_types)

    precursor_frame_builder = TimsTofSyntheticPrecursorFrameBuilder(
        str(Path(acquisition_builder.path) / "synthetic_data.db")
    )

    selected_p = select_precursors_from_frames(
        precursor_frame_builder=precursor_frame_builder,
        frame_ids=acquisition_builder.frame_table.frame_id.values,  # TODO: placeholder, need to get the frame IDs of the MS1 frames
        **kwargs,
    )
    pasef_meta = transform_selected_precursor_to_pasefmeta(selected_p)
    # TODO: After the precursor table and pasef_meta table are created, the frame_types need to be set to 0 for MS1 frames and 8 for MS2 frames
    """
    # set frame types to unknown
    frame_types = np.array(
        acquisition_builder.frame_table.frame_id.apply(lambda fid: -1).values
    )
    
    # sets the frame types and saves the updated frame table to the blueprint
    acquisition_builder.calculate_frame_types(frame_types=frame_types)
    """
    return pasef_meta, selected_p
    # raise NotImplementedError(
    #     "Generation of DDA PASEF tables precursor and pasef_meta not yet implemented."
    # )


def get_precursor_isolation_window_from_frame(frame, ce_bias=54.1984, ce_slope=-0.0345):
    """Get precursor isolation window from a frame

    Args:
        frame: pd.DataFrame
        ce_bias: float, the bias of the linear regression fit for CE = slope * ScanNumApex + bias
        ce_slope: float, the slope of the linear regression fit for CE = slope * ScanNumApex + bias

    Returns:
        pd.DataFrame of precursor isolation windows for all available precursors
    """
    # Aggregate basic stats
    available_precursors = (
        frame.groupby(["peptide_id", "charge_state"])
        .agg(
            scan_min=("scan", "min"),
            scan_max=("scan", "max"),
            mz_min=("mz", "min"),
            mz_max=("mz", "max"),
            intensity_sum=("intensity", "sum"),
        )
        .reset_index()
    )

    # Find the scan corresponding to the max intensity
    max_intensity_scans = frame.loc[
        frame.groupby(["peptide_id", "charge_state"])["intensity"].idxmax(),
        ["peptide_id", "charge_state", "scan"],
    ].rename(columns={"scan": "ScanNumApex"})

    # Merge max intensity scan number back
    available_precursors = available_precursors.merge(
        max_intensity_scans, on=["peptide_id", "charge_state"], how="left"
    )

    # Rename columns for consistency
    available_precursors.rename(
        columns={
            "scan_min": "ScanNumMin",
            "scan_max": "ScanNumMax",
            "mz_min": "mz_min",
            "mz_max": "mz_max",
            "intensity_sum": "intensity",
        },
        inplace=True,
    )

    # Compute Isolation Width with a minimum value of 2 or 3 based on threshold
    available_precursors["IsolationWidth"] = np.where(
        (available_precursors["mz_max"] - available_precursors["mz_min"]) < 2, 2, 3
    )

    # Compute Isolation Mz
    available_precursors["IsolationMz"] = (
        available_precursors["mz_min"] + available_precursors["mz_max"]
    ) / 2
    available_precursors["ScanNumEnd"] = available_precursors["ScanNumApex"] + 9
    available_precursors["ScanNumBegin"] = available_precursors["ScanNumApex"] - 9
    available_precursors["CollisionEnergy"] = (
        ce_bias + ce_slope * available_precursors["ScanNumApex"]
    )
    # TODO: possible to automatically infer regression slope and bias? Not sure if it depends on some other settings in the acquisition method
    return available_precursors


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
        Maximum number of precursors to select, it should be a function of tims ramptime,
        by default 25 (set empriically based on the real data)
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
    exclusion_width,
    excluded_precursors_df: pd.DataFrame = None,
    max_precursors=25,
    intensity_threshold=1200,
):
    """Select precursors for a list of frames

    Args:
        precursor_builder: Precursor builder object
        frame_ids: List of frame IDs
        exclusion_width: Width, i.e. number of frames, for dynamic exclusion of precursors
        excluded_precursors_df: DataFrame with excluded precursors because of dynamic exclusion
        max_precursors: Maximum number of precursors to select
        intensity_threshold: Intensity threshold for precursors

    Returns:
        pd.DataFrame: DataFrame with selected precursors from all frame_ids given
    """
    select_precursors_all_frames = pd.DataFrame()
    if excluded_precursors_df is None:
        excluded_precursors_df = pd.DataFrame(columns=["peptide_id", "Frame"])
    for f in tqdm(frame_ids, total=len(frame_ids), desc="Selecting Precursors"):
        Logger.debug("Selecting precursors for frame %d", f)
        excluded_precursors_df = excluded_precursors_df.loc[
            (excluded_precursors_df["Frame"] - f) <= exclusion_width
        ]
        logging.info(
            "time %s",
        )
        frame = precursor_frame_builder.build_precursor_frame_annotated(f)
        precursors = get_precursor_isolation_window_from_frame(frame.df)
        selected_precursors, excluded_precursors_df = select_precursors_pasef(
            precursors,
            f,
            excluded_precursors=excluded_precursors_df,
            max_precursors=max_precursors,
            intensity_threshold=intensity_threshold,
        )  # TODO: can support other selection methods here
        select_precursors_all_frames = pd.concat(
            [select_precursors_all_frames, selected_precursors]
        )

    return select_precursors_all_frames


def transform_selected_precursor_to_pasefmeta(selected_precursors):
    """Transform selected precursor DataFrame to PASEF meta DataFrame

    Args:
        selected_precursors: DataFrame with selected precursors

    Returns:
        DataFrame: DataFrame with PASEF meta information
    """
    selected_precursors["Precursor"] = selected_precursors["peptide_id"]
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
    return pasef_meta
