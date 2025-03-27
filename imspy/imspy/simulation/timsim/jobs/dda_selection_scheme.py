import json
from typing import Tuple
import logging
from itertools import count

import numpy as np
import pandas as pd

from imspy.data.spectrum import MzSpectrum
from imspy.simulation.acquisition import TimsTofAcquisitionBuilder
from tqdm import tqdm

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
        intensity_threshold: float,
        max_precursors: int,
        selection_mode: str = "topN",
        precursor_exclusion_width: int = 25
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate DDA selection scheme with dynamic exclusion.
    Each ion is excluded for `exclusion_frames` frames after being scheduled.
    Also, each precursor is assigned a new unique id.
    """
    if verbose:
        print("Simulating dda-PASEF selection scheme...")
        print(f"precursors_every: {precursors_every}")
        print(f"intensity_threshold: {intensity_threshold}")
        print(f"max_precursors: {max_precursors}")
        print(f"selection_mode: {selection_mode}")
        print(f"exclusion_frames: {precursor_exclusion_width}")

    # retrieve all frame IDs and initialize frame types (default: 8 for fragmentation)
    frames = acquisition_builder.frame_table.frame_id.values

    scan_max = acquisition_builder.synthetics_handle.get_table("scans").scan.max()
    frame_types = np.full(len(frames), 8, dtype=int)

    for idx in range(len(frames)):
        if idx % precursors_every == 0:
            frame_types[idx] = 0

    # get the last precursor frame ID
    max_frame_id = frames[frame_types == 0][-1]

    acquisition_builder.calculate_frame_types(frame_types=frame_types)

    handle = acquisition_builder.synthetics_handle
    ms_1_frames = set(handle.get_table("frames")[handle.get_table("frames").ms_type == 0].frame_id.values)

    # load peptides and ions
    peptides = handle.get_table("peptides")
    ions = handle.get_table("ions")
    ions = pd.merge(peptides, ions.drop(columns=["sequence"]), on=["peptide_id"])

    # transform occurrences and abundances to lists
    ions["frame_occurrence"] = ions.frame_occurrence.apply(json.loads)
    ions["frame_abundance"] = ions.frame_abundance.apply(json.loads)

    ions["scan_occurrence"] = ions.scan_occurrence.apply(json.loads)
    ions["scan_abundance"] = ions.scan_abundance.apply(json.loads)

    # transform json string spectrum to MzSpectrum
    ions["simulated_spectrum"] = ions.simulated_spectrum.apply(MzSpectrum.from_jsons)

    X = create_ion_table(ions, ms_1_frames, intensity_min=intensity_threshold)

    pasef_meta_list = []
    precursors_list = []

    # {ion_id: last_frame_scheduled}
    global_scheduled_ion_tracker = {}
    unique_id_generator = count(start=1)

    for frame in tqdm(np.sort(list(ms_1_frames)), ncols=80, desc="Selecting precursors"):
        X_tmp = X[X.frame_id == frame]
        if len(X_tmp) > 0 and frame < max_frame_id:
            pasef_meta, precursors = schedule_precursors(
                X_tmp,
                k=precursors_every - 1,
                n=max_precursors,
                w=11,
                selection_mode=selection_mode,
                scan_max=scan_max,
                scheduled_ion_tracker=global_scheduled_ion_tracker,
                exclusion_frames=precursor_exclusion_width,
                unique_id_generator=unique_id_generator
            )
            pasef_meta_list.append(pasef_meta)
            precursors_list.append(precursors)

    pasef_meta_df = pd.concat(pasef_meta_list)
    precursors_df = pd.concat(precursors_list)

    # map column names to the inverse of the PASEF_META_COLUMNS_MAPPING and PRECURSOR_MAPPING
    pasef_meta_df = pasef_meta_df.rename(columns={v: k for k, v in PASEF_META_COLUMNS_MAPPING.items()})
    precursors_df = precursors_df.rename(columns={v: k for k, v in PRECURSOR_MAPPING.items()})

    return pasef_meta_df, precursors_df


def create_ion_table(ions, ms_1_frames, intensity_min: float = 1500.0):
    row_list = []

    for index, row in tqdm(ions.iterrows(), total=len(ions), ncols=80, desc="Pre-filtering"):

        frame_start, frame_end = row.frame_occurrence[0], row.frame_occurrence[-1]
        scan_start, scan_end = row.scan_occurrence[0], row.scan_occurrence[-1]

        spectrum = row.simulated_spectrum
        mz = spectrum.mz
        intensity = spectrum.intensity

        # remove all mz where intensity is below 5% of the maximum intensity
        mz = mz[intensity > 0.05 * np.max(intensity)]

        mz_mono = mz[0]
        mz_max = mz[-1]
        mz_max_contrib = row.simulated_spectrum.mz[np.argmax(row.simulated_spectrum.intensity)]

        for frame, frame_abu in zip(row.frame_occurrence, row.frame_abundance):
            if (frame_abu > 0) and (frame in ms_1_frames):
                contrib = row.relative_abundance * row.events * frame_abu
                row_list.append({
                    "frame_id": frame,
                    "ion_id": row.ion_id,
                    "charge": row.charge,
                    "scan_apex": row.scan_occurrence[np.argmax(row.scan_abundance)],
                    "scan_start": scan_start,
                    "scan_end": scan_end,
                    "mz_mono": mz_mono,
                    "mz_max": mz_max,
                    "mz_max_contrib": mz_max_contrib,
                    "ion_intensity": contrib
                })
    T = pd.DataFrame(row_list).sort_values("frame_id")
    return T[T.ion_intensity >= intensity_min]


def schedule_precursors(
        ions,
        k=7,
        n=15,
        w=13,
        ce_bias: float = 54.1984,
        ce_slope: float = -0.0345,
        selection_mode: str = "topN",
        scan_max: int = 913,
        scheduled_ion_tracker: dict = None,
        exclusion_frames: int = 25,
        unique_id_generator=None
):
    """
    Schedules precursors for a given MS1 frame.

    - scheduled_ion_tracker: a dict mapping ion_id -> last frame (from ions.frame_id) where it was scheduled.
      An ion is only eligible if (current_frame - last_scheduled_frame) >= exclusion_frames.
    - unique_id_generator: a generator that yields unique IDs for precursors.
    """
    frame_id_precursor = ions.frame_id.values[0]
    current_frame = frame_id_precursor  # current MS1 frame id

    known_selection_modes = ["topN", "random"]

    if selection_mode.lower() == "topn":
        ions_sorted = ions.sort_values(by="ion_intensity", ascending=False).copy()
    elif selection_mode.lower() == "random":
        ions_sorted = ions.sample(frac=1.0, random_state=None).copy()
    else:
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be one of {known_selection_modes}")

    # Step 2: Initialize list of k empty fragment frames
    fragment_frames = [[] for _ in range(k)]

    # Ensure scheduled_ion_tracker is a dict
    if scheduled_ion_tracker is None:
        scheduled_ion_tracker = {}

    scheduled_rows = []
    precursor_rows = []

    for _, ion in ions_sorted.iterrows():
        # check if ion was scheduled and if it was scheduled in the last exclusion_frames frames
        if ion.ion_id in scheduled_ion_tracker:
            last_frame = scheduled_ion_tracker[ion.ion_id]
            if (current_frame - last_frame) < exclusion_frames:
                continue

        assigned = False
        new_start = ion.scan_apex - w
        new_end = ion.scan_apex + w

        if new_start < 0:
            new_start = 0

        if new_end > scan_max:
            new_end = scan_max

        for fragment_frame_index in range(k):
            current_frame_list = fragment_frames[fragment_frame_index]

            if len(current_frame_list) >= n:
                continue

            conflict = any(
                not (new_end < (existing_apex - w) or new_start > (existing_apex + w))
                for existing_apex, _ in current_frame_list
            )

            if not conflict:
                current_frame_list.append((ion.scan_apex, ion.ion_id))
                # Calculate the frame for the scheduled precursor (offset by fragment frame index)
                frame_id = frame_id_precursor + fragment_frame_index + 1

                mz_max_contrib = ion.mz_max_contrib
                mz_mono = ion.mz_mono
                mz_avg = (mz_mono + ion.mz_max) / 2.0
                isolation_width = 3.0 if (ion.mz_max - mz_mono) > 2.0 else 2.0

                # Get a new unique ID for this precursor
                new_precursor_id = next(unique_id_generator) if unique_id_generator is not None else ion.ion_id

                scheduled_rows.append({
                    "Frame": frame_id,
                    "ScanNumBegin": new_start,
                    "ScanNumEnd": new_end,
                    "IsolationMz": mz_max_contrib,
                    "IsolationWidth": isolation_width,
                    "CollisionEnergy": ce_bias + ce_slope * ion.scan_apex,
                    "Precursor": new_precursor_id  # using new unique id instead of ion id
                })

                precursor_rows.append({
                    "Id": new_precursor_id,  # unique id replacing ion.ion_id
                    "LargestPeakMz": mz_max_contrib,
                    "AverageMz": mz_avg,
                    "MonoisotopicMz": mz_mono,
                    "Charge": ion.charge,
                    "ScanNumber": ion.scan_apex,
                    "Intensity": ion.ion_intensity,
                    "Parent": frame_id_precursor
                })

                # Update the last scheduled frame for this ion
                scheduled_ion_tracker[ion.ion_id] = current_frame
                assigned = True
                break

        if not assigned:
            Logger.debug("Ion id %s could not be scheduled in any fragment frame.", ion.ion_id)

    # If no ions were scheduled, return empty DataFrames with the expected columns
    if scheduled_rows:
        schedule_df = pd.DataFrame(scheduled_rows).sort_values(by=["Frame", "ScanNumBegin"])
    else:
        schedule_df = pd.DataFrame(
            columns=["Frame", "ScanNumBegin", "ScanNumEnd", "IsolationMz", "IsolationWidth", "CollisionEnergy",
                     "Precursor"])

    if precursor_rows:
        precursors_df = pd.DataFrame(precursor_rows).sort_values(by="ScanNumber")
    else:
        precursors_df = pd.DataFrame(
            columns=["Id", "LargestPeakMz", "AverageMz", "MonoisotopicMz", "Charge", "ScanNumber", "Intensity",
                     "Parent"])

    # Ensure selected columns are integers if the DataFrames are not empty
    if not schedule_df.empty:
        schedule_df["Precursor"] = schedule_df["Precursor"].astype(int)
        schedule_df["Frame"] = schedule_df["Frame"].astype(int)
        schedule_df["ScanNumBegin"] = schedule_df["ScanNumBegin"].astype(int)
        schedule_df["ScanNumEnd"] = schedule_df["ScanNumEnd"].astype(int)

    if not precursors_df.empty:
        precursors_df["Id"] = precursors_df["Id"].astype(int)
        precursors_df["Parent"] = precursors_df["Parent"].astype(int)
        precursors_df["Charge"] = precursors_df["Charge"].astype(int)

    return schedule_df, precursors_df