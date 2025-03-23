import json

from typing import Tuple, Any
import logging

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

    # retrieve all frame IDs and initialize frame types (default: 8 for fragmentation)
    frames = acquisition_builder.frame_table.frame_id.values
    max_frame_id = np.max(frames)
    frame_types = np.full(len(frames), 8, dtype=int)

    for idx in range(len(frames)):
        if idx % precursors_every == 0:
            frame_types[idx] = 0

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

    for frame in tqdm(np.sort(list(ms_1_frames)), ncols=80, desc="Selecting precursors"):
        X_tmp = X[X.frame_id == frame]
        if len(X_tmp) > 0 and frame < max_frame_id:
            pasef_meta, precursors = schedule_precursors(X_tmp, k=precursors_every - 1, n=max_precursors, w=13)
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

    for index, row in tqdm(ions.iterrows(), total=len(ions), ncols=80):

        frame_start, frame_end = row.frame_occurrence[0], row.frame_occurrence[-1]
        scan_start, scan_end = row.scan_occurrence[0], row.scan_occurrence[-1]

        mz_mono = row.simulated_spectrum.mz[0]
        mz_max = row.simulated_spectrum.mz[-1]
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
        ce_slope: float = -0.0345):
    frame_id_precursor = ions.frame_id.values[0]

    # Step 1: Sort ions by intensity (descending)
    ions_sorted = ions.sort_values(by="ion_intensity", ascending=False).copy()

    # Step 2: Initialize list of k empty fragment frames
    fragment_frames = [[] for _ in range(k)]

    # Step 3: Assignment loop
    scheduled_rows = []
    precursor_rows = []

    for _, ion in ions_sorted.iterrows():
        assigned = False

        new_start = ion.scan_apex - w
        new_end = ion.scan_apex + w

        for fragment_frame_index in range(k):
            current_frame = fragment_frames[fragment_frame_index]

            # skip full frames
            if len(current_frame) >= n:
                continue

            # check scan overlap with existing ions in this frame
            conflict = any(
                not (new_end < (existing_apex - w) or new_start > (existing_apex + w))
                for existing_apex, _ in current_frame
            )

            if not conflict:
                current_frame.append((ion.scan_apex, ion.ion_id))

                frame_id = frame_id_precursor + fragment_frame_index + 1

                mz_max_contrib = ion.mz_max_contrib
                mz_mono = ion.mz_mono
                mz_avg = (mz_mono + ion.mz_max) / 2.0
                isolation_width = 3.0 if (ion.mz_max - mz_mono) > 2.0 else 2.0

                scheduled_rows.append({
                    "Frame": frame_id,
                    "ScanNumBegin": new_start,
                    "ScanNumEnd": new_end,
                    "IsolationMz": mz_max_contrib,
                    "IsolationWidth": isolation_width,
                    "CollisionEnergy": ce_bias + ce_slope * ion.scan_apex,
                    "Precursor": ion.ion_id
                })

                precursor_rows.append({
                    "Id": ion.ion_id,
                    "LargestPeakMz": mz_max_contrib,
                    "AverageMz": mz_avg,
                    "MonoisotopicMz": mz_mono,
                    "Charge": ion.charge,
                    "ScanNumber": ion.scan_apex,
                    "Intensity": ion.ion_intensity,
                    "Parent": frame_id_precursor
                })

                assigned = True
                break

        if not assigned:
            pass

    schedule_df = pd.DataFrame(scheduled_rows).sort_values(by=["Frame", "ScanNumBegin"])
    precursors_df = pd.DataFrame(precursor_rows).sort_values(by="ScanNumber")

    # ensure selected columns are integers
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
