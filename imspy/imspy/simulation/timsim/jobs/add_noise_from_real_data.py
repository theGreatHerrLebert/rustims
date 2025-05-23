import pandas as pd
from typing import List, Optional

from imspy.simulation.acquisition import TimsTofAcquisitionBuilderDIA
from imspy.timstof.frame import TimsFrame

import numpy as np

def get_center_scans_per_frame_id(frame_id, pasef_meta) -> List[int]:
    """
    Get the center scans for a given frame id.
    Args:
        frame_id:
        pasef_meta:

    Returns:
        List[int]: List of center scans.
    """
    if pasef_meta is None:
        return []

    pasef_meta_f = pasef_meta[pasef_meta.frame == frame_id]

    if len(pasef_meta_f) == 0:
        return []

    centers = []

    for index, row in pasef_meta_f.iterrows():
        window_mid = int(np.round((row.scan_end - row.scan_start) / 2))
        centers.append(int(row.scan_start) + window_mid)

    return centers


def add_real_data_noise_to_frames(
        acquisition_builder: TimsTofAcquisitionBuilderDIA,
        frames: List[TimsFrame],
        intensity_max_precursor: float = 30,
        intensity_max_fragment: float = 30,
        precursor_sample_fraction: float = 0.2,
        fragment_sample_fraction: float = 0.2,
        num_precursor_frames: int = 5,
        num_fragment_frames: int = 5,
        acquisition_mode: str = 'DIA',
        pasef_meta: Optional[pd.DataFrame] = None,
) -> List[TimsFrame]:
    """Add noise to frame.

    Args:
        acquisition_builder: Acquisition builder.
        frames (List[TimsFrame]): Frames.
        intensity_max_precursor (float): Maximum intensity for precursor.
        intensity_max_fragment (float): Maximum intensity for fragment.
        precursor_sample_fraction (float): Sample fraction for precursor.
        fragment_sample_fraction (float): Sample fraction for fragment.
        num_precursor_frames (int): Number of precursor frames.
        num_fragment_frames (int): Number of fragment frames.
        acquisition_mode (str): Acquisition mode.
        pasef_meta (Optional[pd.DataFrame]): PASEF metadata.

    Returns:
        List[TimsFrame]: Frames.
    """
    r_list = []

    # DDA noise not yet implemented
    if acquisition_mode == 'DDA':
        # Get the maximum number of scans to avoid picking scan values out of range during noise generation
        max_scan = acquisition_builder.tdf_writer.helper_handle.num_scans

        if pasef_meta is None:
            raise ValueError("PASEF metadata is required for DDA acquisition mode.")

        for frame in frames:
            # if the frame is a fragment frame, we need to get the center scans
            if frame.ms_type == 8:
                # get the center scans for the frame
                scan_center_list = get_center_scans_per_frame_id(frame.frame_id, pasef_meta)
                # if scan_center_list is empty, append the frame
                if len(scan_center_list) == 0:
                    r_list.append(frame)
                    continue

                # extract real data noise which will be moved to the center scans
                noise = acquisition_builder.tdf_writer.helper_handle.sample_pasef_fragments_random(
                    scan_center_list, max_scan
                )

                r_list.append(frame + noise)

            # if the frame is a precursor frame, we need to simply add noise the same way as in DIA
            else:
                noise = acquisition_builder.tdf_writer.helper_handle.sample_precursor_signal(
                    num_frames=num_precursor_frames,
                    max_intensity=intensity_max_precursor,
                    take_probability=precursor_sample_fraction
                )
                r_list.append(frame + noise)

    else:
        d = acquisition_builder.frames_to_window_groups
        window_group_dict = dict(zip(d['frame'], d['window_group']))
        fragment_frames = set(window_group_dict.keys())

        for frame in frames:
            if frame.frame_id in fragment_frames:
                window_group = window_group_dict[frame.frame_id]

                noise = acquisition_builder.tdf_writer.helper_handle.sample_fragment_signal(
                    num_frames=num_fragment_frames,
                    window_group=window_group,
                    max_intensity=intensity_max_fragment,
                    take_probability=fragment_sample_fraction
                )
                r_list.append(frame + noise)
            else:
                noise = acquisition_builder.tdf_writer.helper_handle.sample_precursor_signal(
                    num_frames=num_precursor_frames,
                    max_intensity=intensity_max_precursor,
                    take_probability=precursor_sample_fraction
                )
                r_list.append(frame + noise)

    return r_list
