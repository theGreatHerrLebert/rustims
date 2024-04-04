from typing import List

import numpy as np

from imspy.simulation.acquisition import TimsTofAcquisitionBuilderDIA
from imspy.timstof.frame import TimsFrame


def get_precursor_noise(handle, sample_fraction: float = 0.5, max_intensity: float = 30) -> TimsFrame:
    """Get noise from precursor frames.

    Args:
        handle: TimsTOF handle.
        sample_fraction (float): Sample fraction.
        max_intensity (float): Maximum intensity.

    Returns:
        TimsFrame: Frame.
    """
    F_A = handle[np.random.choice(handle.precursor_frames[150:-150])]
    F_B = handle[np.random.choice(handle.precursor_frames[150:-150])]

    frame_noise_a = F_A.filter(intensity_max=max_intensity)
    frame_sampled_a = frame_noise_a.random_subsample_frame(sample_fraction)

    frame_noise_b = F_B.filter(intensity_max=max_intensity)
    frame_sampled_b = frame_noise_b.random_subsample_frame(sample_fraction)

    return frame_sampled_a + frame_sampled_b


def get_fragment_noise(handle, target_wg, frame_to_window_group, sample_fraction: float = 0.5, max_intensity: float = 30) -> TimsFrame:
    """Get noise from fragment frames.

    Args:
        handle: TimsTOF handle.
        target_wg: Target window group.
        frame_to_window_group: Frame to window group.
        sample_fraction (float): Sample fraction.
        max_intensity (float): Maximum intensity.

    Returns:
        TimsFrame: Frame.
    """
    filtered_dict = dict(filter(lambda item: item[1] == target_wg, frame_to_window_group.items()))
    frame_ids = [item[0] for item in filtered_dict.items()]

    F_A = handle[np.random.choice(frame_ids)]
    F_B = handle[np.random.choice(frame_ids)]

    frame_noise_a = F_A.filter(intensity_max=max_intensity)
    frame_sampled_a = frame_noise_a.random_subsample_frame(sample_fraction)

    frame_noise_b = F_B.filter(intensity_max=max_intensity)
    frame_sampled_b = frame_noise_b.random_subsample_frame(sample_fraction)

    return frame_sampled_a + frame_sampled_b


def add_real_data_noise_to_frames(
        acquisition_builder: TimsTofAcquisitionBuilderDIA,
        frames: List[TimsFrame],
        intensity_max: float = 30,
        sample_fraction: float = 0.5,
) -> List[TimsFrame]:
    """Add noise to frame.

    Args:
        acquisition_builder: Acquisition builder.
        frames (List[TimsFrame]): Frames.
        intensity_max (float): Maximum intensity.
        sample_fraction (float): Sample fraction.

    Returns:
        List[TimsFrame]: Frames.
    """
    r_list = []

    d = acquisition_builder.frames_to_window_groups
    window_group_dict = dict(zip(d['frame'], d['window_group']))
    fragment_frames = set(window_group_dict.keys())

    wg = acquisition_builder.tdf_writer.helper_handle.dia_ms_ms_info
    frame_to_window_group = dict(zip(wg.Frame, wg.WindowGroup))

    for frame in frames:
        if frame.frame_id in fragment_frames:
            target_wg = window_group_dict[frame.frame_id]
            noise = get_fragment_noise(acquisition_builder.tdf_writer.helper_handle, target_wg, frame_to_window_group,
                                       sample_fraction, intensity_max)
            r_list.append(frame + noise)
        else:
            noise = get_precursor_noise(acquisition_builder.tdf_writer.helper_handle, sample_fraction, intensity_max)
            r_list.append(frame + noise)

    return r_list
