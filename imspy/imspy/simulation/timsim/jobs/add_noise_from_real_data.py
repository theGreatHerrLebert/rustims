from typing import List

from imspy.simulation.acquisition import TimsTofAcquisitionBuilderDIA
from imspy.timstof.frame import TimsFrame


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

    Returns:
        List[TimsFrame]: Frames.
    """
    r_list = []

    # DDA noise not yet implemented
    if acquisition_mode == 'DDA':
        print("Warning: DDA noise not yet implemented, no noise will be added added.")
        for frame in frames:
            r_list.append(frame)

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
