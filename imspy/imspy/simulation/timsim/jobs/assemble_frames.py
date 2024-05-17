from pathlib import Path

import pandas as pd
from tqdm import tqdm

from examples.simulation.jobs.add_noise_from_real_data import add_real_data_noise_to_frames
from imspy.simulation.acquisition import TimsTofAcquisitionBuilderDIA
from imspy.simulation.experiment import TimsTofSyntheticFrameBuilderDIA


def assemble_frames(
        acquisition_builder: TimsTofAcquisitionBuilderDIA,
        frames: pd.DataFrame,
        batch_size: int,
        verbose: bool = False,
        mz_noise_precursor: bool = False,
        mz_noise_uniform: bool = False,
        precursor_noise_ppm: float = 5.,
        mz_noise_fragment: bool = False,
        fragment_noise_ppm: float = 5.,
        num_threads: int = 4,
        add_real_data_noise: bool = False,
        reference_noise_intensity_max: float = 30,
) -> None:

    if verbose:
        print("Starting frame assembly...")

        if add_real_data_noise:
            print("Real data noise will be added to the frames.")

    batch_size = batch_size
    num_batches = len(frames) // batch_size + 1
    frame_ids = frames.frame_id.values

    frame_builder = TimsTofSyntheticFrameBuilderDIA(
        db_path=str(Path(acquisition_builder.path) / 'synthetic_data.db'),
        with_annotations=False,
        num_threads=num_threads,
    )

    wg = acquisition_builder.tdf_writer.helper_handle.dia_ms_ms_info
    frame_to_window_group = dict(zip(wg.Frame, wg.WindowGroup))

    # go over all frames in batches
    for b in tqdm(range(num_batches), total=num_batches, desc='frame assembly', ncols=100):
        start_index = b * batch_size
        stop_index = (b + 1) * batch_size
        ids = frame_ids[start_index:stop_index]

        built_frames = frame_builder.build_frames(
            ids,
            mz_noise_precursor=mz_noise_precursor,
            mz_noise_uniform=mz_noise_uniform,
            precursor_noise_ppm=precursor_noise_ppm,
            mz_noise_fragment=mz_noise_fragment,
            fragment_noise_ppm=fragment_noise_ppm,
            num_threads=num_threads
        )

        if add_real_data_noise:
            built_frames = add_real_data_noise_to_frames(acquisition_builder, built_frames, frame_to_window_group,
                                                         intensity_max=reference_noise_intensity_max)

        for frame in built_frames:
            acquisition_builder.tdf_writer.write_frame(frame, scan_mode=9)

    if verbose:
        print("Writing frame meta data to database...")

    # write frame meta data to database
    acquisition_builder.tdf_writer.write_frame_meta_data()
    # write frame ms/ms info to database
    acquisition_builder.tdf_writer.write_dia_ms_ms_info(
        acquisition_builder.synthetics_handle.get_table('dia_ms_ms_info'))
    # write frame ms/ms windows to database
    acquisition_builder.tdf_writer.write_dia_ms_ms_windows(
        acquisition_builder.synthetics_handle.get_table('dia_ms_ms_windows'))
