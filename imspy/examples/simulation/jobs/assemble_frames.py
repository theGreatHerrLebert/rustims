from pathlib import Path

import pandas as pd
from tqdm import tqdm

from imspy.simulation.acquisition import TimsTofAcquisitionBuilderDIA
from imspy.simulation.experiment import TimsTofSyntheticFrameBuilderDIA


def assemble_frames(
        acquisition_builder: TimsTofAcquisitionBuilderDIA,
        frames: pd.DataFrame,
        batch_size: int,
        verbose: bool = False,
        num_threads: int = 4
) -> None:

    if verbose:
        print("Starting frame assembly...")

    batch_size = batch_size
    num_batches = len(frames) // batch_size + 1
    frame_ids = frames.frame_id.values

    frame_builder = TimsTofSyntheticFrameBuilderDIA(
        db_path=str(Path(acquisition_builder.path) / 'synthetic_data.db'),
        num_threads=num_threads,
    )

    # go over all frames in batches
    for b in tqdm(range(num_batches), total=num_batches, desc='frame assembly', ncols=100):
        start_index = b * batch_size
        stop_index = (b + 1) * batch_size
        ids = frame_ids[start_index:stop_index]

        built_frames = frame_builder.build_frames(ids, num_threads=num_threads)
        acquisition_builder.tdf_writer.write_frames(built_frames, scan_mode=9, num_threads=num_threads)

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
