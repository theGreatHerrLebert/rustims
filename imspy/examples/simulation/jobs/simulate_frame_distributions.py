import pandas as pd
from tqdm import tqdm

from imspy.simulation.utility import get_frames_numba, accumulated_intensity_cdf_numba, get_z_score_for_percentile, \
    python_list_to_json_string


def simulate_frame_distributions(
        peptides: pd.DataFrame,
        frames: pd.DataFrame,
        z_score: float,
        std_rt: float,
        rt_cycle_length: float,
        verbose: bool = False
) -> pd.DataFrame:
    # distribution parameters
    z_score = get_z_score_for_percentile(target_score=z_score)
    frames_np = frames.time.values

    time_dict = dict(zip(frames.frame_id.values, frames.time.values))

    peptide_rt = peptides

    first_occurrence = []
    last_occurrence = []

    total_list_frames = []
    total_list_frame_contributions = []

    if verbose:
        print("Calculating frame and scan distributions...")

    # generate frame_occurrence and frame_abundance columns
    for _, row in tqdm(peptide_rt.iterrows(), total=peptide_rt.shape[0], desc='frame distribution', ncols=100, disable=not verbose):
        frame_occurrence, frame_abundance = [], []

        rt_value = row.retention_time_gru_predictor
        contributing_frames = get_frames_numba(rt_value, frames_np, std_rt, z_score)

        for frame in contributing_frames:
            time = time_dict[frame]
            start = time - rt_cycle_length
            i = accumulated_intensity_cdf_numba(start, time, rt_value, std_rt)

            # TODO: ADD NOISE HERE?

            frame_occurrence.append(frame)
            frame_abundance.append(i)

        first_occurrence.append(frame_occurrence[0])
        last_occurrence.append(frame_occurrence[-1])

        total_list_frames.append(frame_occurrence)
        total_list_frame_contributions.append(frame_abundance)

    if verbose:
        print("Serializing frame distributions to json...")

    peptide_rt['frame_occurrence_start'] = first_occurrence
    peptide_rt['frame_occurrence_end'] = last_occurrence
    peptide_rt['frame_occurrence'] = [list(x) for x in total_list_frames]
    peptide_rt['frame_abundance'] = [list(x) for x in total_list_frame_contributions]

    peptide_rt['frame_occurrence'] = peptide_rt['frame_occurrence'].apply(
        lambda x: python_list_to_json_string(x, as_float=False))

    peptide_rt['frame_abundance'] = peptide_rt['frame_abundance'].apply(python_list_to_json_string)
    peptide_rt = peptides.sort_values(by=['frame_occurrence_start', 'frame_occurrence_end'])

    return peptide_rt
