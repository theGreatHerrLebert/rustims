from pathlib import Path

import numpy as np
from tqdm import tqdm

from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper
from imspy.simulation.acquisition import TimsTofAcquisitionBuilder
from imspy.simulation.handle import TimsTofSyntheticsDataHandleRust
from imspy.simulation.utility import flatten_prosit_array, flat_intensity_to_sparse, \
    python_list_to_json_string, set_percentage_to_zero


def simulate_fragment_intensities(
        path: str,
        name: str,
        acquisition_builder: TimsTofAcquisitionBuilder,
        batch_size: int,
        verbose: bool,
        num_threads: int,
        down_sample_factor: int = 0.5,
        dda: bool = False
) -> None:
    """Simulate fragment ion intensity distributions.

    Args:
        path: Path to the synthetic data.
        name: Name of the synthetic data.
        acquisition_builder: Acquisition builder object.
        batch_size: Batch size for frame assembly, i.e. how many frames are assembled at once.
        verbose: Verbosity.
        num_threads: Number of threads for frame assembly.
        down_sample_factor: Down sample factor for fragment ion intensity distributions.
        dda: Data dependent acquisition mode.

    Returns:
        None, writes frames to disk and metadata to database.
    """

    if verbose:
        print("Simulating fragment ion intensity distributions...")

    assert 0 <= down_sample_factor < 1, "down_sample_factor must be in the range (0, 1]"

    native_path = Path(path) / name / 'synthetic_data.db'

    native_handle = TimsTofSyntheticsDataHandleRust(str(native_path))

    if verbose:
        print("Calculating precursor ion transmissions and collision energies...")

    transmitted_fragment_ions = native_handle.get_transmitted_ions(num_threads=num_threads, dda=dda)

    IntensityPredictor = Prosit2023TimsTofWrapper()

    i_pred = IntensityPredictor.simulate_ion_intensities_pandas_batched(transmitted_fragment_ions,
                                                                        batch_size_tf_ds=batch_size)

    if verbose:
        print("Mapping fragment ion intensity distributions to b and y ions...")

    n = int(5e4)
    batch_counter = 0

    for batch_indices in tqdm(
            np.array_split(i_pred.index, np.ceil(len(i_pred) / n)),
            total=int(np.ceil(len(i_pred) / n)),
            desc='flattening prosit predicted intensities',
            ncols=100,
            disable=(not verbose)
    ):

        batch = i_pred.loc[batch_indices].reset_index(drop=True)
        batch['intensity_flat'] = batch.apply(lambda r: set_percentage_to_zero(flatten_prosit_array(r.intensity),
                                                                               percentage=down_sample_factor), axis=1)

        batch = batch[['peptide_id', 'ion_id', 'collision_energy', 'charge', 'intensity_flat']]

        R = batch.apply(lambda r: flat_intensity_to_sparse(r.intensity_flat), axis=1)
        R = R.apply(lambda r: (python_list_to_json_string(r[0], as_float=False), python_list_to_json_string(r[1])))

        batch['indices'] = R.apply(lambda r: r[0])
        batch['values'] = R.apply(lambda r: r[1])
        batch = batch[['peptide_id', 'ion_id', 'collision_energy', 'charge', 'indices', 'values']]

        if batch_counter == 0:
            acquisition_builder.synthetics_handle.create_table(
                table=batch,
                table_name='fragment_ions'
            )
        else:
            acquisition_builder.synthetics_handle.append_table(
                table=batch,
                table_name='fragment_ions'
            )

        batch_counter += 1
