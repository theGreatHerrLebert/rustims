from pathlib import Path

import numpy as np
from tqdm import tqdm

from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper
from imspy.simulation.acquisition import TimsTofAcquisitionBuilderDIA
from imspy.simulation.handle import TimsTofSyntheticsDataHandleRust
from imspy.simulation.utility import flatten_prosit_array, flat_intensity_to_sparse, \
    python_list_to_json_string


def simulate_fragment_intensities(
        path: str,
        name: str,
        acquisition_builder: TimsTofAcquisitionBuilderDIA,
        batch_size: int,
        verbose: bool,
        num_threads: int,
) -> None:

    if verbose:
        print("Simulating fragment ion intensity distributions...")

    native_path = Path(path) / name / 'synthetic_data.db'

    native_handle = TimsTofSyntheticsDataHandleRust(str(native_path))

    if verbose:
        print("Calculating precursor ion transmissions and collision energies...")

    transmitted_fragment_ions = native_handle.get_transmitted_ions(num_threads=num_threads)

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
        batch['intensity_flat'] = batch.apply(lambda r: flatten_prosit_array(r.intensity), axis=1)

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
