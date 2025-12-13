from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper
from imspy.simulation.timsim.jobs.peptdeep_utils import (
    simulate_peptdeep_intensities_pandas,
)
from imspy.simulation.acquisition import TimsTofAcquisitionBuilder
from imspy.simulation.handle import TimsTofSyntheticsDataHandleRust
from imspy.simulation.utility import (
    flatten_prosit_array,
    flat_intensity_to_sparse,
    python_list_to_json_string,
    set_percentage_to_zero,
)


def simulate_fragment_intensities(
    path: str,
    name: str,
    acquisition_builder: TimsTofAcquisitionBuilder,
    batch_size: int,
    verbose: bool,
    num_threads: int,
    down_sample_factor: float = 0.5,
    dda: bool = False,
    model_name: Optional[str] = None,
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
        model_name: Optional external MS2 intensity model:
            - None / "prosit": use internal Prosit2023TimsTofWrapper
            - "peptdeep": use AlphaPeptDeep MS2 predictor (already mapped to Prosit-style 174-dim vectors)

    Returns:
        None, writes frames to disk and metadata to database.
    """

    if verbose:
        print("Simulating fragment ion intensity distributions ...")

    assert 0 <= down_sample_factor < 1, "down_sample_factor must be in the range [0, 1)"

    native_path = Path(path) / name / "synthetic_data.db"
    native_handle = TimsTofSyntheticsDataHandleRust(str(native_path))

    if verbose:
        print("Calculating precursor ion transmissions and collision energies ...")

    transmitted_fragment_ions = native_handle.get_transmitted_ions(
        num_threads=num_threads, dda=dda
    )

    # ------------------------------------------------------------------
    # Choose intensity model
    # ------------------------------------------------------------------
    intensity_already_flat = False

    if model_name is None or model_name.lower() == "prosit":
        # default: Prosit wrapper (returns 3D tensors -> needs flattening)
        if verbose:
            print("Using Prosit2023 TIMS-TOF intensity model ...")

        IntensityPredictor = Prosit2023TimsTofWrapper()
        i_pred = IntensityPredictor.simulate_ion_intensities_pandas_batched(
            transmitted_fragment_ions,
            batch_size_tf_ds=batch_size,
        )
        intensity_already_flat = False  # (len,2,3) tensors

    elif model_name.lower() == "peptdeep":
        # PeptDeep based predictor; peptdeep_utils already returns 174-dim vectors
        if verbose:
            print("Using PeptDeep MS2 predictor (Prosit-style 174-dim vectors) ...")

        assert transmitted_fragment_ions["collision_energy"].max() > 5, \
            "PeptDeep expects absolute collision energies (e.g. 20–60)"

        i_pred = simulate_peptdeep_intensities_pandas(
            transmitted_fragment_ions,
            device="gpu",          # or "cpu" if needed
            fill_value=-1.0,
            normalize=True,
        )
        # simulate_peptdeep_intensities_pandas sets i_pred['intensity'] to 174-dim np.ndarray
        intensity_already_flat = True
        # Scale CE from % to absolute for PeptDeep
        i_pred["collision_energy"] = i_pred.collision_energy.apply(lambda ce: ce / 100.0)  # to match Prosit CE scale

    else:
        raise NotImplementedError(
            f"External intensity model '{model_name}' is not implemented."
        )

    # ------------------------------------------------------------------
    # Map to sparse representation and write out
    # ------------------------------------------------------------------
    if verbose:
        print("Mapping fragment ion intensity distributions to b and y ions ...")

    n = int(5e4)
    batch_counter = 0

    for batch_indices in tqdm(
        np.array_split(i_pred.index, np.ceil(len(i_pred) / n)),
        total=int(np.ceil(len(i_pred) / n)),
        desc="flattening predicted intensities",
        ncols=100,
        disable=(not verbose),
    ):
        batch = i_pred.loc[batch_indices].reset_index(drop=True)

        if intensity_already_flat:
            # PeptDeep path: intensity is already a (174,) vector
            batch["intensity_flat"] = batch["intensity"].apply(
                lambda v: set_percentage_to_zero(
                    v.astype(np.float32),
                    percentage=down_sample_factor,
                )
            )
        else:
            # Prosit path: intensity is a (L-1,2,3) tensor -> flatten first
            batch["intensity_flat"] = batch["intensity"].apply(
                lambda t: set_percentage_to_zero(
                    flatten_prosit_array(t),
                    percentage=down_sample_factor,
                )
            )

        batch = batch[
            ["peptide_id", "ion_id", "collision_energy", "charge", "intensity_flat"]
        ]

        R = batch.apply(lambda r: flat_intensity_to_sparse(r.intensity_flat), axis=1)
        R = R.apply(
            lambda r: (
                python_list_to_json_string(r[0], as_float=False),
                python_list_to_json_string(r[1]),
            )
        )

        batch["indices"] = R.apply(lambda r: r[0])
        batch["values"] = R.apply(lambda r: r[1])
        batch = batch[
            ["peptide_id", "ion_id", "collision_energy", "charge", "indices", "values"]
        ]

        if batch_counter == 0:
            acquisition_builder.synthetics_handle.create_table(
                table=batch,
                table_name="fragment_ions",
            )
        else:
            acquisition_builder.synthetics_handle.append_table(
                table=batch,
                table_name="fragment_ions",
            )

        batch_counter += 1