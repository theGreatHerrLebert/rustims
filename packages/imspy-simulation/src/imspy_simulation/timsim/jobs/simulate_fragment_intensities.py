from pathlib import Path
from typing import Optional
import logging

import numpy as np
from tqdm import tqdm

from imspy_predictors.intensity.predictors import Prosit2023TimsTofWrapper
from imspy_simulation.timsim.jobs.peptdeep_utils import (
    simulate_peptdeep_intensities_pandas,
)
from imspy_simulation.acquisition import TimsTofAcquisitionBuilder
from imspy_simulation.data import TransmissionHandle
from imspy_simulation.utility import (
    flatten_prosit_array,
    flat_intensity_to_sparse,
    python_list_to_json_string,
    set_percentage_to_zero,
)

logger = logging.getLogger(__name__)


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
    lazy_loading: bool = False,
    frame_batch_size: int = 500,
) -> None:
    """Simulate fragment ion intensity distributions.

    Args:
        path: Path to the synthetic data.
        name: Name of the synthetic data.
        acquisition_builder: Acquisition builder object.
        batch_size: Batch size for TensorFlow prediction.
        verbose: Verbosity.
        num_threads: Number of threads for parallel processing.
        down_sample_factor: Down sample factor for fragment ion intensity distributions.
        dda: Data dependent acquisition mode.
        model_name: Optional external MS2 intensity model:
            - None / "prosit": use internal Prosit2023TimsTofWrapper
            - "peptdeep": use AlphaPeptDeep MS2 predictor (already mapped to Prosit-style 174-dim vectors)
        lazy_loading: If True, process ions in batches by frame range to reduce memory usage.
        frame_batch_size: Number of frames to process per batch when lazy_loading is True.

    Returns:
        None, writes frames to disk and metadata to database.
    """

    logger.info("Simulating fragment ion intensity distributions ...")

    assert 0 <= down_sample_factor < 1, "down_sample_factor must be in the range [0, 1)"

    native_path = Path(path) / name / "synthetic_data.db"
    native_handle = TransmissionHandle(str(native_path))

    if lazy_loading:
        _simulate_fragment_intensities_lazy(
            native_handle=native_handle,
            acquisition_builder=acquisition_builder,
            batch_size=batch_size,
            num_threads=num_threads,
            down_sample_factor=down_sample_factor,
            dda=dda,
            model_name=model_name,
            frame_batch_size=frame_batch_size,
            verbose=verbose,
        )
    else:
        _simulate_fragment_intensities_standard(
            native_handle=native_handle,
            acquisition_builder=acquisition_builder,
            batch_size=batch_size,
            num_threads=num_threads,
            down_sample_factor=down_sample_factor,
            dda=dda,
            model_name=model_name,
            verbose=verbose,
        )


def _simulate_fragment_intensities_standard(
    native_handle: TransmissionHandle,
    acquisition_builder: TimsTofAcquisitionBuilder,
    batch_size: int,
    num_threads: int,
    down_sample_factor: float,
    dda: bool,
    model_name: Optional[str],
    verbose: bool,
) -> None:
    """Standard (non-lazy) fragment intensity simulation.

    Loads all transmitted ions into memory at once.
    """
    logger.info("Calculating precursor ion transmissions and collision energies ...")

    transmitted_fragment_ions = native_handle.get_transmitted_ions(
        num_threads=num_threads, dda=dda
    )

    # ------------------------------------------------------------------
    # Choose intensity model
    # ------------------------------------------------------------------
    intensity_already_flat = False

    if model_name is None or model_name.lower() == "prosit":
        # default: Prosit wrapper (returns 3D tensors -> needs flattening)
        logger.info("Using Prosit2023 TIMS-TOF intensity model ...")

        IntensityPredictor = Prosit2023TimsTofWrapper()
        i_pred = IntensityPredictor.simulate_ion_intensities_pandas_batched(
            transmitted_fragment_ions,
            batch_size_tf_ds=batch_size,
        )
        intensity_already_flat = False  # (len,2,3) tensors

    elif model_name.lower() == "peptdeep":
        # PeptDeep based predictor; peptdeep_utils already returns 174-dim vectors
        logger.info("Using PeptDeep MS2 predictor (Prosit-style 174-dim vectors) ...")

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
    logger.info("Mapping fragment ion intensity distributions to b and y ions ...")

    n = int(5e4)
    batch_counter = 0

    num_batches = max(1, int(np.ceil(len(i_pred) / n))) if len(i_pred) > 0 else 0
    for batch_indices in tqdm(
        np.array_split(i_pred.index, num_batches) if num_batches > 0 else [],
        total=num_batches,
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


def _simulate_fragment_intensities_lazy(
    native_handle: TransmissionHandle,
    acquisition_builder: TimsTofAcquisitionBuilder,
    batch_size: int,
    num_threads: int,
    down_sample_factor: float,
    dda: bool,
    model_name: Optional[str],
    frame_batch_size: int,
    verbose: bool,
) -> None:
    """Lazy fragment intensity simulation.

    Processes ions in batches by frame range to reduce memory usage.
    Instead of loading all transmitted ions at once, this function:
    1. Gets the total frame range
    2. Processes ions in frame-range batches
    3. Writes each batch to the database incrementally
    """
    logger.info("Using lazy loading mode for fragment intensity simulation ...")

    # Get frame info from the acquisition builder
    frames = acquisition_builder.synthetics_handle.get_frame_meta_data()
    all_frame_ids = frames["frame_id"].values
    min_frame = int(all_frame_ids.min())
    max_frame = int(all_frame_ids.max())

    logger.info(f"Processing frames {min_frame} to {max_frame} in batches of {frame_batch_size}")

    # Initialize intensity predictor once
    if model_name is None or model_name.lower() == "prosit":
        logger.info("Using Prosit2023 TIMS-TOF intensity model ...")
        IntensityPredictor = Prosit2023TimsTofWrapper()
        intensity_already_flat = False
    elif model_name.lower() == "peptdeep":
        logger.info("Using PeptDeep MS2 predictor (Prosit-style 174-dim vectors) ...")
        IntensityPredictor = None  # PeptDeep doesn't use a persistent predictor
        intensity_already_flat = True
    else:
        raise NotImplementedError(
            f"External intensity model '{model_name}' is not implemented."
        )

    # Process in frame batches
    batch_counter = 0
    total_batches = int(np.ceil((max_frame - min_frame + 1) / frame_batch_size))

    for frame_start in tqdm(
        range(min_frame, max_frame + 1, frame_batch_size),
        total=total_batches,
        desc="Processing frame batches",
        ncols=100,
        disable=(not verbose),
    ):
        frame_end = min(frame_start + frame_batch_size - 1, max_frame)

        # Get transmitted ions for this frame range only
        transmitted_fragment_ions = native_handle.get_transmitted_ions_for_frame_range(
            frame_min=frame_start,
            frame_max=frame_end,
            num_threads=num_threads,
            dda=dda,
        )

        if transmitted_fragment_ions.empty:
            logger.debug(f"No transmitted ions in frames {frame_start}-{frame_end}")
            continue

        logger.debug(
            f"Frame batch {frame_start}-{frame_end}: {len(transmitted_fragment_ions)} ions"
        )

        # Predict intensities for this batch
        if model_name is None or model_name.lower() == "prosit":
            i_pred = IntensityPredictor.simulate_ion_intensities_pandas_batched(
                transmitted_fragment_ions,
                batch_size_tf_ds=batch_size,
            )
        elif model_name.lower() == "peptdeep":
            assert transmitted_fragment_ions["collision_energy"].max() > 5, \
                "PeptDeep expects absolute collision energies (e.g. 20–60)"
            i_pred = simulate_peptdeep_intensities_pandas(
                transmitted_fragment_ions,
                device="gpu",
                fill_value=-1.0,
                normalize=True,
            )
            i_pred["collision_energy"] = i_pred.collision_energy.apply(lambda ce: ce / 100.0)

        # Process and write this batch
        n = int(5e4)
        for batch_indices in np.array_split(i_pred.index, max(1, int(np.ceil(len(i_pred) / n)))):
            batch = i_pred.loc[batch_indices].reset_index(drop=True)

            if intensity_already_flat:
                batch["intensity_flat"] = batch["intensity"].apply(
                    lambda v: set_percentage_to_zero(
                        v.astype(np.float32),
                        percentage=down_sample_factor,
                    )
                )
            else:
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

    logger.info(f"Finished processing {batch_counter} batches")