from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from imspy_predictors.intensity.predictors import (
    Prosit2023TimsTofWrapper,
    DeepPeptideIntensityPredictor,
    load_deep_intensity_predictor,
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

# Koina model name mapping for fragment intensity prediction
# Maps short names to full Koina model names
KOINA_INTENSITY_MODELS = {
    "prosit": "Prosit_2023_intensity_timsTOF",
    "alphapeptdeep": "AlphaPeptDeep_ms2_generic",
    "ms2pip": "ms2pip_timsTOF2024",
    "ms2pip_2023": "ms2pip_timsTOF2023",
}

# Models that support phosphorylation
PHOSPHO_COMPATIBLE_MODELS = {"alphapeptdeep", "AlphaPeptDeep_ms2_generic"}


def _predict_intensities_with_koina(
    data: pd.DataFrame,
    model_name: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Predict fragment intensities using a Koina model.

    Args:
        data: DataFrame with 'sequence', 'charge', 'collision_energy' columns.
        model_name: Koina model name (short name or full Koina model name).
        verbose: Verbosity flag.

    Returns:
        DataFrame with 'intensity' column containing flattened 174-dim vectors.
    """
    from imspy_predictors.koina_models import ModelFromKoina

    # Resolve model name
    koina_model_name = KOINA_INTENSITY_MODELS.get(model_name.lower(), model_name)

    if verbose:
        logger.info(f"Using Koina model: {koina_model_name}")

    # Prepare input for Koina
    input_df = pd.DataFrame({
        'peptide_sequences': data['sequence'].values,
        'precursor_charges': data['charge'].values,
        'collision_energies': data['collision_energy'].values / 100.0,  # Normalize CE
        'instrument_types': ['TIMSTOF'] * len(data),
    })

    # Get predictions
    model = ModelFromKoina(model_name=koina_model_name)
    result = model.predict(input_df)

    # Extract intensities and convert to Prosit-style 174-dim vectors
    intensities = _koina_result_to_prosit_vectors(result, data)

    out = data.copy()
    out['intensity'] = list(intensities)

    return out


def _koina_result_to_prosit_vectors(
    koina_result: pd.DataFrame,
    original_data: pd.DataFrame,
    max_len: int = 30,
    fill_value: float = -1.0,
) -> np.ndarray:
    """
    Convert Koina intensity predictions to Prosit-style 174-dim vectors.

    The Koina result contains 'intensities', 'mz', and 'annotation' columns
    where each row is a fragment ion. We need to reorganize into
    Prosit-style (29, 2, 3) tensors then flatten to 174-dim vectors.

    Args:
        koina_result: DataFrame from Koina prediction.
        original_data: Original input DataFrame with sequences.
        max_len: Maximum peptide length (default 30).
        fill_value: Fill value for missing fragments (default -1.0).

    Returns:
        Array of shape (n_peptides, 174) with flattened intensity vectors.
    """
    import re

    n_peps = len(original_data)
    max_frags = max_len - 1  # 29 fragments max

    # Initialize output array
    prosit_mat = np.full((n_peps, 174), fill_value, dtype=np.float32)

    # Check if we have the expected columns
    if 'intensities' not in koina_result.columns:
        # Some models return different column names - try to find intensity column
        intensity_cols = [c for c in koina_result.columns if 'intens' in c.lower()]
        if intensity_cols:
            intensity_col = intensity_cols[0]
        else:
            raise ValueError(f"Could not find intensity column in Koina result. Columns: {koina_result.columns}")
    else:
        intensity_col = 'intensities'

    # Check if intensities are already in array format per peptide
    if koina_result[intensity_col].iloc[0] is not None and hasattr(koina_result[intensity_col].iloc[0], '__len__'):
        # Intensities are arrays - process per peptide
        for i in range(n_peps):
            intensities = np.array(koina_result[intensity_col].iloc[i])

            # If we have annotation, use it to map to correct positions
            if 'annotation' in koina_result.columns:
                annotations = koina_result['annotation'].iloc[i]
                tensor = np.full((max_frags, 2, 3), fill_value, dtype=np.float32)

                for j, (ann, inten) in enumerate(zip(annotations, intensities)):
                    if ann is None or inten <= 0:
                        continue
                    # Parse annotation like b'y5+1' or b'b3+2'
                    ann_str = ann.decode() if isinstance(ann, bytes) else str(ann)
                    match = re.match(r'([by])(\d+)\+(\d+)', ann_str)
                    if match:
                        ion_type = match.group(1)
                        frag_idx = int(match.group(2)) - 1  # 0-indexed
                        charge_idx = int(match.group(3)) - 1  # 0-indexed

                        if frag_idx < max_frags and charge_idx < 3:
                            ion_type_idx = 0 if ion_type == 'y' else 1
                            tensor[frag_idx, ion_type_idx, charge_idx] = inten

                # Flatten tensor to 174-dim vector
                prosit_mat[i, :] = tensor.flatten()
            else:
                # No annotation - assume already in correct format
                if len(intensities) == 174:
                    prosit_mat[i, :] = intensities
                else:
                    # Try to reshape if possible
                    prosit_mat[i, :len(intensities)] = intensities[:174]

            # Normalize to max = 1
            max_val = prosit_mat[i, :].max()
            if max_val > 0:
                prosit_mat[i, :] = prosit_mat[i, :] / max_val

    return prosit_mat


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
    phospho_mode: bool = False,
) -> None:
    """Simulate fragment ion intensity distributions.

    Args:
        path: Path to the synthetic data.
        name: Name of the synthetic data.
        acquisition_builder: Acquisition builder object.
        batch_size: Batch size for prediction.
        verbose: Verbosity.
        num_threads: Number of threads for parallel processing.
        down_sample_factor: Down sample factor for fragment ion intensity distributions.
        dda: Data dependent acquisition mode.
        model_name: Optional MS2 intensity model:
            - None / "local": use local PyTorch model (PROSPECT fine-tuned, default)
            - "prosit": use Prosit_2023_intensity_timsTOF via Koina
            - "alphapeptdeep": use AlphaPeptDeep_ms2_generic via Koina (phospho-compatible)
            - "ms2pip": use ms2pip_timsTOF2024 via Koina
            - Or specify full Koina model name directly
        lazy_loading: If True, process ions in batches by frame range to reduce memory usage.
        frame_batch_size: Number of frames to process per batch when lazy_loading is True.
        phospho_mode: If True and model doesn't support phospho, auto-switch to AlphaPeptDeep.

    Returns:
        None, writes frames to disk and metadata to database.
    """

    logger.info("Simulating fragment ion intensity distributions ...")

    assert 0 <= down_sample_factor < 1, "down_sample_factor must be in the range [0, 1)"

    # Handle phospho mode - auto-switch to AlphaPeptDeep if needed
    effective_model_name = model_name
    if phospho_mode:
        model_key = (model_name or "prosit").lower()
        if model_key not in PHOSPHO_COMPATIBLE_MODELS and model_key not in ["alphapeptdeep"]:
            logger.warning(
                f"Phospho mode enabled but model '{model_name or 'prosit'}' may not support phosphorylation. "
                "Switching to 'alphapeptdeep' (AlphaPeptDeep_ms2_generic) which supports all modifications."
            )
            effective_model_name = "alphapeptdeep"

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
            model_name=effective_model_name,
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
            model_name=effective_model_name,
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
    model_key = (model_name or "local").lower()

    if model_key in (None, "", "local"):
        # Default: Local PyTorch model (PROSPECT fine-tuned)
        logger.info("Using local PyTorch intensity model (PROSPECT fine-tuned) ...")

        try:
            IntensityPredictor = DeepPeptideIntensityPredictor(verbose=verbose)
            i_pred = IntensityPredictor.simulate_ion_intensities_pandas_batched(
                transmitted_fragment_ions,
                batch_size_tf_ds=batch_size,
            )
            intensity_already_flat = False  # (len,2,3) tensors
        except (FileNotFoundError, ImportError) as e:
            logger.warning(f"Local intensity model not available: {e}. Falling back to Koina (Prosit).")
            IntensityPredictor = Prosit2023TimsTofWrapper()
            i_pred = IntensityPredictor.simulate_ion_intensities_pandas_batched(
                transmitted_fragment_ions,
                batch_size_tf_ds=batch_size,
            )
            intensity_already_flat = False

    elif model_key == "prosit":
        # Prosit via Koina
        logger.info("Using Prosit2023 TIMS-TOF intensity model via Koina ...")

        IntensityPredictor = Prosit2023TimsTofWrapper()
        i_pred = IntensityPredictor.simulate_ion_intensities_pandas_batched(
            transmitted_fragment_ions,
            batch_size_tf_ds=batch_size,
        )
        intensity_already_flat = False  # (len,2,3) tensors

    elif model_key in KOINA_INTENSITY_MODELS or model_key in ["alphapeptdeep", "ms2pip", "ms2pip_2023"]:
        # Koina-based intensity prediction (returns 174-dim vectors)
        koina_model_name = KOINA_INTENSITY_MODELS.get(model_key, model_name)
        logger.info(f"Using Koina model: {koina_model_name} ...")

        i_pred = _predict_intensities_with_koina(
            transmitted_fragment_ions,
            model_name=model_key,
            verbose=verbose,
        )
        intensity_already_flat = True

    else:
        # Try to use model_name as a direct Koina model name
        logger.info(f"Trying Koina model: {model_name} ...")
        try:
            i_pred = _predict_intensities_with_koina(
                transmitted_fragment_ions,
                model_name=model_name,
                verbose=verbose,
            )
            intensity_already_flat = True
        except Exception as e:
            raise NotImplementedError(
                f"Intensity model '{model_name}' is not recognized. "
                f"Available short names: {list(KOINA_INTENSITY_MODELS.keys())}. "
                f"Or use a full Koina model name. Error: {e}"
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

    # Determine model type
    model_key = (model_name or "local").lower()
    use_koina_direct = model_key in KOINA_INTENSITY_MODELS or model_key in ["alphapeptdeep", "ms2pip", "ms2pip_2023"]

    # Initialize intensity predictor once
    if model_key in (None, "", "local"):
        # Default: Local PyTorch model (PROSPECT fine-tuned)
        logger.info("Using local PyTorch intensity model (PROSPECT fine-tuned) ...")
        try:
            IntensityPredictor = DeepPeptideIntensityPredictor(verbose=verbose)
            intensity_already_flat = False
        except (FileNotFoundError, ImportError) as e:
            logger.warning(f"Local intensity model not available: {e}. Falling back to Koina (Prosit).")
            IntensityPredictor = Prosit2023TimsTofWrapper()
            intensity_already_flat = False
    elif model_key == "prosit":
        logger.info("Using Prosit2023 TIMS-TOF intensity model via Koina ...")
        IntensityPredictor = Prosit2023TimsTofWrapper()
        intensity_already_flat = False
    elif use_koina_direct:
        koina_model_name = KOINA_INTENSITY_MODELS.get(model_key, model_name)
        logger.info(f"Using Koina model: {koina_model_name} ...")
        IntensityPredictor = None  # Will use _predict_intensities_with_koina
        intensity_already_flat = True
    else:
        # Try to use model_name as a direct Koina model name
        logger.info(f"Trying Koina model: {model_name} ...")
        IntensityPredictor = None
        intensity_already_flat = True

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
        if model_key in (None, "", "local", "prosit"):
            i_pred = IntensityPredictor.simulate_ion_intensities_pandas_batched(
                transmitted_fragment_ions,
                batch_size_tf_ds=batch_size,
            )
        else:
            # Use Koina for other models
            i_pred = _predict_intensities_with_koina(
                transmitted_fragment_ions,
                model_name=model_key if model_key in KOINA_INTENSITY_MODELS else model_name,
                verbose=verbose,
            )

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