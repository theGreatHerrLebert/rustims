import logging
from typing import Optional

import numpy as np
import pandas as pd

from imspy_predictors.ccs.utility import load_deep_ccs_std_predictor, to_tf_dataset_with_variance
from imspy_core.chemistry.mobility import ccs_to_one_over_k0

logger = logging.getLogger(__name__)

# Koina CCS model name mapping
KOINA_CCS_MODELS = {
    "alphapeptdeep": "AlphaPeptDeep_ccs_generic",
    "im2deep": "IM2Deep",
}


def simulate_ion_mobilities_and_variance(
        ions: pd.DataFrame,
        im_lower: float,
        im_upper: float,
        verbose: bool = False,
        epsilon: float = 1e-6,
        remove_mods: bool = False,
        use_target_mean_std: bool = True,
        target_std_mean: float = 0.009,
        use_koina_model: Optional[str] = None,
) -> pd.DataFrame:
    """Simulate ion mobilities.

    Args:
        ions: Ions DataFrame.
        im_lower: Lower ion mobility.
        im_upper: Upper ion mobility.
        verbose: Verbosity.
        epsilon: Epsilon value to be added to the variance.
        remove_mods: Remove Unimod annotation.
        use_target_mean_std: Use target mean standard deviation.
        target_std_mean: Target standard deviation mean.
        use_koina_model: Koina model name for CCS prediction:
            - None / "local": use local PyTorch model
            - "alphapeptdeep": use AlphaPeptDeep_ccs_generic via Koina
            - "im2deep": use IM2Deep via Koina
            - Or specify full Koina model name directly

    Returns:
        pd.DataFrame: Ions DataFrame.
    """

    if verbose:
        print("Simulating ion mobilities and variance ...")

    # prepare data
    assert "mz" in ions.columns, "mz column is missing"
    assert "charge" in ions.columns, "charge column is missing"
    assert "sequence" in ions.columns, "sequence column is missing"

    model_key = (use_koina_model or "local").lower()

    if model_key in (None, "", "local"):
        # Use local PyTorch CCS predictor
        if verbose:
            print("Using local PyTorch CCS predictor ...")

        model = load_deep_ccs_std_predictor()

        tf_ds = to_tf_dataset_with_variance(
            mz=ions.mz.values,
            charge=ions.charge.values,
            sequences=ions.sequence.values,
            batch=True,
            shuffle=False,
            remove_unimod=remove_mods,
        )

        ccs, ccs_std, _ = model.predict(tf_ds)

        inverse_mobility = np.array([ccs_to_one_over_k0(c, mz, charge) for c, mz, charge
                                     in zip(ccs, ions.mz.values, ions.charge.values)]).astype(np.float32)
        inverse_mobility_std = np.array([ccs_to_one_over_k0(std, mz, charge) for std, mz, charge
                                         in zip(ccs_std, ions.mz.values, ions.charge.values)]).astype(np.float32)

    else:
        # Use Koina CCS predictor
        koina_model_name = KOINA_CCS_MODELS.get(model_key, use_koina_model)
        if verbose:
            print(f"Using Koina CCS model: {koina_model_name} ...")

        try:
            inverse_mobility, inverse_mobility_std = _predict_ccs_with_koina(
                ions=ions,
                model_name=koina_model_name,
                verbose=verbose,
            )
        except Exception as e:
            logger.warning(f"Koina CCS prediction failed: {e}. Falling back to local model.")
            if verbose:
                print(f"Koina CCS prediction failed: {e}. Falling back to local model.")

            model = load_deep_ccs_std_predictor()

            tf_ds = to_tf_dataset_with_variance(
                mz=ions.mz.values,
                charge=ions.charge.values,
                sequences=ions.sequence.values,
                batch=True,
                shuffle=False,
                remove_unimod=remove_mods,
            )

            ccs, ccs_std, _ = model.predict(tf_ds)

            inverse_mobility = np.array([ccs_to_one_over_k0(c, mz, charge) for c, mz, charge
                                         in zip(ccs, ions.mz.values, ions.charge.values)]).astype(np.float32)
            inverse_mobility_std = np.array([ccs_to_one_over_k0(std, mz, charge) for std, mz, charge
                                             in zip(ccs_std, ions.mz.values, ions.charge.values)]).astype(np.float32)

    if use_target_mean_std:
        # calculate the current mean standard deviation
        current_mean_std = np.mean(inverse_mobility_std)
        # shift the standard deviation to the target mean
        difference = current_mean_std - target_std_mean
        inverse_mobility_std -= difference

        if verbose:
            print(f"Standard deviation distribution apex was shifted from {current_mean_std:.4f} to {target_std_mean:.4f}")

    dp = ions.copy()
    dp["inv_mobility_gru_predictor"] = inverse_mobility
    dp["inv_mobility_gru_predictor_std"] = inverse_mobility_std + epsilon

    # filter by mobility range
    ions = dp[
        (dp.inv_mobility_gru_predictor >= im_lower) &
        (dp.inv_mobility_gru_predictor <= im_upper)
        ]

    return ions


def _predict_ccs_with_koina(
    ions: pd.DataFrame,
    model_name: str,
    verbose: bool = False,
) -> tuple:
    """
    Predict CCS/ion mobility using Koina API.

    Args:
        ions: DataFrame with 'sequence', 'charge', 'mz' columns.
        model_name: Koina model name.
        verbose: Verbosity flag.

    Returns:
        Tuple of (inverse_mobility, inverse_mobility_std) arrays.
    """
    from imspy_predictors.koina_models import ModelFromKoina

    # Prepare input for Koina
    input_df = pd.DataFrame({
        'peptide_sequences': ions['sequence'].values,
        'precursor_charges': ions['charge'].values,
    })

    if verbose:
        print(f"Predicting CCS for {len(input_df)} peptides with Koina model {model_name}")

    model = ModelFromKoina(model_name=model_name)
    result = model.predict(input_df)

    # Extract CCS values
    if 'ccs' in result.columns:
        ccs_values = result['ccs'].values
    elif 'CCS' in result.columns:
        ccs_values = result['CCS'].values
    else:
        # Try to find CCS column
        ccs_cols = [c for c in result.columns if 'ccs' in c.lower()]
        if ccs_cols:
            ccs_values = result[ccs_cols[0]].values
        else:
            raise ValueError(f"Could not find CCS column in Koina result. Columns: {result.columns}")

    # Convert CCS to inverse mobility
    inverse_mobility = np.array([
        ccs_to_one_over_k0(ccs, mz, charge)
        for ccs, mz, charge in zip(ccs_values, ions['mz'].values, ions['charge'].values)
    ]).astype(np.float32)

    # Koina models don't typically provide std, so estimate it
    # Use a fixed percentage of the predicted value as std estimate
    inverse_mobility_std = inverse_mobility * 0.02  # 2% relative std

    return inverse_mobility, inverse_mobility_std
