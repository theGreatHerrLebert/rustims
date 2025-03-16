import numpy as np
import pandas as pd

from imspy.algorithm.ccs.utility import load_deep_ccs_std_predictor, to_tf_dataset_with_variance
from imspy.chemistry.mobility import ccs_to_one_over_k0

def simulate_ion_mobilities_and_variance(
        ions: pd.DataFrame,
        im_lower: float,
        im_upper: float,
        verbose: bool = False,
        epsilon: float = 1e-6,
        remove_mods: bool = False,
        use_target_mean_std: bool = True,
        target_std_mean: float = 0.009,
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

    Returns:
        pd.DataFrame: Ions DataFrame.
    """

    if verbose:
        print("Simulating ion mobilities and variance...")

    # load IM predictor
    model = load_deep_ccs_std_predictor()

    # prepare data
    assert "mz" in ions.columns, "mz column is missing"
    assert "charge" in ions.columns, "charge column is missing"
    assert "sequence" in ions.columns, "sequence column is missing"

    tf_ds = to_tf_dataset_with_variance(
        mz=ions.mz.values,
        charge=ions.charge.values,
        sequences=ions.sequence.values,
        batch=True,
        shuffle=False,
        remove_unimod=remove_mods,
    )

    ccs, ccs_std, _ = model.predict(tf_ds)

    inverse_mobility = np.array([ccs_to_one_over_k0(ccs, mz, charge) for ccs, mz, charge
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
            print(f"Mean standard deviation apex was shifted from {current_mean_std:.4f} to {target_std_mean:.4f}")

    dp = ions.copy()
    dp["inv_mobility_gru_predictor"] = inverse_mobility
    dp["inv_mobility_gru_predictor_std"] = inverse_mobility_std + epsilon

    # filter by mobility range
    ions = dp[
        (dp.inv_mobility_gru_predictor >= im_lower) &
        (dp.inv_mobility_gru_predictor <= im_upper)
        ]

    return ions
