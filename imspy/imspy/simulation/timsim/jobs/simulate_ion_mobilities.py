import pandas as pd

from imspy.algorithm.ccs.predictors import DeepPeptideIonMobilityApex, load_deep_ccs_predictor
from imspy.algorithm.utility import load_tokenizer_from_resources


def simulate_ion_mobilities(
        ions: pd.DataFrame,
        im_lower: float,
        im_upper: float,
        verbose: bool = False
) -> pd.DataFrame:

    if verbose:
        print("Simulating ion mobilities...")

        # load IM predictor
    IMS = DeepPeptideIonMobilityApex(
        model=load_deep_ccs_predictor(),
        tokenizer=load_tokenizer_from_resources('tokenizer-ptm'),
        verbose=verbose
    )

    # simulate mobilities
    dp = IMS.simulate_ion_mobilities_pandas(
        ions
    )

    # filter by mobility range
    ions = dp[
        (dp.inv_mobility_gru_predictor >= im_lower) &
        (dp.inv_mobility_gru_predictor <= im_upper)
        ]

    return ions
