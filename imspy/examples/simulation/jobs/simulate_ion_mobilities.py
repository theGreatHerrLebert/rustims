import pandas as pd

from imspy.algorithm import DeepPeptideIonMobilityApex, load_deep_ccs_predictor, load_tokenizer_from_resources


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
        tokenizer=load_tokenizer_from_resources(),
        verbose=verbose
    )

    # simulate mobilities
    dp = IMS.simulate_ion_mobilities_pandas(
        ions
    )

    # filter by mobility range
    ions = dp[(dp.mobility_gru_predictor >= im_lower) & (
            ions.mobility_gru_predictor <= im_upper)]

    return ions
