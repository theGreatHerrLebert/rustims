from typing import Optional

import pandas as pd

from imspy.algorithm.rt.predictors import (
    DeepChromatographyApex,
    load_deep_retention_time_predictor,
    predict_retention_time_with_koina,
)
from imspy.algorithm.utility import load_tokenizer_from_resources


def simulate_retention_times(
        peptides: pd.DataFrame,
        verbose: bool = False,
        gradient_length: float = 60 * 60,
        use_koina_model: Optional[str] = None,
) -> pd.DataFrame:
    """Simulate retention times.
    Args:
        peptides: Peptides DataFrame.
        verbose: Verbosity.
        gradient_length: Gradient length in seconds.
        use_koina_model: Model name for Koina retention time prediction, if None, use DeepChromatographyApex.

    Returns:
        pd.DataFrame: Peptides DataFrame.
    """

    if verbose:
        print("Simulating retention times...")

    if use_koina_model is not None and use_koina_model != "":
        try:
            if verbose:
                print(f"Using Koina model: {use_koina_model}")

            peptide_rt = predict_retention_time_with_koina(
                model_name=use_koina_model,
                data=peptides,
                gradient_length=gradient_length,
            )

        except Exception as e:
            print(f"Failed to predict retention time with KOINA: {e}")
            print("Falling back to DeepChromatographyApex model ...")

            RTColumn = DeepChromatographyApex(
                model=load_deep_retention_time_predictor(),
                tokenizer=load_tokenizer_from_resources(tokenizer_name="tokenizer-ptm"),
                verbose=verbose,
            )

            peptide_rt = RTColumn.simulate_separation_times_pandas(
                data=peptides,
                gradient_length=gradient_length,
            )

    else:
        # create RTColumn instance
        RTColumn = DeepChromatographyApex(
            model=load_deep_retention_time_predictor(),
            tokenizer=load_tokenizer_from_resources(tokenizer_name="tokenizer-ptm"),
            verbose=verbose,
        )

        # predict rts
        peptide_rt = RTColumn.simulate_separation_times_pandas(
            data=peptides,
            gradient_length=gradient_length,
        )

    return peptide_rt
