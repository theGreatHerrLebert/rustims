import pandas as pd

from imspy.algorithm import DeepChromatographyApex, load_tokenizer_from_resources, load_deep_retention_time


def simulate_retention_times(
        peptides: pd.DataFrame,
        verbose: bool = False,
        gradient_length: float = 60 * 60,
) -> pd.DataFrame:

    if verbose:
        print("Simulating retention times...")

    # create RTColumn instance
    RTColumn = DeepChromatographyApex(
        model=load_deep_retention_time(),
        tokenizer=load_tokenizer_from_resources(),
        verbose=verbose
    )

    # predict rts
    peptide_rt = RTColumn.simulate_separation_times_pandas(
        data=peptides,
        gradient_length=gradient_length,
    )

    return peptide_rt
