import pandas as pd

from imspy.algorithm.utility import load_tokenizer_from_resources
from imspy.algorithm.ionization.predictors import BinomialChargeStateDistributionModel, DeepChargeStateDistribution, \
    load_deep_charge_state_predictor
from imspy.chemistry.utility import calculate_mz


def simulate_charge_states(
        peptides: pd.DataFrame,
        mz_lower: float,
        mz_upper: float,
        p_charge: float = 0.5,
        charge_state_one_probability: float = 0.0,
        min_charge_contrib: float = 0.15,
) -> pd.DataFrame:
    """
    Simulate charge states for peptides.
    Args:
        peptides: Peptides DataFrame.
        mz_lower: Lower m/z value.
        mz_upper: Upper m/z value.
        p_charge: Probability of charge.
        charge_state_one_probability: Probability of charge state one.
        min_charge_contrib: Minimum charge contribution.

    Returns:
        pd.DataFrame: Ions DataFrame.
    """

    IonSource = DeepChargeStateDistribution(
        model=load_deep_charge_state_predictor(),
        tokenizer=load_tokenizer_from_resources(tokenizer_name='tokenizer-ptm'),
    )

    # IonSource = BinomialChargeStateDistributionModel(charged_probability=p_charge)
    peptide_ions = IonSource.simulate_charge_state_distribution_pandas(
        peptides,
        charge_state_one_probability=charge_state_one_probability,
        min_charge_contrib=min_charge_contrib,
    )

    # merge tables to have sequences with ions, remove mz values outside scope
    ions = pd.merge(left=peptide_ions, right=peptides, left_on=['peptide_id'], right_on=['peptide_id'])

    # TODO: CHECK IF THIS TAKES MODIFICATIONS INTO ACCOUNT
    ions['mz'] = ions.apply(lambda r: calculate_mz(r['monoisotopic-mass'], r['charge']), axis=1)
    ions = ions[(ions.mz >= mz_lower) & (ions.mz <= mz_upper)]

    return ions
