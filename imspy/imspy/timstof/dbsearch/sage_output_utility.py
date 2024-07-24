import re
import numpy as np
import pandas as pd

import random

from sagepy.core import Fragments, IonType
from scipy.spatial import distance

from sagepy.utility import get_features
from sagepy.qfdr.tdc import target_decoy_competition_pandas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from numpy.typing import NDArray
from typing import Tuple

from tqdm import tqdm


def break_into_equal_size_sets(sequence_set, k: int = 10):
    """
    Breaks a set of objects into k sets of equal size at random.

    :param sequence_set: Set of sequences to be divided
    :param k: Number of sets to divide the objects into
    :return: A list containing k sets, each with equal number of randomly chosen sequences
    """
    objects_list = list(sequence_set)  # Convert the set to a list

    # Shuffle the objects to ensure randomness
    random.shuffle(objects_list)

    # Calculate the size of each set
    set_size = len(objects_list) // k
    remainder = len(objects_list) % k

    sets = []
    start = 0
    for i in range(k):
        end = start + set_size + (1 if i < remainder else 0)
        sets.append(set(objects_list[start:end]))
        start = end

    return sets


def split_dataframe_randomly(df: pd.DataFrame, n: int) -> list:

    sequences_set = set(df.sequence.values)
    split_sets = break_into_equal_size_sets(sequences_set, n)

    ret_list = []

    for seq_set in split_sets:
        ret_list.append(df[df['sequence'].apply(lambda s: s in seq_set)])

    return ret_list


def generate_training_data(psms: pd.DataFrame, method: str = "psm", q_max: float = 0.01,
                           balance: bool = True) -> Tuple[NDArray, NDArray]:
    """ Generate training data.
    Args:
        psms: List of PeptideSpectrumMatch objects
        method: Method to use for training data generation
        q_max: Maximum q-value allowed for positive examples
        balance: Whether to balance the dataset

    Returns:
        Tuple[NDArray, NDArray]: X_train and Y_train
    """
    # create pandas table from psms
    PSM_pandas = psms

    # calculate q-values to get inital "good" hits
    PSM_q = target_decoy_competition_pandas(PSM_pandas, method=method)
    PSM_pandas = PSM_pandas.drop(columns=["q_value", "score"])

    # merge data with q-values
    TDC = pd.merge(PSM_q, PSM_pandas, left_on=["spec_idx", "match_idx", "decoy"],
                   right_on=["spec_idx", "match_idx", "decoy"])

    # select best positive examples
    TARGET = TDC[(TDC.decoy == False) & (TDC.q_value <= q_max)]
    X_target, Y_target = get_features(TARGET)

    # select all decoys
    DECOY = TDC[TDC.decoy]
    X_decoy, Y_decoy = get_features(DECOY)

    # balance the dataset such that the number of target and decoy examples are equal
    if balance:
        num_target = np.min((len(DECOY), len(TARGET)))
        target_indices = np.random.choice(np.arange(len(X_target)), size=num_target)
        X_target = X_target[target_indices, :]
        Y_target = Y_target[target_indices]

    X_train = np.vstack((X_target, X_decoy))
    Y_train = np.hstack((Y_target, Y_decoy))

    return X_train, Y_train


def re_score_psms(
        psms: pd.DataFrame,
        num_splits: int = 10,
        verbose: bool = True,
        balance: bool = True,
        score: str = "hyperscore",
        positive_example_q_max: float = 0.01,
) -> pd.DataFrame:
    """ Re-score PSMs using LDA.
    Args:
        psms: List of PeptideSpectrumMatch objects
        num_splits: Number of splits
        verbose: Whether to print progress
        balance: Whether to balance the dataset
        score: Score to use for re-scoring
        positive_example_q_max: Maximum q-value allowed for positive examples

    Returns:
        List[PeptideSpectrumMatch]: List of PeptideSpectrumMatch objects
    """

    scaler = StandardScaler()
    X_all, _ = get_features(psms, score=score)
    X_all = np.nan_to_num(X_all, nan=0.0)
    scaler.fit(X_all)

    splits = split_dataframe_randomly(df=psms, n=num_splits)
    predictions, ids = [], []

    for i in tqdm(range(num_splits), disable=not verbose, desc='Re-scoring PSMs', ncols=100):

        target = splits[i]
        ids.extend(target["spec_idx"].values)
        features = []

        for j in range(num_splits):
            if j != i:
                features.append(splits[j])

        if num_splits == 1:
            features = [target]

        X_train, Y_train = generate_training_data(pd.concat(features), balance=balance, q_max=positive_example_q_max)
        X_train, Y_train = np.nan_to_num(X_train, nan=0.0), np.nan_to_num(Y_train, nan=0.0)
        X, _ = get_features(target)
        X = np.nan_to_num(X, nan=0.0)

        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        lda.fit(scaler.transform(X_train), Y_train)

        try:
            # check for flip sign of LDA classification return to be compatible with good score ascending
            score_flip = 1.0 if Y_train[
                                    np.argmax(np.squeeze(lda.transform(scaler.transform(X_train))))] == 1.0 else -1.0
        except:
            score_flip = 1.0

        Y_pred = np.squeeze(lda.transform(scaler.transform(X))) * score_flip
        predictions.extend(Y_pred)

    return pd.DataFrame({"spec_idx": ids, "re_score": predictions})


def cosim_from_dict(observed, predicted):
    intensities_a = []
    intensities_b = []

    for k, v in observed.items():
        if k in predicted:
            intensities_a.append(v)
            intensities_b.append(predicted[k])
        else:
            intensities_a.append(v)
            intensities_b.append(0)

    a, b = np.array(intensities_a), np.array(intensities_b)
    return 1 - distance.cosine(a, b)


def row_to_fragment(r):
    charges = r.fragment_charge
    ion_types = r.fragment_type
    fragment_ordinals = r.fragment_ordinals
    intensities = r.fragment_intensity
    mz_calculated = r.fragment_mz_calculated
    mz_experimental = r.fragment_mz_experimental

    ion_types_parsed = []

    for ion in ion_types:
        if ion == "b":
            ion_types_parsed.append(IonType("b"))
        else:
            ion_types_parsed.append(IonType("y"))

    return Fragments(charges, ion_types_parsed, fragment_ordinals, intensities, mz_calculated, mz_experimental)


def remove_substrings(input_string: str) -> str:
    result = re.sub(r'\[.*?\]', '', input_string)
    return result


replace_tokens = {
    "[+42]": "[UNIMOD:1]",
    "[+42.010565]": "[UNIMOD:1]",
    "[+57.0215]": "[UNIMOD:4]",
    "[+57.021464]": "[UNIMOD:4]",
    "[+79.9663]": "[UNIMOD:21]",
    "[+15.9949]": "[UNIMOD:35]",
    "[+15.994915]": "[UNIMOD:35]",
}


class PatternReplacer:
    def __init__(
            self,
            replacements: dict[str, str],
            pattern: str | re.Pattern = r"\[.*?\]",
    ):
        self.pattern = re.compile(pattern)
        self.replacements = replacements
        for _in, _out in replacements.items():
            assert (
                    len(re.findall(self.pattern, _in)) > 0
            ), f"The submitted replacemnt, `{_in}->{_out}`, cannot be used with pattern `{pattern}`."

    def apply(self, string: str) -> str:
        out_sequence = string
        for _in in set(re.findall(self.pattern, string)):
            try:
                _out = self.replacements[_in]
            except KeyError:
                raise KeyError(
                    f"Modification {_in} not among those specified in the replacements."
                )
            out_sequence = out_sequence.replace(_in, _out)
        return out_sequence


def fragments_to_dict(fragments: Fragments):
    dict = {}
    for charge, ion_type, ordinal, intensity in zip(fragments.charges, fragments.ion_types, fragments.fragment_ordinals,
                                                    fragments.intensities):
        dict[(charge, ion_type, ordinal)] = intensity

    return dict
