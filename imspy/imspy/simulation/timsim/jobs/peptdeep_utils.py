# imspy/algorithm/intensity/peptdeep_wrapper.py

from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd
from peptdeep.pretrained_models import ModelManager

# ----------------------------------------------------------------------
# UNIMOD → peptdeep mapping
# ----------------------------------------------------------------------

UNIMOD_MAP = {
    "UNIMOD:4": "Carbamidomethyl",       # restricted to C
    "UNIMOD:1": "Acetyl@Protein_N-term", # N-term Ac
    "UNIMOD:21": "Phospho",              # S, T, Y
    "UNIMOD:35": "Oxidation",            # M
}


def parse_modified_sequence(mod_seq: str) -> Tuple[str, list[str], list[str]]:
    """
    Parse peptide sequences with UNIMOD mods (e.g., M[UNIMOD:35]EEL...)
    into the peptdeep format: (clean sequence, mods list, mod_sites list).
    """
    seq: list[str] = []
    mods: list[str] = []
    mod_sites: list[str] = []

    pattern = re.compile(r"([A-Z])(?:\[(UNIMOD:\d+)])?")

    pos = 0
    for aa, unimod_code in pattern.findall(mod_seq):
        seq.append(aa)

        if unimod_code:
            if unimod_code not in UNIMOD_MAP:
                raise ValueError(f"Unsupported PTM {unimod_code} in {mod_seq}")

            base = UNIMOD_MAP[unimod_code]

            if base == "Phospho":
                if aa not in ("S", "T", "Y"):
                    raise ValueError(
                        f"Phospho (UNIMOD:21) found on invalid residue {aa} in {mod_seq}"
                    )
                mod_name = f"Phospho@{aa}"

            elif base == "Carbamidomethyl":
                if aa != "C":
                    raise ValueError(
                        f"Carbamidomethyl (UNIMOD:4) found on non-C residue {aa} in {mod_seq}"
                    )
                mod_name = "Carbamidomethyl@C"

            elif base == "Oxidation":
                if aa != "M":
                    raise ValueError(
                        f"Oxidation (UNIMOD:35) found on non-M residue {aa} in {mod_seq}"
                    )
                mod_name = "Oxidation@M"

            else:
                # e.g. Acetyl N-term; treat as mod at AA position for now
                mod_name = base

            mods.append(mod_name)
            mod_sites.append(str(pos))

        pos += 1

    return "".join(seq), mods, mod_sites


def to_peptdeep_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert your synthetic fragment-ion table into peptdeep's peptide input format.
    Expects a 'sequence' and 'charge' column in df.
    """
    records = []
    for _, row in df.iterrows():
        clean_seq, mods, sites = parse_modified_sequence(row["sequence"])
        records.append(
            {
                "sequence": clean_seq,
                "mods": ";".join(mods) if mods else "",
                "mod_sites": ";".join(sites) if sites else "",
                "charge": int(row["charge"]),
            }
        )

    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# PeptDeep MS2 → Prosit-style vector
# ----------------------------------------------------------------------

def peptdeep_ms2_to_prosit_vector(
    b_z1: np.ndarray,
    b_z2: np.ndarray,
    y_z1: np.ndarray,
    y_z2: np.ndarray,
    *,
    max_len: int = 30,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Map PeptDeep MS2 outputs (b1, b2, y1, y2) for a single peptide
    into a Prosit-style flattened 174-dim vector.

    internal tensor shape: (29, 2, 3)
      [frag_idx, ion_type(0=y,1=b), charge_idx(0→z1,1→z2,2→z3)]
    """

    from imspy.simulation.utility import flatten_prosit_array

    if not (len(b_z1) == len(b_z2) == len(y_z1) == len(y_z2)):
        raise ValueError("All four PeptDeep arrays must have the same length.")

    n_frags = len(b_z1)
    max_frags = max_len - 1  # 29

    if n_frags > max_frags:
        raise ValueError(
            f"Got {n_frags} fragments but max supported is {max_frags} "
            f"(peptide length {max_len})."
        )

    tensor = np.full((max_frags, 2, 3), fill_value, dtype=np.float32)

    # fill only valid fragment positions [0..n_frags-1]
    # y ions
    tensor[:n_frags, 0, 0] = y_z1  # y, z=1
    tensor[:n_frags, 0, 1] = y_z2  # y, z=2

    # b ions
    tensor[:n_frags, 1, 0] = b_z1  # b, z=1
    tensor[:n_frags, 1, 1] = b_z2  # b, z=2

    # z=3 channel left at fill_value

    return flatten_prosit_array(tensor)


def to_prosit_arrays(
    peptdeep_input: pd.DataFrame,
    peptdeep_ms2: pd.DataFrame,
    *,
    max_len: int = 30,
    fill_value: float = -1.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert PeptDeep input+output into a matrix of Prosit-style 174-dim vectors,
    with max-normalization such that max intensity = 1.

    Assumes peptdeep_ms2 contains concatenated b_z1/b_z2/y_z1/y_z2 arrays
    in the same order as peptdeep_input.
    """

    bz1 = peptdeep_ms2["b_z1"].to_numpy()
    bz2 = peptdeep_ms2["b_z2"].to_numpy()
    yz1 = peptdeep_ms2["y_z1"].to_numpy()
    yz2 = peptdeep_ms2["y_z2"].to_numpy()

    if not (len(bz1) == len(bz2) == len(yz1) == len(yz2)):
        raise ValueError("PeptDeep MS2 arrays must all have the same concatenated length.")

    n_peps = len(peptdeep_input)
    prosit_mat = np.zeros((n_peps, 174), dtype=np.float32)

    seq_len_counter = 0

    for i, (_, row) in enumerate(peptdeep_input.iterrows()):
        seq = row["sequence"]
        L = len(seq)

        if L > max_len:
            raise ValueError(
                f"Sequence length {L} exceeds max_len {max_len}: {seq!r}"
            )

        n_frags = L - 1
        start = seq_len_counter
        end = seq_len_counter + n_frags

        b1 = bz1[start:end]
        b2 = bz2[start:end]
        y1 = yz1[start:end]
        y2 = yz2[start:end]

        if len(b1) != n_frags:
            raise ValueError(
                f"Fragment count mismatch for peptide {i}: expected {n_frags}, got {len(b1)}"
            )

        vec = peptdeep_ms2_to_prosit_vector(
            b1, b2, y1, y2, max_len=max_len, fill_value=fill_value
        )

        if normalize:
            m = float(vec.max())
            if m > 0.0:
                vec = vec / m

        prosit_mat[i, :] = vec
        seq_len_counter = end

    if seq_len_counter != len(bz1):
        raise ValueError(
            f"Did not consume all PeptDeep intensities: "
            f"used {seq_len_counter}, total {len(bz1)}"
        )

    return prosit_mat


def simulate_peptdeep_intensities_pandas(
    transmitted_fragment_ions: pd.DataFrame,
    *,
    device: str = "cpu",
    fill_value: float = 0.0,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    High-level helper: run PeptDeep MS2 prediction on the transmitted_fragment_ions
    and return a DataFrame compatible with the Prosit wrapper:
        same rows, plus an 'intensity' column with 174-dim np.ndarray per row.
    """

    # 1) Build peptdeep input
    p_input = to_peptdeep_df(transmitted_fragment_ions)

    # 2) Predict MS2
    model_mgr = ModelManager(mask_modloss=True, device=device)
    intensity_df = model_mgr.predict_ms2(p_input)

    # 3) Convert to Prosit-style flattened vectors
    X = to_prosit_arrays(
        peptdeep_input=p_input,
        peptdeep_ms2=intensity_df,
        fill_value=fill_value,
        normalize=normalize,
    )

    # 4) Attach to original DF
    out = transmitted_fragment_ions.copy()
    # one 174-vector per row
    out["intensity"] = list(X)

    return out