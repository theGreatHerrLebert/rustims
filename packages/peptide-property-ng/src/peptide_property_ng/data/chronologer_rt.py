"""Chronologer retention-time database -> RT-pretraining examples.

The Chronologer DB (``Chronologer_DB_220308.gz``, ~2.64 M peptide-RT records
harmonised across 11 community datasets) is the RT pretraining corpus. Its
``HI`` (hydrophobicity index) column is the cross-dataset-harmonised RT target
-- so it sidesteps the per-run RT-scale problem -- and is min-max normalised to
``[0, 1]`` to match the model's RT-head output range and the Sage ``aligned_rt``
fine-tuning target.

Peptides use Sage-style ``[+mass]`` delta notation; they are converted to UNIMOD
form with the same ``sagepy_rescore`` converter the Sage loader uses.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from peptide_property_ng.data.sage_dataset import SagePropertyDataset

DEFAULT_CHRONOLOGER_DB = Path(
    "/home/administrator/Documents/promotion/chronologer/data/Chronologer_DB_220308.gz"
)
# HI range across the full DB — fixed so normalisation is reproducible for any cap.
HI_MIN, HI_MAX = -1.01, 31.07
_UNIMOD_RE = re.compile(r"\[UNIMOD:\d+\]")


def _normalise_hi(hi: float) -> float:
    """HI -> ~[0,1], clipped (a few records sit just outside the fixed bounds).

    A non-finite HI propagates as NaN through ``np.clip`` and is filtered by the
    caller.
    """
    return float(np.clip((float(hi) - HI_MIN) / (HI_MAX - HI_MIN), 0.0, 1.0))


def prepare_chronologer_examples(
    db_path: str | Path = DEFAULT_CHRONOLOGER_DB,
    *,
    cap: int | None = None,
    instrument: int = 0,
    acq_mode: int = 0,
    max_len: int = 64,
    seed: int = 0,
) -> list[dict]:
    """Load the Chronologer DB into RT-pretraining example dicts."""
    import pandas as pd
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer

    from peptide_property_ng.data.mass_to_unimod import parse_delta_mass_peptide

    df = pd.read_csv(db_path, sep="\t", compression="gzip", usecols=["PeptideModSeq", "HI"])
    if cap is not None and len(df) > cap:
        df = df.sample(cap, random_state=seed)
    peptides = df["PeptideModSeq"].tolist()
    hi = df["HI"].to_numpy(dtype=np.float32)

    tok = ProformaTokenizer.with_defaults()
    parsed = [parse_delta_mass_peptide(p) for p in peptides]  # [+mass] -> UNIMOD

    examples: list[dict] = []
    chunk = 20000  # tokenise in chunks (batch tokenisation right-pads to the chunk max)
    for start in range(0, len(parsed), chunk):
        block = parsed[start : start + chunk]
        enc = tok([m for _, m in block])
        ids, attn = enc["input_ids"], enc["attention_mask"]
        for j, (stripped, modseq) in enumerate(block):
            if "[+" in modseq or "[-" in modseq:
                continue  # an unconverted delta mass — skip
            n_real = int(sum(attn[j]))
            residue_ids = ids[j][1 : n_real - 1]  # strip [CLS]/[SEP] + right-padding
            length = len(residue_ids)
            if length != len(stripped) or not 3 <= length <= max_len:
                continue
            rt = _normalise_hi(hi[start + j])
            if not np.isfinite(rt):
                continue  # non-finite HI -> useless as an RT example
            examples.append(
                {
                    "accession": "Chronologer",
                    "stripped": stripped,
                    "modseq": modseq,
                    "tokens": np.asarray(residue_ids, dtype=np.int64),
                    "charge": 2,  # dummy — RT is charge-independent; the RT head ignores charge
                    "precursor_mz": 0.0,
                    "collision_energy": 0.0,
                    "instrument": int(instrument),
                    "acq_mode": int(acq_mode),
                    # masked placeholder — RT pretraining runs only the RT task
                    "intensity_target": np.full((length - 1, 6), -1.0, dtype=np.float32),
                    "ccs_target": float("nan"),
                    "ccs_valid": False,
                    "rt_target": rt,
                    "rt_valid": True,
                }
            )
    return examples


def build_chronologer_dataset(
    db_path: str | Path = DEFAULT_CHRONOLOGER_DB, *, cap: int | None = None, **kwargs
) -> SagePropertyDataset:
    """Prepared Chronologer RT examples wrapped as a Dataset."""
    return SagePropertyDataset(prepare_chronologer_examples(db_path, cap=cap, **kwargs))
