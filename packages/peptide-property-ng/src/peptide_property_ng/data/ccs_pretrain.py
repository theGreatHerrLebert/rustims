"""ionmob CCS datasets -> ion-mobility-pretraining examples.

Uses the deduplicated, UNIMOD-annotated ionmob CCS parquets
(`*_unique_unimod*.parquet`, Meier / Tenzer / Chang / ...): schema
``mz, charge, sequence-tokenized, ccs, name``.

Target-unit note: these provide **CCS** (Angstrom^2); the Sage fine-tuning data
provides **inverse ion mobility** (1/K0). They are deliberately *not* converted
into each other — that physics conversion is error-prone, and it is unnecessary:
pretraining is single-task per stage, so the CCS head simply warms on CCS here
and the unit/scale shift to 1/K0 is absorbed during the campaign fine-tune —
exactly as the RT head pretrains on Chronologer ``HI`` and fine-tunes on Sage
``aligned_rt``.
"""
from __future__ import annotations

import glob
import re

import numpy as np
import pyarrow.parquet as pq

from peptide_property_ng.data.sage_dataset import SagePropertyDataset

DEFAULT_CCS_GLOB = (
    "/home/administrator/Documents/promotion/rust/rustims-submission/"
    "re-scoring/models/unimod/**/*_unique_unimod*.parquet"
)
_UNIMOD_RE = re.compile(r"\[UNIMOD:\d+\]")

# CCS (Angstrom^2) is min-max normalised to ~[0,1] with fixed bounds so the CCS
# head sees the *same target scale* in pretraining (CCS) and fine-tuning (Sage
# 1/K0, normalised the same way in sage_dataset.py) — this avoids the negative
# transfer of feeding a ~400-scale target into a [0,1]-scale head.
CCS_MIN, CCS_MAX = 200.0, 1200.0


def normalize_ccs(ccs: float) -> float:
    """Map a CCS value (Angstrom^2) to [0,1], clipped at the fixed bounds."""
    return float(np.clip((float(ccs) - CCS_MIN) / (CCS_MAX - CCS_MIN), 0.0, 1.0))


def _tokens_to_proforma(tokens: list[str]) -> str | None:
    """ionmob ``sequence-tokenized`` list -> a ProForma string, or ``None`` to skip.

    A ``<START>`` / ``<END>`` marker carrying a terminal modification (e.g.
    ``<START>[UNIMOD:1]``) flags a terminal mod -> skip, consistent with the
    other loaders.
    """
    if len(tokens) < 5 or tokens[0] != "<START>" or tokens[-1] != "<END>":
        return None
    return "".join(tokens[1:-1])


def prepare_ccs_examples(
    data_glob: str = DEFAULT_CCS_GLOB,
    *,
    cap: int | None = None,
    instrument: int = 0,
    acq_mode: int = 0,
    max_len: int = 64,
    seed: int = 0,
) -> list[dict]:
    """Load the ionmob CCS parquets into ion-mobility-pretraining example dicts."""
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer

    files = sorted(set(glob.glob(data_glob, recursive=True)))
    if not files:
        return []
    seqs: list[list[str]] = []
    charge: list[int] = []
    mz: list[float] = []
    ccs: list[float] = []
    for path in files:
        d = pq.read_table(path, columns=["mz", "charge", "sequence-tokenized", "ccs"]).to_pydict()
        seqs += d["sequence-tokenized"]
        charge += d["charge"]
        mz += d["mz"]
        ccs += d["ccs"]

    idx = np.arange(len(seqs))
    if cap is not None and len(idx) > cap:
        idx = np.sort(np.random.RandomState(seed).choice(len(idx), cap, replace=False))

    tok = ProformaTokenizer.with_defaults()
    proformas: list[str] = []
    src: list[int] = []
    for i in idx:
        pf = _tokens_to_proforma(seqs[i])
        if pf is not None:
            proformas.append(pf)
            src.append(int(i))

    examples: list[dict] = []
    chunk = 20000  # tokenise in chunks (batch tokenisation right-pads to the chunk max)
    for start in range(0, len(proformas), chunk):
        block = proformas[start : start + chunk]
        block_src = src[start : start + chunk]
        enc = tok(block)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        for j, pf in enumerate(block):
            i = block_src[j]
            n_real = int(sum(attn[j]))
            residue_ids = ids[j][1 : n_real - 1]
            length = len(residue_ids)
            stripped = _UNIMOD_RE.sub("", pf)
            if length != len(stripped) or not 3 <= length <= max_len:
                continue
            ccs_val, mz_val, z = float(ccs[i]), float(mz[i]), int(charge[i])
            if not (np.isfinite(ccs_val) and ccs_val > 0
                    and np.isfinite(mz_val) and mz_val > 0 and z >= 1):
                continue  # skip malformed rows rather than emit a NaN target
            examples.append(
                {
                    "accession": "ionmob",
                    "stripped": stripped,
                    "modseq": pf,
                    "tokens": np.asarray(residue_ids, dtype=np.int64),
                    "charge": z,
                    "precursor_mz": mz_val,
                    "collision_energy": 0.0,
                    "instrument": int(instrument),
                    "acq_mode": int(acq_mode),
                    # masked placeholder — CCS pretraining runs only the CCS task
                    "intensity_target": np.full((length - 1, 6), -1.0, dtype=np.float32),
                    "ccs_target": normalize_ccs(ccs_val),
                    "ccs_valid": True,
                    "rt_target": float("nan"),
                    "rt_valid": False,
                }
            )
    return examples


def build_ccs_dataset(data_glob: str = DEFAULT_CCS_GLOB, *, cap: int | None = None, **kwargs):
    """Prepared ionmob CCS examples wrapped as a Dataset."""
    return SagePropertyDataset(prepare_ccs_examples(data_glob, cap=cap, **kwargs))
