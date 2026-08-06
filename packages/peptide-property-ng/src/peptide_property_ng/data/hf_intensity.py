"""Wilhelmlab HuggingFace MS2 datasets -> intensity-pretraining examples.

The three Wilhelmlab MS2 datasets — `timsTOF-ms2`, `prospect-ptms-ms2`,
`Prosit-2025-lac-ms2` — share one schema:
  - `intensities_raw`         : the Prosit 174-vector (`-1` = masked)
  - `modified_sequence`       : `[]-...-[]` terminal placeholders + inline `[UNIMOD:N]`
  - `precursor_charge_onehot` : 6-way one-hot charge
  - `collision_energy_aligned_normed` : normalised collision energy

Each row becomes an example dict identical in shape to the Sage loader's, so the
same collate function and model consume it. CCS / RT are absent here (masked).
The intensity target is produced by the *same* `prosit174_to_sites` conversion
the Sage path uses.
"""
from __future__ import annotations

import itertools
import re

import numpy as np

from peptide_property_ng.data.fragment_targets import prosit174_to_sites
from peptide_property_ng.data.sage_dataset import SagePropertyDataset

_UNIMOD_RE = re.compile(r"\[UNIMOD:\d+\]")


def _strip_terminal_placeholders(modseq: str) -> str:
    """Drop empty `[]-` / `-[]` terminal-modification placeholders.

    A non-empty terminal bracket (a real terminal mod) is left in place; such a
    peptide then fails the token-count check and is skipped — same policy as the
    Sage loader.
    """
    modseq = re.sub(r"^\[\]-", "", modseq)
    modseq = re.sub(r"-\[\]$", "", modseq)
    return modseq


def prepare_hf_intensity_examples(
    dataset_name: str,
    *,
    split: str = "train",
    cap: int | None = None,
    instrument: int = 0,
    acq_mode: int = 0,
    max_len: int = 30,
    batch: int = 2048,
) -> list[dict]:
    """Stream a Wilhelmlab HF MS2 dataset into intensity-pretraining examples."""
    from datasets import load_dataset
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer

    tok = ProformaTokenizer.with_defaults()
    rows = load_dataset(dataset_name, split=split, streaming=True)
    if cap is not None:
        rows = itertools.islice(rows, cap)

    examples: list[dict] = []
    buf: list[dict] = []

    def _flush() -> None:
        if not buf:
            return
        modseqs = [_strip_terminal_placeholders(r["modified_sequence"]) for r in buf]
        enc = tok(modseqs)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        for row, modseq, tok_ids, am in zip(buf, modseqs, ids, attn):
            if "-" in modseq or "[+" in modseq:
                continue  # an unresolved terminal mod / delta mass — skip
            n_real = int(sum(am))
            residue_ids = tok_ids[1 : n_real - 1]  # strip [CLS]/[SEP] + right-padding
            length = len(residue_ids)
            stripped = _UNIMOD_RE.sub("", modseq)
            if length != len(stripped) or not 3 <= length <= max_len:
                continue
            onehot = row["precursor_charge_onehot"]
            if sum(onehot) != 1:
                continue  # malformed / all-zero one-hot -> skip rather than default to charge 1
            charge = int(np.argmax(onehot)) + 1
            try:
                target = prosit174_to_sites(row["intensities_raw"], length)
            except ValueError:
                continue
            examples.append(
                {
                    "accession": dataset_name,
                    "stripped": stripped,
                    "modseq": modseq,
                    "tokens": np.asarray(residue_ids, dtype=np.int64),
                    "charge": charge,
                    "precursor_mz": 0.0,  # absent here; unused by the intensity task
                    "collision_energy": float(row["collision_energy_aligned_normed"]),
                    "instrument": int(instrument),
                    "acq_mode": int(acq_mode),
                    "intensity_target": target,
                    "ccs_target": float("nan"), "ccs_valid": False,
                    "rt_target": float("nan"), "rt_valid": False,
                }
            )
        buf.clear()

    for row in rows:
        buf.append(row)
        if len(buf) >= batch:
            _flush()
    _flush()
    return examples


def build_hf_intensity_dataset(
    dataset_name: str, *, split: str = "train", cap: int | None = None, **kwargs
) -> SagePropertyDataset:
    """Prepared HF intensity examples wrapped as a Dataset."""
    return SagePropertyDataset(
        prepare_hf_intensity_examples(dataset_name, split=split, cap=cap, **kwargs)
    )
