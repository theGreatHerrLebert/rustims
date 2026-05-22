"""Sage-parquet -> multi-task training examples.

Reads a dataset's ``results.sage.parquet`` + ``matched_fragments.sage.parquet``,
filters to confident target PSMs, converts the Sage delta-mass peptide strings
to UNIMOD form (via ``sagepy_rescore``'s residue-specificity-aware converter),
tokenises, and builds per-task targets.
"""
from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from peptide_property_ng.data.fragment_targets import prosit174_to_sites
from peptide_property_ng.data.splits import peptide_split

_PROTON = 1.007276

# Inverse ion mobility (1/K0) is min-max normalised to ~[0,1] with fixed bounds,
# matching the CCS normalisation in ccs_pretrain.py so the CCS/IM head sees one
# consistent target scale across pretraining (CCS) and fine-tuning (1/K0).
IM_MIN, IM_MAX = 0.5, 1.9


def normalize_inverse_mobility(one_over_k0: float) -> float:
    """Map an inverse ion mobility (1/K0) value to [0,1], clipped at the fixed bounds."""
    return float(np.clip((float(one_over_k0) - IM_MIN) / (IM_MAX - IM_MIN), 0.0, 1.0))

_RES_COLS = [
    "psm_id", "peptide", "stripped_peptide", "charge", "calcmass",
    "aligned_rt", "ion_mobility", "spectrum_q", "is_decoy", "rank",
    "matched_peaks", "peptide_len",
]
_FRAG_COLS = [
    "psm_id", "fragment_type", "fragment_ordinals", "fragment_charge", "fragment_intensity",
]


def discover_sage_dirs(glob_pattern: str) -> list[Path]:
    """Find non-empty ``.../processed/sage`` directories under a dataset-root glob.

    Datasets whose ``results.sage.parquet`` has zero rows (e.g. the legacy
    ``TimsCompressionType=1`` deposits Sage cannot read) are skipped.
    """
    out: list[Path] = []
    for root in sorted(glob.glob(glob_pattern)):
        res = Path(root) / "processed" / "sage" / "results.sage.parquet"
        if not res.exists():
            continue
        try:
            if pq.read_metadata(res).num_rows == 0:
                continue
        except Exception:
            continue
        out.append(res.parent)
    return out


def _accession(sage_dir: Path) -> str:
    # .../<ACCESSION>/processed/sage  ->  <ACCESSION>
    return sage_dir.parts[-3] if len(sage_dir.parts) >= 3 else sage_dir.name


def prepare_examples(
    sage_dir: str | Path,
    *,
    cap: int = 8000,
    q_max: float = 0.01,
    min_peaks: int = 6,
    max_len: int = 30,  # the Prosit 174-vector intensity encoding caps peptides at 30 aa
    seed: int = 0,
    instrument: int = 0,
    acq_mode: int = 0,
) -> list[dict]:
    """Load and prepare one dataset's PSMs into a list of example dicts."""
    from imspy_predictors.intensity.predictors import observed_fragments_to_intensity_target
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer

    from peptide_property_ng.data.mass_to_unimod import parse_delta_mass_peptide

    sage_dir = Path(sage_dir)
    res_path = sage_dir / "results.sage.parquet"
    frag_path = sage_dir / "matched_fragments.sage.parquet"
    if not res_path.exists():
        return []
    res = pq.read_table(res_path, columns=_RES_COLS)
    if res.num_rows == 0:
        return []

    keep = pc.and_(
        pc.and_(pc.equal(res["is_decoy"], False), pc.equal(res["rank"], 1)),
        pc.and_(
            pc.less_equal(res["spectrum_q"], q_max),
            pc.and_(
                pc.greater_equal(res["matched_peaks"], min_peaks),
                pc.less_equal(res["peptide_len"], max_len),
            ),
        ),
    )
    res = res.filter(keep)
    if res.num_rows == 0:
        return []
    if res.num_rows > cap:
        idx = np.sort(np.random.RandomState(seed).choice(res.num_rows, cap, replace=False))
        res = res.take(pa.array(idx))
    df = res.to_pydict()
    n = len(df["psm_id"])

    # Fragments for the kept PSMs, grouped by psm_id.
    frag = pq.read_table(frag_path, columns=_FRAG_COLS)
    frag = frag.filter(pc.is_in(frag["psm_id"], value_set=pa.array(list(set(df["psm_id"])))))
    fd = frag.to_pydict()
    by_psm: dict[int, tuple[list, list, list, list]] = defaultdict(lambda: ([], [], [], []))
    for pid, ft, fo, fc, fi in zip(
        fd["psm_id"], fd["fragment_type"], fd["fragment_ordinals"],
        fd["fragment_charge"], fd["fragment_intensity"],
    ):
        cols = by_psm[pid]
        cols[0].append(ft); cols[1].append(fo); cols[2].append(fc); cols[3].append(fi)

    # Sage emits delta-mass peptides; convert to UNIMOD, then tokenise in one batch.
    # Batch tokenisation right-pads to the batch-max length, so the attention
    # mask is needed to recover each peptide's true token count.
    parsed = [parse_delta_mass_peptide(p) for p in df["peptide"]]
    tok = ProformaTokenizer.with_defaults()
    _encoded = tok([m for _, m in parsed])
    token_ids, attn_mask = _encoded["input_ids"], _encoded["attention_mask"]

    accession = _accession(sage_dir)
    examples: list[dict] = []
    for k in range(n):
        stripped, modseq = parsed[k]
        if "[+" in modseq or "[-" in modseq:
            continue  # an unconverted delta mass — skip rather than train on a dropped mod
        n_real = int(sum(attn_mask[k]))  # real tokens incl. [CLS]/[SEP], excl. padding
        residue_ids = token_ids[k][1 : n_real - 1]  # strip [CLS]/[SEP] and right-padding
        length = len(residue_ids)
        # Terminal mods become standalone tokens -> token count != residue count;
        # skip those rather than mis-align the fragment indexing.
        if length != int(df["peptide_len"][k]) or length < 3 or length > max_len:
            continue

        charge = int(df["charge"][k])
        mz = (float(df["calcmass"][k]) + charge * _PROTON) / charge
        cols = by_psm.get(df["psm_id"][k])
        if not cols or not cols[0]:
            continue  # no matched fragments -> no intensity target
        # Encode fragments with the proven imspy encoder (Sage fragments ->
        # Prosit 174-vector), then the single shared 174 -> site conversion.
        frag = SimpleNamespace(
            ion_types=cols[0], fragment_ordinals=cols[1],
            charges=cols[2], intensities=cols[3],
        )
        prosit174 = observed_fragments_to_intensity_target(stripped, charge, frag)
        target = prosit174_to_sites(prosit174, length)

        im = df["ion_mobility"][k]
        rt = df["aligned_rt"][k]
        im = float(im) if im is not None else float("nan")
        rt = float(rt) if rt is not None else float("nan")
        examples.append(
            {
                "accession": accession,
                "stripped": stripped,
                "modseq": modseq,
                "tokens": np.asarray(residue_ids, dtype=np.int64),
                "charge": charge,
                "precursor_mz": mz,
                "collision_energy": 0.0,  # not in Sage output for timsTOF diaPASEF
                "instrument": int(instrument),
                "acq_mode": int(acq_mode),
                "intensity_target": target,
                "ccs_target": normalize_inverse_mobility(im),
                "ccs_valid": bool(np.isfinite(im) and im > 0.0),
                "rt_target": rt,
                "rt_valid": bool(np.isfinite(rt)),
            }
        )
    return examples


class SagePropertyDataset(Dataset):
    """A thin wrapper over a list of prepared example dicts."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        return self.examples[i]


def build_split_datasets(
    sage_dirs: list[str | Path],
    *,
    cap: int = 8000,
    seed: int = 0,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    **prepare_kwargs,
) -> dict[str, SagePropertyDataset]:
    """Prepare every dataset and split peptide-level into train / val / test."""
    examples: list[dict] = []
    for sage_dir in sage_dirs:
        examples.extend(prepare_examples(sage_dir, cap=cap, seed=seed, **prepare_kwargs))

    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for ex in examples:
        split = peptide_split(ex["stripped"], val_frac=val_frac, test_frac=test_frac, seed=seed)
        buckets[split].append(ex)
    return {name: SagePropertyDataset(rows) for name, rows in buckets.items()}
