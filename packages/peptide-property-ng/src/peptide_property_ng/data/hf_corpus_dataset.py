"""HF-corpus -> multi-task training examples (validation of the published corpus).

Mirrors ``sage_dataset.prepare_examples`` but sources from the aggregated HF
corpus parquets (TO_HF_CORPUS.md): Tier-1 (per-precursor RT/IM/charge/CE) +
Tier-3 (per-precursor matched b/y fragments). Produces the SAME example dicts
``SagePropertyDataset`` wraps, so it drops into the existing collate + train loop.

Differences vs the Sage-parquet path, by design:
- RT: Tier-1 ships ``rt_seconds`` (raw apex), not Sage ``aligned_rt`` -> we
  **per-accession min-max normalise** RT so the head sees a comparable scale.
- CE: Tier-1 ships real per-precursor ``collision_energy_mean_v`` (volts) ->
  ``CE = clip(volts/100, 0, 1)`` (the production convention), not a flat 0.26.
- The split is already baked into the parquet (``split`` column).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.dataset as pds

from peptide_property_ng.data.fragment_targets import prosit174_to_sites
from peptide_property_ng.data.sage_dataset import (SagePropertyDataset,
                                                   normalize_inverse_mobility)

_PROTON = 1.007276

_T1_COLS = ["accession", "raw_file", "precursor_id", "sage_psm_id",
            "modified_sequence", "sequence", "charge", "mz", "rt_seconds",
            "mobility", "collision_energy_mean_v"]


def load_rt_aligned_lookup(path):
    """{(accession, int sage_psm_id): aligned_rt} from a lookup parquet."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["accession", "psm_id", "aligned_rt"])
    accs = t["accession"].to_pylist()
    pids = t["psm_id"].to_pylist()
    rts = t["aligned_rt"].to_pylist()
    return {(a, int(p)): r for a, p, r in zip(accs, pids, rts) if p is not None}
_T3_COLS = ["accession", "raw_file", "precursor_id", "fragment_type",
            "fragment_ordinal", "fragment_charge", "fragment_intensity"]


def _key(acc, raw, pid):
    return (acc, raw, int(pid))


def prepare_examples_hf(corpus_dir, split, *, cap=4000, max_len=30, seed=0,
                        accessions=None, rt_lookup=None):
    """Build example dicts from ``corpus_dir/{tier1,tier3}/{split}.parquet``.

    rt_lookup: optional {(accession, int sage_psm_id): aligned_rt} (or a path)
    — when a precursor's aligned RT is found it is used (cross-run consistent,
    ~[0,1]); otherwise we fall back to per-accession-normalised raw rt_seconds.
    """
    from imspy_predictors.intensity.predictors import observed_fragments_to_intensity_target
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer

    from peptide_property_ng.data.mass_to_unimod import parse_delta_mass_peptide

    if isinstance(rt_lookup, (str, Path)):
        rt_lookup = load_rt_aligned_lookup(rt_lookup)

    corpus_dir = Path(corpus_dir)
    afilter = (pds.field("accession").isin(list(accessions))
               if accessions else None)

    # Tier-3 is ~391M rows -> NEVER materialise it. Pass 1: stream Tier-1
    # (pushdown + batches), pick up to `cap` precursors/accession + props +
    # per-accession RT range. Pass 2: stream Tier-3, keep only the selected
    # precursors' fragments. Memory stays bounded by the selection (~50k prec).
    t1set = pds.dataset(corpus_dir / "tier1" / f"{split}.parquet")
    sel: dict = {}                       # key -> props
    per_acc_count: dict = defaultdict(int)
    rt_lo, rt_hi = {}, {}
    for rb in t1set.scanner(columns=_T1_COLS, filter=afilter,
                            batch_size=131072).to_batches():
        d = rb.to_pydict()
        for i in range(len(d["precursor_id"])):
            acc = d["accession"][i]
            rt = d["rt_seconds"][i]
            if rt is not None and np.isfinite(rt):
                rt_lo[acc] = rt if acc not in rt_lo else min(rt_lo[acc], rt)
                rt_hi[acc] = rt if acc not in rt_hi else max(rt_hi[acc], rt)
            if per_acc_count[acc] >= cap or not d["modified_sequence"][i]:
                continue
            key = _key(acc, d["raw_file"][i], d["precursor_id"][i])
            sel[key] = {"acc": acc, "modseq_raw": d["modified_sequence"][i],
                        "charge": d["charge"][i], "mz": d["mz"][i],
                        "rt": rt, "mobility": d["mobility"][i],
                        "ce_v": d["collision_energy_mean_v"][i],
                        "sage_psm_id": d["sage_psm_id"][i]}
            per_acc_count[acc] += 1

    by_prec: dict = defaultdict(lambda: ([], [], [], []))
    t3set = pds.dataset(corpus_dir / "tier3" / f"{split}.parquet")
    for rb in t3set.scanner(columns=_T3_COLS, filter=afilter,
                            batch_size=262144).to_batches():
        fd = rb.to_pydict()
        for acc, raw, pid, ft, fo, fc, fi in zip(
                fd["accession"], fd["raw_file"], fd["precursor_id"],
                fd["fragment_type"], fd["fragment_ordinal"],
                fd["fragment_charge"], fd["fragment_intensity"]):
            key = _key(acc, raw, pid)
            if key not in sel:
                continue
            cols = by_prec[key]
            cols[0].append(ft); cols[1].append(int(fo))
            cols[2].append(int(fc)); cols[3].append(float(fi))

    tok = ProformaTokenizer.with_defaults()
    examples: list[dict] = []
    for key, p in sel.items():
        acc = p["acc"]
        cols = by_prec.get(key)
        if not cols or not cols[0]:
            continue  # no matched fragments -> no intensity target
        modseq_raw = p["modseq_raw"]
        stripped, modseq = parse_delta_mass_peptide(modseq_raw)
        if "[+" in modseq or "[-" in modseq:
            continue  # unconverted delta mass
        enc = tok([modseq])
        ids, mask = enc["input_ids"][0], enc["attention_mask"][0]
        n_real = int(sum(mask))
        residue_ids = ids[1:n_real - 1]
        length = len(residue_ids)
        if length < 3 or length > max_len:
            continue

        charge = int(p["charge"])
        mz = float(p["mz"]) if p["mz"] is not None else 0.0
        frag = SimpleNamespace(ion_types=cols[0], fragment_ordinals=cols[1],
                               charges=cols[2], intensities=cols[3])
        prosit174 = observed_fragments_to_intensity_target(stripped, charge, frag)
        target = prosit174_to_sites(prosit174, length)

        im = p["mobility"]
        im = float(im) if im is not None else float("nan")
        # RT target: prefer Sage aligned_rt (cross-run consistent, ~[0,1]);
        # else fall back to per-accession-normalised raw rt_seconds.
        al = None
        if rt_lookup is not None and p["sage_psm_id"] is not None:
            al = rt_lookup.get((acc, int(p["sage_psm_id"])))
        if al is not None and np.isfinite(al):
            rt_norm = float(min(1.0, max(0.0, al)))
            rt_valid = True
        else:
            rt = float(p["rt"]) if p["rt"] is not None else float("nan")
            lo, hi = rt_lo.get(acc), rt_hi.get(acc)
            if np.isfinite(rt) and lo is not None and hi is not None and hi > lo:
                rt_norm = (rt - lo) / (hi - lo)
                rt_valid = True
            else:
                rt_norm, rt_valid = float("nan"), False

        ce = float(np.clip((p["ce_v"] or 26.0) / 100.0, 0.0, 1.0))

        examples.append({
            "accession": acc, "psm_id": int(key[2]),
            "stripped": stripped, "modseq": modseq,
            "tokens": np.asarray(residue_ids, dtype=np.int64),
            "charge": charge, "precursor_mz": mz, "collision_energy": ce,
            "instrument": 0, "acq_mode": 0,
            "intensity_target": target,
            "ccs_target": normalize_inverse_mobility(im),
            "ccs_valid": bool(np.isfinite(im) and im > 0.0),
            "rt_target": rt_norm, "rt_valid": rt_valid,
        })
    return examples


def build_split_datasets_hf(corpus_dir, *, cap=4000, seed=0, accessions=None,
                            rt_lookup=None):
    """{split: SagePropertyDataset} from the HF corpus parquets."""
    if isinstance(rt_lookup, (str, Path)):
        rt_lookup = load_rt_aligned_lookup(rt_lookup)   # load once, reuse per split
    return {sp: SagePropertyDataset(
                prepare_examples_hf(corpus_dir, sp, cap=cap, seed=seed,
                                    accessions=accessions, rt_lookup=rt_lookup))
            for sp in ("train", "val", "test")}
