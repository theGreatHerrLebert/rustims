"""Build the UNIMOD atomic-composition lookup table for the hybrid encoding.

Maps every ProformaTokenizer vocabulary token to a signed element-count vector
and writes ``data/mod_composition_table.npz``:
  - ``counts``   (vocab_size, n_elements) int16 — signed element deltas
  - ``elements`` (n_elements,) str          — element order (== composition.ELEMENTS)
  - ``tokens``   (vocab_size,) str          — id-indexed token strings (provenance)

Run once; re-run when UNIMOD or the tokenizer vocabulary changes — no model
vocab re-mint is needed, only a fresh table:

    python -m peptide_property_ng.modifications.build_table
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from peptide_property_ng.modifications.composition import DATA_DIR, DEFAULT_TABLE, ELEMENTS

# Matches every UNIMOD reference inside a token, e.g. "C[UNIMOD:4]" -> "4".
_UNIMOD_RE = re.compile(r"\[UNIMOD:(\d+)\]")


def build(out_path: Path = DEFAULT_TABLE) -> dict:
    """Build the composition table and write it to ``out_path``. Returns a summary."""
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer
    from sagepy.core.unimod import modification_atomic_composition

    tok = ProformaTokenizer.with_defaults()
    vocab = list(tok.get_vocab())             # id-indexed list of token strings
    comp = modification_atomic_composition()  # {'[UNIMOD:N]': {element: count}}

    # Every element sagepy reports must be in our canonical alphabet.
    seen = {e for c in comp.values() for e in c}
    unknown = seen - set(ELEMENTS)
    if unknown:
        raise RuntimeError(
            f"UNIMOD composition uses elements absent from ELEMENTS: {sorted(unknown)} "
            "— add them to composition.ELEMENTS and rebuild."
        )

    elem_index = {e: i for i, e in enumerate(ELEMENTS)}
    counts = np.zeros((len(vocab), len(ELEMENTS)), dtype=np.int16)

    n_mod_tokens = n_resolved = 0
    unresolved: set[str] = set()
    for tid, token in enumerate(vocab):
        ids = _UNIMOD_RE.findall(token)
        if not ids:
            continue  # bare residue or special token -> zero delta
        n_mod_tokens += 1
        resolved_any = False
        for uid in ids:  # sum all mods on the token (terminal + residue, etc.)
            key = f"[UNIMOD:{uid}]"
            mod = comp.get(key)
            if mod is None:
                unresolved.add(key)
                continue
            resolved_any = True
            for elem, c in mod.items():
                counts[tid, elem_index[elem]] += c
        n_resolved += int(resolved_any)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        counts=counts,
        elements=np.array(ELEMENTS),
        tokens=np.array(vocab),
    )

    summary = {
        "vocab_size": len(vocab),
        "mod_tokens": n_mod_tokens,
        "resolved": n_resolved,
        "unresolved": len(unresolved),
        "n_elements": len(ELEMENTS),
        "nonzero_rows": int((counts != 0).any(axis=1).sum()),
        "path": str(out_path),
    }
    return summary


if __name__ == "__main__":
    s = build()
    print(f"vocab size          : {s['vocab_size']}")
    print(f"modification tokens : {s['mod_tokens']}")
    print(f"  with composition  : {s['resolved']}")
    print(f"  unresolved UNIMOD : {s['unresolved']}  (id absent from sagepy table)")
    print(f"elements            : {s['n_elements']}")
    print(f"nonzero rows        : {s['nonzero_rows']}")
    print(f"written             : {s['path']}")
