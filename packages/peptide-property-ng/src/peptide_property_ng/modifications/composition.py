"""UNIMOD atomic-composition features for the hybrid modification encoding.

Each vocabulary token gets a fixed vector of element-count *deltas* relative to
the bare residue: a modified-residue token (e.g. ``C[UNIMOD:4]``) carries the
modification's atomic composition; a bare residue or a special token carries
zeros. Deltas are *signed* — UNIMOD neutral-loss modifications have negative
counts.

Source: ``sagepy.core.unimod.modification_atomic_composition()`` — the same
UNIMOD view the Sage search engine uses, so it is consistent with the training
labels produced by Sage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Canonical element order. Verified against sagepy's UNIMOD composition tables
# (31 elements incl. stable isotopes and metals); ``build_table.py`` asserts the
# live data is a subset of this set and fails loudly if UNIMOD ever adds one.
ELEMENTS: tuple[str, ...] = (
    "C", "H", "N", "O", "S", "P", "Se",             # common organic
    "2H", "13C", "15N", "18O",                       # stable isotopes (labels)
    "Ag", "Al", "As", "B", "Br", "Ca", "Cl", "Cu",   # metals / halogens / ...
    "F", "Fe", "Hg", "I", "K", "Li", "Mg", "Mo",
    "Na", "Ni", "Si", "Zn",
)
N_ELEMENTS = len(ELEMENTS)

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_TABLE = DATA_DIR / "mod_composition_table.npz"


def signed_log1p(x):
    """Sign-preserving log compression: ``sign(x) * log1p(|x|)``.

    Plain ``log1p`` is invalid for the ~430 UNIMOD modifications with negative
    (neutral-loss) atom counts; this keeps the sign and compresses magnitude so
    a 120-atom glycan does not dwarf a single-atom methylation.
    """
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.log1p(np.abs(x))


class CompositionTable:
    """Per-token atomic-composition delta vectors, indexed by tokenizer id."""

    def __init__(self, counts: np.ndarray, elements: tuple[str, ...]):
        if counts.ndim != 2 or counts.shape[1] != len(elements):
            raise ValueError(
                f"counts shape {counts.shape} inconsistent with {len(elements)} elements"
            )
        self.counts = counts.astype(np.float32)  # (vocab_size, n_elements), signed
        self.elements = tuple(elements)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TABLE) -> "CompositionTable":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"composition table not found at {path}; run "
                "`python -m peptide_property_ng.modifications.build_table` first."
            )
        with np.load(path, allow_pickle=False) as z:
            return cls(z["counts"], tuple(str(e) for e in z["elements"]))

    @property
    def vocab_size(self) -> int:
        return self.counts.shape[0]

    @property
    def n_elements(self) -> int:
        return self.counts.shape[1]

    def as_tensor(self, normalize: bool = True) -> torch.Tensor:
        """Return the ``(vocab_size, n_elements)`` table as a float tensor.

        With ``normalize`` (default) the signed-log1p compression is applied —
        this is the form the model's ``CompositionEncoder`` consumes.
        """
        t = torch.from_numpy(self.counts).float()
        return signed_log1p(t) if normalize else t
