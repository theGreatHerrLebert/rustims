"""Leakage-free dataset splits.

Peptide-level split — a peptide (its *stripped*, modification-free sequence) is
hashed to a fixed bucket, so the same peptide never appears in two splits
regardless of which dataset, charge or run it came from. This is essential:
peptides recur heavily across the deposits.

Also provides the unseen-modification split used by the key generalisation
probe — hold every PSM bearing a chosen modification out of train/val.
"""
from __future__ import annotations

import hashlib
import re

_UNIMOD_RE = re.compile(r"\[UNIMOD:(\d+)\]")


def _bucket(key: str, seed: int) -> float:
    """Stable hash of ``key`` into ``[0, 1)``."""
    digest = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
    return (int(digest[:8], 16) % 1_000_000) / 1_000_000.0


def peptide_split(
    stripped_peptide: str,
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> str:
    """Assign a peptide to ``"train"`` / ``"val"`` / ``"test"`` by stable hash."""
    f = _bucket(stripped_peptide, seed)
    if f < test_frac:
        return "test"
    if f < test_frac + val_frac:
        return "val"
    return "train"


def modifications_in(modified_peptide: str) -> set[int]:
    """UNIMOD ids present in a UNIMOD-format peptide string."""
    return {int(x) for x in _UNIMOD_RE.findall(modified_peptide)}


def has_modification(modified_peptide: str, unimod_id: int) -> bool:
    """True if the peptide carries the given UNIMOD modification."""
    return unimod_id in modifications_in(modified_peptide)
