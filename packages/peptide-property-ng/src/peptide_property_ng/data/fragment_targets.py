"""Fragment-indexed intensity targets from Sage ``matched_fragments``.

Each peptide of length ``L`` yields an ``(L-1, 6)`` target — one row per
cleavage site, six channels for b/y ions at fragment charges 1-3. This is the
variable-length successor to Prosit's fixed 174-vector (no 30-residue cap).

Convention (Prosit / imspy compatible):
  -1   physically impossible channel (fragment charge > precursor charge)
   0   possible but unobserved fragment
  0..1 observed, base-peak-normalised intensity
The masked spectral-angle loss ignores -1 entries.
"""
from __future__ import annotations

import numpy as np

# Channel order: y ions at charge 1/2/3, then b ions at charge 1/2/3.
ION_CHANNELS: tuple[str, ...] = ("y+1", "y+2", "y+3", "b+1", "b+2", "b+3")
N_ION_CHANNELS = len(ION_CHANNELS)
MAX_FRAGMENT_CHARGE = 3


def build_intensity_target(
    peptide_len: int,
    precursor_charge: int,
    frag_type: np.ndarray,       # ('b'|'y') per matched fragment
    frag_ordinal: np.ndarray,    # int
    frag_charge: np.ndarray,     # int
    frag_intensity: np.ndarray,  # float
) -> np.ndarray:
    """Return the ``(peptide_len - 1, 6)`` intensity target for one PSM."""
    n_sites = peptide_len - 1
    target = np.zeros((n_sites, N_ION_CHANNELS), dtype=np.float32)

    # Mark fragment charges above the precursor charge as impossible.
    for fc in range(1, MAX_FRAGMENT_CHARGE + 1):
        if fc > precursor_charge:
            target[:, fc - 1] = -1.0          # y channel
            target[:, 3 + fc - 1] = -1.0      # b channel

    for ftype, ordi, fch, inten in zip(
        frag_type, frag_ordinal, frag_charge, frag_intensity
    ):
        fch = int(fch)
        if fch < 1 or fch > MAX_FRAGMENT_CHARGE:
            continue
        ordi = int(ordi)
        if ftype == "b":
            site, channel = ordi - 1, 3 + fch - 1
        elif ftype == "y":
            # cleavage site i produces y_{n_sites - i + ...}: y_o sits at site n_sites - o
            site, channel = n_sites - ordi, fch - 1
        else:
            continue
        if 0 <= site < n_sites and target[site, channel] >= 0.0:
            target[site, channel] = max(target[site, channel], float(inten))

    # Base-peak normalisation over the observed (positive) entries.
    observed = target > 0.0
    if observed.any():
        peak = float(target[observed].max())
        if peak > 0.0:
            target[observed] /= peak
    return target
