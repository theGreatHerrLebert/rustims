"""Fragment-intensity targets.

The MS2 intensity target the model trains against is **cleavage-site-indexed**:
``(L-1, 6)`` — one row per cleavage site, channels ``[y+1, y+2, y+3, b+1, b+2, b+3]``.

Both data sources reach it through the *same single conversion*,
``prosit174_to_sites``:
  - Public Wilhelmlab HF datasets store intensities natively as the Prosit
    174-vector.
  - Sage search results are turned into that 174-vector by the proven
    ``imspy_predictors`` encoder ``observed_fragments_to_intensity_target``
    (see ``data/sage_dataset.py``) — the fragment→vector encoding is *not*
    re-implemented here.

So there is exactly one place where the ordinal→site remap happens, and it is
this file.

Convention (Prosit / imspy compatible): ``-1`` = impossible/masked channel,
``0`` = possible but unobserved, ``0..1`` = observed (base-peak-normalised).
"""
from __future__ import annotations

import numpy as np

# Channel order — verified against imspy_predictors/intensity/utility.py:211
# ("Ion types in order: y+1, y+2, y+3, b+1, b+2, b+3").
ION_CHANNELS: tuple[str, ...] = ("y+1", "y+2", "y+3", "b+1", "b+2", "b+3")
N_ION_CHANNELS = len(ION_CHANNELS)

# The Prosit 174-vector covers fragment ordinals 1..29 (peptides up to 30 aa).
PROSIT_MAX_ORDINAL = 29
PROSIT_VECTOR_LEN = PROSIT_MAX_ORDINAL * N_ION_CHANNELS  # 174


def prosit174_to_sites(intensities_174, peptide_len: int) -> np.ndarray:
    """Convert a Prosit 174-element intensity vector to our site-indexed target.

    The Prosit vector is **ordinal-indexed**: reshaped to (29, 6), row ``r`` holds
    the ions of fragment ordinal ``r+1`` — ``y_{r+1}`` in columns 0:3, ``b_{r+1}``
    in columns 3:6.

    Our ``(L-1, 6)`` target is **cleavage-site-indexed**: site ``i`` holds the two
    ions produced by cleaving bond ``i`` — the b ion ``b_{i+1}`` and its
    *complementary* y ion ``y_{L-1-i}``. The b and y of one cleavage have
    different ordinals (``b_k`` pairs with ``y_{L-k}``), so this is a genuine
    remap, not a reshape.

    Mapping (``L`` = peptide_len, ``n_sites`` = L-1):
        site i, b channels (3:6)  <-  ordinal i+1    ->  Prosit row i
        site i, y channels (0:3)  <-  ordinal L-1-i  ->  Prosit row (L-1-i)-1 = L-2-i

    The ``-1`` masked-channel markers in the Prosit vector carry through.
    """
    arr = np.asarray(intensities_174, dtype=np.float32).reshape(-1)
    if arr.size != PROSIT_VECTOR_LEN:
        raise ValueError(f"expected {PROSIT_VECTOR_LEN} intensity values, got {arr.size}")
    prosit = arr.reshape(PROSIT_MAX_ORDINAL, N_ION_CHANNELS)  # row r -> fragment ordinal r+1

    n_sites = peptide_len - 1
    if not 1 <= n_sites <= PROSIT_MAX_ORDINAL:
        raise ValueError(f"peptide_len {peptide_len} outside the 174-vector range [2, 30]")

    target = np.full((n_sites, N_ION_CHANNELS), -1.0, dtype=np.float32)
    for i in range(n_sites):
        target[i, 3:6] = prosit[i, 3:6]                  # b_{i+1}
        target[i, 0:3] = prosit[n_sites - 1 - i, 0:3]    # y_{L-1-i}  (Prosit row L-2-i)
    return target
