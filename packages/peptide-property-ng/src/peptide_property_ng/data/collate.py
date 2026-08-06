"""Variable-length collation of prepared examples into model batches."""
from __future__ import annotations

import numpy as np
import torch

from peptide_property_ng.data.fragment_targets import N_ION_CHANNELS


def collate_examples(
    examples: list[dict],
    *,
    pad_token_id: int = 61,
    max_charge: int = 8,
) -> dict[str, torch.Tensor]:
    """Collate a list of example dicts into a padded batch.

    Token sequences pad to the batch-max length with ``pad_token_id``; intensity
    targets pad to ``(batch_max_len - 1, n_ion_channels)`` with ``-1`` (the
    masked-out marker the spectral-angle loss ignores).
    """
    batch_size = len(examples)
    max_len = max(len(e["tokens"]) for e in examples)

    tokens = np.full((batch_size, max_len), pad_token_id, dtype=np.int64)
    intensity = np.full((batch_size, max_len - 1, N_ION_CHANNELS), -1.0, dtype=np.float32)
    for i, e in enumerate(examples):
        length = len(e["tokens"])
        tokens[i, :length] = e["tokens"]
        tgt = e["intensity_target"]
        intensity[i, : tgt.shape[0], :] = tgt

    def _col(key: str, dtype) -> torch.Tensor:
        return torch.tensor([e[key] for e in examples], dtype=dtype)

    charge = _col("charge", torch.long)
    return {
        "tokens": torch.from_numpy(tokens),
        "charge": charge,
        "precursor_mz": _col("precursor_mz", torch.float32),
        "collision_energy": _col("collision_energy", torch.float32),
        "instrument": _col("instrument", torch.long),
        "acq_mode": _col("acq_mode", torch.long),
        # targets
        "intensity_target": torch.from_numpy(intensity),
        "ccs_target": _col("ccs_target", torch.float32),
        "ccs_valid": _col("ccs_valid", torch.bool),
        "rt_target": _col("rt_target", torch.float32),
        "rt_valid": _col("rt_valid", torch.bool),
        # charge head target: class j == charge j+1
        "charge_target": charge.clamp(1, max_charge) - 1,
    }


def make_collate_fn(pad_token_id: int = 61, max_charge: int = 8):
    """Return a ``collate_fn`` bound to the given padding / charge settings."""

    def _fn(examples: list[dict]) -> dict[str, torch.Tensor]:
        return collate_examples(examples, pad_token_id=pad_token_id, max_charge=max_charge)

    return _fn
