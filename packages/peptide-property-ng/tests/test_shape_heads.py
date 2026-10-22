"""Tests for the timsim peak-shape supervision: RT-sigma head + IM-sigma loss.

Covers the additive wiring: a new ``rt_sigma`` head, and an ``im_sigma`` loss
term riding the existing ``ccs`` head's ``std``. Both must be optional — absent
targets ⇒ no term, existing training unchanged.
"""
import dataclasses

import torch

from peptide_property_ng.model.config import SMALL
from peptide_property_ng.model.multitask import UnifiedPeptidePropertyModel
from peptide_property_ng.losses import MultiTaskLoss


def _cfg_with_shape():
    return dataclasses.replace(SMALL, tasks=tuple(SMALL.tasks) + ("rt_sigma",))


def _batch(b: int = 6, length: int = 14) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.randint(1, 60, (b, length)),
        "charge": torch.randint(1, 5, (b,)),
        "precursor_mz": torch.rand(b) * 800 + 300,
        "collision_energy": torch.zeros(b),
        "instrument": torch.randint(0, SMALL.n_instruments, (b,)),
        "acq_mode": torch.randint(0, SMALL.n_acq_modes, (b,)),
    }


def test_rt_sigma_head_emits_positive_scalar():
    model = UnifiedPeptidePropertyModel(_cfg_with_shape()).eval()
    out = model(_batch(6))
    assert "rt_sigma" in out
    assert out["rt_sigma"].shape == (6,)
    assert (out["rt_sigma"] > 0).all()  # softplus keeps sigma strictly positive


def test_shape_loss_terms_present_and_masked():
    b = 6
    model = UnifiedPeptidePropertyModel(_cfg_with_shape())
    out = model(_batch(b))
    # only a subset of peptides carry a measured shape label
    rt_valid = torch.tensor([True, True, False, True, False, False])
    im_valid = torch.tensor([True, False, True, False, False, True])
    batch = {
        "rt_sigma_target": torch.rand(b),
        "rt_sigma_valid": rt_valid,
        "im_sigma_target": torch.rand(b) * 0.02,
        "im_sigma_valid": im_valid,
        # CCS-mean term inactive here (no ccs_valid set true)
        "ccs_valid": torch.zeros(b, dtype=torch.bool),
        "rt_valid": torch.zeros(b, dtype=torch.bool),
        "charge_target": torch.randint(0, SMALL.max_charge, (b,)),
        "intensity_target": torch.zeros(b, 13, SMALL.n_ion_channels),
    }
    total, logd = MultiTaskLoss()(out, batch)
    assert "rt_sigma" in logd and "im_sigma" in logd
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_shape_terms_absent_when_no_target():
    """No shape targets in the batch ⇒ no shape loss terms (backward-compatible)."""
    b = 5
    model = UnifiedPeptidePropertyModel(SMALL)  # default tasks, no rt_sigma head
    out = model(_batch(b))
    batch = {
        "ccs_valid": torch.ones(b, dtype=torch.bool),
        "ccs_target": torch.rand(b),
        "rt_valid": torch.ones(b, dtype=torch.bool),
        "rt_target": torch.rand(b),
        "charge_target": torch.randint(0, SMALL.max_charge, (b,)),
        "intensity_target": torch.zeros(b, 13, SMALL.n_ion_channels),
    }
    _total, logd = MultiTaskLoss()(out, batch)
    assert "rt_sigma" not in logd and "im_sigma" not in logd
