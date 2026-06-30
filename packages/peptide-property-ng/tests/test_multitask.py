"""Tests for the unified multi-task model — output shapes and the leakage guard."""
import torch

from peptide_property_ng.model.config import SMALL
from peptide_property_ng.model.multitask import UnifiedPeptidePropertyModel


def _batch(b: int = 5, length: int = 14) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.randint(1, 60, (b, length)),
        "charge": torch.randint(1, 5, (b,)),
        "precursor_mz": torch.rand(b) * 800 + 300,
        "collision_energy": torch.zeros(b),
        "instrument": torch.randint(0, SMALL.n_instruments, (b,)),
        "acq_mode": torch.randint(0, SMALL.n_acq_modes, (b,)),
    }


def test_forward_all_heads():
    model = UnifiedPeptidePropertyModel(SMALL).eval()
    b, length = 5, 14
    out = model(_batch(b, length))

    assert set(out) == {"intensity", "ccs", "rt", "charge"}
    assert out["intensity"].shape == (b, length - 1, SMALL.n_ion_channels)
    assert (out["intensity"] >= 0).all() and (out["intensity"] <= 1).all()

    mean, std = out["ccs"]
    assert mean.shape == (b,) and std.shape == (b,)
    assert (std > 0).all()

    assert out["rt"].shape == (b,)
    assert out["charge"].shape == (b, SMALL.max_charge)


def test_task_subset():
    model = UnifiedPeptidePropertyModel(SMALL).eval()
    out = model(_batch(), tasks=["intensity"])
    assert set(out) == {"intensity"}


def test_charge_head_does_not_see_charge():
    """The charge head must be invariant to the precursor charge input
    (charge is conditioned only at the intensity/CCS heads, never the encoder)."""
    model = UnifiedPeptidePropertyModel(SMALL).eval()
    batch = _batch()
    with torch.no_grad():
        logits_a = model(batch, tasks=["charge"])["charge"]
        batch2 = dict(batch, charge=torch.full_like(batch["charge"], 5))
        logits_b = model(batch2, tasks=["charge"])["charge"]
    assert torch.allclose(logits_a, logits_b), "charge leaked into the charge head"


def test_backward_runs():
    model = UnifiedPeptidePropertyModel(SMALL)
    out = model(_batch())
    loss = (
        out["intensity"].mean()
        + out["ccs"][0].mean()
        + out["rt"].mean()
        + out["charge"].mean()
    )
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_parameter_count_reported():
    model = UnifiedPeptidePropertyModel(SMALL)
    assert model.num_parameters() > 1_000_000
