"""Per-task evaluation metrics."""
from __future__ import annotations

import torch

from peptide_property_ng.losses import intensity_signal_mask, masked_spectral_angle


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


@torch.no_grad()
def evaluate_split(model, loader, device: str = "cpu") -> dict[str, float]:
    """Run the model over a DataLoader and return per-task metrics.

    Returns: intensity spectral angle (similarity, higher better), CCS / RT
    median absolute error, RT Pearson r, charge accuracy.
    """
    model.eval()
    sa_sum = sa_n = 0.0
    ccs_abs: list[torch.Tensor] = []
    rt_pred: list[torch.Tensor] = []
    rt_true: list[torch.Tensor] = []
    charge_correct = charge_n = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)

        if "intensity" in out:
            sa = 1.0 - masked_spectral_angle(out["intensity"], batch["intensity_target"])
            signal = intensity_signal_mask(batch["intensity_target"])
            sa_sum += float(sa[signal].sum())
            sa_n += int(signal.sum())

        if "ccs" in out:
            mean, _ = out["ccs"]
            v = batch["ccs_valid"]
            if v.any():
                ccs_abs.append((mean[v] - batch["ccs_target"][v]).abs().cpu())

        if "rt" in out:
            v = batch["rt_valid"]
            if v.any():
                rt_pred.append(out["rt"][v].cpu())
                rt_true.append(batch["rt_target"][v].cpu())

        if "charge" in out:
            pred = out["charge"].argmax(dim=-1)
            charge_correct += float((pred == batch["charge_target"]).sum())
            charge_n += pred.numel()

    ccs = torch.cat(ccs_abs) if ccs_abs else torch.tensor([])
    rtp = torch.cat(rt_pred) if rt_pred else torch.tensor([])
    rtt = torch.cat(rt_true) if rt_true else torch.tensor([])
    return {
        "intensity_sa": sa_sum / sa_n if sa_n else float("nan"),
        "ccs_mae": float(ccs.mean()) if ccs.numel() else float("nan"),
        "ccs_median_ae": float(ccs.median()) if ccs.numel() else float("nan"),
        "rt_mae": float((rtp - rtt).abs().mean()) if rtp.numel() else float("nan"),
        "rt_pearson": _pearson(rtp, rtt),
        "charge_acc": charge_correct / charge_n if charge_n else float("nan"),
    }
