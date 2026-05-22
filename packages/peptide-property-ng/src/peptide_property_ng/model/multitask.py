"""Unified multi-task peptide property model — one encoder, four heads."""
from __future__ import annotations

import torch
from torch import nn

from peptide_property_ng.model.config import SMALL, ModelConfig
from peptide_property_ng.model.encoder import PeptidePropertyEncoder
from peptide_property_ng.model.heads.ccs import IonMobilityHead
from peptide_property_ng.model.heads.intensity import IntensityHead
from peptide_property_ng.model.heads.scalar import ChargeHead, RetentionTimeHead
from peptide_property_ng.modifications.composition import CompositionTable


class UnifiedPeptidePropertyModel(nn.Module):
    """Shared Depthcharge-based encoder driving all property heads.

    ``forward(batch)`` returns a dict over the active tasks:
      - ``intensity`` : ``(B, L-1, n_ion_channels)`` in ``[0, 1]``
      - ``ccs``       : ``(mean (B,), std (B,))`` — inverse ion mobility, 1/K0
      - ``rt``        : ``(B,)`` — normalised retention time
      - ``charge``    : ``(B, max_charge)`` logits (class j -> charge j+1)

    The input ``batch`` dict carries: ``tokens`` (B, L) long, ``charge`` (B,)
    long, ``precursor_mz`` (B,) float, ``collision_energy`` (B,) float, and
    optionally ``instrument`` / ``acq_mode`` (B,) long.
    """

    HEAD_TYPES: dict[str, type[nn.Module]] = {
        "intensity": IntensityHead,
        "ccs": IonMobilityHead,
        "rt": RetentionTimeHead,
        "charge": ChargeHead,
    }

    def __init__(
        self,
        cfg: ModelConfig = SMALL,
        composition_table: CompositionTable | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        if composition_table is None:
            composition_table = CompositionTable.load()
        self.encoder = PeptidePropertyEncoder(cfg, composition_table)
        self.heads = nn.ModuleDict(
            {t: self.HEAD_TYPES[t](cfg) for t in cfg.tasks}
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        tasks: list[str] | None = None,
    ) -> dict[str, object]:
        active = list(self.heads) if tasks is None else [t for t in tasks if t in self.heads]
        latent, _ = self.encoder(
            batch["tokens"],
            instrument=batch.get("instrument"),
            acq_mode=batch.get("acq_mode"),
        )
        out: dict[str, object] = {}
        if "intensity" in active:
            out["intensity"] = self.heads["intensity"](
                latent, batch["charge"], batch["collision_energy"]
            )
        if "ccs" in active:
            out["ccs"] = self.heads["ccs"](latent, batch["charge"], batch["precursor_mz"])
        if "rt" in active:
            out["rt"] = self.heads["rt"](latent)
        if "charge" in active:
            out["charge"] = self.heads["charge"](latent)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
