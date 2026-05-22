"""Depthcharge-based shared encoder for the unified peptide property predictor."""
from __future__ import annotations

import torch
from depthcharge.transformers import AnalyteTransformerEncoder
from torch import nn

from peptide_property_ng.model.config import ModelConfig
from peptide_property_ng.model.embedding import HybridResidueEmbedding
from peptide_property_ng.modifications.composition import CompositionTable


class PeptidePropertyEncoder(AnalyteTransformerEncoder):
    """Shared transformer encoder.

    Built on Depthcharge's ``AnalyteTransformerEncoder``, but:
      - the plain token embedding is replaced with the hybrid (token +
        atomic-composition) residue embedding, and
      - conditioning on instrument model + acquisition mode happens via the
        prepended global token. Charge / precursor m-z / collision energy are
        deliberately *not* fed here — they are conditioned at the heads that use
        them, so the charge head (which reads this shared output) never sees its
        own label. One encoder pass, no label leakage.

    ``forward(tokens, instrument, acq_mode) -> (latent, padding_mask)``
      - ``latent``       ``(batch, 1 + seq_len, d_model)`` — index 0 is the global token
      - ``padding_mask`` ``(batch, 1 + seq_len)`` bool — ``True`` where padded
    """

    def __init__(self, cfg: ModelConfig, composition_table: CompositionTable):
        super().__init__(
            n_tokens=cfg.vocab_size,
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            positional_encoder=True,
            padding_int=cfg.pad_token_id,
        )
        self.cfg = cfg

        # The base class built a plain nn.Embedding token encoder; drop it and
        # use the hybrid residue embedding instead.
        del self.token_encoder
        self.hybrid_embedding = HybridResidueEmbedding(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            composition_table=composition_table,
            pad_token_id=cfg.pad_token_id,
            fusion=cfg.comp_fusion,
        )

        # Leak-free conditioning (instrument + acquisition mode only).
        self.instrument_emb = nn.Embedding(cfg.n_instruments, cfg.d_model)
        self.acq_emb = nn.Embedding(cfg.n_acq_modes, cfg.d_model)

    def global_token_hook(  # noqa: D102 (documented on the class)
        self,
        tokens: torch.Tensor,
        *,
        instrument: torch.Tensor | None = None,
        acq_mode: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        g = torch.zeros(
            tokens.shape[0],
            self.d_model,
            device=tokens.device,
            dtype=self.instrument_emb.weight.dtype,
        )
        if instrument is not None:
            g = g + self.instrument_emb(instrument)
        if acq_mode is not None:
            g = g + self.acq_emb(acq_mode)
        return g

    def forward(  # type: ignore[override]
        self,
        tokens: torch.Tensor,
        *,
        instrument: torch.Tensor | None = None,
        acq_mode: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.hybrid_embedding(tokens)  # (B, L, d)
        global_token = self.global_token_hook(
            tokens, instrument=instrument, acq_mode=acq_mode
        )  # (B, d)
        encoded = torch.cat([global_token[:, None, :], encoded], dim=1)  # (B, 1+L, d)

        # Explicit padding mask from the token ids — robust, and the global
        # token at position 0 is never padding.
        pad = tokens == self.cfg.pad_token_id  # (B, L)
        padding_mask = torch.cat(
            [
                torch.zeros(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device),
                pad,
            ],
            dim=1,
        )  # (B, 1+L)

        encoded = self.positional_encoder(encoded)
        latent = self.transformer_encoder(
            encoded, mask=attn_mask, src_key_padding_mask=padding_mask
        )
        return latent, padding_mask
