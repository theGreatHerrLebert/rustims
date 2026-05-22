"""Hybrid residue embedding — learnable token embedding + atomic-composition.

The central design decision: every residue is represented by BOTH
  (a) a learnable per-token embedding (great for common modifications, which
      have plenty of training data), and
  (b) a chemistry feature derived from the modification's atomic composition
      (lets rare / unseen modifications generalise — they inherit meaning from
      chemically similar mods, and the scheme is open-vocab).

A bare residue or special token has an all-zero composition delta; with the
bias-free CompositionEncoder this maps to an exact zero contribution, so
unmodified residues fall back to the pure token embedding.
"""
from __future__ import annotations

import torch
from torch import nn

from peptide_property_ng.modifications.composition import CompositionTable


class CompositionEncoder(nn.Module):
    """Maps a signed atomic-composition vector to a d_model chemistry embedding.

    Bias-free so that an all-zero composition (a bare residue) maps to an exact
    zero vector — the additive-fusion semantics then mean "unmodified residue =
    pure token embedding".
    """

    def __init__(self, n_elements: int, d_model: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_model // 2
        self.net = nn.Sequential(
            nn.Linear(n_elements, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
        )

    def forward(self, comp: torch.Tensor) -> torch.Tensor:  # (..., n_elements) -> (..., d_model)
        return self.net(comp)


class HybridResidueEmbedding(nn.Module):
    """Token embedding + composition embedding, fused per residue.

    Parameters
    ----------
    vocab_size, d_model, pad_token_id
        Tokenizer / model dimensions.
    composition_table
        Per-token atomic-composition deltas (see modifications.composition).
    fusion
        ``"add"`` — ``token + composition`` (default).
        ``"gate"`` — ``token + sigmoid(g) * composition`` with a learnable
        per-channel gate ``g`` (init 0 → gate 0.5); the recommended upgrade if
        the unseen-modification probe shows the composition branch is ignored.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        composition_table: CompositionTable,
        pad_token_id: int,
        fusion: str = "add",
    ):
        super().__init__()
        if fusion not in ("add", "gate"):
            raise ValueError(f"unknown fusion '{fusion}'")
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.fusion = fusion

        # +1 row to match depthcharge's nn.Embedding(n_tokens + 1, ...) convention.
        self.token_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_token_id)
        self.comp_encoder = CompositionEncoder(composition_table.n_elements, d_model)

        # Composition lookup as a frozen buffer, indexed by token id; signed-log1p
        # normalised. One extra zero row to align with token_emb's vocab_size + 1.
        comp = composition_table.as_tensor(normalize=True)  # (vocab_size, n_elements)
        comp = torch.cat([comp, torch.zeros(1, comp.shape[1])], dim=0)
        self.register_buffer("composition", comp, persistent=False)

        if fusion == "gate":
            self.gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, seq_len) long  ->  (batch, seq_len, d_model)."""
        token_vec = self.token_emb(tokens)
        comp_vec = self.comp_encoder(self.composition[tokens])
        if self.fusion == "gate":
            comp_vec = torch.sigmoid(self.gate) * comp_vec
        return token_vec + comp_vec
