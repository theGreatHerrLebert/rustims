"""Tests for the hybrid embedding and the Depthcharge-based encoder."""
import torch

from peptide_property_ng.model.config import SMALL
from peptide_property_ng.model.encoder import PeptidePropertyEncoder
from peptide_property_ng.modifications.composition import CompositionTable

# Random token ids in 1..59 — safely valid and never the pad id (61).
_LO, _HI = 1, 60


def _encoder() -> PeptidePropertyEncoder:
    return PeptidePropertyEncoder(SMALL, CompositionTable.load())


def test_forward_shape():
    enc = _encoder()
    b, length = 4, 12
    tokens = torch.randint(_LO, _HI, (b, length))
    latent, mask = enc(tokens)
    assert latent.shape == (b, 1 + length, SMALL.d_model)
    assert mask.shape == (b, 1 + length)
    assert not mask[:, 0].any()  # global token is never masked


def test_padding_mask():
    enc = _encoder()
    tokens = torch.randint(_LO, _HI, (2, 8))
    tokens[0, 5:] = SMALL.pad_token_id  # pad the last 3 residues of sample 0
    _, mask = enc(tokens)
    assert mask[0, 6:].all()       # +1 offset for the global token
    assert not mask[0, :6].any()
    assert not mask[1].any()


def test_instrument_conditioning_changes_output():
    enc = _encoder().eval()
    tokens = torch.randint(_LO, _HI, (3, 10))
    base, _ = enc(tokens)
    cond, _ = enc(tokens, instrument=torch.tensor([2, 2, 2]))
    assert not torch.allclose(base, cond)


def test_long_peptide_runs():
    """A 50-residue peptide must encode fine — no fixed-length cap."""
    enc = _encoder()
    tokens = torch.randint(_LO, _HI, (1, 50))
    latent, _ = enc(tokens)
    assert latent.shape == (1, 51, SMALL.d_model)


def test_unmodified_residue_is_pure_token_embedding():
    """Bias-free composition encoder => a zero-composition token gets no chemistry term."""
    enc = _encoder()
    he = enc.hybrid_embedding
    zero_rows = (~he.composition.bool().any(dim=1)).nonzero().flatten()
    tid = int(zero_rows[zero_rows > 0][0])
    tokens = torch.tensor([[tid]])
    assert torch.allclose(he(tokens), he.token_emb(tokens), atol=1e-6)


def test_modified_residue_differs_from_token_only():
    """A modified-residue token's hybrid embedding includes a chemistry term."""
    enc = _encoder()
    he = enc.hybrid_embedding
    mod_rows = he.composition.bool().any(dim=1).nonzero().flatten()
    tid = int(mod_rows[0])
    tokens = torch.tensor([[tid]])
    assert not torch.allclose(he(tokens), he.token_emb(tokens), atol=1e-6)
