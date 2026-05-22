"""Model configuration — presets, instrument / acquisition vocabularies.

Conditioning policy (deliberate, see the charge-leakage note):
  - Instrument model and acquisition mode are conditioned *inside* the encoder
    via the prepended global token — they cannot leak any task label.
  - Charge, precursor m/z and collision energy are conditioned at the *heads*
    that legitimately use them (intensity, CCS). They are deliberately kept out
    of the encoder so the charge head — which reads the shared encoder output —
    never sees its own label. One encoder pass, no leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# timsTOF-focused instrument vocabulary (from timstof_catalog.tsv instrument_model).
INSTRUMENTS: tuple[str, ...] = (
    "unknown",
    "timsTOF",
    "timsTOF Pro",
    "timsTOF Pro 2",
    "timsTOF HT",
    "timsTOF SCP",
    "timsTOF Ultra",
    "timsTOF fleX",
)
ACQUISITION_MODES: tuple[str, ...] = (
    "unknown",
    "DDA",
    "DDA-PASEF",
    "DIA",
    "diaPASEF",
    "PRM",
    "MALDI",
)
INSTRUMENT_TO_ID: dict[str, int] = {n: i for i, n in enumerate(INSTRUMENTS)}
ACQ_TO_ID: dict[str, int] = {n: i for i, n in enumerate(ACQUISITION_MODES)}


def instrument_id(name: str | None) -> int:
    """Map an instrument-model string to an id, tolerating unknowns/aliases."""
    if not name:
        return 0
    n = " ".join(str(name).split())  # normalise whitespace
    if n in INSTRUMENT_TO_ID:
        return INSTRUMENT_TO_ID[n]
    low = n.lower()
    for cand, idx in INSTRUMENT_TO_ID.items():
        if cand.lower() == low:
            return idx
    return 0  # "unknown"


def acquisition_id(name: str | None) -> int:
    """Map an acquisition-mode string to an id, tolerating unknowns."""
    if not name:
        return 0
    n = str(name).strip()
    for cand, idx in ACQ_TO_ID.items():
        if cand.lower() == n.lower():
            return idx
    return 0


@dataclass
class ModelConfig:
    """Architecture + vocabulary configuration for the unified predictor."""

    # encoder
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 64  # peptide-length cap — far above the old 174-vector's 30

    # tokenizer / modification vocabulary
    vocab_size: int = 2175
    pad_token_id: int = 61
    n_elements: int = 31  # atomic-composition channels (see modifications.composition)

    # hybrid embedding
    comp_fusion: str = "add"  # "add" | "gate"

    # conditioning
    max_charge: int = 8
    n_instruments: int = len(INSTRUMENTS)
    n_acq_modes: int = len(ACQUISITION_MODES)

    # heads
    n_ion_channels: int = 6  # b/y ions x fragment charges 1-3 (Prosit-compatible)
    tasks: tuple[str, ...] = ("intensity", "ccs", "rt", "charge")

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")


# First-prototype preset — roughly on par with the production BASE encoder so
# iteration is fast on the ~19 datasets available now.
SMALL = ModelConfig(d_model=256, n_layers=6, n_heads=8, dim_feedforward=1024)

# Research preset — larger; for the full ~138-dataset corpus.
RESEARCH = ModelConfig(d_model=384, n_layers=8, n_heads=8, dim_feedforward=1536)

PRESETS: dict[str, ModelConfig] = {"small": SMALL, "research": RESEARCH}
