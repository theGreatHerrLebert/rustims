"""
Task-specific prediction heads for the unified peptide transformer.

Each head takes the encoder output and produces task-specific predictions:
- CCSHead: Collision Cross Section with uncertainty estimation
- RTHead: Indexed Retention Time (iRT)
- ChargeHead: Charge state distribution
- IntensityHead: Fragment ion intensities

All heads support optional instrument type encoding for instrument-specific
predictions (different mass spectrometers have different characteristics).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Supported instrument types
INSTRUMENT_TYPES = [
    "unknown",          # 0 - fallback/unknown
    "timstof",          # 1 - Bruker timsTOF
    "timstof_pro",      # 2 - Bruker timsTOF Pro
    "timstof_flex",     # 3 - Bruker timsTOF fleX
    "timstof_scp",      # 4 - Bruker timsTOF SCP
    "timstof_ht",       # 5 - Bruker timsTOF HT
    "orbitrap",         # 6 - Thermo Orbitrap
    "orbitrap_fusion",  # 7 - Thermo Orbitrap Fusion
    "orbitrap_eclipse", # 8 - Thermo Orbitrap Eclipse
    "exploris",         # 9 - Thermo Exploris
    "astral",           # 10 - Thermo Astral
    "qtof",             # 11 - Generic Q-TOF
    "tof",              # 12 - Generic TOF
    "triple_quad",      # 13 - Triple quadrupole
    "ion_trap",         # 14 - Ion trap
]

NUM_INSTRUMENT_TYPES = len(INSTRUMENT_TYPES)
INSTRUMENT_TO_ID = {name: i for i, name in enumerate(INSTRUMENT_TYPES)}


def get_instrument_id(instrument: str) -> int:
    """Convert instrument name to ID."""
    return INSTRUMENT_TO_ID.get(instrument.lower().replace(" ", "_").replace("-", "_"), 0)


class SquareRootProjectionLayer(nn.Module):
    """
    Physics-based CCS initialization layer.

    This layer implements the relationship between m/z, charge, and CCS
    using a square root projection. The CCS of an ion is approximately
    proportional to sqrt(m/z) scaled by charge-dependent coefficients.

    This provides a strong physics-informed prior that improves training
    stability and accuracy for CCS prediction.

    Args:
        max_charge: Maximum charge state to support (default: 6)
        init_slopes: Initial slope values per charge state
        init_intercepts: Initial intercept values per charge state
        trainable: Whether to train the slopes/intercepts (default: True)

    Reference:
        Based on the relationship: CCS ≈ slope * sqrt(m/z) + intercept
        where coefficients are charge-dependent.
    """

    # Default parameters derived from empirical CCS data
    DEFAULT_SLOPES = [0.0, 8.71, 12.96, 15.61, 17.20, 19.78, 21.28]  # charge 0-6
    DEFAULT_INTERCEPTS = [0.0, 173.4, 134.5, 125.9, 141.1, 91.4, 91.4]

    def __init__(
        self,
        max_charge: int = 6,
        init_slopes: Optional[list] = None,
        init_intercepts: Optional[list] = None,
        trainable: bool = True,
    ):
        super().__init__()

        self.max_charge = max_charge

        # Use default or provided initialization
        slopes = init_slopes if init_slopes is not None else self.DEFAULT_SLOPES
        intercepts = init_intercepts if init_intercepts is not None else self.DEFAULT_INTERCEPTS

        # Ensure we have enough values for max_charge + 1 (including charge 0)
        slopes = list(slopes) + [slopes[-1]] * (max_charge + 1 - len(slopes))
        intercepts = list(intercepts) + [intercepts[-1]] * (max_charge + 1 - len(intercepts))

        # Register as parameters or buffers
        if trainable:
            self.slopes = nn.Parameter(torch.tensor(slopes[:max_charge + 1], dtype=torch.float32))
            self.intercepts = nn.Parameter(torch.tensor(intercepts[:max_charge + 1], dtype=torch.float32))
        else:
            self.register_buffer("slopes", torch.tensor(slopes[:max_charge + 1], dtype=torch.float32))
            self.register_buffer("intercepts", torch.tensor(intercepts[:max_charge + 1], dtype=torch.float32))

    def forward(
        self,
        mz: torch.Tensor,
        charge: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute physics-based CCS estimate.

        Args:
            mz: Precursor m/z values of shape (batch,) or (batch, 1)
            charge: Charge states of shape (batch,) - integer values

        Returns:
            CCS estimate of shape (batch, 1)
        """
        # Ensure proper shapes
        mz = mz.view(-1, 1)
        charge = charge.view(-1).long()

        # Clamp charge to valid range
        charge = charge.clamp(0, self.max_charge)

        # Get charge-specific coefficients
        slope = self.slopes[charge].view(-1, 1)
        intercept = self.intercepts[charge].view(-1, 1)

        # Compute CCS: slope * sqrt(m/z) + intercept
        sqrt_mz = torch.sqrt(mz)
        ccs = slope * sqrt_mz + intercept

        return ccs


class CCSHead(nn.Module):
    """
    CCS prediction head with uncertainty estimation.

    Combines physics-based initialization (SquareRootProjectionLayer)
    with learned residual correction. Outputs both mean CCS and
    uncertainty (standard deviation).

    Supports optional instrument type encoding for instrument-specific
    predictions. Different instruments may have systematic offsets in
    CCS measurements.

    Args:
        d_model: Encoder embedding dimension
        max_charge: Maximum charge state (default: 6)
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout probability (default: 0.1)
        use_instrument: Whether to use instrument type encoding (default: True)

    Returns:
        Tuple of (ccs_mean, ccs_std) with shapes (batch, 1) each
    """

    def __init__(
        self,
        d_model: int,
        max_charge: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_instrument: bool = True,
    ):
        super().__init__()

        self.max_charge = max_charge
        self.use_instrument = use_instrument

        # Physics-based projection
        self.sqrt_proj = SquareRootProjectionLayer(max_charge=max_charge)

        # Instrument embedding
        if use_instrument:
            self.instrument_embed = nn.Embedding(NUM_INSTRUMENT_TYPES, d_model // 4)
            input_dim = d_model + max_charge + 1 + d_model // 4
        else:
            self.instrument_embed = None
            input_dim = d_model + max_charge + 1

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # mean_correction, log_std
        )

    def forward(
        self,
        encoder_out: torch.Tensor,
        mz: torch.Tensor,
        charge: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        instrument: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict CCS with uncertainty.

        Args:
            encoder_out: Encoder output of shape (batch, seq_len, d_model)
            mz: Precursor m/z values of shape (batch,)
            charge: Charge states of shape (batch,) - integer values
            padding_mask: Optional padding mask (unused, for API consistency)
            instrument: Optional instrument type IDs of shape (batch,)
                       Use get_instrument_id() to convert names to IDs.

        Returns:
            Tuple of:
                - ccs_mean: Predicted CCS values of shape (batch, 1)
                - ccs_std: Predicted uncertainty of shape (batch, 1)
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device

        # Get [CLS] token representation
        cls_repr = encoder_out[:, 0, :]  # (batch, d_model)

        # Create charge one-hot encoding
        charge_long = charge.view(-1).long().clamp(0, self.max_charge)
        charge_onehot = F.one_hot(charge_long, num_classes=self.max_charge + 1).float()

        # Concatenate features
        features = [cls_repr, charge_onehot]

        # Add instrument embedding if used
        if self.use_instrument:
            if instrument is None:
                # Default to "unknown" instrument
                instrument = torch.zeros(batch_size, dtype=torch.long, device=device)
            instrument = instrument.view(-1).long().clamp(0, NUM_INSTRUMENT_TYPES - 1)
            inst_emb = self.instrument_embed(instrument)
            features.append(inst_emb)

        features = torch.cat(features, dim=-1)

        # Neural network correction
        output = self.fc(features)  # (batch, 2)
        mean_correction = output[:, :1]  # (batch, 1)
        log_std = output[:, 1:]  # (batch, 1)

        # Compute physics-based baseline
        physics_ccs = self.sqrt_proj(mz, charge)

        # Add learned correction to physics baseline
        ccs_mean = physics_ccs + mean_correction

        # Convert log_std to std (always positive)
        ccs_std = F.softplus(log_std) + 1e-6  # Add small epsilon for numerical stability

        return ccs_mean, ccs_std


class RTHead(nn.Module):
    """
    Retention Time prediction head.

    Regression head that predicts indexed retention time (iRT)
    from the [CLS] token representation.

    Supports optional instrument type encoding for instrument-specific
    RT calibration. Different LC-MS setups may have different RT scales.

    Args:
        d_model: Encoder embedding dimension
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout probability (default: 0.1)
        use_instrument: Whether to use instrument type encoding (default: True)

    Returns:
        Retention time prediction of shape (batch, 1)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_instrument: bool = True,
    ):
        super().__init__()

        self.use_instrument = use_instrument

        # Instrument embedding
        if use_instrument:
            self.instrument_embed = nn.Embedding(NUM_INSTRUMENT_TYPES, d_model // 4)
            input_dim = d_model + d_model // 4
        else:
            self.instrument_embed = None
            input_dim = d_model

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        encoder_out: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        instrument: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict retention time.

        Args:
            encoder_out: Encoder output of shape (batch, seq_len, d_model)
            padding_mask: Optional padding mask (unused)
            instrument: Optional instrument type IDs of shape (batch,)

        Returns:
            Retention time prediction of shape (batch, 1)
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device

        cls_repr = encoder_out[:, 0, :]  # (batch, d_model)

        if self.use_instrument:
            if instrument is None:
                instrument = torch.zeros(batch_size, dtype=torch.long, device=device)
            instrument = instrument.view(-1).long().clamp(0, NUM_INSTRUMENT_TYPES - 1)
            inst_emb = self.instrument_embed(instrument)
            features = torch.cat([cls_repr, inst_emb], dim=-1)
        else:
            features = cls_repr

        return self.fc(features)


class ChargeHead(nn.Module):
    """
    Charge state distribution prediction head.

    Predicts a probability distribution over possible charge states
    from the [CLS] token representation.

    Args:
        d_model: Encoder embedding dimension
        max_charge: Maximum charge state (default: 6)
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout probability (default: 0.1)

    Returns:
        Charge state probabilities of shape (batch, max_charge)
        Note: Outputs charges 1 through max_charge (0 is not a valid charge)
    """

    def __init__(
        self,
        d_model: int,
        max_charge: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.max_charge = max_charge

        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_charge),  # Output: charges 1 to max_charge
        )

    def forward(
        self,
        encoder_out: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Predict charge state distribution.

        Args:
            encoder_out: Encoder output of shape (batch, seq_len, d_model)
            padding_mask: Optional padding mask (unused)
            return_logits: If True, return raw logits instead of probabilities

        Returns:
            Charge state probabilities/logits of shape (batch, max_charge)
        """
        cls_repr = encoder_out[:, 0, :]  # (batch, d_model)
        logits = self.fc(cls_repr)

        if return_logits:
            return logits
        return F.softmax(logits, dim=-1)


class IntensityHead(nn.Module):
    """
    Fragment ion intensity prediction head.

    Uses the full sequence representation (not just [CLS]) with
    attention pooling to predict fragment ion intensities.

    Predicts intensities for b and y ions across multiple charges
    and loss types. The output dimension follows the Prosit format:
    174 = 29 positions * 3 ion types (b, y, y-loss) * 2 charges (1+, 2+)

    Supports instrument type encoding, which is critical for intensity
    prediction as different instruments produce very different
    fragmentation patterns (e.g., HCD vs CID, beam-type CID vs resonance CID).

    Args:
        d_model: Encoder embedding dimension
        max_seq_len: Maximum peptide length (default: 30)
        num_ion_types: Number of ion types to predict (default: 6)
                      Typically: b1+, b2+, y1+, y2+, y-loss1+, y-loss2+
        hidden_dim: Hidden layer dimension (default: 512)
        dropout: Dropout probability (default: 0.1)
        use_instrument: Whether to use instrument type encoding (default: True)

    Returns:
        Fragment intensities of shape (batch, num_outputs)
    """

    # Standard output dimensions for Prosit-compatible models
    PROSIT_NUM_OUTPUTS = 174  # 29 * 6 ion types

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 30,
        num_ion_types: int = 6,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_instrument: bool = True,
    ):
        super().__init__()

        self.num_outputs = (max_seq_len - 1) * num_ion_types
        self.use_instrument = use_instrument

        # Attention pooling over sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            batch_first=True,
            dropout=dropout,
        )

        # Additional inputs: charge one-hot (6) + collision energy (1) + instrument
        self.charge_embed = nn.Linear(6, d_model // 4)
        self.ce_embed = nn.Linear(1, d_model // 4)

        if use_instrument:
            self.instrument_embed = nn.Embedding(NUM_INSTRUMENT_TYPES, d_model // 4)
            input_dim = d_model + d_model // 4 * 3  # pooled + charge + CE + instrument
        else:
            self.instrument_embed = None
            input_dim = d_model + d_model // 2  # pooled + charge + CE

        # Prediction network
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_outputs),
        )

    def forward(
        self,
        encoder_out: torch.Tensor,
        charge: torch.Tensor,
        collision_energy: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        instrument: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict fragment ion intensities.

        Args:
            encoder_out: Encoder output of shape (batch, seq_len, d_model)
            charge: Charge states of shape (batch,) - integer values 1-6
            collision_energy: Normalized collision energy of shape (batch,) or (batch, 1)
            padding_mask: Boolean mask where True indicates padding
            instrument: Optional instrument type IDs of shape (batch,)

        Returns:
            Fragment intensities of shape (batch, num_outputs)
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device

        # Attention pooling
        pooled, _ = self.attention(
            encoder_out,
            encoder_out,
            encoder_out,
            key_padding_mask=padding_mask,
        )
        # Mean pool the attended values
        if padding_mask is not None:
            # Mask out padding positions
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            pooled = (pooled * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = pooled.mean(dim=1)  # (batch, d_model)

        # Embed charge and collision energy
        charge_long = charge.view(-1).long().clamp(1, 6) - 1  # 0-indexed
        charge_onehot = F.one_hot(charge_long, num_classes=6).float()
        charge_emb = self.charge_embed(charge_onehot)  # (batch, d_model // 4)

        ce = collision_energy.view(-1, 1).float()
        ce_emb = self.ce_embed(ce)  # (batch, d_model // 4)

        # Concatenate features
        features = [pooled, charge_emb, ce_emb]

        # Add instrument embedding if used
        if self.use_instrument:
            if instrument is None:
                instrument = torch.zeros(batch_size, dtype=torch.long, device=device)
            instrument = instrument.view(-1).long().clamp(0, NUM_INSTRUMENT_TYPES - 1)
            inst_emb = self.instrument_embed(instrument)
            features.append(inst_emb)

        features = torch.cat(features, dim=-1)

        # Predict intensities
        intensities = self.fc(features)

        # Apply sigmoid to ensure non-negative outputs
        return torch.sigmoid(intensities)
