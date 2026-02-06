"""
Unified Transformer Encoder for Peptide Property Prediction.

This module implements the shared encoder architecture that can be pre-trained
on fragment intensity prediction and fine-tuned for other tasks (CCS, RT, Charge).
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.

    Adds positional information to token embeddings using sine and cosine
    functions of different frequencies, following the original Transformer paper.

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length to support
        dropout: Dropout probability applied after adding positional encoding
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PeptideTransformer(nn.Module):
    """
    Unified Transformer encoder for peptide sequence encoding.

    This encoder can be pre-trained on fragment intensity prediction (largest dataset)
    and then fine-tuned for other tasks by adding task-specific heads.

    Architecture:
        - Token embedding layer
        - Positional encoding
        - Transformer encoder (multiple layers)

    The encoder outputs contextualized representations for all tokens,
    which can be used differently by different task heads:
        - CCS/RT/Charge: Use [CLS] token representation
        - Intensity: Use full sequence representation

    Args:
        vocab_size: Size of the tokenizer vocabulary
        d_model: Embedding dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer encoder layers (default: 6)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        max_seq_len: Maximum sequence length (default: 100)
        dropout: Dropout probability (default: 0.1)
        padding_idx: Index of padding token (default: 0)

    Example:
        >>> encoder = PeptideTransformer(vocab_size=600, d_model=256)
        >>> tokens = torch.randint(0, 600, (32, 50))  # batch of 32, seq_len 50
        >>> mask = (tokens == 0)  # padding mask
        >>> output = encoder(tokens, padding_mask=mask)
        >>> print(output.shape)  # (32, 50, 256)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_seq_len: int = 100,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Layer normalization for final output
        self.output_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode peptide sequences.

        Args:
            tokens: Token IDs of shape (batch, seq_len)
            padding_mask: Boolean mask where True indicates padding positions
                         Shape: (batch, seq_len)

        Returns:
            Encoded representations of shape (batch, seq_len, d_model)
        """
        # Embed tokens and scale by sqrt(d_model)
        x = self.embedding(tokens) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Apply final layer norm
        x = self.output_norm(x)

        return x

    def get_cls_representation(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the [CLS] token representation for classification tasks.

        Assumes [CLS] is the first token in the sequence.

        Args:
            tokens: Token IDs of shape (batch, seq_len)
            padding_mask: Boolean mask where True indicates padding positions

        Returns:
            [CLS] representation of shape (batch, d_model)
        """
        encoded = self.forward(tokens, padding_mask)
        return encoded[:, 0, :]  # First token is [CLS]

    def freeze(self):
        """Freeze all encoder parameters for fine-tuning task heads only."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        map_location: Optional[str] = None,
        **kwargs,
    ) -> "PeptideTransformer":
        """
        Load a pre-trained encoder from a checkpoint.

        Args:
            path: Path to the checkpoint file (.pt)
            map_location: Device to load the model to
            **kwargs: Additional arguments to override saved config

        Returns:
            Loaded PeptideTransformer model
        """
        checkpoint = torch.load(path, map_location=map_location)

        # Get config from checkpoint or use provided kwargs
        config = checkpoint.get("config", {})
        config.update(kwargs)

        # Create model
        model = cls(**config)

        # Load weights
        model.load_state_dict(checkpoint["state_dict"])

        return model

    def save_pretrained(self, path: str):
        """
        Save the encoder to a checkpoint file.

        Args:
            path: Path to save the checkpoint (.pt)
        """
        checkpoint = {
            "config": {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "max_seq_len": self.max_seq_len,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)


class PeptideTransformerConfig:
    """
    Configuration class for PeptideTransformer.

    Provides preset configurations for different model sizes.
    """

    # Small model - faster training, less capacity
    SMALL = {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
    }

    # Base model - balanced size and performance
    BASE = {
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "dropout": 0.1,
    }

    # Large model - higher capacity for complex tasks
    LARGE = {
        "d_model": 512,
        "nhead": 8,
        "num_layers": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1,
    }

    @classmethod
    def get_config(cls, size: str = "base") -> dict:
        """
        Get configuration dictionary for a given model size.

        Args:
            size: One of "small", "base", or "large"

        Returns:
            Configuration dictionary
        """
        configs = {
            "small": cls.SMALL,
            "base": cls.BASE,
            "large": cls.LARGE,
        }
        return configs.get(size.lower(), cls.BASE).copy()
