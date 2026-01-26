"""
Utility functions and classes for model training and resource loading.

This module provides utilities for:
- Loading pretrained models and tokenizers from resources
- In-memory checkpointing during training
- Model saving/loading utilities

Uses PyTorch as the deep learning backend.
"""

import copy
import json
from typing import Optional, Dict, Any
import numpy as np
import importlib.resources as resources
from importlib.abc import Traversable

# Import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def get_model_path(model_name: str) -> Traversable:
    """
    Get the path to a pretrained model.

    Args:
        model_name: The name of the model to load

    Returns:
        The path to the pretrained model
    """
    return resources.files('imspy_predictors.pretrained').joinpath(model_name)


def get_tokenizer_path(tokenizer_name: str = "tokenizer") -> Traversable:
    """
    Get the path to a tokenizer vocabulary file.

    Args:
        tokenizer_name: Name of the tokenizer file (without .json extension)

    Returns:
        Path to the tokenizer JSON file
    """
    return resources.files('imspy_predictors.pretrained').joinpath(f'{tokenizer_name}.json')


def load_tokenizer_from_resources(tokenizer_name: str = "tokenizer"):
    """
    Load a tokenizer from resources.

    For new code, prefer using ProformaTokenizer from imspy_predictors.utilities.tokenizers
    which uses the high-performance Rust implementation.

    Args:
        tokenizer_name: Name of the tokenizer to load

    Returns:
        The pretrained tokenizer (type depends on backend)
    """
    # Use the Rust-based tokenizer
    try:
        from imspy_predictors.utilities.tokenizers import ProformaTokenizer
        tokenizer_path = get_tokenizer_path(tokenizer_name)
        if tokenizer_path.is_file():
            return ProformaTokenizer.load_vocab(str(tokenizer_path))
        else:
            # Return default tokenizer if file doesn't exist
            return ProformaTokenizer.with_defaults()
    except ImportError:
        raise ImportError(
            "Could not load tokenizer. Install imspy-core for ProformaTokenizer."
        )


class InMemoryCheckpoint:
    """
    In-memory model checkpointing for best validation performance.

    Stores the best model weights in memory during training and restores
    them at the end.

    Args:
        validation_target: Metric name to monitor (default: 'val_loss')
        mode: 'min' to minimize metric, 'max' to maximize (default: 'min')
        patience: Number of epochs with no improvement before stopping (default: None = no early stopping)
        verbose: Whether to print improvement messages (default: False)

    Example:
        >>> checkpoint = InMemoryCheckpoint(patience=5)
        >>> for epoch in range(100):
        ...     val_loss = train_epoch(model, ...)
        ...     if checkpoint.step(val_loss, model):
        ...         print("Early stopping")
        ...         break
        >>> checkpoint.restore(model)
    """

    def __init__(
        self,
        validation_target: str = "val_loss",
        mode: str = "min",
        patience: Optional[int] = None,
        verbose: bool = False,
    ):
        self.validation_target = validation_target
        self.mode = mode
        self.patience = patience
        self.verbose = verbose

        self.best_weights = None
        self.initial_weights = None
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0

    def reset(self) -> None:
        """Reset checkpoint state."""
        self.best_weights = None
        self.initial_weights = None
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.wait = 0

    def _is_improvement(self, value: float) -> bool:
        """Check if value is an improvement over best."""
        if self.mode == 'min':
            return value < self.best_value
        else:
            return value > self.best_value

    def step(
        self,
        value: float,
        model: "nn.Module",
    ) -> bool:
        """
        Update checkpoint with new validation result.

        Args:
            value: Current validation metric value
            model: Model to checkpoint

        Returns:
            True if should stop (patience exceeded), False otherwise
        """
        if self.initial_weights is None:
            self._save_initial(model)

        if self._is_improvement(value):
            self.best_value = value
            self._save_best(model)
            self.wait = 0
            if self.verbose:
                print(f"Improved {self.validation_target} to {value:.6f}")
        else:
            self.wait += 1

        if self.patience is not None and self.wait >= self.patience:
            return True  # Should stop

        return False  # Continue training

    def _save_initial(self, model: "nn.Module") -> None:
        """Save initial model weights."""
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            self.initial_weights = copy.deepcopy(model.state_dict())

    def _save_best(self, model: "nn.Module") -> None:
        """Save best model weights."""
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            self.best_weights = copy.deepcopy(model.state_dict())

    def restore(self, model: "nn.Module") -> bool:
        """
        Restore best weights to model.

        Args:
            model: PyTorch model to restore weights to

        Returns:
            True if weights were restored, False if no best weights available
        """
        if self.best_weights is None:
            return False

        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            model.load_state_dict(self.best_weights)
            return True

        return False


def save_model_checkpoint(
    model: "nn.Module",
    path: str,
    optimizer: Optional["torch.optim.Optimizer"] = None,
    epoch: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a PyTorch model checkpoint.

    Args:
        model: PyTorch model to save
        path: Path to save checkpoint
        optimizer: Optional optimizer to save state
        epoch: Optional epoch number
        metadata: Optional additional metadata
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for save_model_checkpoint")

    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if metadata is not None:
        checkpoint['metadata'] = metadata

    torch.save(checkpoint, path)


def load_model_checkpoint(
    path: str,
    model: Optional["nn.Module"] = None,
    optimizer: Optional["torch.optim.Optimizer"] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a PyTorch model checkpoint.

    Args:
        path: Path to checkpoint file
        model: Optional model to load weights into
        optimizer: Optional optimizer to load state into
        map_location: Device to map tensors to (e.g., 'cpu', 'cuda')

    Returns:
        Dictionary with checkpoint contents
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for load_model_checkpoint")

    checkpoint = torch.load(path, map_location=map_location)

    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def count_parameters(model: "nn.Module", trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for count_parameters")

    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_device(prefer_cuda: bool = True) -> str:
    """
    Get the best available device.

    Args:
        prefer_cuda: If True, prefer CUDA if available

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if not TORCH_AVAILABLE:
        return 'cpu'

    if prefer_cuda and torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
