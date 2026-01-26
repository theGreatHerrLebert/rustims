"""
Unified Peptide Property Prediction Model.

This module provides a unified model architecture that combines the shared
PeptideTransformer encoder with task-specific prediction heads for:
- CCS (Collision Cross Section)
- RT (Retention Time)
- Charge (Charge State Distribution)
- Intensity (Fragment Ion Intensities)

The model supports:
- Pre-training on intensity prediction (largest dataset)
- Fine-tuning for specific tasks
- Multi-task learning
- Transfer learning between tasks
- Instrument-specific predictions (different mass spectrometers)
"""

from typing import Dict, Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from imspy_predictors.models.transformer import PeptideTransformer, PeptideTransformerConfig
from imspy_predictors.models.heads import (
    CCSHead, RTHead, ChargeHead, IntensityHead,
    INSTRUMENT_TYPES, INSTRUMENT_TO_ID, get_instrument_id, NUM_INSTRUMENT_TYPES,
)


class UnifiedPeptideModel(nn.Module):
    """
    Unified model for peptide property prediction.

    Combines a shared transformer encoder with task-specific heads.
    Can be used for single-task or multi-task learning.

    Supports instrument-specific predictions by passing instrument type
    to the prediction heads. Different mass spectrometers have different
    characteristics (CCS calibration, fragmentation patterns, RT scales).

    Args:
        vocab_size: Size of tokenizer vocabulary
        encoder_config: Encoder configuration (default: BASE)
        tasks: List of tasks to enable. Options: ["ccs", "rt", "charge", "intensity"]
        max_charge: Maximum charge state (default: 6)
        max_seq_len: Maximum sequence length (default: 100)
        use_instrument: Whether to use instrument type encoding (default: True)

    Example:
        >>> model = UnifiedPeptideModel(vocab_size=600, tasks=["ccs", "rt"])
        >>> tokens = torch.randint(0, 600, (32, 50))
        >>> mz = torch.rand(32) * 2000 + 500
        >>> charge = torch.randint(1, 5, (32,))
        >>> instrument = torch.ones(32, dtype=torch.long)  # timsTOF
        >>> outputs = model(tokens, mz=mz, charge=charge, instrument=instrument, tasks=["ccs"])
        >>> print(outputs["ccs"][0].shape)  # (32, 1) for CCS mean

    Supported Instruments:
        Use get_instrument_id() to convert instrument names to IDs.
        Available instruments: timstof, timstof_pro, orbitrap, exploris, astral, etc.
    """

    # Export instrument utilities at class level
    INSTRUMENT_TYPES = INSTRUMENT_TYPES
    INSTRUMENT_TO_ID = INSTRUMENT_TO_ID
    get_instrument_id = staticmethod(get_instrument_id)

    def __init__(
        self,
        vocab_size: int,
        encoder_config: Union[str, dict] = "base",
        tasks: Optional[List[str]] = None,
        max_charge: int = 6,
        max_seq_len: int = 100,
        use_instrument: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_charge = max_charge
        self.max_seq_len = max_seq_len
        self.use_instrument = use_instrument

        # Get encoder config
        if isinstance(encoder_config, str):
            config = PeptideTransformerConfig.get_config(encoder_config)
        else:
            config = encoder_config.copy()

        config["vocab_size"] = vocab_size
        config["max_seq_len"] = max_seq_len

        self.encoder_config = config
        d_model = config["d_model"]

        # Create encoder
        self.encoder = PeptideTransformer(**config)

        # Default to all tasks if not specified
        if tasks is None:
            tasks = ["ccs", "rt", "charge", "intensity"]
        self.enabled_tasks = set(tasks)

        # Create task-specific heads
        self.heads = nn.ModuleDict()

        if "ccs" in self.enabled_tasks:
            self.heads["ccs"] = CCSHead(
                d_model=d_model,
                max_charge=max_charge,
                use_instrument=use_instrument,
            )

        if "rt" in self.enabled_tasks:
            self.heads["rt"] = RTHead(
                d_model=d_model,
                use_instrument=use_instrument,
            )

        if "charge" in self.enabled_tasks:
            self.heads["charge"] = ChargeHead(
                d_model=d_model,
                max_charge=max_charge,
            )

        if "intensity" in self.enabled_tasks:
            self.heads["intensity"] = IntensityHead(
                d_model=d_model,
                max_seq_len=30,  # Prosit standard
                use_instrument=use_instrument,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mz: Optional[torch.Tensor] = None,
        charge: Optional[torch.Tensor] = None,
        collision_energy: Optional[torch.Tensor] = None,
        instrument: Optional[torch.Tensor] = None,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for specified tasks.

        Args:
            tokens: Token IDs of shape (batch, seq_len)
            padding_mask: Boolean mask where True indicates padding
            mz: Precursor m/z values (required for CCS)
            charge: Charge states (required for CCS, intensity)
            collision_energy: Normalized collision energy (required for intensity)
            instrument: Optional instrument type IDs of shape (batch,).
                       Use UnifiedPeptideModel.get_instrument_id("timstof") to get IDs.
            tasks: List of tasks to compute. If None, computes all enabled tasks.

        Returns:
            Dictionary mapping task names to outputs:
                - "ccs": Tuple of (mean, std) tensors
                - "rt": Single tensor
                - "charge": Probability distribution tensor
                - "intensity": Fragment intensity tensor
        """
        # Encode sequence
        encoder_out = self.encoder(tokens, padding_mask=padding_mask)

        # Determine which tasks to run
        if tasks is None:
            tasks = list(self.enabled_tasks)

        outputs = {}

        for task in tasks:
            if task not in self.heads:
                raise ValueError(f"Task '{task}' not enabled. Enabled tasks: {self.enabled_tasks}")

            if task == "ccs":
                if mz is None or charge is None:
                    raise ValueError("CCS prediction requires 'mz' and 'charge' inputs")
                outputs["ccs"] = self.heads["ccs"](
                    encoder_out, mz, charge, padding_mask, instrument=instrument
                )

            elif task == "rt":
                outputs["rt"] = self.heads["rt"](
                    encoder_out, padding_mask, instrument=instrument
                )

            elif task == "charge":
                outputs["charge"] = self.heads["charge"](encoder_out, padding_mask)

            elif task == "intensity":
                if charge is None or collision_energy is None:
                    raise ValueError("Intensity prediction requires 'charge' and 'collision_energy' inputs")
                outputs["intensity"] = self.heads["intensity"](
                    encoder_out, charge, collision_energy, padding_mask, instrument=instrument
                )

        return outputs

    def predict_ccs(
        self,
        tokens: torch.Tensor,
        mz: torch.Tensor,
        charge: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        instrument: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for CCS prediction.

        Args:
            tokens: Token IDs
            mz: Precursor m/z values
            charge: Charge states
            padding_mask: Optional padding mask
            instrument: Optional instrument type IDs

        Returns:
            Tuple of (ccs_mean, ccs_std)
        """
        outputs = self.forward(
            tokens, padding_mask=padding_mask,
            mz=mz, charge=charge, instrument=instrument, tasks=["ccs"]
        )
        return outputs["ccs"]

    def predict_rt(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        instrument: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience method for RT prediction.

        Args:
            tokens: Token IDs
            padding_mask: Optional padding mask
            instrument: Optional instrument type IDs

        Returns:
            Retention time predictions
        """
        outputs = self.forward(
            tokens, padding_mask=padding_mask,
            instrument=instrument, tasks=["rt"]
        )
        return outputs["rt"]

    def predict_charge(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience method for charge state prediction.

        Note: Charge prediction doesn't use instrument type as it's
        primarily determined by the peptide sequence.

        Returns:
            Charge state probabilities
        """
        outputs = self.forward(tokens, padding_mask=padding_mask, tasks=["charge"])
        return outputs["charge"]

    def predict_intensity(
        self,
        tokens: torch.Tensor,
        charge: torch.Tensor,
        collision_energy: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        instrument: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience method for intensity prediction.

        Args:
            tokens: Token IDs
            charge: Charge states
            collision_energy: Normalized collision energy
            padding_mask: Optional padding mask
            instrument: Optional instrument type IDs (important for intensity!)

        Returns:
            Fragment ion intensities
        """
        outputs = self.forward(
            tokens, padding_mask=padding_mask,
            charge=charge, collision_energy=collision_energy,
            instrument=instrument, tasks=["intensity"]
        )
        return outputs["intensity"]

    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning heads only."""
        self.encoder.freeze()

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        self.encoder.unfreeze()

    def freeze_heads(self, tasks: Optional[List[str]] = None):
        """Freeze head parameters for specified tasks (or all if None)."""
        tasks = tasks or list(self.heads.keys())
        for task in tasks:
            if task in self.heads:
                for param in self.heads[task].parameters():
                    param.requires_grad = False

    def unfreeze_heads(self, tasks: Optional[List[str]] = None):
        """Unfreeze head parameters for specified tasks (or all if None)."""
        tasks = tasks or list(self.heads.keys())
        for task in tasks:
            if task in self.heads:
                for param in self.heads[task].parameters():
                    param.requires_grad = True

    def num_parameters(self, trainable_only: bool = True) -> Dict[str, int]:
        """
        Count parameters by component.

        Returns:
            Dictionary with parameter counts for encoder and each head
        """
        counts = {
            "encoder": sum(
                p.numel() for p in self.encoder.parameters()
                if not trainable_only or p.requires_grad
            )
        }

        for name, head in self.heads.items():
            counts[name] = sum(
                p.numel() for p in head.parameters()
                if not trainable_only or p.requires_grad
            )

        counts["total"] = sum(counts.values())
        return counts

    @classmethod
    def from_pretrained_encoder(
        cls,
        encoder_path: str,
        tasks: Optional[List[str]] = None,
        map_location: Optional[str] = None,
        **kwargs,
    ) -> "UnifiedPeptideModel":
        """
        Create a model with a pre-trained encoder.

        Loads encoder weights from a checkpoint and creates fresh task heads.
        Useful for transfer learning from intensity pre-training.

        Args:
            encoder_path: Path to encoder checkpoint
            tasks: Tasks to enable
            map_location: Device to load weights to
            **kwargs: Additional model configuration

        Returns:
            Model with pre-trained encoder and fresh heads
        """
        # Load encoder
        encoder = PeptideTransformer.from_pretrained(
            encoder_path, map_location=map_location
        )

        # Create model
        model = cls(
            vocab_size=encoder.vocab_size,
            encoder_config={
                "d_model": encoder.d_model,
                "max_seq_len": encoder.max_seq_len,
            },
            tasks=tasks,
            **kwargs,
        )

        # Replace encoder with pre-trained one
        model.encoder = encoder

        return model

    def save_pretrained(self, path: str, save_heads: bool = True):
        """
        Save model to checkpoint.

        Args:
            path: Path to save checkpoint
            save_heads: Whether to save task heads (default: True)
        """
        checkpoint = {
            "encoder_config": self.encoder_config,
            "enabled_tasks": list(self.enabled_tasks),
            "max_charge": self.max_charge,
            "max_seq_len": self.max_seq_len,
            "encoder_state_dict": self.encoder.state_dict(),
        }

        if save_heads:
            checkpoint["heads_state_dict"] = {
                name: head.state_dict()
                for name, head in self.heads.items()
            }

        torch.save(checkpoint, path)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        map_location: Optional[str] = None,
        tasks: Optional[List[str]] = None,
    ) -> "UnifiedPeptideModel":
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            map_location: Device to load weights to
            tasks: Override enabled tasks (loads fresh heads if different)

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=map_location)

        # Use saved tasks if not overriding
        enabled_tasks = tasks or checkpoint.get("enabled_tasks", ["ccs", "rt", "charge", "intensity"])

        # Create model
        model = cls(
            vocab_size=checkpoint["encoder_config"]["vocab_size"],
            encoder_config=checkpoint["encoder_config"],
            tasks=enabled_tasks,
            max_charge=checkpoint.get("max_charge", 6),
            max_seq_len=checkpoint.get("max_seq_len", 100),
        )

        # Load encoder weights
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])

        # Load head weights if available and tasks match
        if "heads_state_dict" in checkpoint:
            for name, state_dict in checkpoint["heads_state_dict"].items():
                if name in model.heads:
                    model.heads[name].load_state_dict(state_dict)

        return model


class TaskLoss(nn.Module):
    """
    Combined loss function for multi-task learning.

    Computes weighted sum of task-specific losses.
    """

    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        # Default equal weights
        self.task_weights = task_weights or {
            "ccs": 1.0,
            "rt": 1.0,
            "charge": 1.0,
            "intensity": 1.0,
        }

    def forward(
        self,
        predictions: Dict[str, Union[torch.Tensor, Tuple]],
        targets: Dict[str, torch.Tensor],
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task-specific and total losses.

        Args:
            predictions: Model outputs for each task
            targets: Ground truth for each task
            reduction: Loss reduction method ("mean" or "sum")

        Returns:
            Dictionary with individual task losses and "total"
        """
        losses = {}

        for task, pred in predictions.items():
            if task not in targets:
                continue

            target = targets[task]
            weight = self.task_weights.get(task, 1.0)

            if task == "ccs":
                # Gaussian NLL loss for uncertainty-aware prediction
                mean, std = pred
                loss = F.gaussian_nll_loss(mean, target, std ** 2, reduction=reduction)
            elif task == "rt":
                loss = F.l1_loss(pred, target, reduction=reduction)
            elif task == "charge":
                loss = F.cross_entropy(pred, target, reduction=reduction)
            elif task == "intensity":
                # Cosine similarity loss for spectra
                loss = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
            else:
                loss = F.mse_loss(pred, target, reduction=reduction)

            losses[task] = loss * weight

        losses["total"] = sum(losses.values())
        return losses
