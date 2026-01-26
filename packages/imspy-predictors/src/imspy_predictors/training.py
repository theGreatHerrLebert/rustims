"""
Training utilities for peptide property prediction models.

Provides training loops, callbacks, and utilities for training
PyTorch models on peptide property prediction tasks.

Features:
- TrainingConfig: Configuration dataclass for training parameters
- EarlyStopping: Callback for early stopping with best model saving
- MetricTracker: Aggregate metrics during training
- PerformanceLogger: TensorBoard and file-based logging
- Trainer: General-purpose trainer with validation and checkpointing
- Task-specific training functions (train_ccs_model, train_rt_model, etc.)
"""

import copy
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)

# Try to import TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Scheduler parameters
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-5

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_best_only: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 100
    eval_interval: int = 1  # Evaluate every N epochs

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Mixed precision
    use_amp: bool = False


class EarlyStopping:
    """
    Early stopping callback.

    Monitors a metric and stops training when it stops improving.
    Also saves the best model weights for restoration.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as an improvement
        mode: "min" for loss, "max" for metrics like accuracy
        verbose: Whether to print messages
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-5,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.should_stop = False

        if mode == "min":
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value
            model: Model to save weights from

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                logger.info(f"EarlyStopping: New best score: {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs"
                )
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logger.info("EarlyStopping: Stopping training")

        return self.should_stop

    def restore_best_weights(self, model: nn.Module):
        """Restore the best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                logger.info(f"Restored best weights (score: {self.best_score:.6f})")


class MetricTracker:
    """Track and aggregate metrics during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], count: int = 1):
        """Update metrics with new values."""
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            self.metrics[name] += value * count
            self.counts[name] += count

    def average(self) -> Dict[str, float]:
        """Get averaged metrics."""
        return {
            name: total / self.counts[name]
            for name, total in self.metrics.items()
        }


class PerformanceLogger:
    """
    Log training performance metrics to TensorBoard and/or files.

    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment
        use_tensorboard: Whether to use TensorBoard logging
        log_to_file: Whether to log to a JSON file
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        log_to_file: bool = True,
    ):
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.log_to_file = log_to_file

        # Create log directory
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = None
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.experiment_dir / "tensorboard"))

        # Initialize log file
        self.log_file = self.experiment_dir / "training_log.json"
        self.log_data = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "epochs": [],
        }

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ):
        """Log metrics for an epoch."""
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train": train_metrics,
            "val": val_metrics or {},
        }
        if lr is not None:
            epoch_data["learning_rate"] = lr

        self.log_data["epochs"].append(epoch_data)

        # TensorBoard logging
        if self.writer is not None:
            for name, value in train_metrics.items():
                self.writer.add_scalar(f"train/{name}", value, epoch)
            if val_metrics:
                for name, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{name}", value, epoch)
            if lr is not None:
                self.writer.add_scalar("learning_rate", lr, epoch)

        # File logging
        if self.log_to_file:
            with open(self.log_file, "w") as f:
                json.dump(self.log_data, f, indent=2)

    def log_hyperparameters(self, config: "TrainingConfig"):
        """Log training hyperparameters."""
        self.log_data["hyperparameters"] = asdict(config)

        if self.writer is not None:
            # TensorBoard hparams
            hparams = {k: v for k, v in asdict(config).items() if v is not None}
            self.writer.add_hparams(hparams, {})

        if self.log_to_file:
            with open(self.log_file, "w") as f:
                json.dump(self.log_data, f, indent=2)

    def log_model_info(
        self,
        model: nn.Module,
        input_shape: Optional[Tuple[int, ...]] = None,
    ):
        """Log model architecture information."""
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.log_data["model_info"] = {
            "total_parameters": num_params,
            "trainable_parameters": num_trainable,
            "architecture": str(model.__class__.__name__),
        }

        if self.log_to_file:
            with open(self.log_file, "w") as f:
                json.dump(self.log_data, f, indent=2)

        logger.info(
            f"Model: {model.__class__.__name__}, "
            f"Parameters: {num_params:,} (trainable: {num_trainable:,})"
        )

    def log_final_metrics(self, metrics: Dict[str, float]):
        """Log final evaluation metrics."""
        self.log_data["final_metrics"] = metrics
        self.log_data["end_time"] = datetime.now().isoformat()

        if self.log_to_file:
            with open(self.log_file, "w") as f:
                json.dump(self.log_data, f, indent=2)

    def close(self):
        """Close the logger."""
        if self.writer is not None:
            self.writer.close()


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute regression metrics for model evaluation.

    Args:
        predictions: Model predictions
        targets: Ground truth targets

    Returns:
        Dictionary with MAE, RMSE, R², MedAE
    """
    predictions = predictions.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()

    # MAE
    mae = np.mean(np.abs(predictions - targets))

    # RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Median Absolute Error
    medae = np.median(np.abs(predictions - targets))

    # Pearson correlation
    if len(predictions) > 1:
        correlation = np.corrcoef(predictions, targets)[0, 1]
    else:
        correlation = 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "medae": float(medae),
        "pearson_r": float(correlation),
    }


def compute_spectral_angle(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Compute masked spectral angle similarity for intensity predictions.

    Invalid positions in targets are marked with -1 and are masked out.

    Args:
        predictions: Predicted intensities
        targets: Target intensities (invalid positions marked with -1)
        eps: Small value for numerical stability

    Returns:
        Mean spectral angle similarity (0 to 1, higher is better)
    """
    # Create mask for valid positions (where target != -1)
    mask = (targets >= 0).float()

    # Apply mask to both predictions and targets
    pred_masked = predictions * mask
    target_masked = torch.clamp(targets, min=0) * mask  # Replace -1 with 0

    # Normalize (only considering valid positions)
    pred_norm = pred_masked / (pred_masked.norm(dim=-1, keepdim=True) + eps)
    target_norm = target_masked / (target_masked.norm(dim=-1, keepdim=True) + eps)

    # Cosine similarity
    cos_sim = torch.sum(pred_norm * target_norm, dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # Spectral angle (as similarity, not angle)
    return float(cos_sim.mean().item())


def compute_intensity_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute metrics for intensity prediction evaluation with masking.

    Invalid positions in targets are marked with -1 and are masked out
    for all metric computations.

    Args:
        predictions: Predicted intensities
        targets: Target intensities (invalid positions marked with -1)

    Returns:
        Dictionary with spectral angle, cosine similarity, PCC
    """
    spectral_angle = compute_spectral_angle(predictions, targets)

    # Pearson correlation (average over batch, only on valid positions)
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()

    correlations = []
    for p, t in zip(pred_np, target_np):
        # Mask invalid positions (where target == -1)
        valid_mask = t >= 0
        p_valid = p[valid_mask]
        t_valid = t[valid_mask]

        if len(p_valid) > 1 and np.std(p_valid) > 0 and np.std(t_valid) > 0:
            corr = np.corrcoef(p_valid, t_valid)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    pcc = np.mean(correlations) if correlations else 0.0

    return {
        "spectral_angle": spectral_angle,
        "cosine_similarity": spectral_angle,  # Same for normalized vectors
        "pearson_correlation": float(pcc),
    }


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute classification metrics for charge state prediction.

    Args:
        predictions: Predicted class probabilities
        targets: Target class labels or distributions

    Returns:
        Dictionary with accuracy, top-2 accuracy, cross-entropy
    """
    pred_classes = predictions.argmax(dim=-1)

    if targets.dim() > 1:
        # Targets are distributions
        target_classes = targets.argmax(dim=-1)
    else:
        target_classes = targets

    # Top-1 accuracy
    accuracy = (pred_classes == target_classes).float().mean().item()

    # Top-2 accuracy
    _, top2 = predictions.topk(2, dim=-1)
    top2_correct = (top2 == target_classes.unsqueeze(-1)).any(dim=-1)
    top2_accuracy = top2_correct.float().mean().item()

    # Cross-entropy
    if targets.dim() > 1:
        ce = torch.nn.functional.cross_entropy(predictions, targets).item()
    else:
        ce = torch.nn.functional.cross_entropy(predictions, target_classes).item()

    return {
        "accuracy": accuracy,
        "top2_accuracy": top2_accuracy,
        "cross_entropy": ce,
    }


class Trainer:
    """
    General-purpose trainer for peptide property prediction models.

    Handles training loops, validation, checkpointing, and callbacks.

    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        scheduler: Optional learning rate scheduler
        loss_fn: Loss function
        config: Training configuration
        device: Device to train on
        logger: Optional PerformanceLogger for logging metrics
        task: Task type for specialized metrics ("ccs", "rt", "charge", "intensity")

    Example:
        >>> trainer = Trainer(model, optimizer, loss_fn=nn.L1Loss())
        >>> history = trainer.train(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        config: Optional[TrainingConfig] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[str] = None,
        perf_logger: Optional[PerformanceLogger] = None,
        task: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config or TrainingConfig()
        self.device = device or self.config.device
        self.perf_logger = perf_logger
        self.task = task

        self.model.to(self.device)

        # Initialize components
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None

        # History with extended metrics
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
            "learning_rate": [],
            "epoch_time": [],
        }

        # Initialize logger
        if self.perf_logger is not None:
            self.perf_logger.log_hyperparameters(self.config)
            self.perf_logger.log_model_info(self.model)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        metric_tracker = MetricTracker()

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()

            # Forward pass with optional mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                loss = self._compute_loss(batch)

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

            # Track metrics
            batch_size = batch["input_ids"].size(0)
            metric_tracker.update({"loss": loss.item()}, count=batch_size)

            # Log progress
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = metric_tracker.average()["loss"]
                logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.6f}"
                )

        return metric_tracker.average()["loss"]

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        compute_metrics: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model and compute task-specific metrics.

        Args:
            val_loader: Validation data loader
            compute_metrics: Whether to compute detailed metrics

        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()
        metric_tracker = MetricTracker()

        # Collect predictions and targets for metrics
        all_preds = []
        all_targets = []

        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                loss = self._compute_loss(batch)

                # Get predictions for metrics
                if compute_metrics and self.task is not None:
                    preds, targets = self._get_predictions(batch)
                    if preds is not None:
                        all_preds.append(preds.cpu())
                        all_targets.append(targets.cpu())

            batch_size = batch["input_ids"].size(0)
            metric_tracker.update({"loss": loss.item()}, count=batch_size)

        avg_loss = metric_tracker.average()["loss"]

        # Compute task-specific metrics
        metrics = {"loss": avg_loss}
        if compute_metrics and all_preds:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            if self.task in ("ccs", "rt"):
                task_metrics = compute_regression_metrics(all_preds, all_targets)
            elif self.task == "intensity":
                task_metrics = compute_intensity_metrics(all_preds, all_targets)
            elif self.task == "charge":
                task_metrics = compute_classification_metrics(all_preds, all_targets)
            else:
                task_metrics = {}

            metrics.update(task_metrics)

        return avg_loss, metrics

    def _get_predictions(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get predictions and targets for metric computation."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        padding_mask = (attention_mask == 0)

        mz = batch.get("mz")
        charge = batch.get("charge")
        collision_energy = batch.get("collision_energy")
        instrument = batch.get("instrument")

        outputs = self.model(
            input_ids,
            padding_mask=padding_mask,
            mz=mz,
            charge=charge,
            collision_energy=collision_energy,
            instrument=instrument,
        )

        if self.task == "ccs" and "ccs" in outputs and "ccs" in batch:
            mean, _ = outputs["ccs"]
            return mean, batch["ccs"].view(-1, 1)
        elif self.task == "rt" and "rt" in outputs and "rt" in batch:
            return outputs["rt"], batch["rt"].view(-1, 1)
        elif self.task == "intensity" and "intensity" in outputs and "intensity" in batch:
            return outputs["intensity"], batch["intensity"]
        elif self.task == "charge" and "charge" in outputs and "charge_dist" in batch:
            return outputs["charge"], batch["charge_dist"]

        return None, None

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for a batch.

        Override this method for custom loss computation.
        """
        # Extract inputs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        padding_mask = (attention_mask == 0)

        # Get optional inputs
        mz = batch.get("mz")
        charge = batch.get("charge")
        collision_energy = batch.get("collision_energy")
        instrument = batch.get("instrument")

        # Forward pass
        outputs = self.model(
            input_ids,
            padding_mask=padding_mask,
            mz=mz,
            charge=charge,
            collision_energy=collision_energy,
            instrument=instrument,
        )

        # Compute loss based on available targets
        total_loss = 0.0
        num_tasks = 0

        if "ccs" in batch and "ccs" in outputs:
            mean, std = outputs["ccs"]
            target_ccs = batch["ccs"].view(-1, 1)
            # L1 loss for CCS mean prediction
            loss = torch.nn.functional.l1_loss(mean, target_ccs)

            # Supervised std loss if ground truth ccs_std is available
            if "ccs_std" in batch:
                target_std = batch["ccs_std"].view(-1, 1)
                # Mask where target_std == -1 (missing values)
                valid_mask = (target_std >= 0).squeeze()
                if valid_mask.any():
                    std_loss = torch.nn.functional.l1_loss(
                        std[valid_mask], target_std[valid_mask]
                    )
                    loss = loss + std_loss  # Equal weight for std loss

            total_loss += loss
            num_tasks += 1

        if "rt" in batch and "rt" in outputs:
            pred = outputs["rt"]
            target = batch["rt"].view(-1, 1)
            loss = torch.nn.functional.l1_loss(pred, target)
            total_loss += loss
            num_tasks += 1

        if "charge_dist" in batch and "charge" in outputs:
            pred = outputs["charge"]
            target = batch["charge_dist"]
            loss = torch.nn.functional.cross_entropy(pred, target)
            total_loss += loss
            num_tasks += 1

        if "intensity" in batch and "intensity" in outputs:
            from imspy_predictors.losses import masked_spectral_distance
            pred = outputs["intensity"]
            target = batch["intensity"]
            # Masked spectral angle loss (dlomix-compatible, handles -1 markers)
            loss = masked_spectral_distance(target, pred)
            total_loss += loss
            num_tasks += 1

        if num_tasks == 0:
            raise ValueError("No matching targets found in batch")

        return total_loss / num_tasks

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs (overrides config)

        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.epochs
        start_time = time.time()

        logger.info(f"Starting training for {epochs} epochs on {self.device}")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            logger.info(f"Epoch {epoch}/{epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            logger.info(f"Train Loss: {train_loss:.6f}, LR: {current_lr:.2e}")

            # Validate
            val_loss = None
            val_metrics = {}
            if val_loader is not None and epoch % self.config.eval_interval == 0:
                val_loss, val_metrics = self.validate(val_loader, compute_metrics=True)
                self.history["val_loss"].append(val_loss)
                self.history["val_metrics"].append(val_metrics)

                # Log validation metrics
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                logger.info(f"Val Metrics: {metrics_str}")

                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Early stopping
                if self.early_stopping(val_loss, self.model):
                    logger.info("Early stopping triggered")
                    break

                # Checkpointing
                if self.config.checkpoint_dir is not None:
                    self._save_checkpoint(epoch, val_loss, val_metrics)

            # Track epoch time
            epoch_time = time.time() - epoch_start
            self.history["epoch_time"].append(epoch_time)

            # Log to PerformanceLogger
            if self.perf_logger is not None:
                self.perf_logger.log_epoch(
                    epoch=epoch,
                    train_metrics={"loss": train_loss},
                    val_metrics=val_metrics if val_metrics else None,
                    lr=current_lr,
                )

        # Restore best weights
        if val_loader is not None:
            self.early_stopping.restore_best_weights(self.model)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")

        # Log final metrics
        if self.perf_logger is not None:
            final_metrics = {
                "total_training_time": total_time,
                "best_val_loss": self.early_stopping.best_score or 0.0,
                "final_epoch": len(self.history["train_loss"]),
            }
            self.perf_logger.log_final_metrics(final_metrics)
            self.perf_logger.close()

        return self.history

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """Save a training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_best_only:
            # Only save if this is the best model
            if val_loss <= min(self.history["val_loss"]):
                path = checkpoint_dir / "best_model.pt"
                self._save_model(path, epoch, val_loss, val_metrics, is_best=True)
        else:
            path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            self._save_model(path, epoch, val_loss, val_metrics, is_best=False)

    def _save_model(
        self,
        path: Path,
        epoch: Optional[int] = None,
        val_loss: Optional[float] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ):
        """
        Save model weights with metadata.

        Saves a comprehensive checkpoint including:
        - Model state dict
        - Optimizer state dict
        - Scheduler state dict (if available)
        - Training history
        - Config parameters
        - Metrics
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": asdict(self.config),
            "task": self.task,
            "timestamp": datetime.now().isoformat(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        if val_loss is not None:
            checkpoint["val_loss"] = val_loss

        if val_metrics is not None:
            checkpoint["val_metrics"] = val_metrics

        checkpoint["is_best"] = is_best

        # Save model architecture info
        checkpoint["model_class"] = self.model.__class__.__name__

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path} (epoch {epoch}, val_loss: {val_loss:.6f})")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[str] = None,
    ) -> Tuple[nn.Module, Dict]:
        """
        Load a training checkpoint.

        Args:
            path: Path to checkpoint file
            model: Model instance to load weights into
            optimizer: Optional optimizer to restore state
            scheduler: Optional scheduler to restore state
            device: Device to load tensors to

        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            f"Loaded checkpoint from {path} "
            f"(epoch {checkpoint.get('epoch', '?')}, "
            f"val_loss: {checkpoint.get('val_loss', '?'):.6f})"
        )

        return model, checkpoint


def train_ccs_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None,
    freeze_encoder: bool = False,
    experiment_name: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Train a CCS prediction model.

    Convenience function for training CCS models with appropriate
    loss function and settings.

    Args:
        model: UnifiedPeptideModel or CCS-specific model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        freeze_encoder: Whether to freeze the encoder
        experiment_name: Name for logging experiment
        log_dir: Directory for logs

    Returns:
        Training history
    """
    config = config or TrainingConfig()

    if freeze_encoder and hasattr(model, "freeze_encoder"):
        model.freeze_encoder()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    # Set up logging
    perf_logger = None
    if log_dir or experiment_name:
        perf_logger = PerformanceLogger(
            log_dir=log_dir or "./logs",
            experiment_name=experiment_name or f"ccs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    def ccs_loss(batch):
        input_ids = batch["input_ids"]
        padding_mask = (batch["attention_mask"] == 0)
        mz = batch["mz"]
        charge = batch["charge"]
        target = batch["ccs"].view(-1, 1)

        mean, std = model.predict_ccs(input_ids, mz, charge, padding_mask)
        return torch.nn.functional.gaussian_nll_loss(mean, target, std ** 2)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=ccs_loss,
        config=config,
        perf_logger=perf_logger,
        task="ccs",
    )

    return trainer.train(train_loader, val_loader)


def train_rt_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None,
    freeze_encoder: bool = False,
    experiment_name: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Train an RT prediction model.

    Args:
        model: UnifiedPeptideModel or RT-specific model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        freeze_encoder: Whether to freeze the encoder
        experiment_name: Name for logging experiment
        log_dir: Directory for logs

    Returns:
        Training history
    """
    config = config or TrainingConfig()

    if freeze_encoder and hasattr(model, "freeze_encoder"):
        model.freeze_encoder()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    # Set up logging
    perf_logger = None
    if log_dir or experiment_name:
        perf_logger = PerformanceLogger(
            log_dir=log_dir or "./logs",
            experiment_name=experiment_name or f"rt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    def rt_loss(batch):
        input_ids = batch["input_ids"]
        padding_mask = (batch["attention_mask"] == 0)
        target = batch["rt"].view(-1, 1)

        pred = model.predict_rt(input_ids, padding_mask)
        return torch.nn.functional.l1_loss(pred, target)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=rt_loss,
        config=config,
        perf_logger=perf_logger,
        task="rt",
    )

    return trainer.train(train_loader, val_loader)


def train_intensity_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None,
    freeze_encoder: bool = False,
    experiment_name: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Train an intensity prediction model.

    Args:
        model: UnifiedPeptideModel or intensity-specific model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        freeze_encoder: Whether to freeze the encoder
        experiment_name: Name for logging experiment
        log_dir: Directory for logs

    Returns:
        Training history
    """
    config = config or TrainingConfig()

    if freeze_encoder and hasattr(model, "freeze_encoder"):
        model.freeze_encoder()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    # Set up logging
    perf_logger = None
    if log_dir or experiment_name:
        perf_logger = PerformanceLogger(
            log_dir=log_dir or "./logs",
            experiment_name=experiment_name or f"intensity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    def intensity_loss(batch):
        from imspy_predictors.losses import masked_spectral_distance
        input_ids = batch["input_ids"]
        padding_mask = (batch["attention_mask"] == 0)
        charge = batch["charge"]
        ce = batch["collision_energy"]
        target = batch["intensity"]

        pred = model.predict_intensity(input_ids, charge, ce, padding_mask)
        # Masked spectral angle loss (dlomix-compatible, handles -1 markers)
        return masked_spectral_distance(target, pred)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=intensity_loss,
        config=config,
        perf_logger=perf_logger,
        task="intensity",
    )

    return trainer.train(train_loader, val_loader)


def train_charge_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None,
    freeze_encoder: bool = False,
    experiment_name: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Train a charge state prediction model.

    Args:
        model: UnifiedPeptideModel or charge-specific model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        freeze_encoder: Whether to freeze the encoder
        experiment_name: Name for logging experiment
        log_dir: Directory for logs

    Returns:
        Training history
    """
    config = config or TrainingConfig()

    if freeze_encoder and hasattr(model, "freeze_encoder"):
        model.freeze_encoder()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    # Set up logging
    perf_logger = None
    if log_dir or experiment_name:
        perf_logger = PerformanceLogger(
            log_dir=log_dir or "./logs",
            experiment_name=experiment_name or f"charge_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    def charge_loss(batch):
        input_ids = batch["input_ids"]
        padding_mask = (batch["attention_mask"] == 0)
        target = batch["charge_dist"]

        pred = model.predict_charge(input_ids, padding_mask)
        return torch.nn.functional.cross_entropy(pred, target)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=charge_loss,
        config=config,
        perf_logger=perf_logger,
        task="charge",
    )

    return trainer.train(train_loader, val_loader)
