#!/usr/bin/env python
"""
Pre-train encoder on intensity prediction.

This script trains the unified peptide model on fragment intensity prediction,
which serves as pre-training for the shared encoder that will be fine-tuned
for other tasks (CCS, RT, Charge).

Usage:
    # Pre-train on PROSPECT MS2 (31M samples)
    python -m imspy_predictors.scripts.train_intensity \
        --dataset prospect-ptms-ms2 \
        --model-config base \
        --batch-size 256 \
        --epochs 30 \
        --output-dir ./checkpoints/pretrain_intensity

    # Fine-tune on timsTOF MS2
    python -m imspy_predictors.scripts.train_intensity \
        --dataset timstof-ms2 \
        --pretrained ./checkpoints/pretrain_intensity/best_model.pt \
        --batch-size 128 \
        --epochs 50 \
        --output-dir ./checkpoints/timstof_intensity
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-train encoder on intensity prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=["prospect-ptms-ms2", "timstof-ms2", "orbitrap-ms2"],
        required=True,
        help="Dataset to train on",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets",
    )

    # Model
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--model-config",
        default="base",
        choices=["small", "base", "large"],
        help="Model configuration size",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Scheduler
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "plateau"])
    parser.add_argument("--scheduler-patience", type=int, default=5)

    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=10)

    # Encoder freezing
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
        help="Number of epochs to freeze encoder (for fine-tuning)",
    )

    # Mixed precision
    parser.add_argument("--use-amp", action="store_true", help="Use mixed precision")

    # Data loading
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=50)

    # Output
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=5, help="Save every N epochs")

    # Device
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Import after parsing args (for faster --help)
    from imspy_predictors.models.unified import UnifiedPeptideModel
    from imspy_predictors.data_utils import (
        load_prospect_ms2_dataset,
        load_timstof_ms2_dataset,
        load_orbitrap_ms2_dataset,
        create_dataloader,
    )
    from imspy_predictors.training import (
        TrainingConfig,
        PerformanceLogger,
        Trainer,
        EarlyStopping,
        compute_intensity_metrics,
    )
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer

    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = ProformaTokenizer.with_defaults()
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    streaming = args.streaming or args.dataset == "prospect-ptms-ms2"

    if args.dataset == "prospect-ptms-ms2":
        train_ds = load_prospect_ms2_dataset(
            tokenizer, split="train", max_length=args.max_seq_length, streaming=streaming
        )
        val_ds = load_prospect_ms2_dataset(
            tokenizer, split="validation", max_length=args.max_seq_length, streaming=streaming
        )
    elif args.dataset == "timstof-ms2":
        train_ds = load_timstof_ms2_dataset(
            tokenizer, split="train", max_length=args.max_seq_length
        )
        val_ds = load_timstof_ms2_dataset(
            tokenizer, split="validation", max_length=args.max_seq_length
        )
    elif args.dataset == "orbitrap-ms2":
        train_ds = load_orbitrap_ms2_dataset(
            tokenizer, split="train", max_length=args.max_seq_length
        )
        val_ds = load_orbitrap_ms2_dataset(
            tokenizer, split="validation", max_length=args.max_seq_length
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create data loaders
    train_loader = create_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    logger.info(f"Train batches: {len(train_loader) if hasattr(train_loader, '__len__') else 'streaming'}")
    logger.info(f"Val batches: {len(val_loader) if hasattr(val_loader, '__len__') else 'streaming'}")

    # Create or load model
    if args.pretrained:
        logger.info(f"Loading pretrained model: {args.pretrained}")
        model = UnifiedPeptideModel.from_pretrained(
            args.pretrained,
            map_location=device,
            tasks=["intensity"],
        )
    else:
        logger.info(f"Creating new model with config: {args.model_config}")
        model = UnifiedPeptideModel(
            vocab_size=tokenizer.vocab_size,
            encoder_config=args.model_config,
            tasks=["intensity"],
            max_seq_len=args.max_seq_length,
        )

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,} (trainable: {trainable_count:,})")

    # Freeze encoder if specified
    if args.freeze_encoder_epochs > 0:
        logger.info(f"Freezing encoder for first {args.freeze_encoder_epochs} epochs")
        model.freeze_encoder()

    # Training config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        use_amp=args.use_amp,
        checkpoint_dir=str(output_dir),
        device=device,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    if args.scheduler == "cosine":
        # Estimate total steps for cosine scheduler
        steps_per_epoch = len(train_loader) if hasattr(train_loader, '__len__') else 1000
        total_steps = args.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-7
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.scheduler_patience,
            min_lr=1e-7,
        )

    # Performance logger
    experiment_name = args.experiment_name or f"intensity_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    perf_logger = PerformanceLogger(
        log_dir=str(output_dir / "logs"),
        experiment_name=experiment_name,
        use_tensorboard=True,
        log_to_file=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=lambda batch: _intensity_loss(model, batch),
        config=config,
        perf_logger=perf_logger,
        task="intensity",
    )

    # Training loop with encoder unfreezing
    logger.info("Starting training...")

    if args.freeze_encoder_epochs > 0:
        # Phase 1: Train with frozen encoder
        logger.info(f"Phase 1: Training head only for {args.freeze_encoder_epochs} epochs")
        config.epochs = args.freeze_encoder_epochs
        trainer.train(train_loader, val_loader, epochs=args.freeze_encoder_epochs)

        # Phase 2: Unfreeze and continue
        logger.info("Phase 2: Unfreezing encoder and continuing training")
        model.unfreeze_encoder()

        # Reset optimizer with lower LR for encoder
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": args.learning_rate / 10},
            {"params": model.heads["intensity"].parameters(), "lr": args.learning_rate},
        ], weight_decay=args.weight_decay)

        trainer.optimizer = optimizer
        remaining_epochs = args.epochs - args.freeze_encoder_epochs
        trainer.train(train_loader, val_loader, epochs=remaining_epochs)
    else:
        trainer.train(train_loader, val_loader)

    # Save final model
    logger.info("Saving final model...")
    model.save_pretrained(str(output_dir / "best_model.pt"))

    # Save encoder separately for transfer learning
    encoder_path = output_dir / "encoder.pt"
    model.encoder.save_pretrained(str(encoder_path))
    logger.info(f"Saved encoder to {encoder_path}")

    # Final evaluation
    logger.info("Running final evaluation...")
    model.eval()
    val_loss, val_metrics = trainer.validate(val_loader, compute_metrics=True)
    logger.info(f"Final validation loss: {val_loss:.6f}")
    for name, value in val_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    logger.info("Training complete!")
    return 0


def _intensity_loss(model, batch):
    """Compute intensity loss for a batch using masked spectral angle."""
    from imspy_predictors.losses import masked_spectral_distance

    input_ids = batch["input_ids"]
    padding_mask = (batch["attention_mask"] == 0)
    charge = batch["charge"]
    ce = batch["collision_energy"]
    target = batch["intensity"]
    instrument = batch.get("instrument")

    pred = model.predict_intensity(
        input_ids, charge, ce,
        padding_mask=padding_mask,
        instrument=instrument,
    )

    # Masked spectral angle loss (dlomix-compatible)
    # Invalid positions in target are marked with -1
    loss = masked_spectral_distance(target, pred)
    return loss


if __name__ == "__main__":
    sys.exit(main())
