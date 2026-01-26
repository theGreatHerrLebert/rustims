#!/usr/bin/env python
"""
Fine-tune charge state prediction head.

This script fine-tunes the charge state prediction head using a pre-trained
encoder from intensity pre-training.

Usage:
    python -m imspy_predictors.scripts.train_charge \
        --pretrained-encoder ./checkpoints/pretrain_intensity/encoder.pt \
        --batch-size 256 \
        --epochs 60 \
        --output-dir ./checkpoints/charge
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune charge state prediction head",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="prospect-ptms-charge",
        help="Dataset to train on",
    )

    # Model
    parser.add_argument("--pretrained-encoder", type=str, default=None)
    parser.add_argument("--pretrained-model", type=str, default=None)
    parser.add_argument("--model-config", default="base", choices=["small", "base", "large"])

    # Training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Scheduler
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)

    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=10)

    # Encoder freezing
    parser.add_argument("--freeze-encoder-epochs", type=int, default=3)

    # Data loading
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=50)

    # Output
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)

    # Device
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-amp", action="store_true")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Import after parsing
    from imspy_predictors.models.unified import UnifiedPeptideModel
    from imspy_predictors.data_utils import load_prospect_charge_dataset, create_dataloader
    from imspy_predictors.training import (
        TrainingConfig,
        PerformanceLogger,
        Trainer,
        compute_classification_metrics,
    )
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer

    # Tokenizer
    tokenizer = ProformaTokenizer.with_defaults()

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    train_ds = load_prospect_charge_dataset(tokenizer, split="train", max_length=args.max_seq_length)
    val_ds = load_prospect_charge_dataset(tokenizer, split="validation", max_length=args.max_seq_length)
    test_ds = load_prospect_charge_dataset(tokenizer, split="test", max_length=args.max_seq_length)

    train_loader = create_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = create_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = create_dataloader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Val samples: {len(val_ds)}")
    logger.info(f"Test samples: {len(test_ds)}")

    # Create model
    if args.pretrained_model:
        model = UnifiedPeptideModel.from_pretrained(
            args.pretrained_model, map_location=device, tasks=["charge"]
        )
    elif args.pretrained_encoder:
        model = UnifiedPeptideModel.from_pretrained_encoder(
            args.pretrained_encoder, tasks=["charge"]
        )
    else:
        model = UnifiedPeptideModel(
            vocab_size=tokenizer.vocab_size,
            encoder_config=args.model_config,
            tasks=["charge"],
            max_seq_len=args.max_seq_length,
        )

    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Logger
    experiment_name = args.experiment_name or f"charge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    perf_logger = PerformanceLogger(
        log_dir=str(output_dir / "logs"),
        experiment_name=experiment_name,
    )

    # Phase 1: Frozen encoder
    if args.freeze_encoder_epochs > 0 and (args.pretrained_encoder or args.pretrained_model):
        logger.info(f"Phase 1: Training head only for {args.freeze_encoder_epochs} epochs")
        model.freeze_encoder()

        config_phase1 = TrainingConfig(
            epochs=args.freeze_encoder_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.freeze_encoder_epochs + 1,
            checkpoint_dir=str(output_dir / "phase1"),
            device=device,
            use_amp=args.use_amp,
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )

        trainer = Trainer(
            model=model, optimizer=optimizer, scheduler=scheduler,
            loss_fn=lambda batch: _charge_loss(model, batch),
            config=config_phase1, perf_logger=perf_logger, task="charge",
        )
        trainer.train(train_loader, val_loader)

    # Phase 2: Fine-tune
    remaining_epochs = args.epochs - args.freeze_encoder_epochs
    if remaining_epochs > 0:
        logger.info(f"Phase 2: Fine-tuning for {remaining_epochs} epochs")
        model.unfreeze_encoder()

        config_phase2 = TrainingConfig(
            epochs=remaining_epochs,
            batch_size=args.batch_size,
            learning_rate=args.encoder_lr,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_dir=str(output_dir),
            device=device,
            use_amp=args.use_amp,
        )

        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.heads["charge"].parameters(), "lr": args.learning_rate},
        ], weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )

        trainer = Trainer(
            model=model, optimizer=optimizer, scheduler=scheduler,
            loss_fn=lambda batch: _charge_loss(model, batch),
            config=config_phase2, perf_logger=perf_logger, task="charge",
        )
        trainer.train(train_loader, val_loader)

    # Save
    model.save_pretrained(str(output_dir / "best_model.pt"))

    # Evaluate
    logger.info("Evaluating on test set...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model.predict_charge(
                batch["input_ids"],
                padding_mask=(batch["attention_mask"] == 0),
            )
            all_preds.append(pred.cpu())
            all_targets.append(batch["charge_dist"].cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_classification_metrics(all_preds, all_targets)

    logger.info("Test set metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    import json
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete!")
    return 0


def _charge_loss(model, batch):
    """Compute charge loss."""
    input_ids = batch["input_ids"]
    padding_mask = (batch["attention_mask"] == 0)
    target = batch["charge_dist"]

    pred = model.predict_charge(input_ids, padding_mask=padding_mask)
    return torch.nn.functional.cross_entropy(pred, target)


if __name__ == "__main__":
    sys.exit(main())
