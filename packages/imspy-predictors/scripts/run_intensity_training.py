#!/usr/bin/env python
"""
Train intensity prediction model on timsTOF MS2 dataset.

This is a standalone training script that validates the full pipeline.
"""

import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
import copy

import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from imspy_predictors.data_utils import load_timstof_ms2_dataset, create_dataloader
from imspy_predictors.utilities import ProformaTokenizer
from imspy_predictors.models.unified import UnifiedPeptideModel
from imspy_predictors.losses import masked_spectral_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, scaler, device, use_amp=True):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        charge = batch['charge'].to(device)
        ce = batch['collision_energy'].unsqueeze(-1).to(device)
        target = batch['intensity'].to(device)

        optimizer.zero_grad()

        # Forward with mixed precision
        with autocast(enabled=use_amp):
            pred = model.predict_intensity(
                input_ids, charge, ce,
                padding_mask=(attn_mask == 0)
            )
            loss = masked_spectral_distance(target, pred)

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, device, use_amp=True):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validating"):
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        charge = batch['charge'].to(device)
        ce = batch['collision_energy'].unsqueeze(-1).to(device)
        target = batch['intensity'].to(device)

        with autocast(enabled=use_amp):
            pred = model.predict_intensity(
                input_ids, charge, ce,
                padding_mask=(attn_mask == 0)
            )
            loss = masked_spectral_distance(target, pred)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/intensity")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = not args.no_amp and device == "cuda"

    logger.info(f"Device: {device}")
    logger.info(f"Mixed precision: {use_amp}")
    logger.info(f"Output: {output_dir}")

    # Initialize tokenizer
    tokenizer = ProformaTokenizer.with_defaults()
    logger.info(f"Tokenizer vocab: {tokenizer.vocab_size}")

    # Load datasets
    logger.info("Loading datasets...")
    train_ds = load_timstof_ms2_dataset(tokenizer, split='train', max_length=50)
    val_ds = load_timstof_ms2_dataset(tokenizer, split='val', max_length=50)

    train_loader = create_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, drop_last=True
    )
    val_loader = create_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4
    )

    logger.info(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_ds)} samples, {len(val_loader)} batches")

    # Create model
    model = UnifiedPeptideModel(
        vocab_size=tokenizer.vocab_size,
        encoder_config='base',
        tasks=['intensity'],
        max_seq_len=50,
    )
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-7
    )
    scaler = GradScaler(enabled=use_amp)

    # Training loop
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, use_amp)

        # Validate
        val_loss = validate(model, val_loader, device, use_amp)

        # Update scheduler
        scheduler.step()

        # Log
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, LR: {lr:.2e}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, output_dir / "best_model.pt")
            logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / f"checkpoint_epoch_{epoch + 1}.pt")

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Final evaluation
    logger.info("\n=== Final Evaluation ===")
    final_val_loss = validate(model, val_loader, device, use_amp)
    logger.info(f"Final val loss: {final_val_loss:.4f}")

    # Save final model
    model.save_pretrained(str(output_dir / "final_model.pt"))
    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")

    # GPU memory stats
    if device == "cuda":
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
