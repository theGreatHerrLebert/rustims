#!/usr/bin/env python
"""
Pre-train intensity prediction model on PROSPECT MS2 dataset (31M samples).

This script pre-trains the encoder on the large PROSPECT dataset using streaming
to handle the dataset size. After pre-training, fine-tune on timsTOF-ms2.

Usage:
    python scripts/pretrain_prospect.py --batch-size 256 --epochs 3 --output-dir ./checkpoints/pretrain
"""

import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
import copy
import math

import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from imspy_predictors.utilities import ProformaTokenizer
from imspy_predictors.models.unified import UnifiedPeptideModel
from imspy_predictors.losses import masked_spectral_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StreamingProspectDataset(IterableDataset):
    """Streaming dataset wrapper for PROSPECT MS2."""

    def __init__(self, tokenizer, split='train', max_length=50):
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self._dataset = None

    def _load_dataset(self):
        """Lazy load the dataset."""
        if self._dataset is None:
            from datasets import load_dataset
            self._dataset = load_dataset(
                "Wilhelmlab/prospect-ptms-ms2",
                split=self.split,
                streaming=True,
            )
        return self._dataset

    def __iter__(self):
        dataset = self._load_dataset()

        for item in dataset:
            # Tokenize sequence
            sequence = item['modified_sequence']
            tokens = self.tokenizer.tokenize(sequence)

            # Truncate if needed
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length - 1] + ["[SEP]"]

            # Encode
            encoded = self.tokenizer.encode(tokens)

            # Pad
            pad_len = self.max_length - len(encoded)
            mask = [1] * len(encoded) + [0] * pad_len
            encoded = encoded + [self.tokenizer.pad_token_id] * pad_len

            # Get charge from one-hot
            charge_onehot = item.get('precursor_charge_onehot', [0, 1, 0, 0, 0, 0])
            charge_int = charge_onehot.index(1) + 1 if 1 in charge_onehot else 2

            # Get collision energy
            ce = item.get('collision_energy_aligned_normed', 0.3)

            # Get intensities
            intensities = item.get('intensities_raw', [0.0] * 174)

            yield {
                'input_ids': torch.tensor(encoded, dtype=torch.long),
                'attention_mask': torch.tensor(mask, dtype=torch.long),
                'charge': torch.tensor(charge_int, dtype=torch.long),
                'collision_energy': torch.tensor(ce, dtype=torch.float32),
                'intensity': torch.tensor(intensities, dtype=torch.float32),
            }


def collate_streaming(batch):
    """Collate function for streaming dataset."""
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'charge': torch.stack([x['charge'] for x in batch]),
        'collision_energy': torch.stack([x['collision_energy'] for x in batch]),
        'intensity': torch.stack([x['intensity'] for x in batch]),
    }


def train_epoch_streaming(model, train_loader, optimizer, scaler, scheduler, device,
                          use_amp=True, max_steps=None, log_interval=100):
    """Train one epoch with streaming data."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", total=max_steps)
    for batch_idx, batch in enumerate(pbar):
        if max_steps and batch_idx >= max_steps:
            break

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
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % log_interval == 0:
            avg_loss = total_loss / num_batches
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_streaming(model, val_loader, device, use_amp=True, max_steps=500):
    """Validate with streaming data (limited steps)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating", total=max_steps)):
        if batch_idx >= max_steps:
            break

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

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (each ~83k steps at batch 256)")
    parser.add_argument("--steps-per-epoch", type=int, default=50000, help="Training steps per epoch")
    parser.add_argument("--val-steps", type=int, default=500, help="Validation steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/pretrain_prospect")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = not args.no_amp and device == "cuda"

    logger.info(f"Device: {device}")
    logger.info(f"Mixed precision: {use_amp}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Steps per epoch: {args.steps_per_epoch}")

    # Initialize tokenizer
    tokenizer = ProformaTokenizer.with_defaults()
    logger.info(f"Tokenizer vocab: {tokenizer.vocab_size}")

    # Create streaming datasets
    logger.info("Creating streaming datasets...")
    train_ds = StreamingProspectDataset(tokenizer, split='train', max_length=50)
    val_ds = StreamingProspectDataset(tokenizer, split='val', max_length=50)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_streaming,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_streaming,
        pin_memory=True,
    )

    # Create model
    model = UnifiedPeptideModel(
        vocab_size=tokenizer.vocab_size,
        encoder_config='base',
        tasks=['intensity'],
        max_seq_len=50,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Optimizer with warmup + cosine decay
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps = args.epochs * args.steps_per_epoch

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=use_amp)

    # Training loop
    best_val_loss = float('inf')

    logger.info(f"Starting pre-training for {args.epochs} epochs ({total_steps:,} total steps)...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # Train
        train_loss = train_epoch_streaming(
            model, train_loader, optimizer, scaler, scheduler, device,
            use_amp=use_amp, max_steps=args.steps_per_epoch
        )

        # Validate
        val_loss = validate_streaming(model, val_loader, device, use_amp=use_amp, max_steps=args.val_steps)

        # Log
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch + 1}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}, LR={lr:.2e}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
        }

        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1}.pt")
        logger.info(f"Saved checkpoint for epoch {epoch + 1}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / "best_model.pt")
            logger.info(f"New best model! (val_loss: {val_loss:.4f})")

    # Save final model
    logger.info("\n=== Pre-training Complete ===")
    model.save_pretrained(str(output_dir / "pretrained_encoder.pt"))
    logger.info(f"Saved pretrained encoder to {output_dir / 'pretrained_encoder.pt'}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    # GPU stats
    if device == "cuda":
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
