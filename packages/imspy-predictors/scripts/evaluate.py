#!/usr/bin/env python
"""
Evaluate trained models on test/holdout sets.

Usage:
    # Evaluate CCS model
    python -m imspy_predictors.scripts.evaluate \
        --model ./checkpoints/ccs/best_model.pt \
        --task ccs \
        --dataset ionmob \
        --split test \
        --output ./results/ccs_test_metrics.json

    # Evaluate on holdout
    python -m imspy_predictors.scripts.evaluate \
        --model ./checkpoints/rt/best_model.pt \
        --task rt \
        --dataset prospect-ptms-irt \
        --split holdout \
        --output ./results/rt_holdout_metrics.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--task",
        choices=["ccs", "rt", "charge", "intensity"],
        required=True,
        help="Task to evaluate",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=50)
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-predictions", action="store_true", help="Save predictions to file")

    return parser.parse_args()


def load_dataset(dataset_name, tokenizer, split, max_length):
    """Load dataset by name."""
    from imspy_predictors.data_utils import (
        load_ionmob_dataset,
        load_prospect_rt_dataset,
        load_prospect_charge_dataset,
        load_prospect_ms2_dataset,
        load_timstof_ms2_dataset,
    )

    if dataset_name == "ionmob":
        return load_ionmob_dataset(tokenizer, split=split, max_length=max_length)
    elif dataset_name == "prospect-ptms-irt":
        return load_prospect_rt_dataset(tokenizer, split=split, max_length=max_length)
    elif dataset_name == "prospect-ptms-charge":
        return load_prospect_charge_dataset(tokenizer, split=split, max_length=max_length)
    elif dataset_name == "prospect-ptms-ms2":
        return load_prospect_ms2_dataset(tokenizer, split=split, max_length=max_length, streaming=False)
    elif dataset_name == "timstof-ms2":
        return load_timstof_ms2_dataset(tokenizer, split=split, max_length=max_length)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate_ccs(model, dataloader, device):
    """Evaluate CCS model."""
    from imspy_predictors.training import compute_regression_metrics

    model.eval()
    all_preds = []
    all_stds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            mean, std = model.predict_ccs(
                batch["input_ids"],
                batch["mz"],
                batch["charge"],
                padding_mask=(batch["attention_mask"] == 0),
                instrument=batch.get("instrument"),
            )
            all_preds.append(mean.cpu())
            all_stds.append(std.cpu())
            all_targets.append(batch["ccs"].cpu())

    preds = torch.cat(all_preds)
    stds = torch.cat(all_stds)
    targets = torch.cat(all_targets).view(-1, 1)

    metrics = compute_regression_metrics(preds, targets)

    # Add uncertainty metrics
    metrics["mean_predicted_std"] = float(stds.mean().item())
    metrics["std_predicted_std"] = float(stds.std().item())

    # Calibration: check if predicted std correlates with actual error
    errors = (preds - targets).abs()
    error_std_corr = np.corrcoef(errors.numpy().flatten(), stds.numpy().flatten())[0, 1]
    metrics["error_std_correlation"] = float(error_std_corr)

    return metrics, {"predictions": preds, "stds": stds, "targets": targets}


def evaluate_rt(model, dataloader, device):
    """Evaluate RT model."""
    from imspy_predictors.training import compute_regression_metrics

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model.predict_rt(
                batch["input_ids"],
                padding_mask=(batch["attention_mask"] == 0),
                instrument=batch.get("instrument"),
            )
            all_preds.append(pred.cpu())
            all_targets.append(batch["rt"].cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets).view(-1, 1)

    metrics = compute_regression_metrics(preds, targets)
    return metrics, {"predictions": preds, "targets": targets}


def evaluate_charge(model, dataloader, device):
    """Evaluate charge model."""
    from imspy_predictors.training import compute_classification_metrics

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model.predict_charge(
                batch["input_ids"],
                padding_mask=(batch["attention_mask"] == 0),
            )
            all_preds.append(pred.cpu())
            all_targets.append(batch["charge_dist"].cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    metrics = compute_classification_metrics(preds, targets)

    # Add per-class accuracy
    pred_classes = preds.argmax(dim=-1)
    target_classes = targets.argmax(dim=-1)
    for c in range(preds.shape[1]):
        mask = target_classes == c
        if mask.sum() > 0:
            class_acc = (pred_classes[mask] == c).float().mean().item()
            metrics[f"accuracy_charge_{c+1}"] = float(class_acc)

    return metrics, {"predictions": preds, "targets": targets}


def evaluate_intensity(model, dataloader, device):
    """Evaluate intensity model."""
    from imspy_predictors.training import compute_intensity_metrics

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model.predict_intensity(
                batch["input_ids"],
                batch["charge"],
                batch["collision_energy"],
                padding_mask=(batch["attention_mask"] == 0),
                instrument=batch.get("instrument"),
            )
            all_preds.append(pred.cpu())
            all_targets.append(batch["intensity"].cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    metrics = compute_intensity_metrics(preds, targets)

    # Add per-ion-type analysis
    # Prosit format: 29 positions × 6 ion types (b1, b2, b3, y1, y2, y3)
    num_positions = 29
    ion_types = ["b1", "b2", "b3", "y1", "y2", "y3"]

    for i, ion_type in enumerate(ion_types):
        start_idx = i * num_positions
        end_idx = (i + 1) * num_positions
        ion_preds = preds[:, start_idx:end_idx]
        ion_targets = targets[:, start_idx:end_idx]
        ion_metrics = compute_intensity_metrics(ion_preds, ion_targets)
        metrics[f"spectral_angle_{ion_type}"] = ion_metrics["spectral_angle"]

    return metrics, {"predictions": preds, "targets": targets}


def main():
    """Main evaluation function."""
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Import after parsing
    from imspy_predictors.models.unified import UnifiedPeptideModel
    from imspy_predictors.data_utils import create_dataloader
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer

    # Tokenizer
    tokenizer = ProformaTokenizer.with_defaults()

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = UnifiedPeptideModel.from_pretrained(
        args.model,
        map_location=device,
        tasks=[args.task],
    )
    model = model.to(device)
    model.eval()

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} ({args.split})")
    dataset = load_dataset(args.dataset, tokenizer, args.split, args.max_seq_length)
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    logger.info(f"Samples: {len(dataset)}")

    # Evaluate
    logger.info(f"Evaluating {args.task}...")
    if args.task == "ccs":
        metrics, data = evaluate_ccs(model, dataloader, device)
    elif args.task == "rt":
        metrics, data = evaluate_rt(model, dataloader, device)
    elif args.task == "charge":
        metrics, data = evaluate_charge(model, dataloader, device)
    elif args.task == "intensity":
        metrics, data = evaluate_intensity(model, dataloader, device)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Print metrics
    logger.info("Results:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # Save metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model,
        "task": args.task,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": len(dataset),
        "metrics": metrics,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {output_path}")

    # Save predictions
    if args.save_predictions:
        pred_path = output_path.with_suffix(".predictions.pt")
        torch.save({k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in data.items()}, pred_path)
        logger.info(f"Saved predictions to {pred_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
