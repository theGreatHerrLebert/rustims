#!/usr/bin/env python
"""
Compare PyTorch intensity model with Koina Prosit 2023 timsTOF.

This script evaluates both models on the timsTOF test set and computes
spectral angle similarity against ground truth.

Usage:
    python scripts/compare_with_koina.py \
        --checkpoint ./checkpoints/timstof_intensity/best_model.pt \
        --n-samples 100

Requirements:
    pip install koinapy
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from koinapy import Koina
from tqdm import tqdm

from imspy_predictors.utilities import ProformaTokenizer
from imspy_predictors.models.unified import UnifiedPeptideModel

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_koina_annotation(ann):
    """
    Parse Koina ion annotation like 'y2+1' or b'y2+1'.

    Returns: (ion_type, position, charge)
        - ion_type: 'y' or 'b'
        - position: fragment number (1-indexed)
        - charge: ion charge state
    """
    if isinstance(ann, bytes):
        ann = ann.decode()
    # Format: {ion_type}{position}+{charge}
    # Example: y2+1 = y-ion at position 2 with charge +1
    ion_type = ann[0]  # 'y' or 'b'
    rest = ann[1:]     # e.g., '2+1'
    pos_str, charge_str = rest.split('+')
    return ion_type, int(pos_str), int(charge_str)


def koina_index_to_174(ion_type, position, charge):
    """
    Convert Koina ion annotation to 174-vector index.

    The 174-vector format is position-major:
    - 29 positions × 6 ion types per position
    - Per position: [y1+, y2+, y3+, b1+, b2+, b3+]
    - Index = (position - 1) * 6 + ion_offset

    Args:
        ion_type: 'y' or 'b'
        position: fragment number (1-29)
        charge: ion charge (1-3)

    Returns:
        Index in 174-vector, or None if invalid
    """
    if ion_type == 'y':
        ion_offset = charge - 1  # y1+ -> 0, y2+ -> 1, y3+ -> 2
    elif ion_type == 'b':
        ion_offset = 3 + (charge - 1)  # b1+ -> 3, b2+ -> 4, b3+ -> 5
    else:
        return None

    idx = (position - 1) * 6 + ion_offset
    if 0 <= idx < 174:
        return idx
    return None


def koina_result_to_174(result_df):
    """
    Convert Koina sparse output to dense 174-dimensional vector.

    Koina returns one row per ion in long format. This function maps
    each ion to the correct position in the 174-vector.

    Args:
        result_df: Koina result DataFrame with 'annotation' and 'intensities' columns

    Returns:
        numpy array of shape (174,)
    """
    vec = np.zeros(174)
    for i in range(len(result_df)):
        row = result_df.iloc[i]
        try:
            ion_type, position, charge = parse_koina_annotation(row['annotation'])
            idx = koina_index_to_174(ion_type, position, charge)
            if idx is not None:
                vec[idx] = row['intensities']
        except Exception:
            pass
    return vec


def spectral_angle_similarity(pred, target, mask=None):
    """
    Compute spectral angle similarity between two spectra.

    Args:
        pred: Predicted intensities
        target: Ground truth intensities
        mask: Optional boolean mask for valid positions (True = valid)

    Returns:
        Spectral angle similarity in range [0, 1]
        - 1.0 = identical spectra
        - 0.5 = 60 degree angle
        - 0.0 = orthogonal spectra
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    pred_norm = np.sqrt(np.sum(pred ** 2))
    target_norm = np.sqrt(np.sum(target ** 2))

    if pred_norm == 0 or target_norm == 0:
        return 0.0

    pred = pred / pred_norm
    target = target / target_norm

    cosine = np.clip(np.dot(pred, target), -1, 1)
    # Convert to spectral angle: SA = 1 - (2 * arccos(cos) / pi)
    return 1 - (2 * np.arccos(cosine) / np.pi)


def clean_sequence(seq):
    """Remove terminal brackets from sequence."""
    return seq.replace('[]-', '').replace('-[]', '')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PyTorch model checkpoint")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of test samples to evaluate")
    parser.add_argument("--koina-model", type=str, default="Prosit_2023_intensity_timsTOF",
                        help="Koina model name")
    parser.add_argument("--koina-server", type=str, default="koina.wilhelmlab.org:443",
                        help="Koina server URL")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional CSV output file for per-sample results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load tokenizer and model
    logger.info("Loading PyTorch model...")
    tokenizer = ProformaTokenizer.with_defaults()

    # Try to load with from_pretrained first (save_pretrained format)
    try:
        model = UnifiedPeptideModel.from_pretrained(
            args.checkpoint,
            map_location=device,
            tasks=['intensity'],
        )
        logger.info(f"Loaded model using from_pretrained")
    except Exception:
        # Fall back to direct state_dict loading
        model = UnifiedPeptideModel(
            vocab_size=tokenizer.vocab_size,
            encoder_config='base',
            tasks=['intensity'],
            max_seq_len=50,
        )
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded model using state_dict")

    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Initialize Koina
    logger.info(f"Connecting to Koina ({args.koina_model})...")
    koina = Koina(args.koina_model, server_url=args.koina_server)

    # Load test dataset
    logger.info("Loading timsTOF test dataset...")
    dataset = load_dataset("Wilhelmlab/timsTOF-ms2", split="test")
    logger.info(f"Test set size: {len(dataset)} samples")

    # Evaluate
    results = []
    pytorch_scores = []
    koina_scores = []

    logger.info(f"\nEvaluating on {args.n_samples} samples...")
    for i in tqdm(range(min(args.n_samples, len(dataset))), desc="Evaluating"):
        sample = dataset[i]

        # Get sequence and metadata
        seq = sample['modified_sequence']
        clean_seq = clean_sequence(seq)

        charge_onehot = sample['precursor_charge_onehot']
        charge_int = charge_onehot.index(1) + 1 if 1 in charge_onehot else 2
        ce = sample['collision_energy_aligned_normed']

        # Ground truth
        ground_truth = np.array(sample['intensities_raw'])
        mask = ground_truth >= 0  # Valid positions (not -1)

        # --- PyTorch prediction ---
        tokens = tokenizer.tokenize(clean_seq)
        encoded = tokenizer.encode(tokens)
        pad_len = 50 - len(encoded)
        attn_mask = [1] * len(encoded) + [0] * pad_len
        encoded = encoded + [tokenizer.pad_token_id] * pad_len

        input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
        attention_mask = torch.tensor([attn_mask], dtype=torch.long).to(device)
        charge_t = torch.tensor([charge_int], dtype=torch.long).to(device)
        ce_t = torch.tensor([[ce]], dtype=torch.float32).to(device)

        with torch.no_grad():
            pytorch_pred = model.predict_intensity(
                input_ids, charge_t, ce_t,
                padding_mask=(attention_mask == 0)
            ).cpu().numpy()[0]

        # --- Koina prediction ---
        koina_inputs = pd.DataFrame({
            'peptide_sequences': [clean_seq],
            'precursor_charges': [charge_int],
            'collision_energies': [ce * 100.0]  # Koina uses 0-100 scale
        })

        try:
            koina_result = koina.predict(koina_inputs)
            koina_pred = koina_result_to_174(koina_result)
        except Exception as e:
            logger.warning(f"Koina failed for sample {i}: {e}")
            koina_pred = np.zeros(174)

        # Compute spectral angles
        pytorch_sa = spectral_angle_similarity(pytorch_pred, ground_truth, mask)
        koina_sa = spectral_angle_similarity(koina_pred, ground_truth, mask)

        pytorch_scores.append(pytorch_sa)
        koina_scores.append(koina_sa)

        results.append({
            'index': i,
            'sequence': clean_seq,
            'length': len(clean_seq),
            'charge': charge_int,
            'pytorch_sa': pytorch_sa,
            'koina_sa': koina_sa,
            'diff': pytorch_sa - koina_sa,
        })

    # Summary statistics
    pytorch_mean = np.mean(pytorch_scores)
    pytorch_std = np.std(pytorch_scores)
    koina_mean = np.mean(koina_scores)
    koina_std = np.std(koina_scores)

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Samples evaluated: {len(pytorch_scores)}")
    logger.info("")
    logger.info(f"PyTorch model:  Mean SA = {pytorch_mean:.4f} +/- {pytorch_std:.4f}")
    logger.info(f"Koina Prosit:   Mean SA = {koina_mean:.4f} +/- {koina_std:.4f}")
    logger.info("")

    diff = pytorch_mean - koina_mean
    if diff > 0.01:
        logger.info(f"PyTorch is {diff:.4f} BETTER than Koina")
    elif diff < -0.01:
        logger.info(f"Koina is {-diff:.4f} BETTER than PyTorch")
    else:
        logger.info(f"Models are comparable (diff = {diff:.4f})")

    logger.info("")
    logger.info("Spectral Angle scale: 1.0 = perfect, 0.5 = 60 deg, 0.0 = orthogonal")
    logger.info("=" * 70)

    # Save detailed results
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        logger.info(f"\nDetailed results saved to {args.output}")

    # Show sample results
    logger.info("\nSample results:")
    for r in results[:10]:
        logger.info(f"  {r['sequence'][:15]:<15} ch={r['charge']} | "
                   f"PyTorch={r['pytorch_sa']:.4f} | Koina={r['koina_sa']:.4f}")


if __name__ == "__main__":
    main()
