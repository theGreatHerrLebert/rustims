"""Evaluate a trained checkpoint on the held-out test split.

    python -m peptide_property_ng.eval.evaluate --checkpoint runs/prototype/best.pt
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from peptide_property_ng.data.collate import make_collate_fn
from peptide_property_ng.data.sage_dataset import build_split_datasets, discover_sage_dirs
from peptide_property_ng.eval.metrics import evaluate_split
from peptide_property_ng.model.config import PRESETS
from peptide_property_ng.model.multitask import UnifiedPeptidePropertyModel
from peptide_property_ng.modifications.composition import CompositionTable


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--datasets-glob", default="/scratch/claudius-proteomics/*")
    ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--max-datasets", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = PRESETS[ckpt.get("preset", "small")]
    model = UnifiedPeptidePropertyModel(cfg, CompositionTable.load()).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    sage_dirs = discover_sage_dirs(args.datasets_glob)
    if args.max_datasets:
        sage_dirs = sage_dirs[: args.max_datasets]
    splits = build_split_datasets(sage_dirs, cap=args.cap, seed=args.seed)
    loader = DataLoader(
        splits[args.split],
        batch_size=args.batch_size,
        collate_fn=make_collate_fn(cfg.pad_token_id, cfg.max_charge),
    )
    metrics = evaluate_split(model, loader, args.device)
    print(json.dumps({"split": args.split, "checkpoint": args.checkpoint, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
