"""Train the unified peptide property model — first prototype.

Example:
    python -m peptide_property_ng.train.train \\
        --datasets-glob '/scratch/claudius-proteomics/*' --cap 4000 --epochs 12
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from peptide_property_ng.data.collate import make_collate_fn
from peptide_property_ng.data.sage_dataset import build_split_datasets, discover_sage_dirs
from peptide_property_ng.eval.metrics import evaluate_split
from peptide_property_ng.losses import MultiTaskLoss
from peptide_property_ng.model.config import PRESETS
from peptide_property_ng.model.multitask import UnifiedPeptidePropertyModel
from peptide_property_ng.modifications.composition import CompositionTable


def _to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(model, loader, loss_fn, optimizer, device) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad()
        loss, parts = loss_fn(model(batch), batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + value
        totals["_total"] = totals.get("_total", 0.0) + float(loss.detach())
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the unified peptide property model.")
    ap.add_argument("--datasets-glob", default="/scratch/claudius-proteomics/*")
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--comp-fusion", default=None,
                    choices=["add", "gate", "token_only", "composition_only"],
                    help="modification-encoding fusion / ablation (default: preset value)")
    ap.add_argument("--cap", type=int, default=4000, help="max PSMs sampled per dataset")
    ap.add_argument("--max-datasets", type=int, default=0, help="limit datasets (0 = all)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/prototype")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] discovering datasets ...", flush=True)
    sage_dirs = discover_sage_dirs(args.datasets_glob)
    if args.max_datasets:
        sage_dirs = sage_dirs[: args.max_datasets]
    print(f"  {len(sage_dirs)} datasets")

    print(f"[{time.strftime('%H:%M:%S')}] preparing examples (cap {args.cap}/dataset) ...", flush=True)
    t0 = time.time()
    splits = build_split_datasets(sage_dirs, cap=args.cap, seed=args.seed)
    for name, ds in splits.items():
        print(f"  {name}: {len(ds):,} examples")
    print(f"  prepared in {time.time() - t0:.0f}s")
    if len(splits["train"]) == 0:
        raise SystemExit("no training examples — check --datasets-glob")

    cfg = PRESETS[args.preset]
    if args.comp_fusion:
        cfg = dataclasses.replace(cfg, comp_fusion=args.comp_fusion)
    collate = make_collate_fn(cfg.pad_token_id, cfg.max_charge)
    loaders = {
        name: DataLoader(
            ds, batch_size=args.batch_size, shuffle=(name == "train"),
            collate_fn=collate, drop_last=(name == "train"),
        )
        for name, ds in splits.items()
    }

    device = args.device
    model = UnifiedPeptidePropertyModel(cfg, CompositionTable.load()).to(device)
    print(f"model: '{args.preset}' preset, {model.num_parameters():,} parameters, device={device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = MultiTaskLoss()

    best_sa, best_epoch, bad = -1.0, -1, 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, loaders["train"], loss_fn, optimizer, device)
        val = evaluate_split(model, loaders["val"], device)
        history.append({"epoch": epoch, "train": tr, "val": val})
        print(
            f"[{time.strftime('%H:%M:%S')}] epoch {epoch:2d}  "
            f"train_loss={tr.get('_total', 0):.4f} "
            f"(int={tr.get('intensity', 0):.3f} ccs={tr.get('ccs', 0):.3f} "
            f"rt={tr.get('rt', 0):.3f} chg={tr.get('charge', 0):.3f})  "
            f"val: SA={val['intensity_sa']:.4f} ccs_mae={val['ccs_mae']:.4f} "
            f"rt_mae={val['rt_mae']:.4f} chg_acc={val['charge_acc']:.3f}  "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
        if val["intensity_sa"] > best_sa:
            best_sa, best_epoch, bad = val["intensity_sa"], epoch, 0
            torch.save(
                {"model_state_dict": model.state_dict(), "preset": args.preset,
                 "epoch": epoch, "val": val},
                out_dir / "best.pt",
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop (no val SA gain for {args.patience} epochs)")
                break

    if best_epoch == -1:
        print("warning: no epoch improved val SA; saving the final model as best.pt")
        torch.save(
            {"model_state_dict": model.state_dict(), "preset": args.preset,
             "epoch": epoch, "val": val},
            out_dir / "best.pt",
        )
        best_epoch = epoch
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test = evaluate_split(model, loaders["test"], device)
    print(f"\n=== test (best epoch {best_epoch}) ===")
    for k, v in test.items():
        print(f"  {k:16s} {v:.4f}")
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {"args": vars(args), "best_epoch": best_epoch, "history": history, "test": test},
            indent=2,
        )
    )
    print(f"\nsaved -> {out_dir}/best.pt , {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
