"""Staged pretraining — warm the shared encoder + heads on large public corpora,
then hand off to ``train.py`` for the multi-task campaign fine-tune.

Default curriculum (each stage is single-task; the shared encoder accumulates):
  1. intensity  — Wilhelmlab/prospect-ptms-ms2  (Orbitrap-HCD, broad)
  2. intensity  — Wilhelmlab/timsTOF-ms2         (timsTOF — instrument domain)
  3. ccs        — ionmob CCS parquets            (timsTOF)
  4. rt         — Chronologer DB                 (harmonised retention time)

    python -m peptide_property_ng.train.pretrain --cap 50000 --epochs 3
    python -m peptide_property_ng.train.train --init-from runs/pretrain/pretrained.pt \\
        --datasets-glob '/scratch/claudius-proteomics/*'
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from peptide_property_ng.data.ccs_pretrain import prepare_ccs_examples
from peptide_property_ng.data.chronologer_rt import prepare_chronologer_examples
from peptide_property_ng.data.collate import make_collate_fn
from peptide_property_ng.data.hf_intensity import prepare_hf_intensity_examples
from peptide_property_ng.data.sage_dataset import SagePropertyDataset
from peptide_property_ng.data.splits import peptide_split
from peptide_property_ng.eval.metrics import evaluate_split
from peptide_property_ng.losses import MultiTaskLoss
from peptide_property_ng.model.config import PRESETS, instrument_id
from peptide_property_ng.model.multitask import UnifiedPeptidePropertyModel
from peptide_property_ng.modifications.composition import CompositionTable

_STAGE_METRIC = {"intensity": "intensity_sa", "ccs": "ccs_mae", "rt": "rt_mae"}


def _default_stages(
    chronologer_db: str | None = None, ccs_glob: str | None = None
) -> list[dict]:
    """Curriculum: each stage = name, task, and a builder taking a per-stage cap.

    The order is deliberate: the auxiliary tasks (RT, CCS) run first and
    intensity runs last, so the shared encoder ends tuned for the priority task
    rather than left drifted toward RT — a catastrophic-forgetting mitigation.
    Within intensity, the broad Orbitrap corpora precede the timsTOF one.
    """
    rt_kw = {} if chronologer_db is None else {"db_path": chronologer_db}
    ccs_kw = {} if ccs_glob is None else {"data_glob": ccs_glob}
    return [
        {
            "name": "rt-chronologer", "task": "rt",
            "build": lambda cap: prepare_chronologer_examples(cap=cap, **rt_kw),
        },
        {
            "name": "ccs-ionmob", "task": "ccs",
            "build": lambda cap: prepare_ccs_examples(
                cap=cap, instrument=instrument_id("timsTOF Pro"), **ccs_kw),
        },
        {
            "name": "intensity-prospect", "task": "intensity",
            "build": lambda cap: prepare_hf_intensity_examples(
                "Wilhelmlab/prospect-ptms-ms2", split="train", cap=cap,
                instrument=instrument_id("orbitrap")),
        },
        {
            "name": "intensity-prosit2025", "task": "intensity",
            "build": lambda cap: prepare_hf_intensity_examples(
                "Wilhelmlab/Prosit-2025-lac-ms2", split="train", cap=cap,
                instrument=instrument_id("orbitrap")),
        },
        {
            "name": "intensity-timstof", "task": "intensity",
            "build": lambda cap: prepare_hf_intensity_examples(
                "Wilhelmlab/timsTOF-ms2", split="train", cap=cap,
                instrument=instrument_id("timsTOF Pro")),
        },
    ]


def _train_task_epoch(model, loader, task, loss_fn, optimizer, device) -> float:
    model.train()
    total, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        loss, _ = loss_fn(model(batch, tasks=[task]), batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach())
        n_batches += 1
    return total / max(n_batches, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Staged pretraining for the unified model.")
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--intensity-head", default=None,
                    choices=["site", "pooled"],
                    help="intensity head topology (default: preset value); "
                         "use 'pooled' to pretrain the v4-style Prosit-174 head")
    ap.add_argument("--cap", type=int, default=50000, help="max examples per stage (0 = all)")
    ap.add_argument("--epochs", type=int, default=3, help="epochs per stage")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stages", default="all", help="comma-separated stage names, or 'all'")
    ap.add_argument("--chronologer-db", default=None,
                    help="path to Chronologer_DB_*.gz (default: the package's known local path)")
    ap.add_argument("--ccs-glob", default=None,
                    help="glob for ionmob *_unique_unimod*.parquet (default: known local path)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/pretrain")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PRESETS[args.preset]
    if args.intensity_head:
        cfg = dataclasses.replace(cfg, intensity_head=args.intensity_head)
    device = args.device
    cap = None if args.cap == 0 else args.cap

    model = UnifiedPeptidePropertyModel(cfg, CompositionTable.load()).to(device)
    loss_fn = MultiTaskLoss()
    collate = make_collate_fn(cfg.pad_token_id, cfg.max_charge)
    print(f"model: '{args.preset}' preset, {model.num_parameters():,} parameters, device={device}")

    stages = _default_stages(args.chronologer_db, args.ccs_glob)
    if args.stages != "all":
        wanted = {s.strip() for s in args.stages.split(",")}
        stages = [s for s in stages if s["name"] in wanted]

    history: list[dict] = []
    for stage in stages:
        name, task = stage["name"], stage["task"]
        t0 = time.time()
        print(f"\n=== stage '{name}' (task={task}) ===", flush=True)
        examples = stage["build"](cap)
        if len(examples) < args.batch_size * 2:
            print(f"  only {len(examples)} examples — skipping", flush=True)
            continue

        # peptide-level hold-out — a peptide never crosses train/val within a
        # stage (stage metrics are pretraining diagnostics, not the final eval).
        train_ex, val_ex = [], []
        for ex in examples:
            bucket = peptide_split(ex["stripped"], val_frac=args.val_frac,
                                   test_frac=0.0, seed=args.seed)
            (val_ex if bucket == "val" else train_ex).append(ex)
        print(f"  {len(train_ex):,} train / {len(val_ex):,} val  [prepared in {time.time()-t0:.0f}s]",
              flush=True)

        train_loader = DataLoader(SagePropertyDataset(train_ex), batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate, drop_last=True)
        val_loader = DataLoader(SagePropertyDataset(val_ex), batch_size=args.batch_size,
                                collate_fn=collate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        metric = _STAGE_METRIC[task]
        for epoch in range(1, args.epochs + 1):
            te = time.time()
            train_loss = _train_task_epoch(model, train_loader, task, loss_fn, optimizer, device)
            # Restrict eval to the current stage's task: other heads' outputs are
            # at shapes the current corpus's targets/placeholders don't share
            # (eg. pooled-head intensity (B,29,6) vs Chronologer's (B,L-1,6) for
            # L up to max_seq_len), and per-stage metrics are diagnostic anyway.
            val = evaluate_split(model, val_loader, device, tasks=[task])
            print(f"  epoch {epoch}: train_loss={train_loss:.4f}  "
                  f"val[{metric}]={val[metric]:.4f}  [{time.time()-te:.0f}s]", flush=True)
            history.append({"stage": name, "epoch": epoch,
                             "train_loss": train_loss, "val": val})

        # Save a per-stage checkpoint (so the campaign fine-tune can pick the
        # handoff point, not just the post-RT state) and the running pretrained.pt.
        ckpt = {
            "model_state_dict": model.state_dict(),
            "preset": args.preset,
            "intensity_head": cfg.intensity_head,
            "stage": name,
        }
        torch.save(ckpt, out_dir / f"after-{name}.pt")
        torch.save(ckpt, out_dir / "pretrained.pt")
        print(f"  saved -> {out_dir}/after-{name}.pt  (+ pretrained.pt)", flush=True)

    (out_dir / "pretrain_history.json").write_text(json.dumps(history, indent=2))
    print(
        f"\npretraining complete. fine-tune on the campaign data with:\n"
        f"  python -m peptide_property_ng.train.train "
        f"--init-from {out_dir}/pretrained.pt --datasets-glob '/scratch/claudius-proteomics/*'"
    )


if __name__ == "__main__":
    main()
