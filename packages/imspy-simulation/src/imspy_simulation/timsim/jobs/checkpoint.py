"""Checkpoint utilities for the timsim simulation pipeline.

Saves and loads intermediate DataFrames as Parquet files so that a failed
simulation can resume from the last completed stage instead of restarting
from scratch.

Checkpoint stages (in pipeline order):
    "proteome" — after data source loading + RT prediction (before frame distributions)
    "ions"     — after all ion properties + DDA selection (before fragment intensities)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = ".checkpoints"
MANIFEST_FILE = "manifest.json"

# Pipeline stages in execution order
STAGES = ("proteome", "ions")


def _checkpoint_dir(save_path: str) -> Path:
    return Path(save_path) / CHECKPOINT_DIR


def save(
    save_path: str,
    stage: str,
    dataframes: Dict[str, pd.DataFrame],
) -> None:
    """Persist DataFrames for a pipeline stage as Parquet files.

    Args:
        save_path: Base output directory for the simulation.
        stage: Pipeline stage name (must be in STAGES).
        dataframes: Mapping of name -> DataFrame to save.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown checkpoint stage '{stage}', expected one of {STAGES}")

    cp_dir = _checkpoint_dir(save_path)
    cp_dir.mkdir(parents=True, exist_ok=True)

    for name, df in dataframes.items():
        path = cp_dir / f"{stage}_{name}.parquet"
        df.to_parquet(path, index=False)

    # Update manifest
    manifest = _load_manifest(save_path)
    manifest[stage] = sorted(dataframes.keys())
    _save_manifest(save_path, manifest)

    logger.info(f"  Checkpoint saved: {stage} ({', '.join(sorted(dataframes.keys()))})")


def load(save_path: str, stage: str) -> Dict[str, pd.DataFrame]:
    """Load DataFrames from a checkpoint stage.

    Args:
        save_path: Base output directory for the simulation.
        stage: Pipeline stage name to load.

    Returns:
        Mapping of name -> DataFrame.

    Raises:
        FileNotFoundError: If no checkpoint exists for the given stage.
    """
    manifest = _load_manifest(save_path)
    if stage not in manifest:
        raise FileNotFoundError(f"No checkpoint found for stage '{stage}'")

    cp_dir = _checkpoint_dir(save_path)
    result = {}
    for name in manifest[stage]:
        path = cp_dir / f"{stage}_{name}.parquet"
        result[name] = pd.read_parquet(path)

    logger.info(f"  Loaded checkpoint: {stage} ({', '.join(sorted(result.keys()))})")
    return result


def latest(save_path: str) -> Optional[str]:
    """Return the most advanced checkpoint stage, or None if no checkpoints exist."""
    manifest = _load_manifest(save_path)
    result = None
    for stage in STAGES:
        if stage in manifest:
            result = stage
    return result


def clear(save_path: str) -> None:
    """Remove all checkpoint files."""
    import shutil

    cp_dir = _checkpoint_dir(save_path)
    if cp_dir.exists():
        shutil.rmtree(cp_dir)
        logger.info("  Cleared all checkpoints")


def _load_manifest(save_path: str) -> dict:
    path = _checkpoint_dir(save_path) / MANIFEST_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_manifest(save_path: str, manifest: dict) -> None:
    path = _checkpoint_dir(save_path) / MANIFEST_FILE
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
