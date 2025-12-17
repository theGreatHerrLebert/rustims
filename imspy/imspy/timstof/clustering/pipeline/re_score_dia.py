#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSM re-scoring pipeline (TOML-configured)

Stages:
1) Load PSM binaries (features + clusters) -> merged PSM collection
2) Predict: Prosit fragment intensities, RT, inverse ion mobility (optional refine)
3) (Optional) overwrite base score (e.g. hyperscore) with custom beta score
4) Re-score with ML (k-fold CV) -> re_score
5) Target-decoy competition -> q-values
6) Save:
   - rescored PSMs as pandas parquet/csv (optional)
   - summary stats as json (optional)
   - (optional) binary PSM dump if you have a serializer available
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# ------------------------- EARLY ENV CONFIG -----------------------------------
def apply_env(cfg: dict) -> None:
    """
    Must run before importing torch / tensorflow.
    - force a specific GPU (e.g. CUDA_VISIBLE_DEVICES=1) so TF never sees bad GPU0
    - quiet TF logs
    """
    run = cfg.get("run", {}) or {}
    tfc = cfg.get("tensorflow", {}) or {}

    cuda_visible = run.get("cuda_visible_devices", None)
    if cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", str(tfc.get("tf_cpp_min_log_level", "2")))

    # Optional: fully disable TF GPU (strongest)
    if bool(tfc.get("force_cpu", False)):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ------------------------- TOML LOADING ---------------------------------------
try:
    import tomllib as toml
except Exception:
    import tomli as toml  # type: ignore


# --------------------------- logging ------------------------------------------
_LOGGER_NAME = "t-tracer"
_logger = logging.getLogger(_LOGGER_NAME)

@contextmanager
def log_timing(label: str, *, sync_cuda: bool = True):
    _logger.info(f"[timing] {label}: start")
    if sync_cuda:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
        dt = time.perf_counter() - t0
        _logger.info(f"[timing] {label}: done in {dt:0.3f} s")


def setup_logging(
    log_file: str | os.PathLike | None,
    level: str = "INFO",
    also_console: bool = True,
    rotate_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(process)d | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        log_path = Path(log_file)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_path, maxBytes=rotate_bytes, backupCount=backup_count)
        fh.setFormatter(fmt)
        fh.setLevel(logger.level)
        logger.addHandler(fh)

    if also_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(logger.level)
        logger.addHandler(ch)

    def _excepthook(exc_type, exc, tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook


def log(msg: str, level: int = logging.INFO) -> None:
    _logger.log(level, msg)


def ensure_dir(p: str | os.PathLike) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def ensure_dir_for_file(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def cuda_gc() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ------------------------- TF GPU MEMORY --------------------------------------
def configure_tensorflow(cfg: dict) -> None:
    """
    Configure TF AFTER env has been applied, BEFORE any predictor imports TF.
    """
    tfc = cfg.get("tensorflow", {}) or {}
    if bool(tfc.get("disable", False)):
        log("[tf] disabled by config (will not import tensorflow)")
        return

    # If TF not installed or not needed, this is harmless.
    try:
        import tensorflow as tf
    except Exception as e:
        log(f"[tf] import failed (ignored): {e}", logging.WARNING)
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        log("[tf] no GPU visible")
        return

    # After CUDA_VISIBLE_DEVICES is set, TF sees only what you allowed.
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        if bool(tfc.get("memory_growth", False)):
            tf.config.experimental.set_memory_growth(gpus[0], True)
            log("[tf] enabled memory growth")
        else:
            limit_gb = tfc.get("memory_limit_gb", None)
            if limit_gb is not None:
                limit_mb = int(float(limit_gb) * 1024)
                tf.config.set_logical_device_configuration(
                    gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)]
                )
                log(f"[tf] set memory limit to {limit_gb} GB")
    except RuntimeError as e:
        log(f"[tf] GPU config too late (already initialized): {e}", logging.WARNING)


# ------------------------- scoring helpers ------------------------------------
import numpy as np
import numba

@numba.njit
def _log_factorial(n: int, k: int) -> float:
    k = max(k, 2)
    result = 0.0
    for i in range(n, k - 1, -1):
        result += np.log(i)
    return result


def beta_score_with_matching(frag_obs, frag_pred) -> float:
    """
    Match fragments by (ion_type, charge, ordinal) and compute your beta-score-like formula.

    frag_obs: p.sage_feature.fragments
    frag_pred: p.prosit_intensities_to_fragments()
    """
    from sagepy.core import IonType

    pred_index = {}
    for ch, ion_type, ord_, inten in zip(
        frag_pred.charges, frag_pred.ion_types, frag_pred.fragment_ordinals, frag_pred.intensities
    ):
        pred_index[(ion_type, ch, ord_)] = float(inten)

    obs_int = []
    pred_int = []
    len_b = 0
    len_y = 0

    b_type = IonType("b")
    y_type = IonType("y")

    for ch, ion_type, ord_, inten in zip(
        frag_obs.charges, frag_obs.ion_types, frag_obs.fragment_ordinals, frag_obs.intensities
    ):
        key = (ion_type, ch, ord_)
        if key not in pred_index:
            continue
        obs_int.append(float(inten))
        pred_int.append(pred_index[key])

        if ion_type == b_type:
            len_b += 1
        elif ion_type == y_type:
            len_y += 1

    if len(obs_int) == 0:
        return 0.0

    obs_arr = np.asarray(obs_int, dtype=float)
    pred_arr = np.asarray(pred_int, dtype=float)
    intensity = float(np.dot(obs_arr, pred_arr))

    i_min = min(len_b, len_y)
    i_max = max(len_b, len_y)

    return float(np.log1p(intensity) + 2.0 * _log_factorial(int(i_min), 2) + _log_factorial(int(i_max), int(i_min) + 1))


# ------------------------- core pipeline --------------------------------------
def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        cfg = toml.load(f)

    for section in ("run", "input", "output"):
        if section not in cfg:
            raise ValueError(f"Missing [{section}] in config.")

    # run defaults
    r = cfg["run"]
    r.setdefault("cuda_visible_devices", "1")  # your safe default
    r.setdefault("device", "cuda")  # for downstream libs that check this
    r.setdefault("num_threads", 0)

    # tensorflow defaults
    cfg.setdefault("tensorflow", {})
    tfc = cfg["tensorflow"]
    tfc.setdefault("disable", False)
    tfc.setdefault("force_cpu", False)
    tfc.setdefault("tf_cpp_min_log_level", "2")
    tfc.setdefault("memory_limit_gb", 8)
    tfc.setdefault("memory_growth", False)

    # predictors defaults
    cfg.setdefault("predict", {})
    p = cfg["predict"]
    p.setdefault("do_prosit", True)
    p.setdefault("do_rt", True)
    p.setdefault("do_im", True)
    p.setdefault("refine_rt_model", True)
    p.setdefault("refine_im_model", True)
    p.setdefault("verbose", False)

    # score defaults
    cfg.setdefault("score", {})
    s = cfg["score"]
    s.setdefault("overwrite_hyperscore_with_beta", False)
    s.setdefault("base_score_column", "hyperscore")  # if you don’t overwrite

    # rescore defaults
    cfg.setdefault("rescore", {})
    rc = cfg["rescore"]
    rc.setdefault("enabled", True)
    rc.setdefault("num_splits", 5)
    rc.setdefault("model", "xgboost")  # "xgboost" | "svm" | "lda"
    rc.setdefault("xgb_params", {})    # pass through to XGBClassifier

    # tdc defaults
    cfg.setdefault("tdc", {})
    t = cfg["tdc"]
    t.setdefault("method", "peptide_psm_peptide")
    t.setdefault("score", "re_score")  # after rescoring; or base score
    t.setdefault("q_value", 0.01)

    # outputs defaults
    out = cfg["output"]
    out.setdefault("dir", "./rescore_out")
    out.setdefault("write_parquet", True)
    out.setdefault("write_csv", False)
    out.setdefault("parquet_name", "psms_rescored.parquet")
    out.setdefault("csv_name", "psms_rescored.csv")
    out.setdefault("summary_name", "summary.json")

    return cfg


def _default_log_path_from_cfg(cfg: dict) -> Path:
    out_dir = cfg["output"]["dir"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(out_dir) / "logs" / f"timsim_rescore_{ts}.log"


def load_psms(cfg: dict):
    from sagepy.utility import decompress_psms

    inp = cfg["input"]
    feat_path = inp.get("psms_features_bin")
    prec_path = inp.get("psms_clusters_bin")

    if not feat_path or not prec_path:
        raise ValueError("Need [input].psms_features_bin and [input].psms_clusters_bin")

    with open(feat_path, "rb") as f:
        features_data = f.read()
    with open(prec_path, "rb") as f:
        precursors_data = f.read()

    psms_features = decompress_psms(features_data)
    psms_precursors = decompress_psms(precursors_data)

    psms = [*psms_features, *psms_precursors]
    log(f"[input] loaded PSMs: features={len(psms_features)} precursors={len(psms_precursors)} total={len(psms)}")
    return psms


def run_predictions(cfg: dict, psms: list):
    pred = cfg.get("predict", {}) or {}

    # Import predictors lazily so env/TF config runs first.
    if bool(pred.get("do_prosit", True)):
        with log_timing("predict.prosit"):
            from imspy.algorithm import predict_intensities_prosit
            predict_intensities_prosit(psm_collection=psms)

    if bool(pred.get("do_rt", True)):
        with log_timing("predict.rt"):
            from imspy.algorithm import predict_retention_time
            predict_retention_time(
                psm_collection=psms,
                refine_model=bool(pred.get("refine_rt_model", True)),
                verbose=bool(pred.get("verbose", False)),
            )

    if bool(pred.get("do_im", True)):
        with log_timing("predict.im"):
            from imspy.algorithm import predict_inverse_ion_mobility
            predict_inverse_ion_mobility(
                psm_collection=psms,
                refine_model=bool(pred.get("refine_im_model", True)),
                verbose=bool(pred.get("verbose", False)),
            )


def maybe_overwrite_score(cfg: dict, psms: list):
    s = cfg.get("score", {}) or {}
    if not bool(s.get("overwrite_hyperscore_with_beta", False)):
        return

    log("[score] overwriting hyperscore with beta_score_with_matching(...)")
    with log_timing("score.beta_overwrite"):
        for p in psms:
            try:
                p.hyperscore = beta_score_with_matching(
                    p.sage_feature.fragments,
                    p.prosit_intensities_to_fragments(),
                )
            except Exception:
                # keep robustness; bad/missing prediction etc.
                p.hyperscore = 0.0


def run_rescoring(cfg: dict, psms: list):
    rc = cfg.get("rescore", {}) or {}
    if not bool(rc.get("enabled", True)):
        log("[rescore] disabled")
        return psms

    model_kind = str(rc.get("model", "xgboost")).lower()
    num_splits = int(rc.get("num_splits", 5))

    from sagepy.rescore.rescore import rescore_psms

    if model_kind == "xgboost":
        from xgboost import XGBClassifier
        params = dict(rc.get("xgb_params", {}) or {})
        model = XGBClassifier(**params)
    elif model_kind == "svm":
        from sklearn.svm import SVC
        model = SVC(probability=True)
    elif model_kind == "lda":
        # If you want explicit LDA: sagepy has helpers, but rescore_psms expects a model-like object.
        # Using rescore_lda directly is another route; keeping this simple here.
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
    else:
        raise ValueError(f"Unknown rescore.model={model_kind}")

    with log_timing(f"rescore.cv{num_splits}.{model_kind}"):
        psms_rescored = rescore_psms(
            psm_collection=psms,
            num_splits=num_splits,
            model=model,
        )

    return psms_rescored


def run_tdc(cfg: dict, psms_rescored: list):
    from sagepy.utility import psm_collection_to_pandas
    from sagepy.qfdr.tdc import target_decoy_competition_pandas

    tdc_cfg = cfg.get("tdc", {}) or {}
    method = str(tdc_cfg.get("method", "peptide_psm_peptide"))
    score_col = str(tdc_cfg.get("score", "re_score"))

    with log_timing("tdc"):
        df = psm_collection_to_pandas(psms_rescored)
        tdc_df = target_decoy_competition_pandas(
            df,
            score=score_col,
            method=method,
        )
        # Merge back all original columns (tdc_df may be reduced)
        key_cols = ["spec_idx", "match_idx", "decoy"]
        tdc_df = tdc_df.merge(df, on=key_cols, how="left", suffixes=("", ""))
    return tdc_df


def write_outputs(cfg: dict, tdc_df):
    out = cfg["output"]
    out_dir = Path(out["dir"])
    ensure_dir(out_dir)
    ensure_dir(out_dir / "logs")

    if bool(out.get("write_parquet", True)):
        path = out_dir / out.get("parquet_name", "psms_rescored.parquet")
        with log_timing("write.parquet", sync_cuda=False):
            ensure_dir_for_file(path)
            tdc_df.to_parquet(path, index=False)
            log(f"[ok] wrote {path}")

    if bool(out.get("write_csv", False)):
        path = out_dir / out.get("csv_name", "psms_rescored.csv")
        with log_timing("write.csv", sync_cuda=False):
            ensure_dir_for_file(path)
            tdc_df.to_csv(path, index=False)
            log(f"[ok] wrote {path}")

    # Summary JSON
    q = float(cfg.get("tdc", {}).get("q_value", 0.01))
    n_total = int(len(tdc_df))
    n_targets = int((tdc_df["decoy"] == False).sum())
    n_pass = int(((tdc_df["decoy"] == False) & (tdc_df["q_value"] <= q)).sum())

    summary = {
        "n_total": n_total,
        "n_targets": n_targets,
        "n_targets_q_le": {str(q): n_pass},
        "score_used": cfg.get("tdc", {}).get("score", "re_score"),
        "method": cfg.get("tdc", {}).get("method", "peptide_psm_peptide"),
        "output_dir": str(out_dir),
    }

    path = out_dir / out.get("summary_name", "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[ok] wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="PSM re-scoring (TOML-configured).")
    parser.add_argument("-c", "--config", required=True, help="Path to config-rescore.toml")
    parser.add_argument("--device", help="Override run.device (e.g. cuda/cpu)")
    parser.add_argument("--cuda-visible-devices", help="Override run.cuda_visible_devices (e.g. '1')")
    parser.add_argument("--log-file", default=None, help="Override log file path")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # CLI overrides *before* env application
    if args.device:
        cfg["run"]["device"] = args.device
    if args.cuda_visible_devices is not None:
        cfg["run"]["cuda_visible_devices"] = args.cuda_visible_devices

    # apply env NOW (before torch/tf)
    apply_env(cfg)

    # logging
    out_dir = cfg["output"]["dir"]
    ensure_dir(out_dir)
    ensure_dir(Path(out_dir) / "logs")
    log_file = args.log_file or _default_log_path_from_cfg(cfg)

    setup_logging(
        log_file=log_file,
        level=args.log_level,
        also_console=True,
    )
    log(f"[log] writing logfile -> {log_file}")

    # environment banner (after torch import is okay)
    try:
        import torch
        log(f"[env] Torch {torch.__version__} | CUDA available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"[env] CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        log(f"[env] torch not usable: {e}", logging.WARNING)

    # configure TF (keeps TF from touching wrong GPU / hogging VRAM)
    configure_tensorflow(cfg)

    # ---------------- pipeline ----------------
    psms = load_psms(cfg)
    cuda_gc()

    run_predictions(cfg, psms)
    cuda_gc()

    maybe_overwrite_score(cfg, psms)
    cuda_gc()

    psms_rescored = run_rescoring(cfg, psms)
    cuda_gc()

    tdc_df = run_tdc(cfg, psms_rescored)

    write_outputs(cfg, tdc_df)

    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())