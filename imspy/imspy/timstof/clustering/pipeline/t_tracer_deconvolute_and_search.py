#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral deconvolution → SAGE search → initial PSMs (TOML-configured).
"""

from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# tomllib (py3.11+) with fallback
try:
    import tomllib as toml
except Exception:  # pragma: no cover
    import tomli as toml  # type: ignore


# --------------------------- logging ------------------------------------------

_LOGGER_NAME = "t-tracer"
_logger = logging.getLogger(_LOGGER_NAME)


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


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_dir_for_file(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def _default_log_path(out_dir: str | os.PathLike) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(out_dir) / "logs" / f"t_tracer_deconv_search_{ts}.log"

import inspect

def _call_with_accepted_kwargs(fn, kwargs: dict[str, object]):
    sig = inspect.signature(fn)
    params = sig.parameters

    # If builder has **kwargs, just pass everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)

    accepted = {k: v for k, v in kwargs.items() if k in params}
    return fn(**accepted)


# --------------------------- config -------------------------------------------

def load_config(path: str | os.PathLike) -> dict:
    with open(path, "rb") as f:
        cfg = toml.load(f)

    for sec in ("input", "run", "output", "pseudospectra", "query", "sage"):
        if sec not in cfg:
            raise ValueError(f"Missing [{sec}] in config.")

    run = cfg["run"]
    run.setdefault("device", "cuda")
    run.setdefault("cuda_visible_devices", None)  # e.g. "1"
    run.setdefault("num_threads", 0)

    ps = cfg["pseudospectra"]
    ps.setdefault("batch_size", 16384)
    ps.setdefault("max_scan_apex_delta", 7)
    ps.setdefault("max_rt_apex_delta_sec", 1.5)
    ps.setdefault("build_from_features", True)
    ps.setdefault("build_from_clusters", True)

    cfg.setdefault("deconv", {})
    cfg["deconv"].setdefault("features", {})
    feat = cfg["deconv"]["features"]
    feat.setdefault("enabled", True)
    feat.setdefault("z_min", 2)
    feat.setdefault("z_max", 4)
    feat.setdefault("iso_ppm_tol", 10.0)
    feat.setdefault("iso_abs_da", 0.003)
    feat.setdefault("min_members", 2)
    feat.setdefault("max_members", 8)
    feat.setdefault("min_raw_sum", 1.0)
    feat.setdefault("min_mz", 150.0)
    feat.setdefault("min_rt_overlap_frac", 0.1)
    feat.setdefault("min_im_overlap_frac", 0.4)
    feat.setdefault("min_cosine", 0.7)
    feat.setdefault("w_spacing_penalty", 1.0)
    feat.setdefault("lut_k", 10)
    feat.setdefault("lut_resolution", 3)
    feat.setdefault("lut_step", 5.0)

    cfg["deconv"].setdefault("ms2_index", {})
    ms2i = cfg["deconv"]["ms2_index"]
    ms2i.setdefault("min_raw_sum", 1.0)

    q = cfg["query"]
    q.setdefault("merge_fragments", True)
    q.setdefault("merge_allow_cross_window_group", True)
    q.setdefault("merge_max_ppm", 10.0)
    q.setdefault("take_top_n", 150)
    q.setdefault("min_fragments", 5)

    # NEW: whether to attach MS1 metadata (inv_mob, rt, intensity) via your builder
    q.setdefault("attach_ms1_metadata", True)

    # NEW: quant policy knobs
    q.setdefault("quant", {})
    qc = q["quant"]
    qc.setdefault("intensity_source", "volume_proxy_then_raw_sum")
    qc.setdefault("feature_agg", "most_intense_member")     # or sum_top_k_members / sum_members
    qc.setdefault("feature_top_k", 3)

    sage = cfg["sage"]
    sage.setdefault("enzyme", {})
    sage.setdefault("mods", {})
    sage.setdefault("db", {})
    sage.setdefault("search", {})

    enz = sage["enzyme"]
    enz.setdefault("missed_cleavages", 2)
    enz.setdefault("min_len", 7)
    enz.setdefault("max_len", 30)
    enz.setdefault("cleave_at", "KR")
    enz.setdefault("restrict", "P")
    enz.setdefault("c_terminal", True)

    mods = sage["mods"]
    mods.setdefault("static_mods", {"C": "[UNIMOD:4]"})
    mods.setdefault("variable_mods", {"M": ["[UNIMOD:35]"]})

    db = sage["db"]
    db.setdefault("generate_decoys", True)
    db.setdefault("bucket_size_pow2", 14)

    se = sage["search"]
    se.setdefault("report_psms", 5)
    se.setdefault("min_matched_peaks", 4)
    se.setdefault("precursor_ppm", [-15.0, 15.0])
    se.setdefault("fragment_ppm", [-7.0, 7.0])
    se.setdefault("use_hyper_score", True)
    se.setdefault("num_threads", 0)

    out = cfg["output"]
    out.setdefault("write_parquet", True)
    out.setdefault("write_debug_bins", False)
    out.setdefault("psms_features_bin", "psms_features.bin")
    out.setdefault("psms_clusters_bin", "psms_clusters.bin")
    out.setdefault("psms_features_parquet", "psms_features.parquet")
    out.setdefault("psms_clusters_parquet", "psms_clusters.parquet")
    out.setdefault("features_bin", "features.bin")
    out.setdefault("query_features_bin", "query_features.bin")
    out.setdefault("query_clusters_bin", "query_clusters.bin")

    return cfg


def print_config_summary(cfg: dict) -> None:
    inp = cfg["input"]
    run = cfg["run"]
    out = cfg["output"]
    ps = cfg["pseudospectra"]
    q = cfg["query"]
    feat = cfg["deconv"]["features"]
    ms2i = cfg["deconv"]["ms2_index"]
    sage = cfg["sage"]
    qc = q.get("quant", {})

    log("──────────────── CONFIG SUMMARY (DECONV+SEARCH) ────────────────")
    log("[input]")
    log(f"  dataset              : {inp.get('dataset')}")
    log(f"  precursor_clusters   : {inp.get('precursor_clusters')}")
    log(f"  fragment_clusters_dir: {inp.get('fragment_clusters_dir')}")
    log(f"  fasta                : {inp.get('fasta')}")
    log("[run]")
    log(f"  device               : {run.get('device')}")
    log(f"  cuda_visible_devices : {run.get('cuda_visible_devices')}")
    log(f"  num_threads          : {run.get('num_threads')}")
    log("[deconv.features]")
    log(f"  enabled              : {bool(feat.get('enabled'))}")
    log(f"  z_min..z_max         : {feat.get('z_min')}..{feat.get('z_max')}")
    log("[deconv.ms2_index]")
    log(f"  min_raw_sum          : {ms2i.get('min_raw_sum')}")
    log("[pseudospectra]")
    log(f"  batch_size           : {ps.get('batch_size')}")
    log(f"  max_scan_apex_delta  : {ps.get('max_scan_apex_delta')}")
    log(f"  max_rt_apex_delta_sec: {ps.get('max_rt_apex_delta_sec')}")
    log(f"  build_from_features  : {bool(ps.get('build_from_features'))}")
    log(f"  build_from_clusters  : {bool(ps.get('build_from_clusters'))}")
    log("[query]")
    log(f"  attach_ms1_metadata  : {bool(q.get('attach_ms1_metadata', True))}")
    log(f"  merge_fragments      : {bool(q.get('merge_fragments'))}")
    log(f"  merge_max_ppm        : {q.get('merge_max_ppm')}")
    log(f"  merge_cross_wg       : {bool(q.get('merge_allow_cross_window_group'))}")
    log(f"  take_top_n           : {q.get('take_top_n')}")
    log(f"  min_fragments        : {q.get('min_fragments')}")
    log("[query.quant]")
    log(f"  intensity_source     : {qc.get('intensity_source')}")
    log(f"  feature_agg          : {qc.get('feature_agg')}")
    log(f"  feature_top_k        : {qc.get('feature_top_k')}")
    log("[sage.search]")
    log(f"  precursor_ppm        : {sage['search'].get('precursor_ppm')}")
    log(f"  fragment_ppm         : {sage['search'].get('fragment_ppm')}")
    log(f"  min_matched_peaks    : {sage['search'].get('min_matched_peaks')}")
    log(f"  report_psms          : {sage['search'].get('report_psms')}")
    log(f"  use_hyper_score      : {bool(sage['search'].get('use_hyper_score'))}")
    log("[output]")
    log(f"  dir                  : {out.get('dir')}")
    log(f"  write_parquet        : {bool(out.get('write_parquet'))}")
    log(f"  write_debug_bins     : {bool(out.get('write_debug_bins'))}")
    log("────────────────────────────────────────────────────────────────")


# --------------------------- imports that depend on env ------------------------

def apply_cuda_visible_devices(cfg: dict) -> None:
    devs = cfg.get("run", {}).get("cuda_visible_devices", None)
    if devs is not None and str(devs).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devs).strip()


def import_torch_and_deps():
    import torch  # noqa
    return torch


def import_imspy_and_sage():
    from imspy.timstof.dia import TimsDatasetDIA  # noqa
    from imspy.timstof.dia import CandidateOpts, FragmentIndex  # noqa
    from imspy.timstof.dia import load_clusters_parquet  # noqa
    from imspy.timstof.dia import save_pseudo_spectra_bin  # noqa
    from imspy.timstof.clustering.feature import save_features_bin  # noqa
    from imspy.timstof.clustering.feature import (  # noqa
        SimpleFeatureParams,
        build_simple_features_from_clusters,
        AveragineLut,
    )
    from sagepy.core import Scorer, EnzymeBuilder, SageSearchConfiguration, Tolerance  # noqa
    from sagepy.core.fdr import sage_fdr_psm  # noqa
    from sagepy.utility import psm_collection_to_pandas, compress_psms  # noqa

    return {
        "TimsDatasetDIA": TimsDatasetDIA,
        "CandidateOpts": CandidateOpts,
        "FragmentIndex": FragmentIndex,
        "load_clusters_parquet": load_clusters_parquet,
        "save_pseudo_spectra_bin": save_pseudo_spectra_bin,
        "save_features_bin": save_features_bin,
        "SimpleFeatureParams": SimpleFeatureParams,
        "build_simple_features_from_clusters": build_simple_features_from_clusters,
        "AveragineLut": AveragineLut,
        "Scorer": Scorer,
        "EnzymeBuilder": EnzymeBuilder,
        "SageSearchConfiguration": SageSearchConfiguration,
        "Tolerance": Tolerance,
        "sage_fdr_psm": sage_fdr_psm,
        "psm_collection_to_pandas": psm_collection_to_pandas,
        "compress_psms": compress_psms,
    }


def import_query_builder():
    """
    Update this import if your builder lives elsewhere.
    """
    try:
        from imspy.timstof.clustering.utility import build_sagepy_queries_from_pseudo_spectra  # type: ignore
        return build_sagepy_queries_from_pseudo_spectra
    except Exception as e:
        raise ImportError(
            "Could not import build_sagepy_queries_from_pseudo_spectra. "
            "Edit import_query_builder() to point to the right module path in your repo."
        ) from e


# --------------------------- core pipeline ------------------------------------

def stable_sort_clusters(clusters: list) -> list:
    # Deterministic order: (raw_sum desc, cluster_id asc)
    return sorted(
        clusters,
        key=lambda c: (float(getattr(c, "raw_sum", 0.0)), -int(getattr(c, "cluster_id", 0))),
        reverse=True,
    )


def build_ms1_index(clusters: list) -> dict[int, object]:
    return {int(c.cluster_id): c for c in clusters}


def build_feature_index(features: list) -> dict[int, object]:
    return {int(getattr(f, "feature_id")): f for f in features}


def build_features(cfg: dict, clusters: list, imspy: dict):
    feat_cfg = cfg["deconv"]["features"]
    if not bool(feat_cfg.get("enabled", True)):
        log("[features] disabled")
        return []

    AveragineLut = imspy["AveragineLut"]
    SimpleFeatureParams = imspy["SimpleFeatureParams"]
    build_simple_features_from_clusters = imspy["build_simple_features_from_clusters"]

    lut = AveragineLut(
        k=int(feat_cfg["lut_k"]),
        resolution=int(feat_cfg["lut_resolution"]),
        step=float(feat_cfg["lut_step"]),
    )

    params = SimpleFeatureParams(
        z_min=int(feat_cfg["z_min"]),
        z_max=int(feat_cfg["z_max"]),
        iso_ppm_tol=float(feat_cfg["iso_ppm_tol"]),
        iso_abs_da=float(feat_cfg["iso_abs_da"]),
        min_members=int(feat_cfg["min_members"]),
        max_members=int(feat_cfg["max_members"]),
        min_raw_sum=float(feat_cfg["min_raw_sum"]),
        min_mz=float(feat_cfg["min_mz"]),
        min_rt_overlap_frac=float(feat_cfg["min_rt_overlap_frac"]),
        min_im_overlap_frac=float(feat_cfg["min_im_overlap_frac"]),
        min_cosine=float(feat_cfg["min_cosine"]),
        w_spacing_penalty=float(feat_cfg["w_spacing_penalty"]),
    )

    log(f"[features] building from {len(clusters)} precursor clusters …")
    features = build_simple_features_from_clusters(clusters, params, lut=lut)
    features = sorted(
        features,
        key=lambda f: (float(getattr(f, "raw_sum", 0.0)), int(getattr(f, "feature_id", 0))),
        reverse=True,
    )
    log(f"[features] built {len(features)} features")
    return features


def split_leftover_clusters(clusters: list, features: list) -> list:
    if not features:
        return clusters
    used = set()
    for ft in features:
        for cid in getattr(ft, "member_cluster_ids", []):
            used.add(int(cid))
    left = [c for c in clusters if int(c.cluster_id) not in used]
    log(f"[clusters] leftover clusters: {len(left)} / {len(clusters)} (not in any feature)")
    return left


def build_fragment_index(cfg: dict, ds, imspy: dict):
    FragmentIndex = imspy["FragmentIndex"]
    CandidateOpts = imspy["CandidateOpts"]

    frag_dir = cfg["input"]["fragment_clusters_dir"]
    opts = CandidateOpts(min_raw_sum=float(cfg["deconv"]["ms2_index"]["min_raw_sum"]))
    log(f"[ms2] building FragmentIndex from {frag_dir} …")
    ms2_index = FragmentIndex.from_parquet_dir(ds, frag_dir, opts)
    log("[ms2] FragmentIndex ready")
    return ms2_index


def build_pseudospectra(cfg: dict, ms2_index, *, features: list, clusters_left: list):
    from tqdm import tqdm

    ps_cfg = cfg["pseudospectra"]
    batch_size = int(ps_cfg["batch_size"])
    max_scan_apex_delta = int(ps_cfg["max_scan_apex_delta"])
    max_rt_apex_delta_sec = float(ps_cfg["max_rt_apex_delta_sec"])

    spec_features: list = []
    spec_clusters: list = []

    if bool(ps_cfg.get("build_from_features", True)) and features:
        log(f"[pseudospectra] scoring features → pseudospectra (n={len(features)})")
        n_batches = (len(features) + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc="features→pseudospectra", ncols=100):
            u, l = i * batch_size, min((i + 1) * batch_size, len(features))
            chunk = features[u:l]
            results = ms2_index.score_features_to_pseudospectra(
                features=chunk,
                max_scan_apex_delta=max_scan_apex_delta,
                max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            )
            if results:
                spec_features.extend(results)
        log(f"[pseudospectra] feature pseudospectra: {len(spec_features)}")

    if bool(ps_cfg.get("build_from_clusters", True)) and clusters_left:
        log(f"[pseudospectra] scoring leftover clusters → pseudospectra (n={len(clusters_left)})")
        n_batches = (len(clusters_left) + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc="clusters→pseudospectra", ncols=100):
            u, l = i * batch_size, min((i + 1) * batch_size, len(clusters_left))
            chunk = clusters_left[u:l]
            results = ms2_index.score_precursors_to_pseudospectra(
                precursor_clusters=chunk,
                max_scan_apex_delta=max_scan_apex_delta,
                max_rt_apex_delta_sec=max_rt_apex_delta_sec,
            )
            if results:
                spec_clusters.extend(results)
        log(f"[pseudospectra] cluster pseudospectra: {len(spec_clusters)}")

    return spec_features, spec_clusters


def build_queries(
    cfg: dict,
    builder_fn,
    *,
    spectra: list,
    use_charge: bool,
    ms1_index: dict[int, object],
    feature_index: dict[int, object] | None,
    ds,
):
    q = cfg["query"]
    if not spectra:
        return None

    # Base args (old builder signature)
    kwargs: dict[str, object] = dict(
        spectra=spectra,
        use_charge=bool(use_charge),
        merge_fragments=bool(q["merge_fragments"]),
        merge_allow_cross_window_group=bool(q["merge_allow_cross_window_group"]),
        merge_max_ppm=float(q["merge_max_ppm"]),
        take_top_n=int(q["take_top_n"]),
        min_fragments=int(q["min_fragments"]),
    )

    # Optional MS1 metadata + new quant knobs (new builder signature)
    if bool(q.get("attach_ms1_metadata", True)):
        qc = q.get("quant", {})
        kwargs.update(
            ms1_index=ms1_index,
            feature_index=feature_index,
            ds=ds,
            intensity_source=str(qc.get("intensity_source", "volume_proxy_then_raw_sum")),
            feature_agg=str(qc.get("feature_agg", "most_intense_member")),
            feature_top_k=int(qc.get("feature_top_k", 3)),
        )

    try:
        qc = q.get("quant", {})
        log(f"[query.quant] feature_agg={qc.get('feature_agg')} feature_top_k={qc.get('feature_top_k')} intensity_source={qc.get('intensity_source')}")
        return _call_with_accepted_kwargs(builder_fn, kwargs)
    except TypeError as e:
        raise TypeError(
            "build_sagepy_queries_from_pseudo_spectra failed. "
            "This is likely an internal builder error (not a signature mismatch anymore). "
            f"Args passed: {sorted(kwargs.keys())}"
        ) from e


def build_sage_db(cfg: dict, imspy: dict):
    SageSearchConfiguration = imspy["SageSearchConfiguration"]
    EnzymeBuilder = imspy["EnzymeBuilder"]

    fasta_path = cfg["input"]["fasta"]
    with open(fasta_path, "r") as f:
        fasta = f.read()

    enz_cfg = cfg["sage"]["enzyme"]
    enzyme_builder = EnzymeBuilder(
        missed_cleavages=int(enz_cfg["missed_cleavages"]),
        min_len=int(enz_cfg["min_len"]),
        max_len=int(enz_cfg["max_len"]),
        cleave_at=str(enz_cfg["cleave_at"]),
        restrict=str(enz_cfg["restrict"]),
        c_terminal=bool(enz_cfg["c_terminal"]),
    )

    mods_cfg = cfg["sage"]["mods"]
    static_mods = dict(mods_cfg["static_mods"])
    variable_mods = dict(mods_cfg["variable_mods"])

    db_cfg = cfg["sage"]["db"]
    bucket_pow2 = int(db_cfg["bucket_size_pow2"])
    bucket_size = int(2**bucket_pow2)

    log(f"[sage.db] building index (bucket_size=2^{bucket_pow2}={bucket_size}) …")
    sage_config = SageSearchConfiguration(
        fasta=fasta,
        static_mods=static_mods,
        variable_mods=variable_mods,
        enzyme_builder=enzyme_builder,
        generate_decoys=bool(db_cfg["generate_decoys"]),
        bucket_size=bucket_size,
    )
    indexed_db = sage_config.generate_indexed_database()
    log("[sage.db] indexed database ready")
    return indexed_db, static_mods, variable_mods


def run_sage_search(cfg: dict, imspy: dict, indexed_db, queries, *, static_mods: dict, variable_mods: dict):
    if queries is None:
        return None

    Scorer = imspy["Scorer"]
    Tolerance = imspy["Tolerance"]
    sage_fdr_psm = imspy["sage_fdr_psm"]

    se = cfg["sage"]["search"]
    ppm_prec = [float(se["precursor_ppm"][0]), float(se["precursor_ppm"][1])]
    ppm_frag = [float(se["fragment_ppm"][0]), float(se["fragment_ppm"][1])]

    scorer = Scorer(
        report_psms=int(se["report_psms"]),
        min_matched_peaks=int(se["min_matched_peaks"]),
        variable_mods=variable_mods,
        static_mods=static_mods,
        precursor_tolerance=Tolerance(ppm=(ppm_prec[0], ppm_prec[1])),
        fragment_tolerance=Tolerance(ppm=(ppm_frag[0], ppm_frag[1])),
    )

    num_threads = int(se.get("num_threads", 0))
    log(f"[sage.search] searching (threads={num_threads}) …")
    results = scorer.score_collection_psm(
        db=indexed_db,
        spectrum_collection=queries,
        num_threads=num_threads,
    )

    log("[sage.fdr] computing q-values …")
    sage_fdr_psm(results, indexed_db, use_hyper_score=bool(se["use_hyper_score"]))
    return results


def flatten_results_dict(results: dict) -> list:
    out = []
    for _, lst in results.items():
        out.extend(lst)
    return out


def write_psms(cfg: dict, imspy: dict, results: dict, *, out_dir: Path, prefix: str):
    compress_psms = imspy["compress_psms"]
    psm_collection_to_pandas = imspy["psm_collection_to_pandas"]

    out_cfg = cfg["output"]
    write_parquet = bool(out_cfg["write_parquet"])

    bin_name = out_cfg["psms_features_bin"] if prefix == "features" else out_cfg["psms_clusters_bin"]
    bin_path = out_dir / bin_name
    ensure_dir_for_file(bin_path)

    flat = flatten_results_dict(results)
    log(f"[out] {prefix}: total PSM objects = {len(flat)}")

    psm_bin = compress_psms(flat)
    with open(bin_path, "wb") as f:
        f.write(bytearray(psm_bin))
    log(f"[out] wrote {prefix} PSMs (bin) -> {bin_path}")

    if write_parquet:
        parq_name = out_cfg["psms_features_parquet"] if prefix == "features" else out_cfg["psms_clusters_parquet"]
        parq_path = out_dir / parq_name
        ensure_dir_for_file(parq_path)
        df = psm_collection_to_pandas(flat)
        df.to_parquet(parq_path, index=False)
        log(f"[out] wrote {prefix} PSMs (parquet) -> {parq_path}")


def maybe_write_debug_bins(cfg: dict, imspy: dict, *, out_dir: Path, features: list, spec_features: list, spec_clusters: list):
    if not bool(cfg["output"].get("write_debug_bins", False)):
        return

    save_features_bin = imspy["save_features_bin"]
    save_pseudo_spectra_bin = imspy["save_pseudo_spectra_bin"]
    out_cfg = cfg["output"]

    if features:
        path = out_dir / out_cfg["features_bin"]
        ensure_dir_for_file(path)
        save_features_bin(path=str(path), features=features)
        log(f"[debug] wrote features bin -> {path}")

    if spec_features:
        path = out_dir / out_cfg["query_features_bin"]
        ensure_dir_for_file(path)
        save_pseudo_spectra_bin(path=str(path), spectra=spec_features)
        log(f"[debug] wrote query_features bin -> {path}")

    if spec_clusters:
        path = out_dir / out_cfg["query_clusters_bin"]
        ensure_dir_for_file(path)
        save_pseudo_spectra_bin(path=str(path), spectra=spec_clusters)
        log(f"[debug] wrote query_clusters bin -> {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Spectral deconvolution → SAGE search → initial PSMs.")
    parser.add_argument("-c", "--config", required=True, help="Path to config TOML")
    parser.add_argument("--log-file", default=None, help="Override log file path")
    parser.add_argument("--log-level", default="INFO", help="Log level (DEBUG/INFO/WARNING/ERROR)")
    parser.add_argument("--no-console-log", action="store_true", help="Disable console logging")

    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = Path(cfg["output"]["dir"])
    ensure_dir(out_dir)

    log_file = args.log_file or _default_log_path(out_dir)
    setup_logging(
        log_file=log_file,
        level=args.log_level,
        also_console=not args.no_console_log,
    )
    log(f"[log] writing logfile -> {log_file}")

    print_config_summary(cfg)

    # IMPORTANT: set CUDA_VISIBLE_DEVICES before torch import
    apply_cuda_visible_devices(cfg)

    torch = import_torch_and_deps()
    imspy = import_imspy_and_sage()
    build_sagepy_queries_from_pseudo_spectra = import_query_builder()

    log(f"[env] Python {sys.version.split()[0]}")
    log(f"[env] Torch {torch.__version__} | CUDA available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            log(f"[env] CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    TimsDatasetDIA = imspy["TimsDatasetDIA"]
    load_clusters_parquet = imspy["load_clusters_parquet"]

    inp = cfg["input"]
    ds = TimsDatasetDIA(
        inp["dataset"],
        use_bruker_sdk=bool(inp.get("use_bruker_sdk", False)),
    )

    prec_clusters = load_clusters_parquet(inp["precursor_clusters"])
    prec_clusters = stable_sort_clusters(list(prec_clusters))
    log(f"[ms1] loaded precursor clusters: {len(prec_clusters)}")

    ms1_index = build_ms1_index(prec_clusters)

    ms2_index = build_fragment_index(cfg, ds, imspy)

    features = build_features(cfg, prec_clusters, imspy)
    feature_index = build_feature_index(features) if features else None
    clusters_left = split_leftover_clusters(prec_clusters, features)

    spec_features, spec_clusters = build_pseudospectra(
        cfg,
        ms2_index,
        features=features,
        clusters_left=clusters_left,
    )

    maybe_write_debug_bins(
        cfg, imspy,
        out_dir=out_dir,
        features=features,
        spec_features=spec_features,
        spec_clusters=spec_clusters,
    )

    log("[query] building SAGE query spectra …")
    queries_features = None
    queries_clusters = None

    if spec_features:
        queries_features = build_queries(
            cfg,
            build_sagepy_queries_from_pseudo_spectra,
            spectra=spec_features,
            use_charge=True,
            ms1_index=ms1_index,
            feature_index=feature_index,
            ds=ds,
        )
        log("[query] features queries ready")

    if spec_clusters:
        queries_clusters = build_queries(
            cfg,
            build_sagepy_queries_from_pseudo_spectra,
            spectra=spec_clusters,
            use_charge=False,
            ms1_index=ms1_index,
            feature_index=feature_index,
            ds=ds,
        )
        log("[query] cluster queries ready")

    if queries_features is None and queries_clusters is None:
        raise RuntimeError("No query spectra were produced (no pseudo-spectra). Check thresholds / configs.")

    indexed_db, static_mods, variable_mods = build_sage_db(cfg, imspy)

    results_features = None
    results_clusters = None

    if queries_features is not None:
        results_features = run_sage_search(cfg, imspy, indexed_db, queries_features, static_mods=static_mods, variable_mods=variable_mods)

    if queries_clusters is not None:
        results_clusters = run_sage_search(cfg, imspy, indexed_db, queries_clusters, static_mods=static_mods, variable_mods=variable_mods)

    if results_features is not None:
        write_psms(cfg, imspy, results_features, out_dir=out_dir, prefix="features")
    if results_clusters is not None:
        write_psms(cfg, imspy, results_clusters, out_dir=out_dir, prefix="clusters")

    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())