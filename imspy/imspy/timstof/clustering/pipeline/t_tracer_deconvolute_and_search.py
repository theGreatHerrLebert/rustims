#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral deconvolution → PseudoSpectra → SAGE search → initial PSMs (TOML-configured).

NEW:
- Searches *combined* query spectra (features + clusters) in ONE pass.
- Works for both classic DB path and chunked-db path.
- Writes a single combined output (bin + parquet).
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    import tomllib as toml
except Exception:  # pragma: no cover
    import tomli as toml  # type: ignore


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


def get_memory_gb() -> float:
    """Get current process memory in GB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except ImportError:
        return 0.0


def fmt_num(n: int) -> str:
    """Format number with thousand separators."""
    return f"{n:,}"


class timed_stage:
    """Context manager for timing pipeline stages with memory tracking."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.start_mem = None

    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        self.start_mem = get_memory_gb()
        log(f"[{self.name}] starting... (RAM: {self.start_mem:.2f} GB)")
        return self

    def __exit__(self, *args):
        import time
        elapsed = time.perf_counter() - self.start_time
        end_mem = get_memory_gb()
        delta_mem = end_mem - self.start_mem
        sign = "+" if delta_mem >= 0 else ""
        log(f"[{self.name}] done in {elapsed:.1f}s (RAM: {end_mem:.2f} GB, {sign}{delta_mem:.2f} GB)")


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_dir_for_file(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def _default_log_path(out_dir: str | os.PathLike) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(out_dir) / "logs" / f"t_tracer_deconv_search_{ts}.log"


def _call_with_accepted_kwargs(fn, kwargs: dict[str, object]):
    sig = inspect.signature(fn)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)

    accepted = {k: v for k, v in kwargs.items() if k in params}
    return fn(**accepted)


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "rb") as f:
        cfg = toml.load(f)

    for sec in ("input", "run", "output", "pseudospectra", "query", "sage"):
        if sec not in cfg:
            raise ValueError(f"Missing [{sec}] in config.")

    run = cfg["run"]
    run.setdefault("device", "cuda")
    run.setdefault("cuda_visible_devices", None)
    run.setdefault("num_threads", 0)

    ps = cfg["pseudospectra"]
    ps.setdefault("batch_size", 16384)
    ps.setdefault("max_scan_apex_delta", 7)
    ps.setdefault("max_rt_apex_delta_sec", 1.5)
    ps.setdefault("build_from_features", True)
    ps.setdefault("build_from_clusters", True)
    ps.setdefault("mode", "geom")
    ps.setdefault("min_score", 0.0)
    ps.setdefault("reject_frag_inside_precursor_tile", True)
    ps.setdefault("min_im_overlap_scans", 1)
    ps.setdefault("require_tile_compat", True)
    ps.setdefault("min_fragments", 4)
    ps.setdefault("max_fragments", 512)

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
    ms2i.setdefault("use_slim_index", True)  # Low RAM mode: ~80% less memory

    cfg["deconv"].setdefault("ms1_index", {})
    ms1i = cfg["deconv"]["ms1_index"]
    ms1i.setdefault("use_slim_precursors", True)  # Low RAM mode for precursor clusters

    q = cfg["query"]
    q.setdefault("merge_fragments", True)
    q.setdefault("merge_allow_cross_window_group", True)
    q.setdefault("merge_max_ppm", 10.0)
    q.setdefault("take_top_n", 150)
    q.setdefault("min_fragments", 5)
    q.setdefault("attach_ms1_metadata", True)
    q.setdefault("quant", {})
    qc = q["quant"]
    qc.setdefault("intensity_source", "raw_sum")
    qc.setdefault("feature_agg", "most_intense_member")
    qc.setdefault("feature_top_k", 3)

    sage = cfg["sage"]
    sage.setdefault("enzyme", {})
    sage.setdefault("mods", {})
    sage.setdefault("db", {})
    sage.setdefault("search", {})
    sage.setdefault("chunked", {})

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

    ch = sage["chunked"]
    ch.setdefault("enabled", False)
    ch.setdefault("chunk_size", 50_000)
    ch.setdefault("low_memory", False)
    ch.setdefault("max_hits", None)

    out = cfg["output"]
    out.setdefault("write_parquet", True)
    out.setdefault("write_debug_bins", False)

    # keep your old names, but also allow combined
    out.setdefault("psms_combined_bin", "psms_combined.bin")
    out.setdefault("psms_combined_parquet", "psms_combined.parquet")

    out.setdefault("features_bin", "features.bin")
    out.setdefault("query_features_bin", "query_features.bin")
    out.setdefault("query_clusters_bin", "query_clusters.bin")

    return cfg


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
    from imspy.timstof.dia import PrecursorIndex, SlimPrecursor  # noqa
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
        "PrecursorIndex": PrecursorIndex,
        "SlimPrecursor": SlimPrecursor,
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
    from imspy.timstof.clustering.utility import build_sagepy_queries_from_pseudo_spectra  # type: ignore
    return build_sagepy_queries_from_pseudo_spectra


def stable_sort_clusters(clusters: list) -> list:
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

    log(f"[features] building from {fmt_num(len(clusters))} precursor clusters …")
    features = build_simple_features_from_clusters(clusters, params, lut=lut)
    features = sorted(
        features,
        key=lambda f: (float(getattr(f, "raw_sum", 0.0)), int(getattr(f, "feature_id", 0))),
        reverse=True,
    )
    log(f"[features] built {fmt_num(len(features))} features")
    return features


def split_leftover_clusters(clusters: list, features: list) -> list:
    if not features:
        return clusters
    used = set()
    for ft in features:
        for cid in getattr(ft, "member_cluster_ids", []):
            used.add(int(cid))
    left = [c for c in clusters if int(c.cluster_id) not in used]
    log(f"[clusters] leftover: {fmt_num(len(left))} / {fmt_num(len(clusters))} clusters not in any feature")
    return left


def build_fragment_index(cfg: dict, ds, imspy: dict):
    FragmentIndex = imspy["FragmentIndex"]
    CandidateOpts = imspy["CandidateOpts"]

    frag_dir = cfg["input"]["fragment_clusters_dir"]
    ms2i_cfg = cfg["deconv"]["ms2_index"]
    opts = CandidateOpts(min_raw_sum=float(ms2i_cfg["min_raw_sum"]))
    use_slim = bool(ms2i_cfg.get("use_slim_index", True))

    if use_slim:
        log(f"[ms2] building FragmentIndex (slim mode) from {frag_dir} …")
        ms2_index = FragmentIndex.from_parquet_dir_slim(ds, frag_dir, opts)
        log(f"[ms2] FragmentIndex ready: {fmt_num(ms2_index.len())} clusters (slim, has_full_data={ms2_index.has_full_data()})")
    else:
        log(f"[ms2] building FragmentIndex (full mode) from {frag_dir} …")
        ms2_index = FragmentIndex.from_parquet_dir(ds, frag_dir, opts)
        log(f"[ms2] FragmentIndex ready: {fmt_num(ms2_index.len())} clusters (full)")

    return ms2_index


def build_pseudospectra(cfg: dict, ms2_index, *, features: list, clusters_left: list, slim_precursors: list | None = None):
    """
    Build pseudo-spectra from features and/or leftover clusters.

    Args:
        cfg: Config dict.
        ms2_index: FragmentIndex for scoring.
        features: List of SimpleFeature objects (for feature-based pseudo-spectra).
        clusters_left: List of ClusterResult1D objects not assigned to features.
        slim_precursors: Optional list of SlimPrecursor for memory-efficient cluster scoring.
                        If provided and clusters_left is empty/None, uses slim scoring.
    """
    from tqdm import tqdm

    ps_cfg = cfg["pseudospectra"]

    batch_size = int(ps_cfg["batch_size"])
    max_scan_apex_delta = int(ps_cfg["max_scan_apex_delta"])
    max_rt_apex_delta_sec = float(ps_cfg["max_rt_apex_delta_sec"])

    mode = str(ps_cfg.get("mode", "geom"))
    min_score = float(ps_cfg.get("min_score", 0.0))
    reject_inside = bool(ps_cfg.get("reject_frag_inside_precursor_tile", True))
    min_im_overlap_scans = int(ps_cfg.get("min_im_overlap_scans", 1))
    require_tile_compat = bool(ps_cfg.get("require_tile_compat", True))
    min_frags = int(ps_cfg.get("min_fragments", 4))
    max_frags = int(ps_cfg.get("max_fragments", 512))

    # Pseudo-spectrum construction now works with slim data (geom scoring).
    # Full data is only needed for XIC scoring mode.
    if ms2_index.has_full_data():
        log("[pseudospectra] using full cluster data (XIC scoring available)")
    else:
        log("[pseudospectra] using slim cluster data (geom scoring only, RAM-efficient)")

    spec_features: list = []
    spec_clusters: list = []

    if bool(ps_cfg.get("build_from_features", True)) and features:
        log(f"[pseudospectra] scoring {fmt_num(len(features))} features → pseudospectra")
        n_batches = (len(features) + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc="features→pseudospectra", ncols=100, unit="batch"):
            u, l = i * batch_size, min((i + 1) * batch_size, len(features))
            chunk = features[u:l]
            results = ms2_index.score_features_to_pseudospectra(
                features=chunk,
                mode=mode,
                min_score=min_score,
                reject_frag_inside_precursor_tile=reject_inside,
                max_rt_apex_delta_sec=max_rt_apex_delta_sec,
                max_scan_apex_delta=max_scan_apex_delta,
                min_im_overlap_scans=min_im_overlap_scans,
                require_tile_compat=require_tile_compat,
                min_fragments=min_frags,
                max_fragments=max_frags,
            )
            if results:
                spec_features.extend(results)
        log(f"[pseudospectra] feature pseudospectra: {fmt_num(len(spec_features))}")

    # Use slim precursors if provided, otherwise use full clusters
    use_slim_precs = slim_precursors is not None and len(slim_precursors) > 0
    precs_to_score = slim_precursors if use_slim_precs else clusters_left

    if bool(ps_cfg.get("build_from_clusters", True)) and precs_to_score:
        mode_label = "slim" if use_slim_precs else "full"
        log(f"[pseudospectra] scoring {fmt_num(len(precs_to_score))} {mode_label} precursors → pseudospectra")
        n_batches = (len(precs_to_score) + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc="clusters→pseudospectra", ncols=100, unit="batch"):
            u, l = i * batch_size, min((i + 1) * batch_size, len(precs_to_score))
            chunk = precs_to_score[u:l]

            if use_slim_precs:
                # Use slim precursor scoring (geom only, memory-efficient)
                results = ms2_index.score_slim_precursors_to_pseudospectra(
                    slim_precursors=chunk,
                    min_score=min_score,
                    reject_frag_inside_precursor_tile=reject_inside,
                    max_rt_apex_delta_sec=max_rt_apex_delta_sec,
                    max_scan_apex_delta=max_scan_apex_delta,
                    min_im_overlap_scans=min_im_overlap_scans,
                    require_tile_compat=require_tile_compat,
                    min_fragments=min_frags,
                    max_fragments=max_frags,
                )
            else:
                # Use full cluster scoring
                results = ms2_index.score_precursors_to_pseudospectra(
                    precursor_clusters=chunk,
                    mode=mode,
                    min_score=min_score,
                    reject_frag_inside_precursor_tile=reject_inside,
                    max_rt_apex_delta_sec=max_rt_apex_delta_sec,
                    max_scan_apex_delta=max_scan_apex_delta,
                    min_im_overlap_scans=min_im_overlap_scans,
                    require_tile_compat=require_tile_compat,
                    min_fragments=min_frags,
                    max_fragments=max_frags,
                )
            if results:
                spec_clusters.extend(results)
        log(f"[pseudospectra] cluster pseudospectra: {fmt_num(len(spec_clusters))}")

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

    kwargs: dict[str, object] = dict(
        spectra=spectra,
        use_charge=bool(use_charge),
        merge_fragments=bool(q["merge_fragments"]),
        merge_allow_cross_window_group=bool(q["merge_allow_cross_window_group"]),
        merge_max_ppm=float(q["merge_max_ppm"]),
        take_top_n=int(q["take_top_n"]),
        min_fragments=int(q["min_fragments"]),
    )

    if bool(q.get("attach_ms1_metadata", True)):
        qc = q.get("quant", {})
        kwargs.update(
            ms1_index=ms1_index,
            feature_index=feature_index,
            ds=ds,
            intensity_source=str(qc.get("intensity_source", "raw_sum")),
            feature_agg=str(qc.get("feature_agg", "most_intense_member")),
            feature_top_k=int(qc.get("feature_top_k", 3)),
        )

    return _call_with_accepted_kwargs(builder_fn, kwargs)


def build_sage_config(cfg: dict, imspy: dict):
    SageSearchConfiguration = imspy["SageSearchConfiguration"]
    EnzymeBuilder = imspy["EnzymeBuilder"]

    with open(cfg["input"]["fasta"], "r") as f:
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

    ch = cfg["sage"].get("chunked", {})
    chunked_enabled = bool(ch.get("enabled", False))
    chunk_size = int(ch.get("chunk_size", 50_000))
    low_memory = bool(ch.get("low_memory", False))

    log(f"[sage.cfg] building SageSearchConfiguration (bucket_size=2^{bucket_pow2}={bucket_size}) …")
    sage_config = SageSearchConfiguration(
        fasta=fasta,
        static_mods=static_mods,
        variable_mods=variable_mods,
        enzyme_builder=enzyme_builder,
        generate_decoys=bool(db_cfg["generate_decoys"]),
        bucket_size=bucket_size,

        # only meaningful for chunked mode
        prefilter=True if chunked_enabled else None,
        prefilter_chunk_size=chunk_size if chunked_enabled else None,
        prefilter_low_memory=low_memory if chunked_enabled else None,
    )
    return sage_config, static_mods, variable_mods


def run_sage_search_combined(cfg: dict, imspy: dict, sage_config, indexed_db_or_none, queries_all, *, static_mods: dict, variable_mods: dict):
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

    ch = cfg["sage"].get("chunked", {})
    chunked_enabled = bool(ch.get("enabled", False))

    if chunked_enabled:
        chunk_size = int(ch.get("chunk_size", 50_000))
        low_memory = bool(ch.get("low_memory", False))
        max_hits = ch.get("max_hits", None)
        if max_hits is None:
            max_hits = int(se["report_psms"])
        else:
            max_hits = int(max_hits)

        log(f"[sage.search] chunked-db COMBINED search (n={len(queries_all)}, chunk_size={chunk_size}, low_memory={low_memory}, threads={num_threads}, max_hits={max_hits}) …")
        indexed_db, results, num_kept = sage_config.prefilter_build_and_search_psm(
            queries_all,
            scorer,
            chunk_size=chunk_size,
            low_memory=low_memory,
            num_threads=num_threads,
            max_hits=max_hits,
        )
        log(f"[sage.search] chunked-db kept spectra: {num_kept} / {len(queries_all)}")
    else:
        indexed_db = indexed_db_or_none
        if indexed_db is None:
            raise RuntimeError("indexed_db is None but chunked search is disabled (script bug).")

        log(f"[sage.search] classic COMBINED search (n={len(queries_all)}, threads={num_threads}) …")
        results = scorer.score_collection_psm(
            db=indexed_db,
            spectrum_collection=queries_all,
            num_threads=num_threads,
        )

    log("[sage.fdr] computing q-values …")
    sage_fdr_psm(results, indexed_db, use_hyper_score=bool(se["use_hyper_score"]))
    return results, indexed_db


def flatten_results_dict(results: dict) -> list:
    out = []
    for _, lst in results.items():
        out.extend(lst)
    return out


def write_psms_combined(cfg: dict, imspy: dict, results: dict, *, out_dir: Path):
    compress_psms = imspy["compress_psms"]
    psm_collection_to_pandas = imspy["psm_collection_to_pandas"]

    out_cfg = cfg["output"]
    write_parquet = bool(out_cfg.get("write_parquet", True))

    bin_name = out_cfg.get("psms_combined_bin", "psms_combined.bin")
    parq_name = out_cfg.get("psms_combined_parquet", "psms_combined.parquet")

    bin_path = out_dir / bin_name
    ensure_dir_for_file(bin_path)

    flat = flatten_results_dict(results)
    log(f"[out] combined: total PSM objects = {len(flat)}")

    psm_bin = compress_psms(flat)
    with open(bin_path, "wb") as f:
        f.write(bytearray(psm_bin))
    log(f"[out] wrote combined PSMs (bin) -> {bin_path}")

    if write_parquet:
        parq_path = out_dir / parq_name
        ensure_dir_for_file(parq_path)
        df = psm_collection_to_pandas(flat)
        df.to_parquet(parq_path, index=False)
        log(f"[out] wrote combined PSMs (parquet) -> {parq_path}")


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
    setup_logging(log_file=log_file, level=args.log_level, also_console=not args.no_console_log)
    log(f"[log] writing logfile -> {log_file}")

    apply_cuda_visible_devices(cfg)

    torch = import_torch_and_deps()
    imspy = import_imspy_and_sage()
    build_sagepy_queries_from_pseudo_spectra = import_query_builder()

    log(f"[env] Python {sys.version.split()[0]}")
    log(f"[env] Torch {torch.__version__} | CUDA available={torch.cuda.is_available()}")

    TimsDatasetDIA = imspy["TimsDatasetDIA"]
    load_clusters_parquet = imspy["load_clusters_parquet"]

    inp = cfg["input"]

    # --- Stage 1: Load dataset ---
    with timed_stage("dataset"):
        ds = TimsDatasetDIA(inp["dataset"], use_bruker_sdk=bool(inp.get("use_bruker_sdk", False)))

    # --- Stage 2: Load precursor clusters ---
    # Check if we should use slim precursor mode
    ms1_cfg = cfg["deconv"].get("ms1_index", {})
    use_slim_precursors = bool(ms1_cfg.get("use_slim_precursors", True))
    features_enabled = bool(cfg["deconv"]["features"].get("enabled", True))

    # Slim precursor mode only works when features are disabled (feature building needs full data)
    can_use_slim = use_slim_precursors and not features_enabled
    slim_precursors = None
    prec_clusters = None

    with timed_stage("ms1.load"):
        if can_use_slim:
            # Slim mode: load minimal precursor data (~40 bytes per precursor)
            PrecursorIndex = imspy["PrecursorIndex"]
            prec_index = PrecursorIndex.from_parquet_dir_slim(inp["precursor_clusters"])
            slim_precursors = prec_index.get_all_slim()
            log(f"[ms1] loaded {fmt_num(len(slim_precursors))} slim precursors (low RAM mode)")
            # Build MS1 index from slim data for query building
            ms1_index = {int(p.cluster_id): p for p in slim_precursors}
        else:
            # Full mode: load complete cluster data
            prec_clusters = stable_sort_clusters(list(load_clusters_parquet(inp["precursor_clusters"])))
            log(f"[ms1] loaded {fmt_num(len(prec_clusters))} precursor clusters")
            ms1_index = build_ms1_index(prec_clusters)

    # --- Stage 3: Build fragment index ---
    with timed_stage("ms2.index"):
        ms2_index = build_fragment_index(cfg, ds, imspy)

    # --- Stage 4: Build features ---
    with timed_stage("features"):
        if can_use_slim:
            # Slim mode: no features (they require full cluster data)
            log("[features] skipped (slim precursor mode)")
            features = []
            feature_index = None
            clusters_left = []  # All precursors handled as slim
        else:
            features = build_features(cfg, prec_clusters, imspy)
            feature_index = build_feature_index(features) if features else None
            clusters_left = split_leftover_clusters(prec_clusters, features)

    # --- Stage 5: Build pseudospectra ---
    with timed_stage("pseudospectra"):
        spec_features, spec_clusters = build_pseudospectra(
            cfg, ms2_index,
            features=features,
            clusters_left=clusters_left,
            slim_precursors=slim_precursors,
        )

    # --- Stage 6: Build queries ---
    with timed_stage("queries"):
        queries_features = None
        queries_clusters = None

        if spec_features:
            log(f"[query] building queries from {fmt_num(len(spec_features))} feature pseudospectra...")
            queries_features = build_queries(
                cfg,
                build_sagepy_queries_from_pseudo_spectra,
                spectra=spec_features,
                use_charge=True,
                ms1_index=ms1_index,
                feature_index=feature_index,
                ds=ds,
            )
            log(f"[query] feature queries: {fmt_num(len(queries_features) if queries_features else 0)}")

        if spec_clusters:
            log(f"[query] building queries from {fmt_num(len(spec_clusters))} cluster pseudospectra...")
            queries_clusters = build_queries(
                cfg,
                build_sagepy_queries_from_pseudo_spectra,
                spectra=spec_clusters,
                use_charge=False,
                ms1_index=ms1_index,
                feature_index=feature_index,
                ds=ds,
            )
            log(f"[query] cluster queries: {fmt_num(len(queries_clusters) if queries_clusters else 0)}")

    # --- Combine queries ---
    queries_all = []
    if queries_features:
        queries_all.extend(list(queries_features))
    if queries_clusters:
        queries_all.extend(list(queries_clusters))

    if not queries_all:
        raise RuntimeError("No query spectra were produced (no pseudo-spectra). Check thresholds / configs.")

    log(f"[query] combined: {fmt_num(len(queries_all))} queries (features={fmt_num(len(queries_features or []))}, clusters={fmt_num(len(queries_clusters or []))})")

    # --- Stage 7: Build SAGE database ---
    chunked_enabled = bool(cfg["sage"].get("chunked", {}).get("enabled", False))
    indexed_db = None

    if not chunked_enabled:
        with timed_stage("sage.db"):
            sage_config, static_mods, variable_mods = build_sage_config(cfg, imspy)
            indexed_db = sage_config.generate_indexed_database()
    else:
        log("[sage.db] chunked-db mode: DB will be built inside search call")
        sage_config, static_mods, variable_mods = build_sage_config(cfg, imspy)

    # --- Stage 8: SAGE search ---
    with timed_stage("sage.search"):
        results_all, indexed_db = run_sage_search_combined(
            cfg,
            imspy,
            sage_config,
            indexed_db,
            queries_all,
            static_mods=static_mods,
            variable_mods=variable_mods,
        )

    # --- Stage 9: Write output ---
    with timed_stage("output"):
        write_psms_combined(cfg, imspy, results_all, out_dir=out_dir)

    # --- Final summary ---
    log(f"[done] final RAM: {get_memory_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())