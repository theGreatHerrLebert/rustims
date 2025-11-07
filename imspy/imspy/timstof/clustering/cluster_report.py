# cluster_report.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, Callable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ArrayLike = Sequence[float]

# ---------- helpers: robust column resolution ----------

def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def canonicalize_cluster_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make plotting/summary robust across schema variants.
    Ensures presence of: intensity, rt_sigma, im_sigma, mz_mu, mz_sigma, mz_ppm, ms_level
    Derives: log10_intensity, rt_fwhm, im_fwhm
    """
    out = df.copy()

    # intensity
    if "intensity" not in out.columns:
        src = _first_present(out, ["raw_sum", "volume_proxy", "rt_area", "im_area", "mz_area",
                                   "area_raw", "area", "apex_raw", "apex_smoothed"])
        out["intensity"] = pd.to_numeric(out[src], errors="coerce") if src else np.nan

    # rt_sigma
    if "rt_sigma" not in out.columns:
        src = _first_present(out, ["rt_scale", "rt_std", "rt_width"])
        out["rt_sigma"] = pd.to_numeric(out[src], errors="coerce") if src else np.nan

    # im_sigma
    if "im_sigma" not in out.columns:
        src = _first_present(out, ["scan_scale", "im_scale", "im_std", "im_width"])
        out["im_sigma"] = pd.to_numeric(out[src], errors="coerce") if src else np.nan

    # mz_sigma
    if "mz_sigma" not in out.columns:
        src = _first_present(out, ["mz_scale", "mz_std", "mz_width"])
        out["mz_sigma"] = pd.to_numeric(out[src], errors="coerce") if src else np.nan
    else:
        out["mz_sigma"] = pd.to_numeric(out["mz_sigma"], errors="coerce")

    # mz_mu
    if "mz_mu" in out.columns:
        out["mz_mu"] = pd.to_numeric(out["mz_mu"], errors="coerce")

    # ms_level
    if "ms_level" not in out.columns:
        out["ms_level"] = 1

    # mz_ppm
    if "mz_ppm" not in out.columns:
        if "mz_error_ppm" in out.columns:
            out["mz_ppm"] = pd.to_numeric(out["mz_error_ppm"], errors="coerce").abs()
        elif "mz_sigma" in out.columns and "mz_mu" in out.columns:
            mu = pd.to_numeric(out["mz_mu"], errors="coerce").replace(0, np.nan)
            out["mz_ppm"] = (pd.to_numeric(out["mz_sigma"], errors="coerce") / mu) * 1e6
        else:
            out["mz_ppm"] = np.nan

    # log10 intensity (quietly; zeros/negatives -> NaN)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log10_intensity"] = np.log10(pd.to_numeric(out["intensity"], errors="coerce"))
    out.loc[~np.isfinite(out["log10_intensity"]), "log10_intensity"] = np.nan

    # FWHMs if sigmas exist: FWHM ≈ 2.3548 * σ
    out["rt_fwhm"] = 2.354820045 * pd.to_numeric(out["rt_sigma"], errors="coerce")
    out["im_fwhm"] = 2.354820045 * pd.to_numeric(out["im_sigma"], errors="coerce")

    # coerce common numeric cols
    for col in ["rt_sigma", "im_sigma", "mz_sigma", "intensity", "mz_ppm",
                "rt_fwhm", "im_fwhm", "rt_mu", "im_mu"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# ---------- summaries ----------

def _nanpct(s: pd.Series) -> float:
    return float(np.mean(pd.isna(pd.to_numeric(s, errors="coerce"))) * 100.0)

def _percentiles(x: ArrayLike, qs=(5, 25, 50, 75, 95)) -> Dict[str, float]:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    pr = np.nanpercentile(arr, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, pr)}

def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    df = canonicalize_cluster_df(df)

    res: Dict[str, Any] = {}
    res["n_clusters"] = int(len(df))

    # TICs (overall and per level)
    res["tic_sum"] = float(np.nansum(df["intensity"]))
    for lvl in sorted(pd.unique(pd.to_numeric(df["ms_level"], errors="coerce").fillna(0))):
        sub = df[df["ms_level"] == lvl]
        res[f"ms{int(lvl)}_tic"] = float(np.nansum(sub["intensity"]))

    # medians
    for k in ["intensity", "log10_intensity", "rt_sigma", "im_sigma", "mz_ppm"]:
        if k in df.columns:
            res[f"{k}_median"] = float(np.nanmedian(df[k]))

    # percentiles
    res["intensity_percentiles"] = _percentiles(df["intensity"], qs=(1,5,25,50,75,95,99))
    for k in ["rt_sigma", "im_sigma", "mz_ppm"]:
        if k in df.columns:
            res[f"{k}_percentiles"] = _percentiles(df[k])

    # flags
    for k in ["has_rt_axis","has_im_axis","has_mz_axis","empty_rt","empty_im","empty_mz","any_empty_dim","raw_empty"]:
        if k in df.columns:
            res[f"{k}_rate_pct"] = float(np.mean(pd.to_numeric(df[k], errors="coerce").fillna(0) != 0) * 100.0)

    # NaN rates for important cols
    for k in ["intensity","rt_sigma","im_sigma","mz_mu","mz_sigma","mz_ppm"]:
        if k in df.columns:
            res[f"{k}_nan_pct"] = _nanpct(df[k])

    return res


# ---------- plotting primitives ----------

def _hist(ax, data, bins=60, title=None, xlabel=None, log=False):
    arr = pd.to_numeric(pd.Series(data), errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size:
        ax.hist(arr, bins=bins)
        if log:
            ax.set_yscale("log")
    ax.set_title(title or "")
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel("count")

def _hex(ax, x, y, title=None, xlabel=None, ylabel=None, gridsize=50):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() > 0:
        hb = ax.hexbin(x[m], y[m], gridsize=gridsize, bins="log")
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label("log10 density")
    ax.set_title(title or "")
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")

def _scatter(ax, x, y, c=None, title=None, xlabel=None, ylabel=None, s=6, alpha=0.5):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    if c is not None:
        c = pd.to_numeric(pd.Series(c), errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
        if m.sum() > 0:
            sc = ax.scatter(x[m], y[m], c=c[m], s=s, alpha=alpha)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("color")
    else:
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 0:
            ax.scatter(x[m], y[m], s=s, alpha=alpha)
    ax.set_title(title or "")
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")

# ---------- high-level figures ----------

def plot_distributions(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = canonicalize_cluster_df(df)

    # Intensity (linear + log10)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    _hist(axs[0], df["intensity"], bins=80, title="Intensity", xlabel="a.u.")
    _hist(axs[1], df["log10_intensity"], bins=80, title="log10(Intensity)", xlabel="log10(a.u.)")
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_panels.png", dpi=150)
    plt.close(fig)

    # Widths / errors
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    _hist(axs[0], df["rt_sigma"], bins=80, title="RT σ", xlabel="seconds", log=True)
    _hist(axs[1], df["im_sigma"], bins=80, title="IM σ", xlabel="scans", log=True)
    _hist(axs[2], df["mz_ppm"],   bins=80, title="m/z error", xlabel="ppm (abs)")
    fig.tight_layout()
    fig.savefig(out_dir / "widths_ppm_hist.png", dpi=150)
    plt.close(fig)

def plot_relationships(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = canonicalize_cluster_df(df)

    # RTσ vs IMσ colored by log10 intensity
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, df["rt_sigma"], df["im_sigma"], df["log10_intensity"],
             title="RT σ vs IM σ", xlabel="RT σ (s)", ylabel="IM σ (scans)")
    fig.tight_layout()
    fig.savefig(out_dir / "rt_im_sigma_scatter.png", dpi=150)
    plt.close(fig)

    # mz_ppm vs m/z apex
    if "mz_mu" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        _hex(ax, df["mz_mu"], df["mz_ppm"], title="m/z error vs m/z apex", xlabel="m/z apex", ylabel="ppm")
        fig.tight_layout()
        fig.savefig(out_dir / "mzppm_vs_mz_hex.png", dpi=150)
        plt.close(fig)

    # intensity vs RTσ (hex)
    fig, ax = plt.subplots(figsize=(6, 5))
    _hex(ax, df["rt_sigma"], df["log10_intensity"], title="log10(I) vs RT σ",
         xlabel="RT σ (s)", ylabel="log10(I)")
    fig.tight_layout()
    fig.savefig(out_dir / "logI_vs_rtsigma_hex.png", dpi=150)
    plt.close(fig)

def plot_correlations(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = canonicalize_cluster_df(df)

    cols = [c for c in [
        "log10_intensity","intensity","rt_sigma","im_sigma","mz_ppm",
        "rt_fwhm","im_fwhm","rt_r2","im_r2","mz_r2"
    ] if c in df.columns]
    if not cols:
        return

    # Coerce each column separately; drop columns that are entirely NaN
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().any(axis=0)]

    # Drop constant columns (<=1 unique non-NaN value)
    const_cols = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    if const_cols:
        num = num.drop(columns=const_cols)

    # Need at least 2 columns and some non-NaN rows
    if num.shape[1] < 2 or num.dropna(how="all").shape[0] == 0:
        # Still write an empty CSV for reproducibility
        (out_dir / "correlations_spearman.csv").write_text("")
        return

    C = num.corr(method="spearman")
    C.to_csv(out_dir / "correlations_spearman.csv")

    fig, ax = plt.subplots(figsize=(0.9*len(C.columns)+2, 0.9*len(C.columns)+2))
    im = ax.imshow(C.values, aspect="auto")
    ax.set_xticks(range(len(C.columns))); ax.set_xticklabels(C.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(C.columns))); ax.set_yticklabels(C.columns)
    cb = plt.colorbar(im, ax=ax); cb.set_label("Spearman ρ")
    ax.set_title("Correlation matrix (Spearman)")
    fig.tight_layout()
    fig.savefig(out_dir / "correlations_heatmap.png", dpi=150)
    plt.close(fig)

def plot_per_level(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = canonicalize_cluster_df(df)
    levels = sorted(pd.unique(pd.to_numeric(df["ms_level"], errors="coerce").dropna()))
    for lvl in levels:
        sub = df[df["ms_level"] == lvl]
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        _hist(axs[0], sub["log10_intensity"], bins=80, title=f"MS{int(lvl)} log10(I)", xlabel="log10(a.u.)")
        _hist(axs[1], sub["rt_sigma"], bins=80, title=f"MS{int(lvl)} RT σ", xlabel="s", log=True)
        _hist(axs[2], sub["im_sigma"], bins=80, title=f"MS{int(lvl)} IM σ", xlabel="scans", log=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"ms{int(lvl)}_panels.png", dpi=150)
        plt.close(fig)


# ---------- example cluster rendering (optional) ----------

def _render_example_cluster(
    rec: pd.Series,
    ds: Any,
    extractor: Callable[..., Optional[np.ndarray]],
    out_dir: Path,
    is_precursor: bool,
    mz_pad: float = 0.0,
    vmax_pct: float = 99.5,
) -> Optional[str]:
    try:
        M = extractor(rec, ds, is_precursor=is_precursor, mz_pad=mz_pad)
        if M is None or not isinstance(M, np.ndarray) or M.size == 0:
            return None
        # Plot heatmap + RT/IM marginals
        fig = plt.figure(figsize=(6.8, 4.8))
        gs = fig.add_gridspec(2, 2, width_ratios=[4,1], height_ratios=[1,4], hspace=0.05, wspace=0.05)
        ax_top = fig.add_subplot(gs[0,0]); ax_right = fig.add_subplot(gs[1,1])
        ax_main = fig.add_subplot(gs[1,0])

        vmax = np.nanpercentile(M[np.isfinite(M)], vmax_pct) if np.isfinite(M).any() else None
        im = ax_main.imshow(M, origin="lower", aspect="auto", vmax=vmax)
        cb = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
        cb.set_label("intensity (a.u.)")

        ax_top.plot(np.nansum(M, axis=1))
        ax_right.plot(np.nansum(M, axis=0), np.arange(M.shape[1]))
        ax_top.set_xticks([]); ax_right.set_yticks([])
        ax_top.set_ylabel("RT marginal"); ax_right.set_xlabel("IM marginal")
        ax_main.set_xlabel("IM scans"); ax_main.set_ylabel("RT frames")

        fname = f"example_c_{int(rec.get('parent_im_id', 0))}_{int(rec.get('parent_rt_id', 0))}.png"
        fig.suptitle(f"Example cluster — ms{int(rec.get('ms_level',1))}  I={rec.get('intensity',np.nan):.2g}")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        return fname
    except Exception:
        return None

def render_example_clusters(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    ds: Any = None,
    extractor: Optional[Callable[..., Optional[np.ndarray]]] = None,
    mode: str = "ms1",
    n_examples: int = 8,
    mz_pad: float = 0.0,
) -> Dict[str, Any]:
    """
    Writes up to n_examples PNGs of representative clusters and returns a dict
    with file names and selection metadata. If extractor or ds is None, returns {}.
    """
    if ds is None or extractor is None:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    df = canonicalize_cluster_df(df).copy()

    # choose by intensity quantiles to get diversity
    df = df.sort_values("intensity", ascending=False)
    qs = np.linspace(0.05, 0.95, num=max(1, n_examples))
    picks = [int(np.clip(int(q*len(df)), 0, len(df)-1)) for q in qs] if len(df) else []
    chosen = df.iloc[picks] if len(picks) else df.head(0)

    files = []
    for _, rec in chosen.iterrows():
        fn = _render_example_cluster(
            rec, ds, extractor, out_dir,
            is_precursor=(mode.lower() == "ms1"),
            mz_pad=mz_pad
        )
        if fn:
            files.append(fn)

    return {"examples": files, "count": len(files)}


# ---------- top-level API ----------

def save_run_report(
    df: pd.DataFrame,
    out_dir: Path | str,
    *,
    title: Optional[str] = None,
    mode: str = "ms1",
    extra_meta: Optional[Dict[str, Any]] = None,
    ds: Any = None,
    slice_extractor: Optional[Callable[..., Optional[np.ndarray]]] = None,
    n_examples: int = 8,
    example_mz_pad: float = 0.0,
) -> None:
    """
    Save JSON summary, PNG figures, and an HTML dashboard.
    Optional: pass ds + slice_extractor(rec, ds, is_precursor=True/False, mz_pad=...)
              to render example cluster tiles.
    """
    out_dir = Path(out_dir)
    figs = out_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    # Normalize once
    df_norm = canonicalize_cluster_df(df)

    # Summary JSON
    summary = compute_summary(df_norm)
    if extra_meta:
        summary["meta"] = extra_meta
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # CSV dump of normalized DF (lightweight)
    df_norm.to_csv(out_dir / "clusters.normalized.csv.gz", index=False, compression="gzip")

    # Figures
    plot_distributions(df_norm, figs)
    plot_relationships(df_norm, figs)
    plot_correlations(df_norm, figs)
    plot_per_level(df_norm, figs)

    # Example clusters (optional)
    examples = render_example_clusters(
        df_norm, figs, ds=ds, extractor=slice_extractor,
        mode=mode, n_examples=n_examples, mz_pad=example_mz_pad
    )

    # Lightweight HTML
    cards = [
        ('Intensity panels', 'figs/intensity_panels.png'),
        ('Widths & m/z error', 'figs/widths_ppm_hist.png'),
        ('RTσ vs IMσ (colored by log10 I)', 'figs/rt_im_sigma_scatter.png'),
        ('m/z error vs m/z apex', 'figs/mzppm_vs_mz_hex.png'),
        ('log10(I) vs RTσ (hex)', 'figs/logI_vs_rtsigma_hex.png'),
        ('Correlation heatmap', 'figs/correlations_heatmap.png'),
    ]
    per_level_imgs = []
    for p in figs.glob("ms*_panels.png"):
        per_level_imgs.append((p.stem.replace("_", " ").upper(), f"figs/{p.name}"))

    example_imgs = examples.get("examples", [])
    example_html = "".join(
        f'<div class="card"><img src="figs/{fn}"/></div>' for fn in example_imgs
    ) if example_imgs else "<p>No examples rendered.</p>"

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>{(title or 'Cluster Report')}</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 1.5rem; }}
      .grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(360px,1fr)); gap:14px; }}
      .row {{ display:flex; flex-wrap:wrap; gap:12px; }}
      .card {{ border:1px solid #ddd; padding:12px; border-radius:8px; background:#fff; }}
      img {{ max-width: 100%; height:auto; display:block; }}
      pre {{ background:#f7f7f7; padding:8px 10px; border-radius:6px; overflow:auto; }}
      a {{ color:#06c; text-decoration:none; }}
    </style>
  </head>
  <body>
    <h1>{(title or 'Cluster Report')}</h1>
    <h3>Mode: {mode.upper()}</h3>
    <p>
      <a href="summary.json">summary.json</a> ·
      <a href="clusters.normalized.csv.gz">clusters.normalized.csv.gz</a> ·
      <a href="figs/correlations_spearman.csv">correlations_spearman.csv</a>
    </p>
    <h2>Summary</h2>
    <pre>{json.dumps(summary, indent=2)}</pre>

    <h2>Distributions</h2>
    <div class="grid">
      {''.join(f'<div class="card"><h3>{t}</h3><img src="{src}"/></div>' for t,src in cards if (out_dir / src).exists())}
      {''.join(f'<div class="card"><h3>{t}</h3><img src="{src}"/></div>' for t,src in per_level_imgs)}
    </div>

    <h2>Example clusters</h2>
    <div class="grid">
      {example_html}
    </div>
  </body>
</html>"""
    (out_dir / "report.html").write_text(html)

from pathlib import Path
import json

def save_sweep_index(root_out: Path | str) -> None:
    """
    Build a simple index.html that links to each run subdir containing a report.
    A run directory is any subfolder with both report.html and summary.json.
    """
    root = Path(root_out)
    cards = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        rep = sub / "report.html"
        summ = sub / "summary.json"
        if rep.exists() and summ.exists():
            try:
                meta = json.loads(summ.read_text())
            except Exception:
                meta = {}
            n = meta.get("n_clusters", "NA")
            tic = meta.get("tic_sum", meta.get("intensity_sum", "NA"))
            cards.append(f"""
            <div class="card">
              <h3>{sub.name}</h3>
              <p>clusters: <b>{n}</b> &nbsp; TIC: <b>{tic}</b></p>
              <a href="{sub.name}/report.html">Open report</a>
            </div>""")

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Cluster sweep index</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 1.5rem; }}
      .grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); gap:12px; }}
      .card {{ border:1px solid #ddd; padding:12px; border-radius:8px; background:#fff; }}
      a {{ text-decoration:none; color:#06c; }}
    </style>
  </head>
  <body>
    <h1>Cluster sweep index</h1>
    <div class="grid">
      {''.join(cards) if cards else '<p>No reports found.</p>'}
    </div>
  </body>
</html>"""
    (root / "index.html").write_text(html)