# cluster_report.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    """
    out = df.copy()

    # 1) intensity
    if "intensity" not in out.columns:
        src = _first_present(out, ["raw_sum", "area_raw", "area", "apex_raw", "apex_smoothed"])
        if src is not None:
            out["intensity"] = pd.to_numeric(out[src], errors="coerce")
        else:
            out["intensity"] = np.nan

    # 2) rt_sigma / im_sigma / mz_sigma naming variants (e.g., *scale)
    if "rt_sigma" not in out.columns:
        if "rt_scale" in out.columns:
            out["rt_sigma"] = pd.to_numeric(out["rt_scale"], errors="coerce")
        else:
            out["rt_sigma"] = np.nan

    if "im_sigma" not in out.columns:
        if "scan_scale" in out.columns:
            out["im_sigma"] = pd.to_numeric(out["scan_scale"], errors="coerce")
        elif "im_scale" in out.columns:
            out["im_sigma"] = pd.to_numeric(out["im_scale"], errors="coerce")
        else:
            out["im_sigma"] = np.nan

    if "mz_sigma" not in out.columns and "mz_scale" in out.columns:
        out["mz_sigma"] = pd.to_numeric(out["mz_scale"], errors="coerce")

    # 3) mz_mu (apex) must be numeric if present
    if "mz_mu" in out.columns:
        out["mz_mu"] = pd.to_numeric(out["mz_mu"], errors="coerce")

    # 4) ms_level default
    if "ms_level" not in out.columns:
        out["ms_level"] = 1  # sensible default for precursor; MS2 callers will override upstream

    # 5) mz_ppm: prefer explicit error; else derive from sigma/mu
    if "mz_ppm" not in out.columns:
        if "mz_error_ppm" in out.columns:
            out["mz_ppm"] = pd.to_numeric(out["mz_error_ppm"], errors="coerce").abs()
        elif "mz_sigma" in out.columns and "mz_mu" in out.columns:
            mu = out["mz_mu"].replace(0, np.nan)
            out["mz_ppm"] = (pd.to_numeric(out["mz_sigma"], errors="coerce") / mu) * 1e6
        else:
            out["mz_ppm"] = np.nan

    # Coerce common numeric fields just in case
    for col in ["rt_sigma", "im_sigma", "mz_sigma", "intensity", "mz_ppm"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# ---------- summaries ----------

def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    df = canonicalize_cluster_df(df)

    inten     = df["intensity"].to_numpy()
    rt_sigma  = df["rt_sigma"].to_numpy()
    im_sigma  = df["im_sigma"].to_numpy()
    mz_ppm    = df["mz_ppm"].to_numpy()

    res: Dict[str, Any] = {}
    res["n_clusters"]        = int(len(df))
    res["intensity_sum"]     = float(np.nansum(inten))
    res["intensity_median"]  = float(np.nanmedian(inten))
    res["rt_sigma_median"]   = float(np.nanmedian(rt_sigma))
    res["im_sigma_median"]   = float(np.nanmedian(im_sigma))
    res["mz_ppm_median"]     = float(np.nanmedian(mz_ppm))

    # Per-MS-level TIC (robust to missing 'intensity')
    for lvl in sorted(pd.unique(pd.to_numeric(df["ms_level"], errors="coerce").fillna(0))):
        sub = df[df["ms_level"] == lvl]
        res[f"ms{int(lvl)}_tic"] = float(np.nansum(pd.to_numeric(sub["intensity"], errors="coerce")))

    return res


# ---------- plotting (use canonical columns) ----------

def _hist(ax, data, bins=50, title=None, xlabel=None):
    data = pd.to_numeric(pd.Series(data), errors="coerce").to_numpy()
    data = data[np.isfinite(data)]
    if data.size:
        ax.hist(data, bins=bins)
    ax.set_title(title or "")
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel("count")

def _kde_or_hist(ax, data, title=None, xlabel=None):
    # Simple fallback to hist; keep it deterministic
    _hist(ax, data, bins=50, title=title, xlabel=xlabel)

def plot_basic_panels(df: pd.DataFrame, out_dir: Path):
    df = canonicalize_cluster_df(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_int, ax_int = plt.subplots(figsize=(6,4))
    _hist(ax_int, df["intensity"], bins=60, title="Intensity", xlabel="intensity (a.u.)")
    fig_int.tight_layout()
    fig_int.savefig(out_dir / "intensity_hist.png", dpi=150)
    plt.close(fig_int)

    fig_rt, ax_rt = plt.subplots(figsize=(6,4))
    _kde_or_hist(ax_rt, df["rt_sigma"], title="RT σ", xlabel="seconds")
    fig_rt.tight_layout()
    fig_rt.savefig(out_dir / "rt_sigma_hist.png", dpi=150)
    plt.close(fig_rt)

    fig_im, ax_im = plt.subplots(figsize=(6,4))
    _kde_or_hist(ax_im, df["im_sigma"], title="IM σ", xlabel="scans")
    fig_im.tight_layout()
    fig_im.savefig(out_dir / "im_sigma_hist.png", dpi=150)
    plt.close(fig_im)

    fig_mz, ax_mz = plt.subplots(figsize=(6,4))
    _kde_or_hist(ax_mz, df["mz_ppm"], title="m/z error", xlabel="ppm (abs)")
    fig_mz.tight_layout()
    fig_mz.savefig(out_dir / "mz_ppm_hist.png", dpi=150)
    plt.close(fig_mz)


# ---------- top-level API ----------

def save_run_report(
    df: pd.DataFrame,
    out_dir: Path | str,
    *,
    title: Optional[str] = None,
    mode: str = "ms1",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize first
    df_norm = canonicalize_cluster_df(df)

    # Summary JSON
    summary = compute_summary(df_norm)
    if extra_meta:
        summary["meta"] = extra_meta
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Figures
    plot_basic_panels(df_norm, out_dir)

    # Lightweight HTML
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>{(title or 'Cluster Report')}</title>
    <style>
      body {{ font-family: sans-serif; margin: 1.5rem; }}
      .row {{ display:flex; flex-wrap:wrap; gap:12px; }}
      .card {{ border:1px solid #ddd; padding:12px; border-radius:8px; }}
      img {{ max-width: 100%; height:auto; }}
      .grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap:12px; }}
      code, pre {{ background:#f7f7f7; padding:4px 6px; border-radius:4px; }}
    </style>
  </head>
  <body>
    <h1>{(title or 'Cluster Report')}</h1>
    <h3>Mode: {mode.upper()}</h3>
    <pre>{json.dumps(summary, indent=2)}</pre>
    <div class="grid">
      <div class="card"><h3>Intensity</h3><img src="intensity_hist.png"/></div>
      <div class="card"><h3>RT σ</h3><img src="rt_sigma_hist.png"/></div>
      <div class="card"><h3>IM σ</h3><img src="im_sigma_hist.png"/></div>
      <div class="card"><h3>m/z error (ppm)</h3><img src="mz_ppm_hist.png"/></div>
    </div>
  </body>
</html>"""
    (out_dir / "report.html").write_text(html)


def save_sweep_index(root_out: Path | str) -> None:
    """
    Build a simple index.html that links to each run subdir containing a report.
    """
    root = Path(root_out)
    cards = []
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        rep = sub / "report.html"
        summ = sub / "summary.json"
        if rep.exists() and summ.exists():
            try:
                meta = json.loads(summ.read_text())
            except Exception:
                meta = {}
            n = meta.get("n_clusters", "NA")
            tic = meta.get("intensity_sum", "NA")
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
      body {{ font-family: sans-serif; margin: 1.5rem; }}
      .grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); gap:12px; }}
      .card {{ border:1px solid #ddd; padding:12px; border-radius:8px; }}
      a {{ text-decoration:none; }}
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