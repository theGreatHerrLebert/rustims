# imspy_pipeline/report.py
from __future__ import annotations
import base64, io, json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- helpers ----
def _b64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _safe_quantile(s: pd.Series, q: float) -> float:
    try:
        return float(np.nanquantile(s.to_numpy(), q))
    except Exception:
        return float("nan")

def _ppm_from_sigma(row) -> Optional[float]:
    mz_mu, mz_sigma = row.get("mz_apex"), row.get("mz_scale")
    if mz_mu is None or mz_sigma is None or not np.isfinite(mz_mu) or mz_mu <= 0:
        return np.nan
    return float(mz_sigma / mz_mu * 1e6)

# ---- core metrics ----
def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    res["n_clusters"] = int(len(df))
    inten = pd.to_numeric(df.get("intensity"), errors="coerce")
    res["tic_sum"] = float(np.nansum(inten))
    res["intensity_median"] = float(np.nanmedian(inten))
    res["intensity_p90"] = float(_safe_quantile(inten, 0.90))

    # widths / scales
    rt_sigma = pd.to_numeric(df.get("rt_scale"), errors="coerce")
    im_sigma = pd.to_numeric(df.get("scan_scale"), errors="coerce")
    res["rt_sigma_median"] = float(np.nanmedian(rt_sigma))
    res["im_sigma_median"] = float(np.nanmedian(im_sigma))

    # ppm proxy
    mz_ppm = df.apply(_ppm_from_sigma, axis=1)
    res["mz_ppm_median"] = float(np.nanmedian(mz_ppm))

    # ms level / groups
    if "ms_level" in df:
        for lvl, sub in df.groupby("ms_level"):
            res[f"ms{int(lvl)}_clusters"] = int(len(sub))
            res[f"ms{int(lvl)}_tic"] = float(np.nansum(pd.to_numeric(sub["intensity"], errors="coerce")))
    if "window_group" in df:
        res["n_window_groups"] = int(df["window_group"].nunique())

    return res

# ---- plotting ----
def _hist(ax, s: pd.Series, title: str, bins=60, log=False):
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s)]
    ax.hist(s, bins=bins, log=log)
    ax.set_title(title)

def _scatter(ax, x, y, title: str, logy=False, alpha=0.5):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], s=6, alpha=alpha)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)

def _bar_by_group(ax, df: pd.DataFrame, col: str, agg="count", title="per window_group"):
    if "window_group" not in df.columns:
        ax.axis("off"); return
    g = df.groupby("window_group")[col]
    val = g.count() if agg == "count" else g.sum(min_count=1)
    val.plot(kind="bar", ax=ax)
    ax.set_title(title)

def make_figures(df: pd.DataFrame, mode: str) -> Dict[str, str]:
    figs: Dict[str, str] = {}

    # 1) histogram of intensities (log y)
    fig, ax = plt.subplots(figsize=(5,3))
    _hist(ax, df["intensity"], "Intensity (log y)", bins=80, log=True)
    ax.set_xlabel("intensity"); ax.set_ylabel("count")
    figs["hist_intensity"] = _b64_png(fig)

    # 2) widths
    fig, axs = plt.subplots(1,2, figsize=(8,3))
    _hist(axs[0], df["rt_scale"], "RT σ (frames/sec)")
    _hist(axs[1], df["scan_scale"], "IM σ (scans)")
    figs["hist_widths"] = _b64_png(fig)

    # 3) ppm proxy
    ppm = df.apply(_ppm_from_sigma, axis=1)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(ppm[np.isfinite(ppm)], bins=60)
    ax.set_title("m/z width proxy (ppm)"); ax.set_xlabel("ppm"); ax.set_ylabel("count")
    figs["hist_ppm"] = _b64_png(fig)

    # 4) apex scatter
    fig, axs = plt.subplots(1,2, figsize=(8,3))
    _scatter(axs[0], df["rt_apex"], df["scan_apex"], "RT apex vs IM apex", alpha=0.3)
    _scatter(axs[1], df["rt_scale"], df["intensity"], "RT σ vs intensity (log y)", logy=True, alpha=0.3)
    axs[0].set_xlabel("rt_apex (s or frames)"); axs[0].set_ylabel("scan_apex")
    axs[1].set_xlabel("rt_scale"); axs[1].set_ylabel("intensity")
    figs["scatter_apex"] = _b64_png(fig)

    # 5) per-group bars for MS2
    if str(mode).lower() == "ms2" and "window_group" in df.columns:
        fig, axs = plt.subplots(1,2, figsize=(9,3))
        _bar_by_group(axs[0], df, col="intensity", agg="count", title="#clusters per window_group")
        _bar_by_group(axs[1], df, col="intensity", agg="sum",   title="TIC per window_group")
        figs["bars_groups"] = _b64_png(fig)

    return figs

# ---- HTML emit ----
_HTML = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
body{{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:24px;}}
h1,h2{{margin:0 0 8px 0}}
.card{{border:1px solid #ddd; border-radius:10px; padding:12px; margin-bottom:16px}}
.grid{{display:grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap:12px}}
.kv td{{padding:2px 8px; vertical-align:top}}
.kv td:first-child{{opacity:.7}}
img{{max-width:100%}}
code{{background:#f6f8fa; padding:2px 6px; border-radius:6px}}
</style>
<h1>{title}</h1>
<div class="card">
  <h2>Summary</h2>
  <table class="kv">
    {rows}
  </table>
  <p><b>Dataset:</b> <code>{dataset}</code></p>
</div>
<div class="grid">
  <div class="card"><h3>Intensity</h3><img src="data:image/png;base64,{hist_intensity}"></div>
  <div class="card"><h3>Widths</h3><img src="data:image/png;base64,{hist_widths}"></div>
  <div class="card"><h3>m/z width (ppm proxy)</h3><img src="data:image/png;base64,{hist_ppm}"></div>
  <div class="card"><h3>Apex & Width vs Intensity</h3><img src="data:image/png;base64,{scatter_apex}"></div>
  {maybe_groups}
</div>
"""

_GROUPS_CARD = """<div class="card"><h3>Per window_group</h3><img src="data:image/png;base64,{bars_groups}"></div>"""

def save_run_report(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    title: str = "Cluster Report",
    mode: str = "ms1",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Persist DF if not already
    if not (out_dir / "clusters.parquet").exists():
        df.to_parquet(out_dir / "clusters.parquet", index=False)

    # Compute summary & figures
    summary = compute_summary(df)
    figs = make_figures(df, mode=mode)

    # Write summary.json
    meta = {"summary": summary, "mode": mode}
    if extra_meta: meta["meta"] = extra_meta
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))

    # Save key PNGs to disk (thumbnails etc.)
    for key, b64 in figs.items():
        (out_dir / f"{key}.png").write_bytes(base64.b64decode(b64))

    # Build rows
    rows = "\n".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary.items()
    )
    maybe_groups = _GROUPS_CARD.format(bars_groups=figs["bars_groups"]) if "bars_groups" in figs else ""

    html = _HTML.format(
        title=title,
        rows=rows,
        dataset=(extra_meta or {}).get("dataset", ""),
        maybe_groups=maybe_groups,
        **figs,
    )
    (out_dir / "report.html").write_text(html, encoding="utf-8")

def _load_summary(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads((p / "summary.json").read_text())
    except Exception:
        return None

def save_sweep_index(root: Path) -> None:
    root = Path(root)
    items = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        meta = _load_summary(d)
        if not meta: continue
        thumb = "hist_intensity.png" if (d / "hist_intensity.png").exists() else None
        items.append((d.name, meta["summary"], thumb))

    html = ["<!doctype html><meta charset='utf-8'><title>Sweep index</title><style>body{font-family:system-ui;margin:24px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px} .card{border:1px solid #ddd;border-radius:10px;padding:12px}</style><h1>Sweep index</h1><div class='grid'>"]
    for name, s, thumb in items:
        html.append("<div class='card'>")
        html.append(f"<h3>{name}</h3>")
        if thumb:
            html.append(f"<img style='max-width:100%' src='{name}/{thumb}'/>")
        html.append("<ul>")
        for k in ["n_clusters","tic_sum","intensity_median","mz_ppm_median","rt_sigma_median","im_sigma_median"]:
            if k in s:
                html.append(f"<li><b>{k}</b>: {s[k]}</li>")
        html.append("</ul>")
        html.append(f"<a href='{name}/report.html'>open report</a>")
        html.append("</div>")
    html.append("</div>")
    (root / "index.html").write_text("\n".join(html), encoding="utf-8")