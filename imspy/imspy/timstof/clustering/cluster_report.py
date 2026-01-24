# cluster_report.py
from __future__ import annotations
import argparse, json, math, os, sys
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ----------------------------- Config ---------------------------------

NUMERIC_COLS = [
    "rt_mu","rt_sigma","rt_height","rt_baseline","rt_area","rt_r2","rt_n",
    "im_mu","im_sigma","im_height","im_baseline","im_area","im_r2","im_n",
    "mz_mu","mz_sigma","mz_height","mz_baseline","mz_area","mz_r2","mz_n",
    "volume_proxy","frame_count","raw_sum","raw_points_n",
]
BOOL_COLS = [
    "has_rt_axis","has_im_axis","has_mz_axis",
    "empty_rt","empty_im","empty_mz","any_empty_dim","raw_points_attached","raw_empty",
]
CATEGORICAL_COLS = ["ms_level","window_group","parent_im_id","parent_rt_id"]

# pairs for density hexbins (log1p where appropriate)
DENSITY_PAIRS = [
    ("rt_area", "im_area", True, True, "log1p(RT area)", "log1p(IM area)"),
    ("im_area", "volume_proxy", True, True, "log1p(IM area)", "log1p(Volume proxy)"),
    ("rt_area", "volume_proxy", True, True, "log1p(RT area)", "log1p(Volume proxy)"),
    ("rt_sigma", "im_sigma", False, False, "RT σ (s)", "IM σ"),
    ("mz_sigma", "rt_sigma", False, False, "m/z σ", "RT σ (s)"),
    ("im_mu", "rt_mu", False, False, "IM μ", "RT μ (s)"),
    ("mz_mu", "im_mu", False, False, "m/z μ", "IM μ"),
    ("mz_mu", "rt_mu", False, False, "m/z μ", "RT μ (s)"),
]

# univariate hists: (column, log1p?, xlabel, nbins)
HIST_SPECS = [
    ("volume_proxy", True, "log1p(Volume proxy)", 200),
    ("raw_sum", True, "log1p(Raw sum)", 200),
    ("frame_count", False, "Frame count", 100),
    ("rt_area", True, "log1p(RT area)", 200),
    ("im_area", True, "log1p(IM area)", 200),
    ("mz_area", True, "log1p(m/z area)", 200),
    ("rt_sigma", False, "RT σ (s)", 150),
    ("im_sigma", False, "IM σ", 150),
    ("mz_sigma", False, "m/z σ", 150),
    ("rt_r2", False, "RT R²", 100),
    ("im_r2", False, "IM R²", 100),
    ("mz_r2", False, "m/z R²", 100),
    ("raw_points_n", True, "log1p(# raw points)", 150),
]

# correlations on a sampled subset
CORR_COLS = [
    "rt_area","im_area","mz_area","volume_proxy","frame_count",
    "rt_sigma","im_sigma","mz_sigma","rt_height","im_height","mz_height",
]
MAX_POINTS_HEX = 2_000_000  # cap for 2D density inputs (subsample if above)
SAMPLE_FOR_CORR = 1_000_000
RANDOM_SEED = 1337


# ----------------------------- Utils ----------------------------------

def safe_log1p(x: np.ndarray) -> np.ndarray:
    """log1p(max(x, 0)). Preserves NaNs."""
    out = x.copy()
    mask = np.isfinite(out)
    out[mask] = np.log1p(np.maximum(out[mask], 0.0))
    return out

def finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def maybe_log1p(arr: np.ndarray, do: bool) -> np.ndarray:
    return safe_log1p(arr) if do else arr

def downsample_idx(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return rng.choice(n, size=max_n, replace=False)

def quantiles(a: np.ndarray, qs=(0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1.0)) -> dict:
    a = a[np.isfinite(a)]
    if a.size == 0: return {}
    qv = np.quantile(a, qs)
    return {f"q{int(q*100):02d}": float(v) for q, v in zip(qs, qv)}

def bool_counts(df: pd.DataFrame, cols: Sequence[str]) -> dict:
    out = {}
    for c in cols:
        if c in df:
            vc = df[c].value_counts(dropna=False)
            out[c] = {str(k): int(v) for k, v in vc.items()}
    return out

def value_counts_top(df: pd.DataFrame, col: str, k: int = 30) -> list[tuple]:
    if col not in df: return []
    vc = df[col].value_counts(dropna=False).head(k)
    return [(str(i), int(n)) for i, n in vc.items()]

def pathology_rate(x: np.ndarray, predicate) -> float:
    m = np.isfinite(x)
    if not m.any(): return float("nan")
    return float(np.mean(predicate(x[m])))


# ----------------------------- Plots -----------------------------------

def fig_hist(ax, data: np.ndarray, bins: int, xlabel: str) -> None:
    data = data[np.isfinite(data)]
    if data.size == 0:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
        ax.set_xlabel(xlabel)
        return
    ax.hist(data, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

def fig_hexbin(ax, x: np.ndarray, y: np.ndarray, gridsize: int,
               xlabel: str, ylabel: str) -> None:
    x, y = finite_pair(x, y)
    if x.size == 0:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        return
    hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Density")


# ----------------------------- Core ------------------------------------

@dataclass
class ReportResult:
    json_summary: dict
    out_pdf_path: str


def build_cluster_report(
    df: pd.DataFrame,
    out_pdf_path: str,
    json_out_path: Optional[str] = None,
    title: str = "timsTOF Cluster Summary",
) -> ReportResult:
    """Builds a multi-page PDF + JSON summary from the provided DataFrame."""
    need_cols = set(NUMERIC_COLS + BOOL_COLS + CATEGORICAL_COLS)
    ensure_columns(df, [c for c in need_cols if c in df.columns])  # allow missing cats

    rng = np.random.default_rng(RANDOM_SEED)

    # Casts that reduce memory but keep numeric fidelity
    for c in BOOL_COLS:
        if c in df and df[c].dtype != "bool":
            df[c] = df[c].astype("bool", copy=False)

    # -------- Summary JSON --------
    js = {}
    js["n_rows"] = int(len(df))
    js["n_cols"] = int(len(df.columns))

    # boolean tallies
    js["bool_counts"] = bool_counts(df, BOOL_COLS)

    # categorical head counts
    for c in ["ms_level","window_group"]:
        js[f"value_counts_{c}"] = value_counts_top(df, c, k=50)

    # quantiles for key numerics
    js["quantiles"] = {}
    for c in NUMERIC_COLS:
        if c in df:
            js["quantiles"][c] = quantiles(df[c].to_numpy())

    # pathology checks (rates)
    for c in ["rt_mu","im_mu","mz_mu"]:
        if c in df:
            js[f"{c}_is_zero_rate"] = pathology_rate(df[c].to_numpy(), lambda v: v == 0)
    for c in ["rt_sigma","im_sigma","mz_sigma"]:
        if c in df:
            js[f"{c}_nonpos_rate"] = pathology_rate(df[c].to_numpy(), lambda v: v <= 0)
    for c in ["rt_r2","im_r2","mz_r2"]:
        if c in df:
            js[f"{c}_outside_0_1_rate"] = pathology_rate(
                df[c].to_numpy(), lambda v: (v < 0) | (v > 1)
            )

    # correlations on a sample (robust to huge N)
    corr_sample = df[[c for c in CORR_COLS if c in df]].dropna()
    if len(corr_sample) > SAMPLE_FOR_CORR:
        corr_sample = corr_sample.sample(SAMPLE_FOR_CORR, random_state=RANDOM_SEED)
    if not corr_sample.empty:
        C = corr_sample.corr(numeric_only=True)
        js["correlations"] = C.round(4).to_dict()
    else:
        js["correlations"] = {}

    # -------- PDF Report --------
    os.makedirs(os.path.dirname(out_pdf_path) or ".", exist_ok=True)
    with PdfPages(out_pdf_path) as pdf:

        # Title / overview
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.axis("off")
        txt = [
            title,
            f"Rows: {js['n_rows']:,}",
            f"Columns: {js['n_cols']:,}",
            "",
            "Boolean tallies:",
        ]
        ax.text(0.01, 0.95, "\n".join(txt), va="top", fontsize=14)
        y = 0.75
        for c, vc in js["bool_counts"].items():
            ax.text(0.03, y, f"• {c}: {vc}", va="top", fontsize=10)
            y -= 0.05
        pdf.savefig(fig); plt.close(fig)

        # Categorical counts
        for cname, label in [("ms_level","ms_level"), ("window_group","window_group")]:
            if cname in df:
                vc = df[cname].value_counts().head(40)
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                ax.bar(vc.index.astype(str), vc.values)
                ax.set_title(f"Top {len(vc)} by {label}")
                ax.set_ylabel("Count")
                ax.set_xticks(range(len(vc)))
                ax.set_xticklabels([str(x) for x in vc.index], rotation=90)
                plt.tight_layout()
                pdf.savefig(fig); plt.close(fig)

        # Univariate histograms
        for col, do_log, xlabel, bins in HIST_SPECS:
            if col not in df: continue
            a = df[col].to_numpy()
            a = maybe_log1p(a, do_log)
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            fig_hist(ax, a, bins=bins, xlabel=xlabel)
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Density (hexbin) plots with downsampling
        n_total = len(df)
        idx = downsample_idx(n_total, MAX_POINTS_HEX, rng)
        df2d = df.iloc[idx]

        for xcol, ycol, logx, logy, xl, yl in DENSITY_PAIRS:
            if xcol not in df2d or ycol not in df2d: continue
            x = maybe_log1p(df2d[xcol].to_numpy(), logx)
            y = maybe_log1p(df2d[ycol].to_numpy(), logy)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            fig_hexbin(ax, x, y, gridsize=200, xlabel=xl, ylabel=yl)
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Pathology dashboards (zeros/NaNs/nonpositive)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        labels, vals = [], []
        for k, v in js.items():
            if k.endswith(("_is_zero_rate","_nonpos_rate","_outside_0_1_rate")) and np.isfinite(v):
                labels.append(k); vals.append(v)
        if vals:
            order = np.argsort(vals)[::-1]
            labels = [labels[i] for i in order]
            vals = [vals[i] for i in order]
            ax.bar(range(len(vals)), vals)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_ylabel("Rate")
            ax.set_title("Pathology Rates")
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Correlation heatmap (coarse, no styling)
        if js["correlations"]:
            C = pd.DataFrame(js["correlations"]).reindex(columns=CORR_COLS, index=CORR_COLS)
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111)
            im = ax.imshow(C.to_numpy(), vmin=-1, vmax=1)
            ax.set_xticks(range(len(C.columns))); ax.set_xticklabels(C.columns, rotation=90)
            ax.set_yticks(range(len(C.index))); ax.set_yticklabels(C.index)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Pearson r")
            ax.set_title("Correlation matrix (sampled)")
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    # JSON out
    if json_out_path:
        with open(json_out_path, "w") as f:
            json.dump(js, f, indent=2)

    return ReportResult(json_summary=js, out_pdf_path=out_pdf_path)


# ----------------------------- IO & CLI --------------------------------

def load_any(path: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path, columns=columns)
    if ext in (".feather", ".ft"):
        return pd.read_feather(path, columns=columns)
    if ext in (".csv", ".tsv", ".txt"):
        sep = "," if ext == ".csv" else "\t"
        # note: CSV is memory heavy; consider converting to parquet beforehand
        return pd.read_csv(path, sep=sep, usecols=columns)
    raise ValueError(f"Unsupported file extension: {ext}")

def main():
    p = argparse.ArgumentParser(description="timsTOF cluster summary report")
    p.add_argument("--in", dest="inp", required=True, help="Input table (parquet/feather/csv)")
    p.add_argument("--out", dest="out_pdf", required=True, help="Output PDF path")
    p.add_argument("--json", dest="json_out", default=None, help="Optional JSON summary path")
    p.add_argument("--title", dest="title", default="timsTOF Cluster Summary")
    args = p.parse_args()

    needed = list(set(NUMERIC_COLS + BOOL_COLS + CATEGORICAL_COLS) & set(pd.read_parquet(args.inp, columns=[]).columns if args.inp.endswith(".parquet") else []))
    # Load full table (Parquet/Feather recommended)
    df = load_any(args.inp)  # Use column pruning above if needed

    res = build_cluster_report(df, args.out_pdf, json_out_path=args.json_out, title=args.title)
    print(f"Wrote: {res.out_pdf_path}")
    if args.json_out:
        print(f"Wrote: {args.json_out}")

if __name__ == "__main__":
    main()