"""Reviewer 2 views (i) and (iii): binned marginals from raw MS1 signal and from the simulated ions.

(i)   every MS1 frame of a .d, read with imspy (Bruker SDK calibration). Intensity-weighted histograms of
      RT, 1/K0 and m/z, plus a peak-count histogram of log10 intensity. No identification is involved.
(iii) the ions in synthetic_data.db: RT, 1/K0, m/z and charge, counted and abundance-weighted.

Runs inside the timsim container (imspy-core). Writes one JSON of histograms per run.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

# Fixed bins, the same for every run and both sides (claims/marginals.yaml).
BINS = {
    "rt": np.arange(0, 125.0001, 0.25),        # min
    "im": np.arange(0.6, 1.6001, 0.005),       # 1/K0
    "mz": np.arange(100, 1700.0001, 2.0),      # Th
    "log10_intensity": np.arange(0, 7.0001, 0.05),
}


def outside(x: np.ndarray, k: str, w: np.ndarray | None = None) -> float:
    """Mass outside the fixed bins, so nothing is dropped silently."""
    m = (x < BINS[k][0]) | (x > BINS[k][-1])
    return float(m.sum() if w is None else w[m].sum())


def raw_histograms(d_path: Path) -> dict:
    from imspy_core.timstof import TimsDataset  # only needed in the container

    ds = TimsDataset(str(d_path), use_bruker_sdk=True)
    meta = ds.get_table("Frames")
    ms1 = meta[meta["MsMsType"] == 0]["Id"].to_numpy()
    acc = {k: np.zeros(len(v) - 1) for k, v in BINS.items()}
    n_peaks, total = 0, 0.0
    out_of_range = {k: 0.0 for k in ("rt", "im", "mz")}
    for fid in ms1:
        f = ds.get_tims_frame(int(fid))
        inten = np.asarray(f.intensity, dtype=float)
        if inten.size == 0:
            continue
        rt = np.full(inten.size, f.retention_time / 60.0)
        acc["rt"] += np.histogram(rt, BINS["rt"], weights=inten)[0]
        acc["im"] += np.histogram(np.asarray(f.mobility), BINS["im"], weights=inten)[0]
        acc["mz"] += np.histogram(np.asarray(f.mz), BINS["mz"], weights=inten)[0]
        acc["log10_intensity"] += np.histogram(np.log10(inten), BINS["log10_intensity"])[0]
        for k, x in (("rt", rt), ("im", np.asarray(f.mobility)), ("mz", np.asarray(f.mz))):
            out_of_range[k] += outside(x, k, inten)
        n_peaks += inten.size
        total += float(inten.sum())
    return {"view": "raw_ms1", "n_ms1_frames": int(len(ms1)), "n_peaks": n_peaks, "total_intensity": total,
            "out_of_range_intensity": out_of_range,
            "weighting": {"rt": "intensity", "im": "intensity", "mz": "intensity", "log10_intensity": "peak count"},
            "hist": {k: v.tolist() for k, v in acc.items()}}


def truth_histograms(db: Path) -> dict:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        cols = {r[1] for r in c.execute("pragma table_info(peptides)")}
        rt_col = "rt_mu" if "rt_mu" in cols else "retention_time_gru_predictor"
        rows = c.execute(
            f"select i.mz, i.charge, i.inv_mobility_gru_predictor, p.{rt_col}, p.events * i.relative_abundance "
            "from ions i join peptides p on p.peptide_id = i.peptide_id").fetchall()
    a = np.asarray(rows, dtype=float)
    mz, z, im, rt, w = a.T
    rt_min = rt / 60.0 if np.nanmax(rt) > 300 else rt  # TimSim stores seconds; a 2 h gradient never exceeds 300 min
    out = {"view": "simulated_ions", "n_ions": int(len(a)), "rt_column": rt_col, "hist": {}, "hist_weighted": {}}
    for k, x in (("rt", rt_min), ("im", im), ("mz", mz)):
        out["hist"][k] = np.histogram(x, BINS[k])[0].tolist()
        out["hist_weighted"][k] = np.histogram(x, BINS[k], weights=w)[0].tolist()
    out["out_of_range"] = {k: outside(x, k) for k, x in (("rt", rt_min), ("im", im), ("mz", mz))}
    out["charge"] = {str(int(k)): int(v) for k, v in zip(*np.unique(z, return_counts=True))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=Path, help="raw .d directory (view i)")
    ap.add_argument("--db", type=Path, help="synthetic_data.db (view iii)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {"bins": {k: v.tolist() for k, v in BINS.items()}}
    if a.d:
        res["raw"] = raw_histograms(a.d)
    if a.db:
        res["truth"] = truth_histograms(a.db)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res) + "\n")
    print(a.out, {k: v.get("n_peaks", v.get("n_ions")) for k, v in res.items() if k != "bins"})


if __name__ == "__main__":
    main()
