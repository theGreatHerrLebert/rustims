"""Per-peak intensity distribution of a timsTOF ``.d``, separated by frame type (MS1 vs MS2).

The measurement behind ``timsim-cli/RENDER_CALIBRATION.md``: it decodes raw peaks (validated exact
against the stored ``Frames.NumPeaks/MaxIntensity/SummedIntensities``) and reports the floor, a
percentile ladder, dynamic range, and peak density — separately for precursor (MS1) and fragment (MS2)
frames, because they are very different distributions. Point it at a real ``.d`` or a rendered one to
compare against the calibration target.

    python -m imspy_simulation.timsim.validate.peak_distribution <path.d> [n_frames]
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np


def frame_intensity_stats(path: str, ms_ms_type: int, n_frames: int = 120) -> dict:
    """Pooled per-peak intensity stats for ``n_frames`` frames of one ``MsMsType`` (0=MS1, 9=MS2)."""
    from imspy_core.timstof import TimsDatasetDIA

    d = TimsDatasetDIA(path)
    con = sqlite3.connect(path + "/analysis.tdf")
    frames = [
        (r[0], r[2])
        for r in con.execute(
            "SELECT Id, MsMsType, NumScans FROM Frames WHERE MsMsType=? ORDER BY Id", (ms_ms_type,)
        )
    ]
    if not frames:
        return {}
    frames = frames[:: max(1, len(frames) // n_frames)][:n_frames]
    pooled, n_scans = [], 0
    for fid, ns in frames:
        iv = np.asarray(d.get_tims_frame(fid).intensity, dtype=np.float64)
        pooled.append(iv[iv > 0])
        n_scans += ns
    a = np.concatenate(pooled)
    pct = [1, 25, 50, 75, 90, 99, 99.9]
    q = dict(zip((f"p{p}" for p in pct), np.percentile(a, pct)))
    return {
        "frames": len(frames),
        "n_peaks": int(len(a)),
        "peaks_per_frame": len(a) / len(frames),
        "peaks_per_scan": len(a) / max(n_scans, 1),
        "floor": float(a.min()),
        "frac_at_floor": float(np.mean(a == a.min())),
        "frac_within_2x_floor": float(np.mean(a <= 2 * a.min())),
        **{k: float(v) for k, v in q.items()},
        "max": float(a.max()),
        "dynamic_range": float(q["p99.9"] / max(q["p1"], 1.0)),
    }


def _print(name: str, s: dict) -> None:
    if not s:
        print(f"  {name}: (no frames)")
        return
    print(
        f"  {name}: {s['frames']} frames | {s['n_peaks']:,} peaks | "
        f"{s['peaks_per_frame']:.0f} peaks/frame | {s['peaks_per_scan']:.1f} peaks/scan"
    )
    print(
        f"     floor={s['floor']:.0f}  within-2x-floor={100*s['frac_within_2x_floor']:.1f}%  "
        f"p50={s['p50']:.0f}  p99={s['p99']:.0f}  p99.9={s['p99.9']:.0f}  max={s['max']:.0f}  "
        f"dyn={s['dynamic_range']:.0f}x"
    )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    path = argv[0]
    n = int(argv[1]) if len(argv) > 1 else 120
    print(f"=== {path} ===")
    _print("MS1 precursor", frame_intensity_stats(path, 0, n))
    _print("MS2 fragment ", frame_intensity_stats(path, 9, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
