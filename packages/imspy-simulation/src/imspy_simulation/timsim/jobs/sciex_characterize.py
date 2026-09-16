"""Offline characterization of a SCIEX ZenoTOF `.wiff` template into a *profile* the native
`.wiff.scan` writer can author from exactly.

The writer's runtime heuristics — the metadata-length role model and the closed-form seed
`hdr = 8*cut_n + Q(a,b)` — are fit to one template (K562 standard SWATH) and do NOT hold across
instrument units / acquisition methods (a different unit's calibration shifts `Q`; a different
method changes the metadata structure). A *profile* sidesteps both: it records, per physical
block, the exact MS level and low-mass-cutoff seed `cut_n` derived from ProteoWizard's own
reading of the template. With a profile, ANY pwiz-readable template becomes authorable.

Physical blocks and pwiz spectra are NOT 1:1 in file order. pwiz reports spectra that have no
block behind them (the ZenoTOF 7600 60VW CC0 template: 49,741 blocks vs 64,921 spectra), and the
divergence starts mid-run, not at the end. Pairing block `i` with spectrum `i` is therefore
correct only up to the first divergence and silently wrong after it — on that template the base
peak of the paired spectrum drifts from ~10 mDa to 159-368 Da past block ~30,000, and the K562
template (95,372 blocks vs 95,552 spectra) degrades over its last ~20% the same way. The
resulting profile still *looks* valid (every cycle still covers N distinct windows), so nothing
downstream catches it. Blocks are instead aligned to spectra by base-peak m/z, which is directly comparable between the
two (TIC is not: pwiz sums PROFILE points against a block's centroids, and that factor is ~1.08 on
one ZenoTOF 7600 template and ~100 on another).

This is a one-time, offline prep step (like a Thermo `.raw` template is a prepared asset). It
needs the template converted to mzML once (``msconvert``); the runtime writer never touches pwiz.

Usage (CLI):
    python -m imspy_simulation.timsim.jobs.sciex_characterize TEMPLATE.wiff TEMPLATE.mzML OUT.json
"""
from __future__ import annotations

import base64
import json
import math
import re
import struct
import zlib
from typing import Optional

import numpy as np

# Alignment tuning. A block matches a spectrum when their base-peak m/z agree within MZ_TOL_DA.
# Isolated misses are tolerated; the offset only moves when ALIGN_NEED of the next ALIGN_WINDOW
# blocks agree at the new offset, so one spurious match cannot drag the alignment off track.
MZ_TOL_DA = 0.02
ALIGN_WINDOW = 12
ALIGN_NEED = 9
ALIGN_MAX_JUMP = 40000
# A block that failed to decode reports a negative base-peak m/z; it cannot be matched and must be
# stepped over without disturbing the offset.
MZ_FLOOR = 0.0


def _parse_mzml(path: str):
    """Stream a pwiz mzML → list of (ms_level, isolation_center, min_reported_mz, tic,
    base_peak_mz) per spectrum, in file order. Streamed one spectrum at a time: these run to GBs."""
    out = []
    buf: list[str] = []
    inside = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "<spectrum " in line:
                inside = True
                buf = [line]
                continue
            if not inside:
                continue
            buf.append(line)
            if "</spectrum>" in line:
                out.append(_parse_spectrum("".join(buf)))
                inside = False
                buf = []
    return out


def _parse_spectrum(head: str):
    ml = re.search(r'MS:1000511"[^>]*value="(\d+)"', head)
    ml = int(ml.group(1)) if ml else 0
    iso = re.search(r'MS:1000827"[^>]*value="([\d.]+)"', head)
    iso = float(iso.group(1)) if iso else None
    tic = re.search(r'MS:1000285"[^>]*value="([\d.eE+-]+)"', head)
    tic = float(tic.group(1)) if tic else -1.0
    bmz = re.search(r'MS:1000504"[^>]*value="([\d.eE+-]+)"', head)
    bmz = float(bmz.group(1)) if bmz else -1.0
    mz_min = None
    for b in re.findall(r"<binaryDataArray.*?</binaryDataArray>", head, re.S):
        accs = set(re.findall(r'accession="(MS:\d+)"', b))
        m = re.search(r"<binary>(.*?)</binary>", b, re.S)
        if not m or "MS:1000514" not in accs:  # m/z array only
            continue
        raw = base64.b64decode(m.group(1).strip())
        if "MS:1000574" in accs:  # zlib compression
            raw = zlib.decompress(raw)
        # 64- vs 32-bit float (MS:1000523 / MS:1000521). NB: decide the width from the accession,
        # not from the format string — "<%df" contains a 'd' (the %d placeholder), so a
        # `"d" in fmt` test is always true and mis-sizes every 32-bit array.
        is64 = "MS:1000523" in accs
        width, code = (8, "d") if is64 else (4, "f")
        v = struct.unpack("<%d%s" % (len(raw) // width, code), raw[: (len(raw) // width) * width])
        if len(v):
            mz_min = float(v[0])  # spectra are m/z-ascending; [0] = the low-mass cutoff
    return (ml, iso, mz_min, tic, bmz)


def align_blocks_to_spectra(block_mzs, scan_mzs, *, tol_da: float = MZ_TOL_DA):
    """Map each physical block to its pwiz spectrum index (-1 = unmatched).

    Blocks and spectra share an order but not an indexing: spectra may appear with no block, so
    the offset only ever grows. Matching is on base-peak m/z, which is directly comparable between
    a block and pwiz's reported value (~1-10 mDa when paired correctly, hundreds of Da when not).
    The offset moves only on sustained agreement, so one chance collision cannot desynchronise the
    walk. Undecodable blocks are stepped over, leaving the offset untouched.
    """
    st = np.asarray(scan_mzs, dtype=float)
    bt = np.asarray(block_mzs, dtype=float)
    nb, ns = len(bt), len(st)
    out = np.full(nb, -1, dtype=np.int64)
    if nb == 0 or ns == 0:
        return out

    def ok(i: int, d: int) -> bool:
        j = i + d
        if j < 0 or j >= ns or bt[i] <= MZ_FLOOR or st[j] <= MZ_FLOOR:
            return False
        return abs(bt[i] - st[j]) <= tol_da

    inf = np.nonzero(bt > MZ_FLOOR)[0]  # only decodable blocks drive the alignment
    delta, p = 0, 0
    while p < len(inf):
        i = int(inf[p])
        if ok(i, delta):
            out[i] = i + delta
            p += 1
            continue
        probe = [int(x) for x in inf[p : p + ALIGN_WINDOW]]
        if sum(1 for t in probe if not ok(t, delta)) <= len(probe) - ALIGN_NEED:
            p += 1  # isolated miss (a block pwiz did not report) — keep the offset
            continue
        lo, hi = i + delta, min(ns, i + delta + ALIGN_MAX_JUMP)  # monotonic: forward only
        best, best_d = -1, None
        if hi > lo:
            seg = st[lo:hi]
            cand = np.nonzero(np.abs(seg - bt[i]) <= tol_da)[0]
            for c in cand[:400]:
                d = int(lo + c) - i
                sc = sum(1 for t in probe if ok(t, d))
                if sc > best:
                    best, best_d = sc, d
                if sc >= len(probe):
                    break
        if best_d is not None and best >= min(ALIGN_NEED, len(probe)):
            delta = best_d
            continue
        p += 1
    return out


def fill_offsets(amap, n_blocks: int):
    """Extend a sparse block→spectrum map across unmatched blocks.

    The offset is piecewise-constant, so a block sitting between two anchors that agree on the
    offset inherits it. Where the bracketing anchors DISAGREE the offset changed somewhere inside
    the gap and there is no evidence for where — those blocks stay unassigned and get cleared,
    rather than guessed. Leading/trailing runs have only one anchor and are likewise left alone.
    """
    filled = np.full(n_blocks, -1, dtype=np.int64)
    anchors = [k for k in range(n_blocks) if amap[k] >= 0]
    for k in anchors:
        filled[k] = amap[k]
    for a, b in zip(anchors, anchors[1:]):
        da, db = amap[a] - a, amap[b] - b
        if da == db:
            for k in range(a + 1, b):
                filled[k] = k + da
    return filled


def characterize(scan_path: str, mzml_path: str, out_path: str, *, verbose: bool = True) -> dict:
    """Write a native-writer profile for `scan_path` (a `.wiff.scan`) using its pwiz `mzml_path`.

    Returns the profile dict. Blocks are enumerated by the connector (the exact same enumeration
    the writer uses) and aligned to the mzML spectra by TIC — not by index, which they do not
    share. Only whole, physically contiguous `1 + N` cycles are emitted; every other block is
    left out of `authored` and so gets cleared by the writer.
    """
    import imspy_connector

    cals = imspy_connector.py_acquisition.sciex_scan_blocks(scan_path)  # [(cal_a, cal_b), ...]
    bmzs = imspy_connector.py_acquisition.sciex_scan_block_basepeaks(scan_path)
    scans = _parse_mzml(mzml_path)
    if not cals or not scans:
        raise ValueError("no blocks/spectra — is the mzML the conversion of this template?")

    # SWATH window count N = distinct MS2 isolation centers.
    n_windows = len(
        {round(iso, 2) for ml, iso, _mz, _t, _b in scans if ml == 2 and iso is not None}
    )
    if n_windows == 0:
        raise ValueError("no MS2 isolation windows found in the mzML (not a SWATH template?)")
    period = 1 + n_windows

    amap = align_blocks_to_spectra(bmzs, [s[4] for s in scans])
    matched = int((amap >= 0).sum())
    amap = fill_offsets(amap, len(cals))
    assigned = int((amap >= 0).sum())
    informative = int(sum(1 for t in bmzs if t > MZ_FLOOR))
    if matched * 2 < informative:
        raise ValueError(
            f"could not align template to its mzML: only {matched} of {informative} decodable "
            f"blocks matched a spectrum by base-peak m/z. Is this mzML the conversion of THIS "
            f"template, and does the closed-form seed Q hold for this instrument?"
        )

    # Per-matched-block (ms, cut_n) from the aligned spectrum.
    labelled: dict[int, tuple[int, int]] = {}
    for k in range(len(cals)):
        j = int(amap[k])
        if j < 0:
            continue
        ml, _iso, mz_min, _tic, _bmz = scans[j]
        cal_a, cal_b = cals[k]
        mz = mz_min if (mz_min and mz_min > 0) else (cal_b * cal_b)  # fallback: n=0 -> mz=b^2
        cut_n = int(round(5.0 * (math.sqrt(max(mz, 1e-6)) - cal_b) / cal_a))
        labelled[k] = (ml, cut_n)

    # Emit only whole cycles whose blocks are physically CONTIGUOUS: a cycle with a block missing
    # would otherwise absorb the next cycle's first block and shift every window in it.
    authored: list[list[int]] = []
    run: list[int] = []

    def flush(run):
        if len(run) == period and labelled[run[0]][0] == 1:
            if all(labelled[b][0] == 2 for b in run[1:]):
                authored.extend([[b, labelled[b][0], labelled[b][1]] for b in run])

    for k in sorted(labelled):
        ml = labelled[k][0]
        if run and (k != run[-1] + 1 or ml == 1):
            flush(run)
            run = []
        run.append(k)
    flush(run)

    prof = {"n_windows": n_windows, "authored": authored}
    with open(out_path, "w") as fh:
        json.dump(prof, fh)
    if verbose:
        print(
            f"sciex profile: {n_windows} windows, {len(cals)} blocks, {len(scans)} spectra, "
            f"{matched} matched + {assigned - matched} inferred = {assigned} assigned "
            f"({100.0 * assigned / len(cals):.1f}%), "
            f"{len(authored)} authored blocks ({len(authored) // period} full cycles) -> {out_path}"
        )
    return prof


def main(argv: Optional[list] = None) -> None:
    import sys

    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 3:
        print("usage: sciex_characterize TEMPLATE.wiff.scan TEMPLATE.mzML OUT_profile.json")
        raise SystemExit(2)
    characterize(argv[0], argv[1], argv[2])


if __name__ == "__main__":
    main()
