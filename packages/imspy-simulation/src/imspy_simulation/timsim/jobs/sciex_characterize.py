"""Offline characterization of a SCIEX ZenoTOF `.wiff` template into a *profile* the native
`.wiff.scan` writer can author from exactly.

The writer's runtime heuristics — the metadata-length role model and the closed-form seed
`hdr = 8*cut_n + Q(a,b)` — are fit to one template (K562 standard SWATH) and do NOT hold across
instrument units / acquisition methods (a different unit's calibration shifts `Q`; a different
method changes the metadata structure). A *profile* sidesteps both: it records, per physical
block, the exact MS level and low-mass-cutoff seed `cut_n` derived from ProteoWizard's own
reading of the template. With a profile, ANY pwiz-readable template becomes authorable.

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


def _parse_mzml(path: str):
    """Stream a pwiz mzML → list of (ms_level, isolation_center, min_reported_mz) per spectrum,
    in file order (== physical block order for a SCIEX `.wiff.scan`)."""
    text = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for chunk in text.split("<spectrum ")[1:]:
        head = chunk.split("</spectrum>")[0]
        ml = re.search(r'MS:1000511"[^>]*value="(\d+)"', head)
        ml = int(ml.group(1)) if ml else 0
        iso = re.search(r'MS:1000827"[^>]*value="([\d.]+)"', head)
        iso = float(iso.group(1)) if iso else None
        mz_min = None
        for b in re.findall(r"<binaryDataArray.*?</binaryDataArray>", head, re.S):
            accs = set(re.findall(r'accession="(MS:\d+)"', b))
            m = re.search(r"<binary>(.*?)</binary>", b, re.S)
            if not m or "MS:1000514" not in accs:  # m/z array only
                continue
            raw = base64.b64decode(m.group(1).strip())
            if "MS:1000574" in accs:  # zlib compression
                raw = zlib.decompress(raw)
            fmt = "<%dd" if "MS:1000523" in accs else "<%df"  # 64- vs 32-bit
            v = struct.unpack(fmt % (len(raw) // (8 if "d" in fmt else 4)), raw)
            if len(v):
                mz_min = float(v[0])  # spectra are m/z-ascending; [0] = the low-mass cutoff
        out.append((ml, iso, mz_min))
    return out


def characterize(scan_path: str, mzml_path: str, out_path: str, *, verbose: bool = True) -> dict:
    """Write a native-writer profile for `scan_path` (a `.wiff.scan`) using its pwiz `mzml_path`.

    Returns the profile dict. Blocks are enumerated by the connector (the exact same enumeration
    the writer uses), aligned 1:1 with the mzML spectra in file order.
    """
    import imspy_connector

    cals = imspy_connector.py_acquisition.sciex_scan_blocks(scan_path)  # [(cal_a, cal_b), ...]
    scans = _parse_mzml(mzml_path)
    n = min(len(cals), len(scans))
    if n == 0:
        raise ValueError("no aligned blocks/spectra — is the mzML the conversion of this template?")

    # SWATH window count N = distinct MS2 isolation centers.
    n_windows = len({round(iso, 2) for ml, iso, _ in scans[:n] if ml == 2 and iso is not None})
    if n_windows == 0:
        raise ValueError("no MS2 isolation windows found in the mzML (not a SWATH template?)")
    period = 1 + n_windows

    # Group into cycles by MS1 (each MS1 starts a cycle); keep whole 1+N cycles.
    authored: list[list[int]] = []
    cur: list[list[int]] = []

    def flush(cur):
        if len(cur) == period:
            authored.extend(cur)

    for k in range(n):
        ml, _iso, mz_min = scans[k]
        cal_a, cal_b = cals[k]
        mz = mz_min if (mz_min and mz_min > 0) else (cal_b * cal_b)  # fallback: n=0 -> mz=b^2
        cut_n = int(round(5.0 * (math.sqrt(max(mz, 1e-6)) - cal_b) / cal_a))
        if ml == 1 and cur:
            flush(cur)
            cur = []
        cur.append([k, ml, cut_n])
    flush(cur)

    prof = {"n_windows": n_windows, "authored": authored}
    with open(out_path, "w") as fh:
        json.dump(prof, fh)
    if verbose:
        print(
            f"sciex profile: {n_windows} windows, {len(authored)} authored blocks "
            f"({len(authored) // period} full cycles) -> {out_path}"
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
