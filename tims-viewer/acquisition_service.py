#!/usr/bin/env python3
"""Acquisition sidecar — stream annotated DIA frames for live playback in tims-viewer.

Wraps TimSim's ``DIAFrameBuilder(with_annotations=True)`` over a prepared synthetic-simulation DB and
serves, per frame, the annotated points (m/z, 1/K0, RT, intensity + peptide_id + precursor/fragment
flag). The viewer plays the acquisition over time at the device's real frame rate and colors each
peak by its precursor — a precursor's MS1 signal and all its MS2 fragments share one ``peptide_id``,
so you can watch DIA fragment convolution build up.

A decoupled, optional sidecar (same shape as ``cluster_service.py``): the viewer auto-detects it via
a GET health probe and only enables the "live acquisition" mode when it's up.

Wire format (``/acq/frame``): little-endian records, 24 bytes/peak —
    [ mz f32, im f32, rt f32, intensity f32, peptide_id u32, flags u32 ]
``flags`` bit0 = fragment (MS2); ``peptide_id`` = 0xFFFFFFFF for unassigned/noise peaks (the viewer
renders those grey). Positions are real units; the viewer normalizes them to its cube.

Run (needs the imspy_simulation env: DIAFrameBuilder + its Rust bindings):

    python tims-viewer/acquisition_service.py --db <synthetic_sim.db> --port 8092

Then load the viewer with ``?acq=http://localhost:8092`` (or it auto-detects the default port).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from imspy_simulation.builders.dia import DIAFrameBuilder

# NOTE: the Rust builder (PyTimsTofSyntheticsFrameBuilderDIA) is *unsendable* — it must stay on the
# thread that created it — so this server is single-threaded (plain HTTPServer). That's fine: the DIA
# build is internally Rayon-parallel and is the bottleneck, and the viewer pulls frames sequentially
# (with its own prefetch), so request-level concurrency would add nothing.

# u32::MAX — matches the viewer's NO_CLUSTER sentinel, so unassigned peaks render in the neutral grey.
NO_PEPTIDE = 0xFFFFFFFF

# Per-peak record dtype sent over the wire (little-endian; 24 bytes).
_REC = np.dtype([
    ("mz", "<f4"), ("im", "<f4"), ("rt", "<f4"),
    ("intensity", "<f4"), ("peptide_id", "<u4"), ("flags", "<u4"),
])


class Acquisition:
    """Holds the eager annotated builder + precomputed run metadata."""

    def __init__(self, db_path: str, num_threads: int, fragmentation: bool):
        self.fragmentation = fragmentation
        # Eager builder WITH annotations (lazy + annotations is not supported yet). This reads/builds
        # all peptides + annotated fragment spectra up front — fine for small/medium sims.
        print(f"loading annotated builder from {db_path} (this reads all peptides up front)…", flush=True)
        t = time.perf_counter()
        self.builder = DIAFrameBuilder(db_path, num_threads=num_threads, with_annotations=True)
        print(f"builder ready in {time.perf_counter() - t:.1f}s", flush=True)

        # Frame metadata from the synthetic DB's `frames` table (frame_id, time, ms_type).
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute("SELECT * FROM frames ORDER BY frame_id").fetchall()
        finally:
            con.close()
        if not rows:
            raise SystemExit("no rows in the `frames` table — is this a prepared TimSim synthetic DB?")
        cols = set(rows[0].keys())
        id_col = "frame_id" if "frame_id" in cols else "id"
        time_col = "time" if "time" in cols else ("rt" if "rt" in cols else "retention_time")
        ms_col = next((c for c in ("ms_type", "ms_ms_type", "scan_mode") if c in cols), None)
        self.frame_ids = [int(r[id_col]) for r in rows]
        times = [float(r[time_col]) for r in rows]
        self.ms_types = [int(r[ms_col]) for r in rows] if ms_col else [0] * len(rows)
        self.n_frames = len(self.frame_ids)
        self.rt_cycle_length = float(np.mean(np.diff(times))) if len(times) > 1 else 0.0
        self.rt_min, self.rt_max = (min(times), max(times)) if times else (0.0, 1.0)

        # Sample a few frames for axis bounds + an intensity p99 (the eager builder is already loaded).
        self._compute_bounds()

    def _build(self, frame_id: int):
        return self.builder.build_frame_annotated(frame_id, fragment=self.fragmentation)

    def _compute_bounds(self):
        mz_lo = im_lo = float("inf")
        mz_hi = im_hi = float("-inf")
        ints = []
        step = max(1, self.n_frames // 8)
        for fid in self.frame_ids[::step][:8]:
            f = self._build(fid)
            mz = np.asarray(f.mz, dtype=np.float64)
            im = np.asarray(f.inv_mobility, dtype=np.float64)
            if mz.size:
                mz_lo, mz_hi = min(mz_lo, mz.min()), max(mz_hi, mz.max())
                im_lo, im_hi = min(im_lo, im.min()), max(im_hi, im.max())
                ints.append(np.asarray(f.intensity, dtype=np.float64))
        self.mz_min, self.mz_max = (mz_lo, mz_hi) if np.isfinite(mz_lo) else (100.0, 1700.0)
        self.im_min, self.im_max = (im_lo, im_hi) if np.isfinite(im_lo) else (0.6, 1.6)
        allint = np.concatenate(ints) if ints else np.array([1.0])
        self.i_p99 = float(np.percentile(allint, 99)) if allint.size else 1.0

    def meta_json(self) -> bytes:
        return json.dumps({
            "n_frames": self.n_frames,
            "rt_cycle_length": self.rt_cycle_length,
            "mz_min": self.mz_min, "mz_max": self.mz_max,
            "im_min": self.im_min, "im_max": self.im_max,
            "rt_min": self.rt_min, "rt_max": self.rt_max,
            "i_p99": self.i_p99,
            "ms_types": self.ms_types,
        }).encode()

    def frame_bytes(self, frame_id: int) -> bytes:
        f = self._build(frame_id)
        mz = np.asarray(f.mz, dtype="<f4")
        n = mz.size
        if n == 0:
            return b""
        im = np.asarray(f.inv_mobility, dtype="<f4")
        intensity = np.asarray(f.intensity, dtype="<f4")
        rt = np.full(n, float(f.retention_time), dtype="<f4")
        # peptide_ids_first_only: the (Rust-side, fast) dominant-or-first contributor per peak.
        # Negative ids (unassigned/noise) → the grey sentinel.
        pid = np.asarray(f.peptide_ids_first_only, dtype=np.int64)
        pid = np.where(pid < 0, NO_PEPTIDE, pid).astype("<u4")
        is_fragment = 1 if int(f.ms_type_numeric) == 9 else 0  # frame-level: MS2 ⇒ all peaks fragments
        flags = np.full(n, is_fragment, dtype="<u4")
        rec = np.empty(n, dtype=_REC)
        rec["mz"], rec["im"], rec["rt"] = mz, im, rt
        rec["intensity"], rec["peptide_id"], rec["flags"] = intensity, pid, flags
        return rec.tobytes()


class Handler(BaseHTTPRequestHandler):
    acq: Acquisition = None  # set in main()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/acq/meta":
            self._send(200, self.acq.meta_json(), "application/json")
            return
        if path == "/acq/frame":
            q = parse_qs(urlparse(self.path).query)
            try:
                fid = int(q.get("id", ["?"])[0])
            except ValueError:
                self._send(400, b"bad frame id", "text/plain")
                return
            t = time.perf_counter()
            try:
                body = self.acq.frame_bytes(fid)
            except Exception as exc:  # out-of-range id / build failure — don't drop the connection
                print(f"frame {fid} failed: {exc}", flush=True)
                self._send(404, f"frame {fid}: {exc}".encode(), "text/plain")
                return
            dt = time.perf_counter() - t
            print(f"frame {fid}: {len(body)//24} peaks in {dt*1000:.0f} ms", flush=True)
            self._send(200, body, "application/octet-stream")
            return
        # health probe (the viewer auto-detects the sidecar via this)
        self._send(200, b"tims-viewer acquisition service: ok\n", "text/plain")

    def log_message(self, *_args):
        pass


def main():
    ap = argparse.ArgumentParser(description="Annotated-frame acquisition sidecar for tims-viewer")
    ap.add_argument("--db", required=True, help="prepared TimSim synthetic simulation DB")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--threads", type=int, default=-1, help="builder threads (-1 = auto)")
    ap.add_argument("--no-fragmentation", action="store_true", help="precursor-only (skip MS2 fragments)")
    args = ap.parse_args()

    Handler.acq = Acquisition(args.db, args.threads, not args.no_fragmentation)
    a = Handler.acq
    print(
        f"acquisition service on http://localhost:{args.port}  "
        f"({a.n_frames} frames, rt_cycle_length={a.rt_cycle_length:.4f}s, "
        f"mz [{a.mz_min:.0f},{a.mz_max:.0f}] 1/K0 [{a.im_min:.3f},{a.im_max:.3f}])",
        flush=True,
    )
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
