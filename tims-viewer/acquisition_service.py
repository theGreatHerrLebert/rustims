#!/usr/bin/env python3
"""Acquisition sidecar (EXPERIMENTAL) — stream annotated SIMULATED DIA frames for live playback.

STATUS: experimental. This streams a *simulated* acquisition (TimSim synthetic data), not real
instrument output; the wire format and endpoints may still change, and it's an optional, off-by-
default sidecar. The viewer flags the "Live acquisition" mode as experimental to match.

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
renders those grey). Positions are real units; the viewer normalizes them to its cube. The ``id`` query
param is a 0-based frame INDEX (``0..n_frames-1``), not a DB frame id — the client plays the run in
order without needing to know the actual frame ids.

``GET /acq/frames?start=N&count=K`` builds a contiguous batch of K frames IN PARALLEL (Rayon) in one
call and returns ``u32 count``, then ``count × [u32 frame_index, u32 n_peaks]``, then the concatenated
records — so the viewer can prefetch ahead of the cursor cheaply instead of paying a round-trip +
per-call build per frame.

Run (needs the imspy_simulation env: DIAFrameBuilder + its Rust bindings):

    python tims-viewer/acquisition_service.py --db <synthetic_sim.db> --port 8092

Then load the viewer with ``?acq=http://localhost:8092`` (or it auto-detects the default port).
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sqlite3
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from imspy_simulation.builders.dia import DIAFrameBuilder


def build_subset_db(src_path: str, max_peptides: int) -> str:
    """Write a temp DB with only the first `max_peptides` peptides (+ their ions/fragment_ions),
    keeping every other table whole (frames/scans/dia windows are the acquisition schedule, not
    peptide data). Lets the eager annotated builder init in seconds over a slice of a huge sim instead
    of minutes over all of it — a no-Rust "semi-lazy" startup. Returns the temp DB path (auto-removed
    at exit)."""
    fd, tmp = tempfile.mkstemp(suffix=".db", prefix="acq_subset_")
    os.close(fd)
    os.remove(tmp)  # let sqlite create it fresh
    t = time.perf_counter()
    con = sqlite3.connect(tmp)
    try:
        con.execute("ATTACH DATABASE ? AS src", (src_path,))
        tables = [
            r[0] for r in con.execute(
                "SELECT name FROM src.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        # peptides first (ions/fragment_ions reference it); CREATE AS SELECT copies data, no indexes —
        # fine for the small subset the builder reads.
        con.execute(
            f"CREATE TABLE peptides AS SELECT * FROM src.peptides ORDER BY peptide_id LIMIT {int(max_peptides)}"
        )
        for t_name in ("ions", "fragment_ions"):
            if t_name in tables:
                con.execute(
                    f"CREATE TABLE {t_name} AS SELECT * FROM src.{t_name} "
                    "WHERE peptide_id IN (SELECT peptide_id FROM peptides)"
                )
        for t_name in tables:
            if t_name not in ("peptides", "ions", "fragment_ions"):
                con.execute(f"CREATE TABLE {t_name} AS SELECT * FROM src.{t_name}")
        con.commit()
    finally:
        con.close()
    atexit.register(lambda: os.path.exists(tmp) and os.remove(tmp))
    print(f"subset DB ({max_peptides} peptides) built in {time.perf_counter() - t:.1f}s", flush=True)
    return tmp

# NOTE: the Rust builder (PyTimsTofSyntheticsFrameBuilderDIA) is *unsendable* — it must stay on the
# thread that created it — so this server is single-threaded (plain HTTPServer). That's fine: the DIA
# build is internally Rayon-parallel and is the bottleneck, and the viewer pulls frames sequentially
# (with its own prefetch), so request-level concurrency would add nothing.

# u32::MAX — matches the viewer's NO_CLUSTER sentinel, so unassigned peaks render in the neutral grey.
NO_PEPTIDE = 0xFFFFFFFF

# Largest batch a single /acq/frames request may build (bounds per-request work/memory; the client
# prefetches in batches well under this).
MAX_BATCH = 128

# Per-peak record dtype sent over the wire (little-endian; 24 bytes).
_REC = np.dtype([
    ("mz", "<f4"), ("im", "<f4"), ("rt", "<f4"),
    ("intensity", "<f4"), ("peptide_id", "<u4"), ("flags", "<u4"),
])


class Acquisition:
    """Holds the eager annotated builder + precomputed run metadata."""

    def __init__(self, db_path: str, num_threads: int, fragmentation: bool, max_peptides: int = 0):
        self.fragmentation = fragmentation
        # Threads for the (Rayon-parallel) batch build — resolve -1 to the core count.
        self.num_threads = num_threads if num_threads and num_threads > 0 else (os.cpu_count() or 4)
        # Optional "semi-lazy" startup: build over a peptide subset so the eager builder inits in
        # seconds (the frame schedule stays whole, so playback covers the full run).
        builder_db = db_path
        if max_peptides > 0:
            print(f"subsetting to the first {max_peptides} peptides for a fast startup…", flush=True)
            builder_db = build_subset_db(db_path, max_peptides)
        # Eager builder WITH annotations (lazy + annotations is not supported yet). This reads/builds
        # all (subset) peptides + annotated fragment spectra up front.
        print(f"loading annotated builder from {builder_db}…", flush=True)
        t = time.perf_counter()
        self.builder = DIAFrameBuilder(builder_db, num_threads=self.num_threads, with_annotations=True)
        print(f"builder ready in {time.perf_counter() - t:.1f}s", flush=True)
        # Frame metadata comes from the (subset or full) builder DB — both carry the full frames table.
        db_path = builder_db

        # Frame metadata from the synthetic DB's `frames` table — detect columns FIRST (some DBs use
        # `id`/`rt`), then order by the detected id column.
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            fcols = {r[1] for r in con.execute("PRAGMA table_info(frames)").fetchall()}
            if not fcols:
                raise SystemExit("no `frames` table — is this a prepared TimSim synthetic DB?")
            id_col = next((c for c in ("frame_id", "id") if c in fcols), None)
            time_col = next((c for c in ("time", "rt", "retention_time") if c in fcols), None)
            ms_col = next((c for c in ("ms_type", "ms_ms_type", "scan_mode") if c in fcols), None)
            if not id_col or not time_col:
                raise SystemExit(f"`frames` missing id/time column (have: {sorted(fcols)})")
            rows = con.execute(f"SELECT * FROM frames ORDER BY {id_col}").fetchall()
            # Authoritative full 1/K0 range from the `scans` table (mobility per scan), if present —
            # far more reliable than sampling a few frames.
            im_range = None
            scols = {r[1] for r in con.execute("PRAGMA table_info(scans)").fetchall()}
            if "mobility" in scols:
                lo, hi = con.execute("SELECT MIN(mobility), MAX(mobility) FROM scans").fetchone()
                if lo is not None and hi is not None:
                    im_range = (float(lo), float(hi))
        finally:
            con.close()
        if not rows:
            raise SystemExit("the `frames` table is empty")
        self.frame_ids = [int(r[id_col]) for r in rows]
        times = [float(r[time_col]) for r in rows]
        self.ms_types = [int(r[ms_col]) for r in rows] if ms_col else [0] * len(rows)
        self.n_frames = len(self.frame_ids)
        # Per-FRAME interval = the playback cadence (reveal one frame every rt_cycle_length seconds).
        self.rt_cycle_length = float(np.mean(np.diff(times))) if len(times) > 1 else 0.0
        self.rt_min, self.rt_max = (min(times), max(times)) if times else (0.0, 1.0)

        # m/z + intensity p99 from a spread of sampled frames; 1/K0 from the scans table when available.
        self._compute_bounds(im_range)

    def _build(self, frame_id: int):
        return self.builder.build_frame_annotated(frame_id, fragment=self.fragmentation)

    @staticmethod
    def _records(f) -> np.ndarray:
        """Pack one built annotated frame into the 24-byte/peak wire records."""
        mz = np.asarray(f.mz, dtype="<f4")
        n = mz.size
        if n == 0:
            return np.empty(0, dtype=_REC)
        rec = np.empty(n, dtype=_REC)
        rec["mz"] = mz
        rec["im"] = np.asarray(f.inv_mobility, dtype="<f4")
        rec["rt"] = np.full(n, float(f.retention_time), dtype="<f4")
        rec["intensity"] = np.asarray(f.intensity, dtype="<f4")
        # peptide_ids_first_only: the FIRST contributor per peak (fast Rust ndarray). The dominant
        # max-intensity contributor would be truer for convolved peaks but needs a per-peak Python walk
        # over `contributions` — too slow at frame cadence. Negative ids (noise) → the grey sentinel.
        pid = np.asarray(f.peptide_ids_first_only, dtype=np.int64)
        rec["peptide_id"] = np.where(pid < 0, NO_PEPTIDE, pid).astype("<u4")
        rec["flags"] = np.uint32(1 if int(f.ms_type_numeric) == 9 else 0)  # frame-level: MS2 ⇒ fragments
        return rec

    def _compute_bounds(self, im_range):
        mz_lo, mz_hi = float("inf"), float("-inf")
        im_lo, im_hi = float("inf"), float("-inf")
        ints = []
        step = max(1, self.n_frames // 24)
        for fid in self.frame_ids[::step][:24]:
            f = self._build(fid)
            mz = np.asarray(f.mz, dtype=np.float64)
            if mz.size:
                mz_lo, mz_hi = min(mz_lo, mz.min()), max(mz_hi, mz.max())
                im = np.asarray(f.inv_mobility, dtype=np.float64)
                im_lo, im_hi = min(im_lo, im.min()), max(im_hi, im.max())
                ints.append(np.asarray(f.intensity, dtype=np.float64))
        # m/z: sampled, padded ~1% (sampling can miss the extremes; padding avoids edge-clipping).
        if np.isfinite(mz_lo):
            pad = 0.01 * (mz_hi - mz_lo) + 0.5
            self.mz_min, self.mz_max = mz_lo - pad, mz_hi + pad
        else:
            self.mz_min, self.mz_max = 100.0, 1700.0
        # 1/K0: prefer the authoritative full range from the scans table; else fall back to samples.
        if im_range:
            self.im_min, self.im_max = min(im_range), max(im_range)
        elif np.isfinite(im_lo):
            self.im_min, self.im_max = im_lo, im_hi
        else:
            self.im_min, self.im_max = 0.6, 1.6
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

    def frame_bytes(self, index: int) -> bytes:
        # `index` is a 0-based position into the (frame_id-ordered) frame list — the client plays
        # 0..n_frames-1 and never needs to know the actual DB frame ids (which may not be 0-based).
        return self._records(self._build(self.frame_ids[index])).tobytes()

    def frames_bytes(self, start: int, count: int) -> bytes:
        """Build a CONTIGUOUS batch of frames in parallel (Rayon, `num_threads`) and pack them into one
        response so the client can prefetch ahead of the cursor without per-frame round-trips.

        Wire layout (little-endian):
            u32 count
            count × [u32 frame_index, u32 n_peaks]    -- the per-frame table, in order
            then the concatenated 24-byte records for frame[0], frame[1], … (n_peaks each)

        NOTE: the annotated build is STOCHASTIC per call (the simulator's shot-noise model), so
        re-fetching a frame yields slightly different peaks. That's realistic, but it means the client
        must CACHE streamed frames for scrub/replay rather than re-fetching them.
        """
        indices = list(range(start, start + count))
        frame_ids = [self.frame_ids[i] for i in indices]
        frames = self.builder.build_frames_annotated(
            frame_ids, fragment=self.fragmentation, num_threads=self.num_threads
        )
        recs = [self._records(f) for f in frames]
        table = np.empty(count, dtype=[("idx", "<u4"), ("n", "<u4")])
        table["idx"] = np.asarray(indices, dtype="<u4")
        table["n"] = np.asarray([r.size for r in recs], dtype="<u4")
        head = np.array([count], dtype="<u4").tobytes() + table.tobytes()
        body = b"".join(r.tobytes() for r in recs)
        return head + body


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
            if not (0 <= fid < self.acq.n_frames):  # `id` is a 0-based frame index
                self._send(404, f"frame index {fid} out of range [0,{self.acq.n_frames})".encode(), "text/plain")
                return
            t = time.perf_counter()
            try:
                body = self.acq.frame_bytes(fid)
            except Exception as exc:  # a real build/packing failure — surface it, don't drop the conn
                print(f"frame {fid} build failed: {exc}", flush=True)
                self._send(500, f"frame {fid}: {exc}".encode(), "text/plain")
                return
            dt = time.perf_counter() - t
            print(f"frame {fid}: {len(body)//24} peaks in {dt*1000:.0f} ms", flush=True)
            self._send(200, body, "application/octet-stream")
            return
        if path == "/acq/frames":  # parallel batch — the client prefetches ahead with this
            q = parse_qs(urlparse(self.path).query)
            try:
                start = int(q.get("start", ["?"])[0])
                count = int(q.get("count", ["?"])[0])
            except ValueError:
                self._send(400, b"bad start/count", "text/plain")
                return
            if count <= 0:
                self._send(400, b"count must be > 0", "text/plain")
                return
            if start < 0 or start >= self.acq.n_frames:
                self._send(404, f"start {start} out of range [0,{self.acq.n_frames})".encode(), "text/plain")
                return
            # Clamp to the run end AND a sane max so one request can't build the whole run into memory.
            count = min(count, self.acq.n_frames - start, MAX_BATCH)
            t = time.perf_counter()
            try:
                body = self.acq.frames_bytes(start, count)
            except Exception as exc:
                print(f"batch [{start},{start+count}) build failed: {exc}", flush=True)
                self._send(500, f"batch [{start},{start+count}): {exc}".encode(), "text/plain")
                return
            dt = time.perf_counter() - t
            print(f"batch [{start},{start+count}): {len(body)} B in {dt*1000:.0f} ms "
                  f"({dt*1000/count:.1f} ms/frame, {self.acq.num_threads} threads)", flush=True)
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
    ap.add_argument("--max-peptides", type=int, default=0,
                    help="semi-lazy: build over only the first N peptides for a fast startup (0 = all)")
    args = ap.parse_args()

    Handler.acq = Acquisition(args.db, args.threads, not args.no_fragmentation, args.max_peptides)
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
