#!/usr/bin/env python3
"""Minimal sklearn-DBSCAN clustering service for the tims-viewer web client.

A feasibility step toward running clustering through Python (sklearn today; HDBSCAN /
the MIDIA 4D pipeline later) instead of the in-wasm DBSCAN. The web client POSTs the
exact (already axis-scaled) points it would have clustered locally, so the result is
directly comparable to the wasm path.

Wire format (matches the wasm worker's): the body is the little-endian float32 point
buffer ``[x,y,z, x,y,z, ...]``; the reply is little-endian int32 labels (one per point,
``-1`` = noise). eps / min_samples ride the query string.

    POST /cluster?eps=<f>&min=<n>        body: float32 xyz triples
      -> int32 labels

Run (needs numpy + scikit-learn, both in the imspy env):

    python tims-viewer/cluster_service.py --port 8091

Then load the viewer with ``?cluster=http://localhost:8091/cluster`` to route Run
through this service; without that query param the viewer uses its built-in wasm DBSCAN.
"""
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN

# Single-flight: DBSCAN(n_jobs=-1) already uses every core, so serialize requests (the browser does
# not abort superseded fetches) to avoid oversubscribing the CPU with overlapping runs.
_DBSCAN_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        # The trunk page is served from a different origin than this service.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self) -> None:  # CORS preflight
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # health probe — the web auto-enables the Python backend if this is 200
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self._cors()
        self.end_headers()
        self.wfile.write(b"tims-viewer sklearn cluster service: ok\n")

    def do_POST(self) -> None:
        print(f"POST {self.path}  ({self.headers.get('Content-Length', '?')} bytes)", flush=True)
        if urlparse(self.path).path != "/cluster":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        q = parse_qs(urlparse(self.path).query)
        method = q.get("method", ["dbscan"])[0]
        try:
            n = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):  # malformed header -> clean 400, don't drop the connection
            self.send_response(400)
            self._cors()
            self.end_headers()
            return
        buf = bytearray()
        while len(buf) < n:  # the socket can hand back the body in chunks
            chunk = self.rfile.read(n - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
        # Guard the length BEFORE np.frombuffer (which raises on non-4-byte-aligned input): xyz f4
        # triples = 12 bytes/point, so a valid body is a non-empty multiple of 12.
        if len(buf) != n or len(buf) == 0 or len(buf) % 12 != 0:
            self.send_response(400)
            self._cors()
            self.end_headers()
            return
        pts = np.frombuffer(bytes(buf), dtype="<f4").reshape(-1, 3).astype(np.float64)
        n_pts = pts.shape[0]
        t = time.perf_counter()
        # Never drop the connection on a bad/degenerate input: clamp params to the point count and,
        # on any sklearn error, fall back to all-noise so the client always gets valid labels.
        with _DBSCAN_LOCK:  # serialize the heavy compute across overlapping requests
            try:
                if method == "hdbscan":
                    mcs = max(2, min(int(q.get("mcs", ["7"])[0]), n_pts))
                    ms_raw = int(q.get("ms", ["0"])[0])
                    ms = min(ms_raw, n_pts) if ms_raw > 0 else None
                    cse = float(q.get("cse", ["0"])[0])
                    if n_pts < 2:
                        labels = np.full(n_pts, -1, dtype="<i4")
                    else:
                        labels = HDBSCAN(
                            min_cluster_size=mcs,
                            min_samples=ms,
                            cluster_selection_epsilon=cse,
                            copy=True,  # don't mutate our buffer (+ silence the 1.10 FutureWarning)
                        ).fit(pts).labels_
                    desc = f"HDBSCAN mcs={mcs} ms={ms or 'auto'} cse={cse}"
                else:
                    eps = float(q.get("eps", ["0.012"])[0])
                    min_samples = int(q.get("min", ["8"])[0])
                    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(pts).labels_
                    desc = f"DBSCAN eps={eps} min={min_samples}"
            except Exception as exc:  # degenerate input, etc. — return all-noise, don't 500
                print(f"clustering failed ({method}, {n_pts} pts): {exc}", flush=True)
                labels = np.full(n_pts, -1, dtype="<i4")
                desc = f"{method} (fallback all-noise)"
        k = int(labels.max()) + 1 if labels.size else 0
        dt = time.perf_counter() - t
        print(f"{desc}: {pts.shape[0]} pts -> {k} clusters in {dt:.2f}s", flush=True)
        out = labels.astype("<i4").tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(out)))
        self._cors()
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *_args) -> None:  # quiet the default per-request logging
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="sklearn-DBSCAN clustering service for tims-viewer web")
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()
    print(f"sklearn-DBSCAN cluster service on http://localhost:{args.port}/cluster")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
