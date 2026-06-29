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
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from sklearn.cluster import DBSCAN


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

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/cluster":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        q = parse_qs(urlparse(self.path).query)
        eps = float(q.get("eps", ["0.012"])[0])
        min_samples = int(q.get("min", ["8"])[0])
        n = int(self.headers.get("Content-Length", 0))
        buf = self.rfile.read(n) if n else b""
        pts = np.frombuffer(buf, dtype="<f4")
        if pts.size == 0 or pts.size % 3 != 0:
            self.send_response(400)
            self._cors()
            self.end_headers()
            return
        pts = pts.reshape(-1, 3).astype(np.float64)
        t = time.perf_counter()
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(pts).labels_
        k = int(labels.max()) + 1 if labels.size else 0
        dt = time.perf_counter() - t
        print(f"DBSCAN: {pts.shape[0]} pts, eps={eps} min={min_samples} -> {k} clusters in {dt:.2f}s")
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
