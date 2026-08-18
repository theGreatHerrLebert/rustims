#!/usr/bin/env bash
# Launch the WEB tims-viewer on a MOBILion .mbi file: point server + trunk-served wasm app.
#
# Usage:
#   ./scripts/view-mbi.sh                          # serves the default HeLa CE-ramp run
#   ./scripts/view-mbi.sh /path/to/other.mbi       # serves a specific .mbi (or a .d / folder)
#   BUDGET=25000000 ./scripts/view-mbi.sh          # bigger point budget
#   SERVE_PORT=9090 WEB_PORT=9080 ./scripts/view-mbi.sh
#
# The .mbi reader is behind the opt-in `mbi` cargo feature (it links HDF5), so this
# script rebuilds with that feature when needed. HDF5 comes from Homebrew; the build
# script only finds it with HDF5_DIR set. Ctrl-C stops both servers.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MBI="${1:-/Users/davidteschner/Promotion/ims/mobilion/CERamp-25ngHeLa-14.52.54.mbi}"
BUDGET="${BUDGET:-12000000}"
SERVE_PORT="${SERVE_PORT:-8090}"     # point server (native binary, --serve)
WEB_PORT="${WEB_PORT:-8080}"         # trunk dev server (wasm app)
CLUSTER_PORT="${CLUSTER_PORT:-8091}" # sklearn clustering sidecar (auto-detected by the UI)

if [[ ! -e "$MBI" ]]; then
    echo "error: no such dataset: $MBI" >&2
    exit 1
fi

# HDF5 (brew) — required at build time and runtime by the mbi feature.
if ! brew --prefix hdf5 >/dev/null 2>&1; then
    echo "HDF5 not found — installing via Homebrew..."
    brew install hdf5
fi
export HDF5_DIR="$(brew --prefix hdf5)"

# Web toolchain: trunk + the wasm target.
if ! command -v trunk >/dev/null 2>&1; then
    echo "trunk not found — installing (cargo install trunk)..."
    cargo install trunk
fi
rustup target list --installed | grep -q wasm32-unknown-unknown \
    || rustup target add wasm32-unknown-unknown

echo "building tims-viewer (--features mbi)..."
cargo build -p tims-viewer --release --features mbi --manifest-path "$REPO/Cargo.toml"

# Point server + optional clustering sidecar in the background; trunk in the foreground.
# Kill the background servers on exit.
"$REPO/target/release/tims-viewer" "$MBI" --serve "$SERVE_PORT" --budget "$BUDGET" &
SERVER_PID=$!
CLUSTER_PID=""
# sklearn clustering sidecar: much faster than the in-wasm DBSCAN on big regions. The UI
# auto-detects it at startup and prefers it. Optional — skipped when sklearn is missing.
if python3 -c "import sklearn" >/dev/null 2>&1; then
    python3 "$REPO/tims-viewer/cluster_service.py" --port "$CLUSTER_PORT" &
    CLUSTER_PID=$!
else
    echo "note: python3 has no sklearn — clustering will use the slower in-browser DBSCAN"
fi
trap 'kill "$SERVER_PID" $CLUSTER_PID 2>/dev/null || true' EXIT

URL="http://localhost:$WEB_PORT/?port=$SERVE_PORT"
[[ "$CLUSTER_PORT" != 8091 ]] && URL="$URL&clusterport=$CLUSTER_PORT"
echo
echo "point server: http://localhost:$SERVE_PORT   web app: $URL"
# The point server only starts listening AFTER the eager dataset build (~10 s for a big
# .mbi), and trunk needs a moment for the wasm build. Open the browser only once BOTH
# answer — a page loaded before the point server is up falls back to the demo cloud.
(
    for _ in $(seq 1 120); do
        curl -sf -o /dev/null "http://localhost:$SERVE_PORT/datasets" \
            && curl -sf -o /dev/null "http://localhost:$WEB_PORT/" \
            && { open "$URL"; exit 0; }
        sleep 1
    done
    echo "warning: servers not up after 120 s — open $URL manually" >&2
) &

cd "$REPO/tims-viewer/web"
trunk serve --release --port "$WEB_PORT"
