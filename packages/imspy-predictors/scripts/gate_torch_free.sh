#!/usr/bin/env bash
# P0 gate — prove `imspy-predictors` installs and runs WITHOUT PyTorch.
#
# Builds a throwaway venv, installs imspy-predictors with NO extras (so torch is
# excluded), and asserts the torch-decoupling contract. `imspy-core` must be
# installable (it needs the compiled `imspy_connector` wheel — build it first
# with `maturin build --release` in ../../imspy_connector and point pip at it,
# or `pip install imspy-core` if published).
#
# Usage:  scripts/gate_torch_free.sh [/path/to/python3.11]
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${1:-python3.11}"
VENV="$(mktemp -d)/torchfree-venv"

echo ">> creating torch-free venv at $VENV (python: $PY)"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip

echo ">> installing imspy-predictors with NO extras (torch must NOT be pulled)"
pip install -q -e "$PKG_DIR" pytest

echo ">> asserting torch is genuinely absent"
if python -c "import torch" 2>/dev/null; then
  echo "!! FAIL: torch is installed in the supposedly torch-free env" >&2
  exit 1
fi
echo "   ok: 'import torch' fails as expected"

echo ">> import + every console entry point must load torch-free"
python -c "import imspy_predictors; print('   ok: import imspy_predictors')"

echo ">> running the torch-optional gate suite"
pytest -q "$PKG_DIR/tests/test_torch_optional.py"

echo ">> PASS: imspy-predictors is torch-free; local models raise the [local] hint"
deactivate
rm -rf "$(dirname "$VENV")"
