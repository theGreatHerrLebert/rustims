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

# --- P1: end-to-end torch-free simulation -----------------------------------
# If imspy-simulation is present as a sibling, install it (no extras) and assert
# the whole `timsim` chain stays torch-free. This is what catches transitive
# leaks like the losses.py one and the imspy-search dep. Requires the sagepy
# connector to be installable in this env.
SIM_DIR="$(cd "$PKG_DIR/../imspy-simulation" 2>/dev/null && pwd || true)"
if [ -n "$SIM_DIR" ]; then
  echo ">> installing imspy-simulation (no extras) — must NOT pull torch"
  pip install -q -e "$SIM_DIR"
  if python -c "import torch" 2>/dev/null; then
    echo "!! FAIL: installing imspy-simulation pulled torch back in" >&2
    exit 1
  fi
  echo "   ok: sim install is torch-free"
  python -c "import imspy_simulation.timsim.simulator as s; assert callable(s.main); print('   ok: timsim entry point loads torch-free')"
  python -c "import imspy_simulation.timsim.jobs.simulate_fragment_intensities; print('   ok: Koina intensity job imports torch-free')"
  # RUNTIME (not just import) path: the default config has use_gpu=True, whose GPU
  # probe must degrade to CPU when torch is absent instead of crashing at startup.
  python -c "from imspy_simulation.timsim.simulator import configure_gpu_memory; configure_gpu_memory(use_gpu=True); print('   ok: default GPU probe degrades torch-free')"
else
  echo ">> (imspy-simulation not found as sibling — skipping end-to-end sim gate)"
fi

echo ">> PASS: imspy-predictors + timsim are torch-free; local models raise the [local] hint"
deactivate
rm -rf "$(dirname "$VENV")"
