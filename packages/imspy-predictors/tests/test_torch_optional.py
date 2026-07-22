"""P0 gate: torch is an *optional* extra (`imspy-predictors[local]`).

These tests assert the torch-decoupling contract:
  1. `import imspy_predictors` works with no torch installed.
  2. The Koina (remote) path and its wrappers construct with no torch.
  3. Any *local* model routes a missing torch through `require_torch`, which
     raises an actionable `pip install 'imspy-predictors[local]'` hint — never
     a bare `ModuleNotFoundError: No module named 'torch'` or a silent `None`.

To stay runnable even in a torch-*installed* dev env, `block_torch()` makes
`import torch` fail in-process (meta-path finder + sys.modules eviction). The
fresh-env variant of this gate lives in `scripts/gate_torch_free.sh`.
"""

import sys

import pytest


class block_torch:
    """Context manager: make ``import torch`` raise ImportError in-process.

    Evicts already-imported ``torch`` submodules and installs a meta-path
    finder that refuses them, so lazy ``import torch`` at point-of-use behaves
    exactly as it would in an env where torch was never installed.
    """

    def __init__(self):
        self._saved = {}

    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"No module named '{name}' (blocked by block_torch)")
        return None

    def __enter__(self):
        for mod in list(sys.modules):
            if mod == "torch" or mod.startswith("torch."):
                self._saved[mod] = sys.modules.pop(mod)
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *exc):
        sys.meta_path.remove(self)
        sys.modules.update(self._saved)
        self._saved.clear()
        return False


def _evict(prefix):
    for mod in list(sys.modules):
        if mod == prefix or mod.startswith(prefix + "."):
            del sys.modules[mod]


def test_package_imports_without_torch():
    """`import imspy_predictors` must succeed with torch unavailable."""
    with block_torch():
        _evict("imspy_predictors")
        import imspy_predictors  # noqa: F401

        assert imspy_predictors is not None


def test_require_torch_raises_actionable_hint():
    """The helper names the fix and reassures that Koina needs no torch."""
    from imspy_predictors.utility import require_torch

    with block_torch():
        with pytest.raises(ImportError) as ei:
            require_torch("some local model")

    msg = str(ei.value)
    assert "imspy-predictors[local]" in msg
    assert "Koina" in msg
    # never leak the raw failure as the primary message
    assert "some local model" in msg


@pytest.mark.parametrize(
    "module, cls",
    [
        ("imspy_predictors.ccs.predictors", "DeepPeptideIonMobilityApex"),
        ("imspy_predictors.rt.predictors", "DeepChromatographyApex"),
        ("imspy_predictors.intensity.predictors", "DeepPeptideIntensityPredictor"),
    ],
)
def test_local_predictor_instantiation_raises_hint(module, cls):
    """Instantiating a local (torch) predictor without torch → the hint."""
    import importlib

    with block_torch():
        mod = importlib.import_module(module)
        predictor_cls = getattr(mod, cls)
        with pytest.raises(ImportError) as ei:
            predictor_cls()

    assert "imspy-predictors[local]" in str(ei.value)


def test_koina_wrapper_constructs_without_torch():
    """The Prosit/Koina intensity wrapper is torch-free to construct."""
    with block_torch():
        from imspy_predictors.intensity.predictors import Prosit2023TimsTofWrapper

        wrapper = Prosit2023TimsTofWrapper()  # default use_koina=True; no torch touched
        assert wrapper.use_koina is True
