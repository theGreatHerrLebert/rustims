"""
Download and cache pretrained model files from GitHub Releases.

Models are downloaded on first use and cached locally at:
    $IMSPY_CACHE_DIR  or  ~/.cache/imspy/models/v{MODEL_VERSION}/

Uses torch.hub.download_url_to_file() for downloads (progress bar included),
with a fallback to urllib if torch.hub is unavailable.
"""

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version & release metadata
# ---------------------------------------------------------------------------

MODEL_VERSION = "0.5.0"
GITHUB_RELEASE_TAG = "models-v0.5.0"

_RELEASE_BASE = (
    f"https://github.com/theGreatHerrLebert/rustims/releases/download/{GITHUB_RELEASE_TAG}"
)

# Maps the *package-relative* path (as used by callers of get_model_path)
# to its download URL and expected SHA-256 hash.
MODELS = {
    "ccs/best_model.pt": {
        "filename": "ccs-best_model.pt",
        "sha256": "f06282ffcc071bd046bfd054ac0db78c18a0ce405918046de494b51eb26f75e4",
    },
    "rt/best_model.pt": {
        "filename": "rt-best_model.pt",
        "sha256": "6f318fcfe2d37c1a24b1bea32db6c9fa8eb87d2c545a20e93ae8e422f901ad60",
    },
    "charge/best_model.pt": {
        "filename": "charge-best_model.pt",
        "sha256": "e0e2ab6d43f028718d7e09aa5c2946059d8f67cf6cd165785a6fc2038cf64ecf",
    },
    "intensity/best_model.pt": {
        "filename": "intensity-best_model.pt",
        "sha256": "6849d857bf03d1205716940206ad49f8c3ecbf36e3668429883b468274b9d3fc",
    },
    "pretrained_encoder.pt": {
        "filename": "pretrained_encoder.pt",
        "sha256": "43ccc2f836bf3d81943ddce353ade9628e7d036421ba5b5c182bf163e496385e",
    },
}


def get_cache_dir() -> Path:
    """Return the local cache directory for pretrained models.

    Respects the ``IMSPY_CACHE_DIR`` environment variable.  Falls back to
    ``~/.cache/imspy/models/v{MODEL_VERSION}``.
    """
    env = os.environ.get("IMSPY_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "imspy" / "models" / f"v{MODEL_VERSION}"


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest*, preferring torch.hub for its progress bar."""
    try:
        from torch.hub import download_url_to_file

        download_url_to_file(url, str(dest), progress=True)
    except Exception:
        # Fallback: plain urllib (no progress bar, but no extra deps)
        import urllib.request

        logger.info("Downloading %s (urllib fallback) ...", url)
        urllib.request.urlretrieve(url, str(dest))


def _find_bundled_model(model_name: str) -> Path | None:
    """Check for a model file co-located with this module (source / editable installs)."""
    local = Path(__file__).resolve().parent / model_name
    if local.is_file():
        return local
    return None


def ensure_model(model_name: str) -> Path:
    """Return the path to a cached model, downloading it first if necessary.

    Resolution order:

    1. Bundled alongside this module (editable / source installs).
    2. Local cache (``$IMSPY_CACHE_DIR`` or ``~/.cache/imspy/models/``).
    3. Download from GitHub Releases.

    Parameters
    ----------
    model_name : str
        Package-relative name, e.g. ``"ccs/best_model.pt"``.

    Returns
    -------
    Path
        Absolute path to the ``.pt`` file.

    Raises
    ------
    ValueError
        If *model_name* is not a known model.
    RuntimeError
        If the download succeeds but the SHA-256 check fails.
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Known models: {sorted(MODELS)}"
        )

    # 1. Bundled copy (works for editable / source installs where .pt files
    #    live next to this module even though they are excluded from wheels).
    bundled = _find_bundled_model(model_name)
    if bundled is not None:
        return bundled

    meta = MODELS[model_name]
    cache_dir = get_cache_dir()
    cached_path = cache_dir / model_name

    # 2. Fast path: already cached and hash matches.
    if cached_path.exists():
        digest = _sha256(cached_path)
        if digest == meta["sha256"]:
            return cached_path
        logger.warning(
            "Cached model %s has SHA-256 %s (expected %s); re-downloading.",
            cached_path,
            digest,
            meta["sha256"],
        )

    # 3. Download into a temp file in the same filesystem, then atomic-rename.
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_RELEASE_BASE}/{meta['filename']}"
    logger.info("Downloading model '%s' from %s ...", model_name, url)

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(cached_path.parent), suffix=".download"
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    try:
        _download(url, tmp_path)

        digest = _sha256(tmp_path)
        if digest != meta["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {model_name}: "
                f"got {digest}, expected {meta['sha256']}"
            )

        shutil.move(str(tmp_path), str(cached_path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info("Cached model at %s", cached_path)
    return cached_path
