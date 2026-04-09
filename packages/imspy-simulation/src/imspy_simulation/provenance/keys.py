"""Ed25519 key generation, storage, and loading for TimSim provenance.

The signing key is a *software* key. It lives at::

    ~/.config/timsim/keys/signing_key.pem    (Ed25519 private, PKCS#8, no passphrase)
    ~/.config/timsim/keys/verifying_key.pem  (Ed25519 public)
    ~/.config/timsim/keys/key_id             (text file with the stable key id)

This is NOT a substitute for instrument-rooted attestation as proposed in
SIGNING.md §7 Step 1. The Phase 0 prototype demonstrates the structural chain
of custody — not hardware-equivalent guarantees. Anyone with read access to
the signing key file can forge signatures from this key.

Key id derivation (stable across processes / machines for the same key):

    key_id = "timsim-local-" + base32(blake2b(public_key_bytes, digest_size=10))

This format makes key ids 26 characters total, fits in URLs and JSON,
and is bit-for-bit deterministic.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import NamedTuple, Union

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from imspy_simulation.provenance.errors import KeyNotFoundError, MalformedKey

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)

_KEY_ID_PREFIX = "timsim-local-"
_BLAKE2B_DIGEST_SIZE = 10  # bytes; -> 16 base32 chars
_PRIVATE_KEY_FILENAME = "signing_key.pem"
_PUBLIC_KEY_FILENAME = "verifying_key.pem"
_KEY_ID_FILENAME = "key_id"


class KeyPair(NamedTuple):
    """An Ed25519 key pair plus its derived stable id."""

    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey
    key_id: str


def default_key_dir() -> Path:
    """Return the default key directory: ``~/.config/timsim/keys/``."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "timsim" / "keys"


def _public_key_raw_bytes(public_key: Ed25519PublicKey) -> bytes:
    """Return the 32 raw bytes of an Ed25519 public key."""
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def derive_key_id(public_key: Ed25519PublicKey) -> str:
    """Derive a stable, deterministic key id from an Ed25519 public key.

    Format: ``timsim-local-{base32(blake2b(pubkey)[:10])}`` (28 chars total
    including the prefix; 16 chars of base32-encoded digest).
    """
    raw = _public_key_raw_bytes(public_key)
    digest = hashlib.blake2b(raw, digest_size=_BLAKE2B_DIGEST_SIZE).digest()
    encoded = base64.b32encode(digest).decode("ascii").rstrip("=").lower()
    return _KEY_ID_PREFIX + encoded


def generate_keypair() -> KeyPair:
    """Generate a fresh Ed25519 keypair in memory (not written to disk)."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return KeyPair(
        private_key=private_key,
        public_key=public_key,
        key_id=derive_key_id(public_key),
    )


def write_keypair(keypair: KeyPair, key_dir: PathLike | None = None) -> Path:
    """Write a keypair to ``key_dir`` (default: ~/.config/timsim/keys/).

    Writes signing_key.pem (PKCS#8, no passphrase), verifying_key.pem,
    and a ``key_id`` text file. Returns the directory path. Sets mode
    0600 on the private key file.
    """
    key_dir = Path(key_dir) if key_dir is not None else default_key_dir()
    key_dir.mkdir(parents=True, exist_ok=True)

    private_pem = keypair.private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = keypair.public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    sk_path = key_dir / _PRIVATE_KEY_FILENAME
    pk_path = key_dir / _PUBLIC_KEY_FILENAME
    id_path = key_dir / _KEY_ID_FILENAME

    sk_path.write_bytes(private_pem)
    try:
        os.chmod(sk_path, 0o600)
    except OSError:
        # On filesystems where chmod is not meaningful (e.g. some CI
        # mounts), we tolerate the failure but log it.
        logger.warning("could not chmod signing key to 0600: %s", sk_path)

    pk_path.write_bytes(public_pem)
    id_path.write_text(keypair.key_id + "\n", encoding="ascii")

    return key_dir


def load_private_key(path: PathLike) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM file.

    Raises
    ------
    KeyNotFoundError
        The file does not exist.
    MalformedKey
        The file exists but cannot be parsed as an Ed25519 PKCS#8 PEM
        (corrupt, wrong armor, wrong algorithm). MalformedKey is a
        ``ProvenanceError`` subclass so the simulator hook's
        ``except ProvenanceError`` catches it and honors the
        ``[provenance] required`` contract.
    """
    path = Path(path)
    if not path.is_file():
        raise KeyNotFoundError(f"private key not found: {path}")
    pem = path.read_bytes()
    try:
        key = serialization.load_pem_private_key(pem, password=None)
    except (ValueError, TypeError) as e:
        raise MalformedKey(
            f"private key at {path} could not be parsed: {e}"
        ) from e
    if not isinstance(key, Ed25519PrivateKey):
        raise MalformedKey(
            f"key at {path} is not an Ed25519 private key "
            f"(got {type(key).__name__})"
        )
    return key


def load_public_key(path: PathLike) -> Ed25519PublicKey:
    """Load an Ed25519 public key from a PEM file.

    Raises
    ------
    KeyNotFoundError
        The file does not exist.
    MalformedKey
        The file exists but cannot be parsed as an Ed25519
        SubjectPublicKeyInfo PEM.
    """
    path = Path(path)
    if not path.is_file():
        raise KeyNotFoundError(f"public key not found: {path}")
    pem = path.read_bytes()
    try:
        key = serialization.load_pem_public_key(pem)
    except (ValueError, TypeError) as e:
        raise MalformedKey(
            f"public key at {path} could not be parsed: {e}"
        ) from e
    if not isinstance(key, Ed25519PublicKey):
        raise MalformedKey(
            f"key at {path} is not an Ed25519 public key "
            f"(got {type(key).__name__})"
        )
    return key


def load_or_create_keypair(key_dir: PathLike | None = None) -> KeyPair:
    """Load a keypair from ``key_dir``; generate and persist a new one if missing.

    On first call this generates a new keypair, persists it, and logs a
    clear "generated new TimSim signing key, this is a software key" message.
    Subsequent calls load the existing keypair.

    Raises ``KeyNotFoundError`` if a path was explicitly given but the
    private key exists in a partial / unreadable state.
    """
    key_dir = Path(key_dir) if key_dir is not None else default_key_dir()
    sk_path = key_dir / _PRIVATE_KEY_FILENAME

    if sk_path.exists():
        private_key = load_private_key(sk_path)
        public_key = private_key.public_key()
        return KeyPair(
            private_key=private_key,
            public_key=public_key,
            key_id=derive_key_id(public_key),
        )

    keypair = generate_keypair()
    write_keypair(keypair, key_dir)
    logger.warning(
        "Generated new TimSim signing key at %s. Key id: %s. "
        "This is a SOFTWARE key — see SIGNING.md §9 for the limitations of "
        "the Phase 0 prototype.",
        sk_path,
        keypair.key_id,
    )
    return keypair


def public_key_to_b64(public_key: Ed25519PublicKey) -> str:
    """Encode a public key as the ``ed25519:base64:...`` string used in sidecars."""
    raw = _public_key_raw_bytes(public_key)
    return "ed25519:base64:" + base64.b64encode(raw).decode("ascii")


def public_key_from_b64(encoded: str) -> Ed25519PublicKey:
    """Decode an ``ed25519:base64:...`` public key string."""
    if not encoded.startswith("ed25519:base64:"):
        raise ValueError(f"unrecognized verifying key format: {encoded[:20]}...")
    raw = base64.b64decode(encoded[len("ed25519:base64:"):])
    return Ed25519PublicKey.from_public_bytes(raw)


def signature_to_b64(signature: bytes) -> str:
    """Encode a raw Ed25519 signature as ``ed25519:base64:...``."""
    return "ed25519:base64:" + base64.b64encode(signature).decode("ascii")


def signature_from_b64(encoded: str) -> bytes:
    """Decode a signature in ``ed25519:base64:...`` form."""
    if not encoded.startswith("ed25519:base64:"):
        raise ValueError(f"unrecognized signature format: {encoded[:20]}...")
    return base64.b64decode(encoded[len("ed25519:base64:"):])
