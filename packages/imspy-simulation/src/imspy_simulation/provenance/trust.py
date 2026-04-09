"""Trusted-keys registry for TimSim provenance verification.

The Phase 0 sidecar embeds its own verifying key. That makes the signature
*internally consistent* — the bytes match the math — but it does NOT prove
identity. Anyone can generate a fresh keypair, sign their own bundle, and
``timsim-verify`` will say VERIFIED because the math checks out.

This module is the bridge from "internally consistent" to "actually
trusted by someone in particular". Two complementary mechanisms:

1. **Ad-hoc pinning** via ``timsim-verify --expected-key-id ID``. The
   user asserts at the command line that they expect *this exact key id*.
   No registry involved.

2. **Trusted-keys registry** at ``~/.config/timsim/trusted_keys.json``,
   populated explicitly by the user via the ``timsim-keys trust ...``
   command. ``timsim-verify --require-trusted`` then checks that the
   sidecar's signing key is in the registry. The registry stores the
   full PEM, not just the key id, so a forged sidecar that *claims* a
   trusted key id but ships a different public key is caught.

The user must add keys explicitly. The registry is **never** populated
by ``timsim-verify`` automatically — that would be a TOFU (trust on
first use) pattern, and TOFU silently grants trust on the first run,
which is exactly the failure mode we are trying to prevent.

File format (JSON, atomically written):

    {
      "schema": "timsim.trusted_keys/v0",
      "keys": [
        {
          "key_id": "timsim-local-q4ffrs376n5yefou",
          "public_key_pem": "-----BEGIN PUBLIC KEY-----\\n...\\n-----END PUBLIC KEY-----\\n",
          "comment": "lab X production server",
          "added_at": "2026-04-09T10:00:00.000Z"
        }
      ]
    }

The schema version is bumped only when the on-disk format changes
incompatibly. Adding new optional fields is a non-breaking change.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from imspy_simulation.provenance.errors import (
    KeyNotFoundError,
    MalformedSidecar,
    ProvenanceError,
)
from imspy_simulation.provenance.keys import (
    derive_key_id,
    load_public_key,
    public_key_from_b64,
)

PathLike = Union[str, Path]

REGISTRY_SCHEMA = "timsim.trusted_keys/v0"
SUPPORTED_REGISTRY_SCHEMAS = frozenset({REGISTRY_SCHEMA})

_REGISTRY_FILENAME = "trusted_keys.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustedKey:
    """A single entry in the trusted-keys registry."""

    key_id: str
    public_key_pem: str
    comment: str
    added_at: str  # ISO 8601 UTC

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrustedKey":
        required = {"key_id", "public_key_pem", "comment", "added_at"}
        missing = required - data.keys()
        if missing:
            raise MalformedSidecar(
                f"trusted-key entry is missing fields: {sorted(missing)}"
            )
        return cls(
            key_id=str(data["key_id"]),
            public_key_pem=str(data["public_key_pem"]),
            comment=str(data["comment"]),
            added_at=str(data["added_at"]),
        )

    def load_public_key(self) -> Ed25519PublicKey:
        """Parse the stored PEM into an Ed25519PublicKey object."""
        from cryptography.hazmat.primitives import serialization
        key = serialization.load_pem_public_key(self.public_key_pem.encode("utf-8"))
        if not isinstance(key, Ed25519PublicKey):
            raise ProvenanceError(
                f"trusted key {self.key_id!r} is not an Ed25519 public key"
            )
        return key


# ---------------------------------------------------------------------------
# Registry persistence
# ---------------------------------------------------------------------------


def default_registry_path() -> Path:
    """Return the default registry path: ``~/.config/timsim/trusted_keys.json``.

    Honors ``XDG_CONFIG_HOME`` if set, matching the convention used by
    keys.py for the signing key location.
    """
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "timsim" / _REGISTRY_FILENAME


@dataclass
class TrustedKeyRegistry:
    """An in-memory view of the trusted-keys registry, with persistence helpers.

    Use ``load`` / ``save`` to round-trip the on-disk JSON. Mutations
    (``add``, ``remove``) update the in-memory state and require an
    explicit ``save`` to persist — no implicit writes.
    """

    path: Path
    keys: list[TrustedKey] = field(default_factory=list)

    @classmethod
    def load(cls, path: PathLike | None = None) -> "TrustedKeyRegistry":
        """Load a registry from disk. Returns an empty registry if the file is missing.

        A missing file is normal: it just means the user has not
        trusted any keys yet. Empty != malformed.
        """
        registry_path = Path(path) if path is not None else default_registry_path()
        if not registry_path.exists():
            return cls(path=registry_path, keys=[])

        try:
            blob = json.loads(registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise MalformedSidecar(
                f"trusted-keys registry at {registry_path} is not valid JSON: {e}"
            ) from e

        if not isinstance(blob, dict):
            raise MalformedSidecar(
                f"trusted-keys registry at {registry_path} root must be an object"
            )
        schema = blob.get("schema")
        if schema not in SUPPORTED_REGISTRY_SCHEMAS:
            raise MalformedSidecar(
                f"trusted-keys registry schema {schema!r} is not supported "
                f"(supported: {sorted(SUPPORTED_REGISTRY_SCHEMAS)})"
            )
        keys_blob = blob.get("keys", [])
        if not isinstance(keys_blob, list):
            raise MalformedSidecar(
                "trusted-keys registry 'keys' field must be a list"
            )
        keys = [TrustedKey.from_dict(k) for k in keys_blob]
        return cls(path=registry_path, keys=keys)

    def save(self) -> None:
        """Atomically write the registry to disk (write temp + rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "schema": REGISTRY_SCHEMA,
            "keys": [k.to_dict() for k in self.keys],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(blob, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            os.fsync(os.open(tmp, os.O_RDONLY))
        except OSError:
            pass
        os.replace(tmp, self.path)

    def add(self, key: TrustedKey) -> None:
        """Add a key. Raises if a different PEM is already stored under the same key_id.

        Adding the *same* key again is a no-op (so ``timsim-keys trust``
        is idempotent). Adding a *different* PEM under an existing key_id
        is an error — that would either be a hash collision or a forgery
        attempt and we refuse rather than silently overwrite.
        """
        for existing in self.keys:
            if existing.key_id == key.key_id:
                if existing.public_key_pem == key.public_key_pem:
                    return  # idempotent: same key, same id, no-op
                raise ProvenanceError(
                    f"refusing to add: a different public key is already trusted "
                    f"under key_id {key.key_id!r}. Untrust it first if this is "
                    f"intentional."
                )
        self.keys.append(key)

    def remove(self, key_id: str) -> bool:
        """Remove a key by id. Returns True if removed, False if not present."""
        for i, k in enumerate(self.keys):
            if k.key_id == key_id:
                del self.keys[i]
                return True
        return False

    def find(self, key_id: str) -> Optional[TrustedKey]:
        """Look up a trusted key by id. Returns None if not present."""
        for k in self.keys:
            if k.key_id == key_id:
                return k
        return None

    def __contains__(self, key_id: str) -> bool:
        return self.find(key_id) is not None

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Iterable[TrustedKey]:
        return iter(self.keys)


# ---------------------------------------------------------------------------
# Helpers used by the verifier and the timsim-keys CLI
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time as a millisecond-precision ISO 8601 string."""
    return (
        _dt.datetime.now(tz=_dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z"
    )


def trusted_key_from_public_key(
    public_key: Ed25519PublicKey,
    *,
    comment: str,
    added_at: str | None = None,
) -> TrustedKey:
    """Build a TrustedKey entry from an Ed25519 public key object.

    The key_id is derived deterministically from the key bytes via the
    same algorithm as keys.derive_key_id, so trust round-trips.
    """
    from cryptography.hazmat.primitives import serialization
    pem_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return TrustedKey(
        key_id=derive_key_id(public_key),
        public_key_pem=pem_bytes.decode("ascii"),
        comment=comment,
        added_at=added_at or utc_now_iso(),
    )


def trusted_key_from_pem_file(path: PathLike, *, comment: str) -> TrustedKey:
    """Load a public key PEM file and wrap it as a TrustedKey entry."""
    public_key = load_public_key(path)
    return trusted_key_from_public_key(public_key, comment=comment)


def trusted_key_from_sidecar_file(sidecar_path: PathLike, *, comment: str) -> TrustedKey:
    """Extract the verifying key from a sidecar JSON file and wrap it as a TrustedKey.

    Useful for the workflow "I just received this signed dataset, and I
    want to trust the key it was signed with going forward". The user
    is asserting trust based on out-of-band confidence in the dataset's
    origin (e.g. they got it from a known collaborator).
    """
    from imspy_simulation.provenance.envelope import Sidecar

    sidecar_path = Path(sidecar_path)
    if not sidecar_path.is_file():
        raise KeyNotFoundError(f"sidecar file not found: {sidecar_path}")
    sidecar = Sidecar.from_json_bytes(sidecar_path.read_bytes())
    try:
        public_key = public_key_from_b64(sidecar.verifying_key)
    except ValueError as e:
        raise MalformedSidecar(
            f"sidecar at {sidecar_path} has an undecodable verifying_key: {e}"
        ) from e
    return trusted_key_from_public_key(public_key, comment=comment)
