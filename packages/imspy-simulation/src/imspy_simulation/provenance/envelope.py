"""Sidecar JSON envelope for TimSim provenance attestations.

The sidecar is the on-disk format that wraps a signed payload. Its layout
is documented in plan §4.3.

The payload fields are alphabetized so the canonical JSON serialization
(used as the bytes that get signed) is deterministic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from imspy_simulation.provenance.errors import MalformedSidecar, UnknownVersion

ATTESTATION_TYPE = "timsim.provenance.v0"
SUPPORTED_TYPES = frozenset({ATTESTATION_TYPE})
SUPPORTED_CANONICALIZATION_VERSIONS = frozenset({"v0"})


@dataclass(frozen=True)
class Payload:
    """The signed inner payload of a provenance sidecar.

    All hash fields are stored as ``"sha256:" + hex_digest`` strings to
    keep the JSON human-inspectable. The bytes that get signed are the
    canonical JSON serialization of this dataclass (sorted keys, no
    whitespace, UTF-8).
    """

    simulator_name: str
    simulator_version: str
    experiment_name: str
    config_hash: str
    d_content_hash: str
    ground_truth_hash: str  # may be empty string if no ground truth was signed
    content_hash: str
    timestamp_utc: str
    key_id: str
    canonicalization_version: str = "v0"

    def to_canonical_json(self) -> bytes:
        """Serialize this payload to deterministic UTF-8 JSON bytes for signing.

        Sorted keys, no whitespace, ensure_ascii=False so unicode is preserved.
        Two payloads with the same field values produce byte-identical output.
        """
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "Payload":
        """Build a Payload from a dict, validating required fields."""
        required = {
            "simulator_name",
            "simulator_version",
            "experiment_name",
            "config_hash",
            "d_content_hash",
            "ground_truth_hash",
            "content_hash",
            "timestamp_utc",
            "key_id",
            "canonicalization_version",
        }
        missing = required - data.keys()
        if missing:
            raise MalformedSidecar(
                f"sidecar payload is missing required fields: {sorted(missing)}"
            )
        if data["canonicalization_version"] not in SUPPORTED_CANONICALIZATION_VERSIONS:
            raise UnknownVersion(
                f"sidecar canonicalization_version "
                f"{data['canonicalization_version']!r} is not supported by this "
                f"build (supported: {sorted(SUPPORTED_CANONICALIZATION_VERSIONS)})"
            )
        return cls(**{k: data[k] for k in required})


@dataclass(frozen=True)
class Sidecar:
    """The full sidecar envelope: type tag + payload + signature + verifying key."""

    payload: Payload
    signature: str  # "ed25519:base64:..."
    verifying_key: str  # "ed25519:base64:..."
    type: str = ATTESTATION_TYPE

    def to_json_bytes(self) -> bytes:
        """Serialize the entire sidecar to indented JSON bytes for disk.

        Indentation is for human inspection only — the bytes that get
        signed are produced by ``Payload.to_canonical_json``, not by this
        method.
        """
        blob = {
            "type": self.type,
            "payload": asdict(self.payload),
            "signature": self.signature,
            "verifying_key": self.verifying_key,
        }
        return json.dumps(blob, indent=2, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "Sidecar":
        """Parse a sidecar from JSON bytes. Raises MalformedSidecar on shape errors."""
        try:
            blob = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MalformedSidecar(f"sidecar is not valid UTF-8 JSON: {e}") from e

        if not isinstance(blob, dict):
            raise MalformedSidecar("sidecar root must be a JSON object")

        type_tag = blob.get("type")
        if type_tag not in SUPPORTED_TYPES:
            raise UnknownVersion(
                f"sidecar type {type_tag!r} is not supported "
                f"(supported: {sorted(SUPPORTED_TYPES)})"
            )

        payload_dict = blob.get("payload")
        if not isinstance(payload_dict, dict):
            raise MalformedSidecar("sidecar.payload must be an object")
        payload = Payload.from_dict(payload_dict)

        signature = blob.get("signature")
        verifying_key = blob.get("verifying_key")
        if not isinstance(signature, str) or not isinstance(verifying_key, str):
            raise MalformedSidecar(
                "sidecar.signature and sidecar.verifying_key must be strings"
            )

        return cls(
            payload=payload,
            signature=signature,
            verifying_key=verifying_key,
            type=type_tag,
        )
