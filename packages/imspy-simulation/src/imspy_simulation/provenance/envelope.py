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
ATTESTATION_TYPE_MZML = "timsim.provenance.mzml.v0"
SUPPORTED_TYPES = frozenset({ATTESTATION_TYPE, ATTESTATION_TYPE_MZML})
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
class MzmlPayload:
    """The signed inner payload of an mzML provenance sidecar.

    Distinct from ``Payload`` because mzML has no ground-truth analog
    and the producer is not necessarily TimSim — any tool that emits
    mzML can sign it (Synthedia, SMITER, msconvert, custom converters).
    The field set is therefore named generically (``tool_*``) rather
    than ``simulator_*``.
    """

    tool_name: str
    tool_version: str
    experiment_name: str
    config_hash: str
    mzml_content_hash: str
    content_hash: str
    timestamp_utc: str
    key_id: str
    canonicalization_version: str = "v0"

    def to_canonical_json(self) -> bytes:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "MzmlPayload":
        required = {
            "tool_name",
            "tool_version",
            "experiment_name",
            "config_hash",
            "mzml_content_hash",
            "content_hash",
            "timestamp_utc",
            "key_id",
            "canonicalization_version",
        }
        missing = required - data.keys()
        if missing:
            raise MalformedSidecar(
                f"mzml sidecar payload is missing required fields: {sorted(missing)}"
            )
        if data["canonicalization_version"] not in SUPPORTED_CANONICALIZATION_VERSIONS:
            raise UnknownVersion(
                f"mzml sidecar canonicalization_version "
                f"{data['canonicalization_version']!r} is not supported"
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
        """Parse a .d sidecar from JSON bytes.

        Raises MalformedSidecar on shape errors. Raises UnknownVersion if
        the sidecar's type tag is not the .d attestation type — in that
        case the caller should dispatch to a different parser (see the
        module-level ``parse_sidecar`` helper).
        """
        try:
            blob = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MalformedSidecar(f"sidecar is not valid UTF-8 JSON: {e}") from e

        if not isinstance(blob, dict):
            raise MalformedSidecar("sidecar root must be a JSON object")

        type_tag = blob.get("type")
        if type_tag != ATTESTATION_TYPE:
            raise UnknownVersion(
                f"sidecar type {type_tag!r} is not the .d attestation type "
                f"({ATTESTATION_TYPE!r}); use parse_sidecar() to dispatch"
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


@dataclass(frozen=True)
class MzmlSidecar:
    """Sidecar envelope for mzML provenance attestations.

    Same structural shape as ``Sidecar`` (type / payload / signature /
    verifying_key) but the payload is an ``MzmlPayload`` (different
    fields) and the type tag is ``timsim.provenance.mzml.v0``.
    """

    payload: MzmlPayload
    signature: str
    verifying_key: str
    type: str = ATTESTATION_TYPE_MZML

    def to_json_bytes(self) -> bytes:
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
    def from_json_bytes(cls, data: bytes) -> "MzmlSidecar":
        try:
            blob = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MalformedSidecar(f"sidecar is not valid UTF-8 JSON: {e}") from e

        if not isinstance(blob, dict):
            raise MalformedSidecar("sidecar root must be a JSON object")

        type_tag = blob.get("type")
        if type_tag != ATTESTATION_TYPE_MZML:
            raise UnknownVersion(
                f"sidecar type {type_tag!r} is not the mzml attestation type "
                f"({ATTESTATION_TYPE_MZML!r}); use parse_sidecar() to dispatch"
            )

        payload_dict = blob.get("payload")
        if not isinstance(payload_dict, dict):
            raise MalformedSidecar("sidecar.payload must be an object")
        payload = MzmlPayload.from_dict(payload_dict)

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


def parse_sidecar(data: bytes) -> "Sidecar | MzmlSidecar":
    """Parse a sidecar JSON blob and return the right concrete type.

    Looks at the ``type`` field at the top of the envelope and dispatches
    to either ``Sidecar.from_json_bytes`` or ``MzmlSidecar.from_json_bytes``.
    Raises ``MalformedSidecar`` on JSON shape errors and ``UnknownVersion``
    if the type tag is not in ``SUPPORTED_TYPES``.
    """
    try:
        blob = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise MalformedSidecar(f"sidecar is not valid UTF-8 JSON: {e}") from e

    if not isinstance(blob, dict):
        raise MalformedSidecar("sidecar root must be a JSON object")

    type_tag = blob.get("type")
    if type_tag == ATTESTATION_TYPE:
        return Sidecar.from_json_bytes(data)
    if type_tag == ATTESTATION_TYPE_MZML:
        return MzmlSidecar.from_json_bytes(data)
    raise UnknownVersion(
        f"sidecar type {type_tag!r} is not supported "
        f"(supported: {sorted(SUPPORTED_TYPES)})"
    )
