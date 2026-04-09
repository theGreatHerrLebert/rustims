"""Verify a TimSim provenance sidecar against the artifacts it describes.

The verifier is pure: it takes a sidecar path, reads the referenced
artifacts, recomputes canonical hashes, and validates the Ed25519
signature. It returns a structured ``VerificationResult`` rather than
raising on every kind of failure — the CLI maps failures to exit codes.

The only exceptions raised are *structural* problems (sidecar missing,
malformed, unknown version, missing referenced artifact). Hash mismatches
and signature mismatches are reported as fields on the result object so
the CLI can render a useful diagnostic listing every check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from cryptography.exceptions import InvalidSignature

from imspy_simulation.provenance.canonicalize import (
    canonicalize_bytes,
    canonicalize_d,
    canonicalize_sqlite,
    compose_content_hash,
)
from imspy_simulation.provenance.envelope import Payload, Sidecar
from imspy_simulation.provenance.errors import (
    MalformedSidecar,
    MissingArtifact,
    Unsigned,
)
from imspy_simulation.provenance.keys import (
    load_public_key,
    public_key_from_b64,
    signature_from_b64,
)

PathLike = Union[str, Path]


@dataclass
class FieldCheck:
    """The verification status of a single hash field in the sidecar."""

    name: str
    expected: str  # the hex digest from the sidecar (e.g. "sha256:abc...")
    actual: str  # the hex digest we recomputed (empty if not checked)
    ok: bool

    def __str__(self) -> str:
        status = "OK" if self.ok else "MISMATCH"
        return f"{self.name:<18} {status}   ({self.expected[:24]}...)"


@dataclass
class VerificationResult:
    """The full result of verifying a sidecar."""

    sidecar_path: Path
    payload: Payload
    checks: list = field(default_factory=list)
    signature_ok: bool = False
    overall_ok: bool = False


def _hex(b: bytes) -> str:
    return "sha256:" + b.hex()


def _decode_hash(field_value: str) -> bytes:
    """Decode a ``"sha256:hex..."`` string to raw bytes. Returns empty bytes if not present."""
    if not field_value:
        return b""
    if not field_value.startswith("sha256:"):
        raise MalformedSidecar(f"hash field is not sha256-prefixed: {field_value!r}")
    try:
        return bytes.fromhex(field_value[len("sha256:"):])
    except ValueError as e:
        raise MalformedSidecar(f"hash field is not valid hex: {field_value!r}") from e


def find_sidecar_for(path: PathLike) -> Path | None:
    """Given a path to a sidecar, an experiment dir, or a .d, return the sidecar path.

    Discovery rules (per plan §6.1):
        - If ``path`` is itself a sidecar JSON file, return it.
        - If ``path`` is a directory, look for ``*.provenance.json`` inside it.
        - If ``path`` is a ``.d`` directory, look for the sidecar one level up.
        - Otherwise return None.
    """
    path = Path(path)

    if path.is_file() and path.suffix == ".json" and ".provenance" in path.name:
        return path

    if path.is_dir():
        # If this directory itself looks like a .d, search its parent first.
        if path.suffix == ".d":
            parent_hits = sorted(path.parent.glob("*.provenance.json"))
            if parent_hits:
                return parent_hits[0]

        hits = sorted(path.glob("*.provenance.json"))
        if hits:
            return hits[0]

    return None


def _find_unique_d(search_root: Path) -> Path | None:
    """Find a unique ``.d`` directory in ``search_root`` or one level deep.

    A ``.d`` is recognized as a directory with ``.d`` suffix that contains
    ``analysis.tdf``. We search the immediate children of ``search_root``
    and one level deeper (to handle the conventional
    ``{save_path}/{exp}/{exp}.d`` layout).

    Path resolution is intentionally INDEPENDENT of the sidecar payload so
    that a tampered ``experiment_name`` field cannot redirect verification
    to a non-existent file. The integrity of the path resolution is what
    lets us treat signature mismatch as a clean diagnostic.

    Returns None if zero or multiple candidates are found.
    """
    candidates: list[Path] = []
    try:
        children = list(search_root.iterdir())
    except OSError:
        return None

    for child in children:
        if not child.is_dir():
            continue
        if child.suffix == ".d" and (child / "analysis.tdf").is_file():
            candidates.append(child)
            continue
        # Look one level deeper for the {exp}/{exp}.d pattern.
        try:
            for grandchild in child.iterdir():
                if (
                    grandchild.is_dir()
                    and grandchild.suffix == ".d"
                    and (grandchild / "analysis.tdf").is_file()
                ):
                    candidates.append(grandchild)
        except OSError:
            continue

    if len(candidates) == 1:
        return candidates[0]
    return None


def verify_sidecar(
    sidecar_path: PathLike,
    *,
    public_key_override: PathLike | None = None,
) -> VerificationResult:
    """Verify a sidecar by recomputing all referenced hashes and checking the signature.

    Raises
    ------
    MalformedSidecar
        Sidecar file exists but is unreadable or shape-incorrect.
    UnknownVersion
        Sidecar has a ``type`` or ``canonicalization_version`` we cannot handle.
    MissingArtifact
        Sidecar references a ``.d`` or ``synthetic_data.db`` that does not exist.

    Hash mismatches and signature mismatches do NOT raise; they appear
    on the returned ``VerificationResult``.
    """
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.is_file():
        raise MalformedSidecar(f"sidecar file does not exist: {sidecar_path}")

    sidecar = Sidecar.from_json_bytes(sidecar_path.read_bytes())
    payload = sidecar.payload

    # Locate the .d INDEPENDENTLY of the payload, so a tampered
    # experiment_name field cannot redirect verification to a phantom
    # path. We accept either the conventional {save_path}/{exp}/{exp}.d
    # layout or the sibling {save_path}/{exp}.d layout — _find_unique_d
    # handles both.
    save_path = sidecar_path.parent
    d_path = _find_unique_d(save_path)
    if d_path is None:
        raise MissingArtifact(
            f"could not find a unique .d directory near {save_path}"
        )

    ground_truth_path = save_path / "synthetic_data.db"

    # Recompute hashes.
    d_hash = canonicalize_d(d_path)
    ground_truth_hash: bytes | None = None
    if payload.ground_truth_hash:
        if not ground_truth_path.is_file():
            raise MissingArtifact(
                f"sidecar references a ground-truth DB but none was found at "
                f"{ground_truth_path}"
            )
        ground_truth_hash = canonicalize_sqlite(ground_truth_path)

    # The config file may not be co-located with the sidecar in all
    # workflows. We can only re-hash it if we can find it. The user can
    # supply it via CLI override in a future iteration; for v0 we attempt
    # discovery and skip the check if absent.
    config_hash: bytes | None = None
    config_check_actual = ""
    candidate_configs = list(save_path.glob("*.toml"))
    if candidate_configs:
        # Use the first match. If multiple TOMLs sit next to the sidecar
        # the user can disambiguate later; for now we accept the first.
        config_hash = canonicalize_bytes(candidate_configs[0].read_bytes())
        config_check_actual = _hex(config_hash)

    checks: list[FieldCheck] = []

    expected_d = payload.d_content_hash
    actual_d = _hex(d_hash)
    checks.append(
        FieldCheck(
            name="d_content_hash",
            expected=expected_d,
            actual=actual_d,
            ok=(expected_d == actual_d),
        )
    )

    if payload.ground_truth_hash:
        expected_g = payload.ground_truth_hash
        actual_g = _hex(ground_truth_hash) if ground_truth_hash is not None else ""
        checks.append(
            FieldCheck(
                name="ground_truth_hash",
                expected=expected_g,
                actual=actual_g,
                ok=(expected_g == actual_g),
            )
        )

    if config_hash is not None:
        checks.append(
            FieldCheck(
                name="config_hash",
                expected=payload.config_hash,
                actual=config_check_actual,
                ok=(payload.config_hash == config_check_actual),
            )
        )

    # Recompute the composed content hash and check it as well.
    composed = compose_content_hash(
        d_hash=d_hash,
        ground_truth_hash=ground_truth_hash,
        config_hash=config_hash if config_hash is not None else _decode_hash(payload.config_hash),
    )
    checks.append(
        FieldCheck(
            name="content_hash",
            expected=payload.content_hash,
            actual=_hex(composed),
            ok=(payload.content_hash == _hex(composed)),
        )
    )

    # Verify the signature against the canonical payload bytes.
    if public_key_override is not None:
        public_key = load_public_key(public_key_override)
    else:
        public_key = public_key_from_b64(sidecar.verifying_key)

    signature_bytes = signature_from_b64(sidecar.signature)
    signed_bytes = payload.to_canonical_json()

    try:
        public_key.verify(signature_bytes, signed_bytes)
        signature_ok = True
    except InvalidSignature:
        signature_ok = False
    except Exception:
        signature_ok = False

    overall_ok = signature_ok and all(c.ok for c in checks)

    return VerificationResult(
        sidecar_path=sidecar_path,
        payload=payload,
        checks=checks,
        signature_ok=signature_ok,
        overall_ok=overall_ok,
    )
