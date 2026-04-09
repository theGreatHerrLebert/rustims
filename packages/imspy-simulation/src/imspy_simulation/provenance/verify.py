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

# Possible statuses for an individual FieldCheck. UNCHECKED means we
# could not recompute this hash (e.g. the artifact is missing) — it is
# DELIBERATELY treated as not-ok by overall_ok so that "we couldn't
# verify it" never gets reported as VERIFIED.
STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_UNCHECKED = "unchecked"


@dataclass
class FieldCheck:
    """The verification status of a single hash field in the sidecar."""

    name: str
    expected: str  # the hex digest from the sidecar (e.g. "sha256:abc...")
    actual: str  # the hex digest we recomputed (empty if not checked)
    status: str  # one of STATUS_OK / STATUS_MISMATCH / STATUS_UNCHECKED
    detail: str = ""  # optional human-readable explanation (e.g. "no config file found")

    @property
    def ok(self) -> bool:
        """True iff the hash was computed AND matched the signed value."""
        return self.status == STATUS_OK

    def __str__(self) -> str:
        label = {
            STATUS_OK: "OK",
            STATUS_MISMATCH: "MISMATCH",
            STATUS_UNCHECKED: "UNCHECKED",
        }.get(self.status, self.status.upper())
        return f"{self.name:<18} {label}   ({self.expected[:24]}...)"


@dataclass
class TrustCheck:
    """Trust-pinning result. Layered ON TOP of the integrity check.

    Trust is conceptually orthogonal to integrity:

      - Integrity (signature_ok + per-field hash checks): "the bytes match
        what was signed by the key embedded in the sidecar". Always
        evaluated.
      - Trust (this struct): "the embedded key is who I expected it to
        be". Only evaluated if the user passed --expected-key-id or
        --require-trusted; otherwise reported as ``not_requested``.

    A bundle that fails integrity but passes trust is still a failure.
    A bundle that passes integrity but fails trust is also a failure.
    Both must be true for overall_ok.

    Status values:
      - "not_requested": no trust check was requested by the caller.
      - "ok":             the embedded key matched the user's expectation.
      - "id_mismatch":    --expected-key-id was given but the sidecar's
                          key_id is different.
      - "not_in_registry": --require-trusted was given but the sidecar's
                           signing key is not in the trusted-keys registry.
      - "registry_pem_mismatch": --require-trusted was given, the
                                 sidecar's key_id IS in the registry, but
                                 the registered PEM differs from the
                                 sidecar's embedded key. This catches
                                 forgeries that reuse a trusted key_id
                                 but ship different bytes.
    """

    status: str = "not_requested"
    detail: str = ""
    expected_key_id: str = ""
    actual_key_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "not_requested")

    @property
    def was_requested(self) -> bool:
        return self.status != "not_requested"


@dataclass
class VerificationResult:
    """The full result of verifying a sidecar."""

    sidecar_path: Path
    payload: Payload
    checks: list = field(default_factory=list)
    signature_ok: bool = False
    overall_ok: bool = False
    trust: TrustCheck = field(default_factory=TrustCheck)


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


def _config_path_for_sidecar(sidecar_path: Path) -> Path:
    """Return the conventional config-copy path for a given sidecar path.

    Convention: a sidecar at ``foo.provenance.json`` has its config copy
    at ``foo.config.toml`` in the same directory. The basename match is
    derived from the sidecar filename, NOT from any payload field, so a
    tampered ``experiment_name`` field cannot redirect this lookup.
    """
    name = sidecar_path.name
    if name.endswith(".provenance.json"):
        stem = name[: -len(".provenance.json")]
    else:
        stem = sidecar_path.stem
    return sidecar_path.parent / f"{stem}.config.toml"


def verify_sidecar(
    sidecar_path: PathLike,
    *,
    public_key_override: PathLike | None = None,
    config_path_override: PathLike | None = None,
    expected_key_id: str | None = None,
    require_trusted: bool = False,
    trusted_registry_path: PathLike | None = None,
) -> VerificationResult:
    """Verify a sidecar by recomputing all referenced hashes and checking the signature.

    Parameters
    ----------
    sidecar_path
        Path to the ``*.provenance.json`` sidecar file.
    public_key_override
        Optional path to a PEM verifying key. If given, this key is used
        instead of the one embedded in the sidecar.
    config_path_override
        Optional path to the TOML config file to hash. If given, this
        path is hashed and compared to ``payload.config_hash``. If not
        given, we look for the conventional ``{stem}.config.toml`` next
        to the sidecar. If neither resolves to a real file, the
        ``config_hash`` check is reported as UNCHECKED — we **never**
        fall back to the signed value, because that would make the
        check tautological.
    expected_key_id
        Optional ad-hoc trust pin. If given, the verifier requires the
        signing key id to equal this value. Mismatch sets the trust
        check status to ``id_mismatch`` and overall_ok to False.
    require_trusted
        If True, the signing key must be present in the trusted-keys
        registry AND its embedded PEM must equal the registered PEM.
        Mismatch sets the trust check status to ``not_in_registry`` or
        ``registry_pem_mismatch`` and overall_ok to False.
    trusted_registry_path
        Optional override for the trusted-keys registry path. Default:
        ``~/.config/timsim/trusted_keys.json``.

    Raises
    ------
    MalformedSidecar
        Sidecar file does not exist, is unreadable, or has the wrong shape.
    UnknownVersion
        Sidecar has a ``type`` or ``canonicalization_version`` we cannot handle.
    MissingArtifact
        Sidecar references a ``.d`` or ``synthetic_data.db`` that does not exist.
    SqliteNotQuiescent
        A SQLite file we need to hash has -wal/-shm/-journal sidecars present.

    Hash mismatches, signature mismatches, and trust mismatches do NOT
    raise; they appear on the returned ``VerificationResult``.
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

    # Recompute the .d hash.
    d_hash = canonicalize_d(d_path)

    # Recompute the ground-truth hash if the payload claims one.
    ground_truth_hash: bytes | None = None
    if payload.ground_truth_hash:
        if not ground_truth_path.is_file():
            raise MissingArtifact(
                f"sidecar references a ground-truth DB but none was found at "
                f"{ground_truth_path}"
            )
        ground_truth_hash = canonicalize_sqlite(ground_truth_path)

    # Resolve the config file: explicit override wins, otherwise look
    # for the conventional copy next to the sidecar. We never fall back
    # to the payload's signed value — that would make the check
    # tautological (compare hash to itself => always passes).
    config_hash: bytes | None = None
    config_check_status = STATUS_UNCHECKED
    config_check_actual = ""
    config_check_detail = ""

    if config_path_override is not None:
        config_resolved = Path(config_path_override)
        if not config_resolved.is_file():
            raise MissingArtifact(
                f"--config override points at a missing file: {config_resolved}"
            )
        config_hash = canonicalize_bytes(config_resolved.read_bytes())
        config_check_actual = _hex(config_hash)
        config_check_status = (
            STATUS_OK if payload.config_hash == config_check_actual else STATUS_MISMATCH
        )
    else:
        conventional_config = _config_path_for_sidecar(sidecar_path)
        if conventional_config.is_file():
            config_hash = canonicalize_bytes(conventional_config.read_bytes())
            config_check_actual = _hex(config_hash)
            config_check_status = (
                STATUS_OK if payload.config_hash == config_check_actual else STATUS_MISMATCH
            )
        else:
            config_check_detail = (
                f"no config file found at {conventional_config} "
                f"(pass --config to override)"
            )

    checks: list[FieldCheck] = []

    expected_d = payload.d_content_hash
    actual_d = _hex(d_hash)
    checks.append(
        FieldCheck(
            name="d_content_hash",
            expected=expected_d,
            actual=actual_d,
            status=STATUS_OK if expected_d == actual_d else STATUS_MISMATCH,
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
                status=STATUS_OK if expected_g == actual_g else STATUS_MISMATCH,
            )
        )

    checks.append(
        FieldCheck(
            name="config_hash",
            expected=payload.config_hash,
            actual=config_check_actual,
            status=config_check_status,
            detail=config_check_detail,
        )
    )

    # Recompute the composed content hash. If we could not hash the
    # config from disk, we mark the composed check UNCHECKED rather than
    # use the signed value — same reasoning as above. The .d and ground
    # truth components have already been recomputed independently, so
    # any tampering on those is caught by their own per-field checks.
    if config_hash is None:
        checks.append(
            FieldCheck(
                name="content_hash",
                expected=payload.content_hash,
                actual="",
                status=STATUS_UNCHECKED,
                detail="cannot recompose content_hash without the config file",
            )
        )
    else:
        composed = compose_content_hash(
            d_hash=d_hash,
            ground_truth_hash=ground_truth_hash,
            config_hash=config_hash,
        )
        checks.append(
            FieldCheck(
                name="content_hash",
                expected=payload.content_hash,
                actual=_hex(composed),
                status=(
                    STATUS_OK if payload.content_hash == _hex(composed) else STATUS_MISMATCH
                ),
            )
        )

    # Verify the signature against the canonical payload bytes. Decode
    # errors on attacker-controlled fields surface as MalformedSidecar
    # so the CLI maps them to a clean exit code rather than a generic
    # error. (Reviewer #5)
    try:
        if public_key_override is not None:
            public_key = load_public_key(public_key_override)
        else:
            public_key = public_key_from_b64(sidecar.verifying_key)
        signature_bytes = signature_from_b64(sidecar.signature)
    except (ValueError, MalformedSidecar) as e:
        raise MalformedSidecar(
            f"sidecar signature/verifying_key field is not decodable: {e}"
        ) from e

    signed_bytes = payload.to_canonical_json()

    try:
        public_key.verify(signature_bytes, signed_bytes)
        signature_ok = True
    except InvalidSignature:
        signature_ok = False
    except Exception:
        signature_ok = False

    # Trust check (layered ON TOP of integrity).
    trust = _evaluate_trust(
        payload=payload,
        sidecar=sidecar,
        expected_key_id=expected_key_id,
        require_trusted=require_trusted,
        trusted_registry_path=trusted_registry_path,
    )

    overall_ok = (
        signature_ok
        and all(c.status == STATUS_OK for c in checks)
        and trust.ok
    )

    return VerificationResult(
        sidecar_path=sidecar_path,
        payload=payload,
        checks=checks,
        signature_ok=signature_ok,
        overall_ok=overall_ok,
        trust=trust,
    )


def _evaluate_trust(
    *,
    payload: Payload,
    sidecar: Sidecar,
    expected_key_id: str | None,
    require_trusted: bool,
    trusted_registry_path: PathLike | None,
) -> TrustCheck:
    """Run the optional trust pinning checks. Returns a TrustCheck struct.

    Both ``expected_key_id`` and ``require_trusted`` are evaluated. If
    both are given and either fails, the FIRST failure to be discovered
    is reported (id mismatch wins because it's a stronger user assertion).
    """
    actual_key_id = payload.key_id

    # No trust pinning requested → always "ok" but flagged as not_requested
    # so the CLI can render it as "(not pinned)" rather than "trusted".
    if expected_key_id is None and not require_trusted:
        return TrustCheck(
            status="not_requested",
            actual_key_id=actual_key_id,
        )

    # Ad-hoc pin via --expected-key-id.
    if expected_key_id is not None:
        if expected_key_id != actual_key_id:
            return TrustCheck(
                status="id_mismatch",
                detail=(
                    f"sidecar was signed by {actual_key_id!r} but caller "
                    f"expected {expected_key_id!r}"
                ),
                expected_key_id=expected_key_id,
                actual_key_id=actual_key_id,
            )

    # Registry-based trust check via --require-trusted.
    if require_trusted:
        # Local import to avoid a circular dep at module load time.
        from imspy_simulation.provenance.trust import TrustedKeyRegistry

        try:
            registry = TrustedKeyRegistry.load(trusted_registry_path)
        except MalformedSidecar as e:
            return TrustCheck(
                status="not_in_registry",
                detail=f"trusted-keys registry is malformed: {e}",
                actual_key_id=actual_key_id,
            )

        entry = registry.find(actual_key_id)
        if entry is None:
            return TrustCheck(
                status="not_in_registry",
                detail=(
                    f"key {actual_key_id!r} is not in the trusted-keys "
                    f"registry at {registry.path}. Add it with "
                    f"'timsim-keys trust ...' if you trust this signer."
                ),
                actual_key_id=actual_key_id,
            )

        # The registry entry exists. Now make sure the embedded PEM
        # matches what is registered, NOT just that the key_id matches.
        # If they differ, someone shipped a sidecar that claims a
        # trusted key id but signed with different bytes — a forgery.
        try:
            sidecar_pubkey = public_key_from_b64(sidecar.verifying_key)
            registered_pubkey = entry.load_public_key()
        except (ValueError, ProvenanceError) as e:
            return TrustCheck(
                status="registry_pem_mismatch",
                detail=f"could not compare sidecar key to registered key: {e}",
                actual_key_id=actual_key_id,
            )

        from cryptography.hazmat.primitives import serialization

        sidecar_raw = sidecar_pubkey.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        registered_raw = registered_pubkey.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        if sidecar_raw != registered_raw:
            return TrustCheck(
                status="registry_pem_mismatch",
                detail=(
                    f"sidecar claims trusted key id {actual_key_id!r} "
                    f"but its embedded public key does not match the one "
                    f"in the registry. This is consistent with a forgery."
                ),
                actual_key_id=actual_key_id,
            )

    # All requested trust checks passed.
    return TrustCheck(
        status="ok",
        expected_key_id=expected_key_id or "",
        actual_key_id=actual_key_id,
    )
