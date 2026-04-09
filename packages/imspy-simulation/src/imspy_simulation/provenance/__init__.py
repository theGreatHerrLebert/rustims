"""TimSim provenance signing — Phase 0 prototype.

Implements the §7 Step 1 vendor-signing pattern from SIGNING.md, applied to
TimSim's own ``.d`` output. This is the working reference implementation
that §9 of SIGNING.md mandates as a prerequisite for vendor conversations.

This is a *software-rooted* prototype. The signing key lives on the user's
filesystem; it is not HSM-backed and does not provide hardware-equivalent
guarantees. The purpose is to demonstrate the structural chain of custody
and to ship a working canonical-hashing scheme that survives container-level
changes (SQLite VACUUM, REINDEX, page-size differences, etc.).

Public API:

    canonicalize_d(d_path) -> bytes
        Return a 32-byte SHA-256 over the canonical content of a ``.d``.

    canonicalize_sqlite(db_path) -> bytes
        Return a 32-byte SHA-256 over the canonical content of a SQLite file.

    sign_simulation_output(...) -> Path
        Sign a TimSim output (``.d`` + ground-truth DB + config) and write
        the sidecar JSON. Returns the sidecar path.

    verify_sidecar(sidecar_path, *, strict=False) -> VerificationResult
        Re-canonicalize, recompute hashes, and verify the Ed25519 signature.

See ``SIGNING.md`` for the conceptual framework and the floating-swinging-tide
plan file for the implementation contract.
"""

from imspy_simulation.provenance.errors import (
    HashMismatch,
    KeyNotFoundError,
    MalformedSidecar,
    MissingArtifact,
    ProvenanceError,
    SignatureMismatch,
    Unsigned,
    UnknownVersion,
)

__all__ = [
    "HashMismatch",
    "KeyNotFoundError",
    "MalformedSidecar",
    "MissingArtifact",
    "ProvenanceError",
    "SignatureMismatch",
    "Unsigned",
    "UnknownVersion",
    "canonicalize_d",
    "canonicalize_sqlite",
    "sign_simulation_output",
    "verify_sidecar",
]


def canonicalize_d(d_path):
    from imspy_simulation.provenance.canonicalize import canonicalize_d as _impl
    return _impl(d_path)


def canonicalize_sqlite(db_path):
    from imspy_simulation.provenance.canonicalize import canonicalize_sqlite as _impl
    return _impl(db_path)


def sign_simulation_output(*args, **kwargs):
    from imspy_simulation.provenance.sign import sign_simulation_output as _impl
    return _impl(*args, **kwargs)


def verify_sidecar(*args, **kwargs):
    from imspy_simulation.provenance.verify import verify_sidecar as _impl
    return _impl(*args, **kwargs)
