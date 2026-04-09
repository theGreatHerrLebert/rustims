"""Typed exceptions for the TimSim provenance subsystem.

Every error path in the provenance module raises one of these. The CLI
maps each exception type to a stable exit code (see ``cli.py`` and
SIGNING-prototype plan §6.3). Tests assert exception types, not strings.
"""


class ProvenanceError(Exception):
    """Base for all provenance errors. Catches "anything went wrong with signing"."""


class KeyNotFoundError(ProvenanceError):
    """A signing or verifying key was requested but not found on disk."""


class MalformedSidecar(ProvenanceError):
    """The sidecar JSON exists but is malformed, missing required fields, or has the wrong shape."""


class UnknownVersion(ProvenanceError):
    """The sidecar declares a ``type`` or ``canonicalization_version`` we do not understand."""


class HashMismatch(ProvenanceError):
    """One or more recomputed content hashes does not match the value in the sidecar.

    This is the tamper-detection signal. The diagnostic should name *which*
    hash diverged so the user can localize the problem.
    """

    def __init__(self, message: str, *, field: str, expected: str, actual: str) -> None:
        super().__init__(message)
        self.field = field
        self.expected = expected
        self.actual = actual


class SignatureMismatch(ProvenanceError):
    """The Ed25519 signature does not validate against the payload + verifying key.

    This is distinct from HashMismatch. SignatureMismatch means *the bytes
    that were signed are not what we now have*; HashMismatch means the bytes
    on disk no longer match the hash that was signed.
    """


class Unsigned(ProvenanceError):
    """A ``.d`` directory was found but no provenance sidecar accompanies it.

    This is informational, not strictly an error. The CLI surfaces it as
    exit code 4 by default; ``--strict`` upgrades it to a failure.
    """


class MissingArtifact(ProvenanceError):
    """The sidecar references an artifact (.d, ground truth) that does not exist on disk."""
