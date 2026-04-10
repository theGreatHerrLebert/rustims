"""imspy_simulation.provenance — moved to mzprov.

The cryptographic provenance signing implementation that used to live in
this module has moved to a standalone package: ``mzprov``. The canonical
home for this code is the mzprov repository.

This file is a thin re-export shim that preserves the existing
``imspy_simulation.provenance`` import surface, so existing TimSim users
and integration code (notably ``imspy_simulation.timsim.simulator``)
continue to work unchanged.

New code should import from ``mzprov`` directly:

    from mzprov import sign_simulation_output, sign_mzml_output, verify_sidecar

For the migration rationale and plan, see ``MIGRATION_PLAN.md`` in the
rustims branch.
"""
from mzprov import (
    HashMismatch,
    KeyNotFoundError,
    MalformedSidecar,
    MissingArtifact,
    ProvenanceError,
    SignatureMismatch,
    Unsigned,
    UnknownVersion,
    canonicalize_d,
    canonicalize_mzml,
    canonicalize_sqlite,
    sign_mzml_output,
    sign_simulation_output,
    verify_sidecar,
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
    "canonicalize_mzml",
    "canonicalize_sqlite",
    "sign_mzml_output",
    "sign_simulation_output",
    "verify_sidecar",
]
