"""``timsim-verify`` command-line interface for the provenance subsystem.

Exit codes (plan §6.3):
    0 — verified
    1 — generic error
    2 — key error (missing or unreadable)
    3 — sidecar error (missing, malformed, unknown version, missing artifact)
    4 — unsigned (only failure if --strict)
    5 — hash mismatch (tamper detected)
    6 — signature mismatch
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from imspy_simulation.provenance.errors import (
    KeyNotFoundError,
    MalformedSidecar,
    MissingArtifact,
    UnknownVersion,
)
from imspy_simulation.provenance.verify import (
    VerificationResult,
    find_sidecar_for,
    verify_sidecar,
)

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_KEY_ERROR = 2
EXIT_SIDECAR_ERROR = 3
EXIT_UNSIGNED = 4
EXIT_HASH_MISMATCH = 5
EXIT_SIGNATURE_MISMATCH = 6


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timsim-verify",
        description=(
            "Verify the cryptographic provenance of a TimSim simulation output. "
            "See SIGNING.md for the conceptual framework and the limitations of "
            "the Phase 0 prototype."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to a sidecar file, an experiment directory, or a .d directory. "
            "The sidecar is discovered automatically."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat 'unsigned' (no sidecar found) as a failure (exit 4 -> 4 instead of warn).",
    )
    parser.add_argument(
        "--print-payload",
        action="store_true",
        help="Print the full sidecar payload as JSON before the verification result.",
    )
    parser.add_argument(
        "--public-key",
        type=Path,
        default=None,
        help=(
            "Override the verifying key to use. Default: use the verifying_key "
            "embedded in the sidecar (suitable for the v0 prototype)."
        ),
    )
    return parser


def _print_header(result: VerificationResult) -> None:
    p = result.payload
    print("TimSim provenance verification")
    print(f"  experiment:        {p.experiment_name}")
    print(f"  simulator:         {p.simulator_name} {p.simulator_version}")
    print(f"  signed at:         {p.timestamp_utc}")
    print(f"  key id:            {p.key_id}")
    print(f"  canonicalization:  {p.canonicalization_version}")
    print()


def _print_checks(result: VerificationResult) -> None:
    width = max((len(c.name) for c in result.checks), default=0)
    for check in result.checks:
        status = "OK      " if check.ok else "MISMATCH"
        marker = " " if check.ok else "*"
        digest_short = check.expected[:24] + "..." if len(check.expected) > 24 else check.expected
        print(f" {marker} {check.name:<{width}}  {status}  ({digest_short})")
        if not check.ok and check.actual:
            actual_short = (
                check.actual[:24] + "..." if len(check.actual) > 24 else check.actual
            )
            print(f"     expected:  {check.expected}")
            print(f"     actual:    {check.actual}")
    sig_status = "OK      " if result.signature_ok else "MISMATCH"
    sig_marker = " " if result.signature_ok else "*"
    print(f" {sig_marker} {'signature':<{width}}  {sig_status}  (ed25519)")
    print()


def _exit_for_failure(result: VerificationResult) -> int:
    """Pick the most specific exit code for a failed verification."""
    if not result.signature_ok:
        return EXIT_SIGNATURE_MISMATCH
    if any(not c.ok for c in result.checks):
        return EXIT_HASH_MISMATCH
    return EXIT_GENERIC


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    sidecar_path = find_sidecar_for(args.path)
    if sidecar_path is None:
        msg = (
            f"timsim-verify: no provenance sidecar found near {args.path}. "
            f"This file or directory is unsigned."
        )
        if args.strict:
            print(msg, file=sys.stderr)
            print("FAILED: --strict and no sidecar present", file=sys.stderr)
        else:
            print(msg)
            print("UNSIGNED")
        return EXIT_UNSIGNED

    try:
        result = verify_sidecar(sidecar_path, public_key_override=args.public_key)
    except KeyNotFoundError as e:
        print(f"timsim-verify: key error: {e}", file=sys.stderr)
        return EXIT_KEY_ERROR
    except (MalformedSidecar, UnknownVersion, MissingArtifact) as e:
        print(f"timsim-verify: sidecar error: {e}", file=sys.stderr)
        return EXIT_SIDECAR_ERROR
    except Exception as e:  # pragma: no cover - safety net
        print(f"timsim-verify: unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_GENERIC

    if args.print_payload:
        print(json.dumps(asdict(result.payload), indent=2, sort_keys=True))
        print()

    _print_header(result)
    _print_checks(result)

    if result.overall_ok:
        print("VERIFIED")
        return EXIT_OK

    print("FAILED")
    return _exit_for_failure(result)


if __name__ == "__main__":
    sys.exit(main())
