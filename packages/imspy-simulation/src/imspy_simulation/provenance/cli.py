"""``timsim-verify`` command-line interface for the provenance subsystem.

Exit codes:
    0 — verified
    1 — generic error
    2 — key error (missing or unreadable)
    3 — sidecar error (missing, malformed, unknown version, missing artifact,
        SQLite not quiescent)
    4 — unsigned (only failure if --strict)
    5 — hash mismatch (tamper detected)
    6 — signature mismatch
    7 — trust check failed (--expected-key-id mismatch, or key not in
        trusted-keys registry under --require-trusted)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from imspy_simulation.provenance.errors import (
    KeyNotFoundError,
    MalformedKey,
    MalformedSidecar,
    MissingArtifact,
    SqliteNotQuiescent,
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
EXIT_KEY_NOT_TRUSTED = 7


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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to the config TOML to verify against. Default: look for "
            "{sidecar-basename}.config.toml next to the sidecar (TimSim copies "
            "this file there at sign time). If neither resolves, the config_hash "
            "check is reported as UNCHECKED."
        ),
    )
    parser.add_argument(
        "--expected-key-id",
        type=str,
        default=None,
        metavar="KEY_ID",
        help=(
            "Pin verification to a specific signing key id (e.g. "
            "'timsim-local-q4ffrs376n5yefou'). If the sidecar's key_id "
            "differs, verification fails with exit code 7. This is the "
            "smallest way to convert integrity into identity: assert out "
            "of band that you expect THIS signer."
        ),
    )
    parser.add_argument(
        "--require-trusted",
        action="store_true",
        help=(
            "Require the sidecar's signing key to be present in the "
            "trusted-keys registry (~/.config/timsim/trusted_keys.json). "
            "Use 'timsim-keys trust ...' to add a key to the registry. "
            "Without this flag, the embedded verifying key proves only "
            "integrity, not identity."
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
    label_for = {
        "ok": "OK       ",
        "mismatch": "MISMATCH ",
        "unchecked": "UNCHECKED",
    }
    for check in result.checks:
        status_label = label_for.get(check.status, check.status.upper().ljust(9))
        marker = " " if check.status == "ok" else "*"
        digest_short = check.expected[:24] + "..." if len(check.expected) > 24 else check.expected
        print(f" {marker} {check.name:<{width}}  {status_label}  ({digest_short})")
        if check.status == "mismatch" and check.actual:
            print(f"     expected:  {check.expected}")
            print(f"     actual:    {check.actual}")
        elif check.status == "unchecked" and check.detail:
            print(f"     note:      {check.detail}")
    sig_status = "OK       " if result.signature_ok else "MISMATCH "
    sig_marker = " " if result.signature_ok else "*"
    print(f" {sig_marker} {'signature':<{width}}  {sig_status}  (ed25519)")

    # Trust line — only printed if a trust check was requested or failed.
    trust = result.trust
    if trust.was_requested or trust.status not in ("ok", "not_requested"):
        trust_label = {
            "ok": "TRUSTED  ",
            "id_mismatch": "ID MISMTCH",
            "not_in_registry": "NOT TRUSTD",
            "registry_pem_mismatch": "PEM FORGED",
            "not_requested": "(not pin) ",
        }.get(trust.status, trust.status.upper().ljust(10))
        marker = " " if trust.ok else "*"
        print(f" {marker} {'trust':<{width}}  {trust_label} ({trust.actual_key_id})")
        if not trust.ok and trust.detail:
            print(f"     note:      {trust.detail}")
    print()


def _exit_for_failure(result: VerificationResult) -> int:
    """Pick the most specific exit code for a failed verification.

    Order of precedence (most specific first):
        - signature mismatch  -> 6
        - hash mismatch       -> 5
        - hash unchecked      -> 5 (we could not verify, treat as failure)
        - trust failure       -> 7
        - other               -> 1
    """
    if not result.signature_ok:
        return EXIT_SIGNATURE_MISMATCH
    if any(c.status == "mismatch" for c in result.checks):
        return EXIT_HASH_MISMATCH
    if any(c.status == "unchecked" for c in result.checks):
        return EXIT_HASH_MISMATCH
    if not result.trust.ok:
        return EXIT_KEY_NOT_TRUSTED
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
            return EXIT_UNSIGNED
        # Non-strict: unsigned is informational, NOT a failure. Print
        # the message and return EXIT_OK so shell/CI users do not get a
        # spurious failure on legitimately unsigned input. (Reviewer #2)
        print(msg)
        print("UNSIGNED")
        return EXIT_OK

    try:
        result = verify_sidecar(
            sidecar_path,
            public_key_override=args.public_key,
            config_path_override=args.config,
            expected_key_id=args.expected_key_id,
            require_trusted=args.require_trusted,
        )
    except (KeyNotFoundError, MalformedKey) as e:
        print(f"timsim-verify: key error: {e}", file=sys.stderr)
        return EXIT_KEY_ERROR
    except SqliteNotQuiescent as e:
        print(f"timsim-verify: artifact error: {e}", file=sys.stderr)
        return EXIT_SIDECAR_ERROR
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
