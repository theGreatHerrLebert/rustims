"""``timsim-keys`` command-line interface for the trust-key registry.

Subcommands:

    show              Print the local signing key id and where it lives.
    export            Print the local public key in PEM (or write to a file).
    trust PATH        Add a key to the trusted-keys registry. PATH may be:
                        - a public key PEM file
                        - a sidecar JSON (the embedded verifying key is used)
                        - a directory containing a *.provenance.json
                      Requires --comment so the user has to think about why
                      they are granting trust.
    list              List all keys currently in the trusted-keys registry.
    untrust KEY_ID    Remove a key from the trusted-keys registry by id.

Exit codes:
    0 — success
    1 — generic error
    2 — key error (missing or unreadable)
    3 — registry error (malformed)
    4 — key not found in registry (untrust on a missing id)

This CLI is the workflow that converts the registry from "a file you
have to edit by hand" into "something a user can actually use".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from imspy_simulation.provenance.errors import (
    KeyNotFoundError,
    MalformedSidecar,
    ProvenanceError,
)
from imspy_simulation.provenance.keys import (
    default_key_dir,
    load_or_create_keypair,
)
from imspy_simulation.provenance.trust import (
    TrustedKeyRegistry,
    default_registry_path,
    trusted_key_from_pem_file,
    trusted_key_from_public_key,
    trusted_key_from_sidecar_file,
)

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_KEY_ERROR = 2
EXIT_REGISTRY_ERROR = 3
EXIT_KEY_NOT_FOUND = 4


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timsim-keys",
        description=(
            "Manage the local TimSim signing key and the trusted-keys "
            "registry used by 'timsim-verify --require-trusted'."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser(
        "show",
        help="Print the local signing key id and where it lives.",
    )
    p_show.add_argument(
        "--key-dir",
        type=Path,
        default=None,
        help="Override the key directory (default: ~/.config/timsim/keys/).",
    )

    p_export = sub.add_parser(
        "export",
        help="Print the local public key in PEM, or write it to --to PATH.",
    )
    p_export.add_argument(
        "--key-dir",
        type=Path,
        default=None,
        help="Override the key directory (default: ~/.config/timsim/keys/).",
    )
    p_export.add_argument(
        "--to",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the PEM to this file instead of stdout.",
    )

    p_trust = sub.add_parser(
        "trust",
        help=(
            "Add a key to the trusted-keys registry. SOURCE may be a "
            "public key PEM file, a sidecar JSON file, or a directory "
            "containing a sidecar."
        ),
    )
    p_trust.add_argument(
        "source",
        type=Path,
        metavar="SOURCE",
        help="Path to a public key PEM, sidecar JSON, or experiment directory.",
    )
    p_trust.add_argument(
        "--comment",
        type=str,
        required=True,
        help=(
            "Required note explaining WHY this key is being trusted. "
            "Forces the user to pause and think before granting trust."
        ),
    )
    p_trust.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Override the registry path (default: ~/.config/timsim/trusted_keys.json).",
    )

    p_list = sub.add_parser(
        "list",
        help="List all keys currently in the trusted-keys registry.",
    )
    p_list.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Override the registry path (default: ~/.config/timsim/trusted_keys.json).",
    )

    p_untrust = sub.add_parser(
        "untrust",
        help="Remove a key from the trusted-keys registry by key id.",
    )
    p_untrust.add_argument("key_id", type=str, metavar="KEY_ID")
    p_untrust.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Override the registry path (default: ~/.config/timsim/trusted_keys.json).",
    )

    return parser


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_show(args) -> int:
    keypair = load_or_create_keypair(args.key_dir)
    key_dir = Path(args.key_dir) if args.key_dir is not None else default_key_dir()
    print(f"key id:    {keypair.key_id}")
    print(f"key dir:   {key_dir}")
    print(f"private:   {key_dir / 'signing_key.pem'}")
    print(f"public:    {key_dir / 'verifying_key.pem'}")
    print()
    print(
        "This is a software-rooted key. It is not a substitute for "
        "instrument attestation. See SIGNING.md §9."
    )
    return EXIT_OK


def _cmd_export(args) -> int:
    from cryptography.hazmat.primitives import serialization

    keypair = load_or_create_keypair(args.key_dir)
    pem = keypair.public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if args.to is not None:
        Path(args.to).write_bytes(pem)
        print(f"public key written to {args.to}")
    else:
        sys.stdout.write(pem.decode("ascii"))
    return EXIT_OK


def _cmd_trust(args) -> int:
    source = Path(args.source)
    if not source.exists():
        print(
            f"timsim-keys: trust source does not exist: {source}",
            file=sys.stderr,
        )
        return EXIT_KEY_ERROR

    # Decide source type by inspection.
    if source.is_dir():
        # Directory: look for a *.provenance.json inside.
        candidates = sorted(source.glob("*.provenance.json"))
        if not candidates:
            print(
                f"timsim-keys: no *.provenance.json found in directory {source}",
                file=sys.stderr,
            )
            return EXIT_KEY_ERROR
        try:
            entry = trusted_key_from_sidecar_file(candidates[0], comment=args.comment)
        except (MalformedSidecar, KeyNotFoundError, ProvenanceError) as e:
            print(f"timsim-keys: {e}", file=sys.stderr)
            return EXIT_KEY_ERROR
    elif source.suffix == ".json" and ".provenance" in source.name:
        try:
            entry = trusted_key_from_sidecar_file(source, comment=args.comment)
        except (MalformedSidecar, KeyNotFoundError, ProvenanceError) as e:
            print(f"timsim-keys: {e}", file=sys.stderr)
            return EXIT_KEY_ERROR
    else:
        # Assume it's a PEM file.
        try:
            entry = trusted_key_from_pem_file(source, comment=args.comment)
        except (KeyNotFoundError, ProvenanceError) as e:
            print(f"timsim-keys: {e}", file=sys.stderr)
            return EXIT_KEY_ERROR

    try:
        registry = TrustedKeyRegistry.load(args.registry)
    except MalformedSidecar as e:
        print(f"timsim-keys: registry error: {e}", file=sys.stderr)
        return EXIT_REGISTRY_ERROR

    try:
        registry.add(entry)
    except ProvenanceError as e:
        print(f"timsim-keys: refused to add: {e}", file=sys.stderr)
        return EXIT_KEY_ERROR

    registry.save()
    print(f"trusted: {entry.key_id}")
    print(f"  comment:  {entry.comment}")
    print(f"  added at: {entry.added_at}")
    print(f"  registry: {registry.path}")
    return EXIT_OK


def _cmd_list(args) -> int:
    try:
        registry = TrustedKeyRegistry.load(args.registry)
    except MalformedSidecar as e:
        print(f"timsim-keys: registry error: {e}", file=sys.stderr)
        return EXIT_REGISTRY_ERROR

    if len(registry) == 0:
        print(f"trusted-keys registry at {registry.path} is empty.")
        print("Add a key with: timsim-keys trust SOURCE --comment '...'")
        return EXIT_OK

    print(f"trusted-keys registry: {registry.path}")
    print(f"  {len(registry)} key(s)")
    print()
    for entry in registry:
        print(f"  {entry.key_id}")
        print(f"    comment:  {entry.comment}")
        print(f"    added at: {entry.added_at}")
        print()
    return EXIT_OK


def _cmd_untrust(args) -> int:
    try:
        registry = TrustedKeyRegistry.load(args.registry)
    except MalformedSidecar as e:
        print(f"timsim-keys: registry error: {e}", file=sys.stderr)
        return EXIT_REGISTRY_ERROR

    if not registry.remove(args.key_id):
        print(
            f"timsim-keys: no key with id {args.key_id!r} in registry "
            f"at {registry.path}",
            file=sys.stderr,
        )
        return EXIT_KEY_NOT_FOUND

    registry.save()
    print(f"untrusted: {args.key_id}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_DISPATCH = {
    "show": _cmd_show,
    "export": _cmd_export,
    "trust": _cmd_trust,
    "list": _cmd_list,
    "untrust": _cmd_untrust,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH[args.cmd]
    try:
        return handler(args)
    except Exception as e:  # pragma: no cover - safety net
        print(f"timsim-keys: unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_GENERIC


if __name__ == "__main__":
    sys.exit(main())
