"""End-to-end smoke test for the TimSim provenance subsystem.

Run as::

    python -m imspy_simulation.provenance.test_smoke

This is the inner-loop command. If smoke breaks, nothing else matters.
The script:

  1. Builds a minimal .d directory in a temp dir.
  2. Builds a minimal synthetic_data.db.
  3. Builds a minimal config TOML.
  4. Signs the bundle (hashes everything, writes a sidecar).
  5. Verifies the bundle (must succeed).
  6. Flips a single byte in analysis.tdf_bin.
  7. Verifies again (must FAIL with HashMismatch on the .d hash).
  8. Restores the bundle, then tampers a SQL value.
  9. Verifies again (must FAIL with HashMismatch).
 10. Prints "SMOKE OK" and exits 0.

Any deviation prints a clear "SMOKE FAILED" with the reason and exits non-zero.

This file is a script, not a pytest test, on purpose: it must run without
any test framework so it can be invoked as a single command from CI, from
a docs example, or from a shell pipe.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# We import from the conftest in the test tree because that is the single
# source of truth for the minimal-fixture builders. Sharing it here keeps
# the smoke test and the unit tests honest about what "minimal .d" means.
_HERE = Path(__file__).resolve()
_PACKAGE_ROOT = _HERE.parents[3]  # imspy-simulation/
_TEST_DIR = _PACKAGE_ROOT / "tests"
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))


def _fail(reason: str) -> None:
    print(f"SMOKE FAILED: {reason}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    try:
        from test_provenance.conftest import (  # type: ignore[import-not-found]
            make_minimal_d,
            make_minimal_ground_truth,
            tamper_byte,
            tamper_sql_value,
        )
    except ImportError as e:
        _fail(f"could not import test fixtures: {e}")

    from imspy_simulation.provenance import (
        HashMismatch,
        ProvenanceError,
        sign_simulation_output,
        verify_sidecar,
    )

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)

        # 1-3. Build artifacts.
        d_path = make_minimal_d(td, name="smoke_experiment")
        ground_truth = make_minimal_ground_truth(td)
        config_path = td / "smoke_config.toml"
        config_path.write_bytes(
            b'[experiment]\nexperiment_name = "smoke_experiment"\n'
        )

        # 4. Sign.
        try:
            sidecar_path = sign_simulation_output(
                d_path=d_path,
                ground_truth_path=ground_truth,
                config_path=config_path,
                experiment_name="smoke_experiment",
                simulator_version="smoke-test",
            )
        except ProvenanceError as e:
            _fail(f"sign_simulation_output raised ProvenanceError: {e}")
        except NotImplementedError as e:
            _fail(f"sign_simulation_output is not implemented yet: {e}")

        if not sidecar_path.exists():
            _fail(f"sidecar was not written at {sidecar_path}")

        # 5. Verify clean.
        try:
            result = verify_sidecar(sidecar_path)
        except NotImplementedError as e:
            _fail(f"verify_sidecar is not implemented yet: {e}")
        if not result.overall_ok:
            _fail(
                f"clean verification failed: signature_ok={result.signature_ok}, "
                f"checks={[(c.name, c.ok) for c in result.checks]}"
            )

        # 6. Tamper analysis.tdf_bin (single byte at offset 64, just past header).
        tamper_byte(d_path / "analysis.tdf_bin", 64)

        # 7. Verify dirty — must fail.
        result = verify_sidecar(sidecar_path)
        if result.overall_ok:
            _fail("tampered .tdf_bin still verified — tamper detection broken")
        bin_check = next((c for c in result.checks if c.name == "d_content_hash"), None)
        if bin_check is None or bin_check.ok:
            _fail("d_content_hash check did not flag the tdf_bin tamper")

        # 8. Rebuild a clean .d, re-sign, then tamper a SQL value.
        shutil.rmtree(d_path)
        d_path = make_minimal_d(td, name="smoke_experiment")
        sidecar_path = sign_simulation_output(
            d_path=d_path,
            ground_truth_path=ground_truth,
            config_path=config_path,
            experiment_name="smoke_experiment",
            simulator_version="smoke-test",
        )
        result = verify_sidecar(sidecar_path)
        if not result.overall_ok:
            _fail("rebuilt clean bundle did not verify before SQL tamper")

        tamper_sql_value(
            d_path / "analysis.tdf",
            table="GlobalMetadata",
            set_column="Value",
            set_value="FAKE-INSTRUMENT",
            where_column="Key",
            where_value="InstrumentName",
        )

        # 9. Verify dirty SQL — must fail.
        result = verify_sidecar(sidecar_path)
        if result.overall_ok:
            _fail("tampered SQL value still verified — canonical hash is not catching content")
        sql_check = next((c for c in result.checks if c.name == "d_content_hash"), None)
        if sql_check is None or sql_check.ok:
            _fail("d_content_hash check did not flag the SQL tamper")

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # pragma: no cover - safety net
        traceback.print_exc()
        print("SMOKE FAILED: unexpected exception", file=sys.stderr)
        sys.exit(2)
