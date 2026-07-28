"""Regression tests for Bruker SDK binary selection (issue #442).

open-tims-bruker-bridge ships the vendor binaries for every platform and orders
``get_so_paths()`` by ``platform.architecture()``. That call shells out to the
``file`` command; in slim Docker images ``file`` is absent, so architecture is
reported as ``('64bit', '')`` and the *Windows* ``timsdata.dll`` comes back
first. Passing it to the Rust reader fails with "invalid ELF header".
"""

import pytest

from imspy_core.timstof import data as data_mod


@pytest.fixture
def so_dir(tmp_path):
    """A fake open-tims-bruker-bridge install containing every vendor binary."""
    (tmp_path / "libtimsdata.so").touch()
    for sub in ("win32", "win64"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "timsdata.dll").touch()
    return tmp_path


def _patch(monkeypatch, so_dir, order, system, machine):
    paths = [str(so_dir / rel) for rel in order]
    monkeypatch.setattr(data_mod.obb, "get_so_paths", lambda: paths)
    monkeypatch.setattr(data_mod.platform, "system", lambda: system)
    monkeypatch.setattr(data_mod.platform, "machine", lambda: machine)


# The order obb returns when `file` is missing: Windows DLL first.
BROKEN_ORDER = ["win64/timsdata.dll", "libtimsdata.so", "win32/timsdata.dll"]
# The order obb returns on a host with `file` installed.
HEALTHY_ORDER = ["libtimsdata.so", "win64/timsdata.dll", "win32/timsdata.dll"]


@pytest.mark.parametrize("order", [BROKEN_ORDER, HEALTHY_ORDER])
def test_linux_never_selects_windows_dll(monkeypatch, so_dir, order):
    _patch(monkeypatch, so_dir, order, "Linux", "x86_64")
    candidates = data_mod.sdk_candidate_paths()
    assert candidates == [str(so_dir / "libtimsdata.so")]


def test_windows_selects_dlls_only(monkeypatch, so_dir):
    _patch(monkeypatch, so_dir, BROKEN_ORDER, "Windows", "AMD64")
    candidates = data_mod.sdk_candidate_paths()
    assert candidates == [str(so_dir / "win64" / "timsdata.dll"),
                          str(so_dir / "win32" / "timsdata.dll")]


@pytest.mark.parametrize("system,machine", [("Darwin", "x86_64"),
                                            ("Darwin", "arm64"),
                                            ("Linux", "aarch64")])
def test_no_candidates_where_sdk_is_unusable(monkeypatch, so_dir, system, machine):
    """macOS and non-amd64 have no loadable Bruker binary: stay on NO_SDK."""
    _patch(monkeypatch, so_dir, BROKEN_ORDER, system, machine)
    assert data_mod.sdk_candidate_paths() == []


def test_missing_bridge_does_not_raise(monkeypatch, so_dir):
    def boom():
        raise RuntimeError("open-tims-bruker-bridge is broken")

    monkeypatch.setattr(data_mod.obb, "get_so_paths", boom)
    monkeypatch.setattr(data_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(data_mod.platform, "machine", lambda: "x86_64")
    assert data_mod.sdk_candidate_paths() == []


def test_nonexistent_paths_are_dropped(monkeypatch, so_dir, tmp_path):
    monkeypatch.setattr(data_mod.obb, "get_so_paths",
                        lambda: [str(tmp_path / "gone" / "libtimsdata.so")])
    monkeypatch.setattr(data_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(data_mod.platform, "machine", lambda: "x86_64")
    assert data_mod.sdk_candidate_paths() == []
