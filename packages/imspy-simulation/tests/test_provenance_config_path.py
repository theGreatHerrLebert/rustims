"""provenance_config_path: the mzPROV config_hash must bind a SCIEX template profile by CONTENT."""
import hashlib
import logging
from types import SimpleNamespace

import toml

from imspy_simulation.timsim.simulator import provenance_config_path


def _cfg(tmp_path, body: str) -> str:
    p = tmp_path / "config.toml"
    p.write_text(body)
    return str(p)


def test_passthrough_without_profile(tmp_path):
    src = _cfg(tmp_path, "[experiment]\nsave_path = 'x'\n")
    cfg = SimpleNamespace(sciex_profile=None)
    assert provenance_config_path(cfg, src, tmp_path, "run", logging.getLogger()) == src
    assert not list(tmp_path.glob("*.effective-config.toml"))


def test_effective_config_binds_profile_by_sha256(tmp_path):
    src = _cfg(tmp_path, "[experiment]\nsave_path = 'x'\nsciex_native = true\n[models]\nrt_model = ''\n")
    prof = tmp_path / "k562.json"
    prof.write_bytes(b'{"n_windows": 60, "authored": []}')
    cfg = SimpleNamespace(sciex_profile=str(prof))
    out = provenance_config_path(cfg, src, tmp_path, "run", logging.getLogger())
    assert out == str(tmp_path / "run.effective-config.toml")
    eff = toml.load(out)
    assert eff["sciex_profile"] == str(prof.resolve())
    assert eff["sciex_profile_sha256"] == hashlib.sha256(prof.read_bytes()).hexdigest()
    # sections flattened, source keys preserved
    assert eff["save_path"] == "x" and eff["sciex_native"] is True and eff["rt_model"] == ""


def test_different_profile_content_changes_effective_config(tmp_path):
    src = _cfg(tmp_path, "save_path = 'x'\n")
    prof = tmp_path / "p.json"
    log = logging.getLogger()
    prof.write_bytes(b"A")
    a = open(provenance_config_path(SimpleNamespace(sciex_profile=str(prof)), src, tmp_path, "r", log), "rb").read()
    prof.write_bytes(b"B")  # same path, different content (the buggy-vs-corrected profile case)
    b = open(provenance_config_path(SimpleNamespace(sciex_profile=str(prof)), src, tmp_path, "r", log), "rb").read()
    assert hashlib.sha256(a).hexdigest() != hashlib.sha256(b).hexdigest()


def test_missing_profile_raises(tmp_path):
    src = _cfg(tmp_path, "save_path = 'x'\n")
    import pytest
    with pytest.raises(FileNotFoundError):
        provenance_config_path(SimpleNamespace(sciex_profile=str(tmp_path / "nope.json")), src, tmp_path, "r", logging.getLogger())
