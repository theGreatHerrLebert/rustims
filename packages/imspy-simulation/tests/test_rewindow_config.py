"""Unit tests for the Tier-2 re-window config plumbing (no .raw / connector needed for
the off-paths). Covers dict-vs-object config, None/0/negative = no-op, bad value raises."""
import logging
import types

import pytest

from imspy_simulation.timsim.simulator import (
    maybe_rewindow_thermo_template,
    thermo_template_path,
    _set_template_path,
)

LOG = logging.getLogger("test")


def _obj(**kw):
    return types.SimpleNamespace(**kw)


@pytest.mark.parametrize("mk", [dict, lambda **kw: _obj(**kw)], ids=["dict", "object"])
@pytest.mark.parametrize("width", [None, 0, 0.0, -1.0])
def test_rewindow_noop_when_off(mk, width):
    """None / 0 / negative width -> no-op: template path unchanged, no connector call."""
    cfg = mk(template_path="/orig/template.raw", dia_rewindow_isolation_width=width)
    maybe_rewindow_thermo_template(cfg, "/tmp/x", "exp", LOG)
    assert thermo_template_path(cfg) == "/orig/template.raw"


@pytest.mark.parametrize("mk", [dict, lambda **kw: _obj(**kw)], ids=["dict", "object"])
def test_rewindow_missing_key_noop(mk):
    cfg = mk(template_path="/orig/template.raw")  # option absent entirely
    maybe_rewindow_thermo_template(cfg, "/tmp/x", "exp", LOG)
    assert thermo_template_path(cfg) == "/orig/template.raw"


@pytest.mark.parametrize("mk", [dict, lambda **kw: _obj(**kw)], ids=["dict", "object"])
def test_rewindow_bad_value_raises(mk):
    cfg = mk(template_path="/orig/template.raw", dia_rewindow_isolation_width="wide")
    with pytest.raises(ValueError):
        maybe_rewindow_thermo_template(cfg, "/tmp/x", "exp", LOG)


@pytest.mark.parametrize("mk", [dict, lambda **kw: _obj(**kw)], ids=["dict", "object"])
def test_set_template_path_both_keys(mk):
    """_set_template_path redirects what thermo_template_path returns (astral takes precedence)."""
    cfg = mk(template_path="/orig.raw", astral_template_path="/orig_astral.raw")
    _set_template_path(cfg, "/new.raw")
    assert thermo_template_path(cfg) == "/new.raw"
