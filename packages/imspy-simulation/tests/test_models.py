"""The model-registry grammar and provenance."""
import pytest
from imspy_simulation.timsim.models import resolve, provenance, DEFAULTS


def test_default_resolves_to_our_model():
    for prop, default in DEFAULTS.items():
        assert resolve(prop, None) == ("local", default)
        assert resolve(prop, "default") == ("local", default)
        assert provenance(prop, None) == default


def test_koina_prefix_selects_koina():
    assert resolve("rt", "koina:AlphaPeptDeep_rt_generic") == ("koina", "AlphaPeptDeep_rt_generic")
    assert provenance("rt", "koina:Prosit_2019_irt") == "koina:Prosit_2019_irt"


def test_named_backend_passes_through_as_local():
    assert resolve("rt", "chronologer") == ("local", "chronologer")


def test_unknown_property_and_empty_koina_name_raise():
    with pytest.raises(ValueError):
        resolve("smell", None)
    with pytest.raises(ValueError):
        resolve("rt", "koina:")
