"""One place to say which model produces each predicted property, and to record what was used.

Every predicted property (RT, CCS, charge, fragment intensity) can come from **our** trained PyTorch
model or from a named **Koina** model. That choice was already expressible per job, but each job spelled
it differently and none of them wrote down what they picked. This module unifies the *grammar* and the
*provenance*; the actual loading stays per-property, because each property's Koina integration is
genuinely different and pretending otherwise would be a leaky abstraction.

# The grammar

A model spec is a short string:

- ``None`` / ``"default"``  → our default model for that property (see ``DEFAULTS``).
- ``"koina:<name>"``        → the named model, served via Koina (needs network).
- any other string          → a named local backend (e.g. ``"chronologer"``), passed through.

# Provenance

``resolve`` returns a ``(kind, name)`` pair whose ``name`` is written into the artifact metadata as
``timsim.<property>.model``. A benchmark that compares two runs can then see that one used Chronologer
and the other Prosit, instead of guessing — the same discipline as recording the id hash or the RNG
seed: a result is only reproducible if you know what produced it.
"""

from __future__ import annotations

from typing import Optional, Tuple

#: Our default model per property — what ``None``/``"default"`` resolves to. These are the trained
#: models this project ships as the goto choice; a Koina model is opt-in.
DEFAULTS = {
    "rt": "chronologer",
    "ccs": "deep-ccs",
    "charge": "site-specific",
    "fragments": "prospect-local",
}

_KOINA_PREFIX = "koina:"


def resolve(property: str, spec: Optional[str]) -> Tuple[str, str]:
    """Resolve a model spec for ``property`` into ``(kind, name)``.

    ``kind`` is ``"koina"`` or ``"local"``; ``name`` is the concrete model identifier and is what
    gets recorded as provenance. Raises on an unknown property so a typo fails loudly rather than
    silently picking a default.
    """
    if property not in DEFAULTS:
        raise ValueError(f"unknown predicted property {property!r}; known: {sorted(DEFAULTS)}")
    if spec is None or spec == "default":
        return ("local", DEFAULTS[property])
    if spec.startswith(_KOINA_PREFIX):
        name = spec[len(_KOINA_PREFIX):].strip()
        if not name:
            raise ValueError(f"empty Koina model name in {spec!r} (expected 'koina:<name>')")
        return ("koina", name)
    return ("local", spec)


def provenance(property: str, spec: Optional[str]) -> str:
    """The provenance string for a resolved spec: the local model name, or ``koina:<name>``."""
    kind, name = resolve(property, spec)
    return f"koina:{name}" if kind == "koina" else name
