"""Delta-mass peptide -> UNIMOD conversion.

Sage and the Chronologer DB write peptides with delta-mass brackets
(``MC[+57.0216]PEPTIDE``); the tokenizer needs UNIMOD form
(``MC[UNIMOD:4]PEPTIDE``). The mass->UNIMOD step is residue-specificity aware
(the residue immediately before the bracket), with a small hardcoded table for
the common search modifications and the quantized sagepy table as the general
case.

This logic is **vendored verbatim** from ``sagepy_rescore`` (``sage_loader.
_parse_sage_peptide`` + ``pipeline._mass_to_unimod`` and its tables) so this
package does not import ``sagepy_rescore.pipeline`` — a heavy module whose
unrelated ``sagepy.core.fdr`` imports are version-fragile across sagepy
releases. The conversion itself is unchanged; it needs only ``sagepy.core.unimod``
and ``sagepy_connector.py_unimod``.
"""
from __future__ import annotations

import re

from sagepy.core.unimod import unimod_to_mass
from sagepy_connector import py_unimod

# Sage / Chronologer write only positive delta masses, e.g. "[+57.0216]".
_BRACKET_MOD = re.compile(r"\[\+([0-9.]+)\]")


def _build_mass_to_unimod() -> dict[float, str]:
    out: dict[float, str] = {}
    for unimod_id, mass in unimod_to_mass().items():
        out[round(float(mass), 4)] = unimod_id
    return out


_MASS_TO_UNIMOD = _build_mass_to_unimod()

# Residue-specificity-disambiguated entries for the common search modifications.
_COMMON_SAGE_MASS_TO_UNIMOD = {
    ("C", py_unimod.quanzied_mass(57.0216)): "[UNIMOD:4]",
    ("M", py_unimod.quanzied_mass(15.9949)): "[UNIMOD:35]",
    ("[", py_unimod.quanzied_mass(42.0)): "[UNIMOD:1]",
}


def _mass_to_unimod(value, specificity: str | None = None):
    """Resolve a delta mass (+ optional residue specificity) to a UNIMOD token."""
    if isinstance(value, list):
        return [_mass_to_unimod(v, specificity=specificity) for v in value]
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return f"[UNIMOD:{value}]"

    quantized = py_unimod.quanzied_mass(float(value))
    common = _COMMON_SAGE_MASS_TO_UNIMOD.get((specificity, quantized))
    if common is not None:
        return common

    candidates = py_unimod.quantized_mass_to_unimod_candidates().get(quantized, [])
    if candidates:
        return candidates[0]

    key = round(float(value), 4)
    if key in _MASS_TO_UNIMOD:
        return _MASS_TO_UNIMOD[key]
    raise ValueError(
        f"Mass {value!r} for specificity {specificity!r} has no UNIMOD entry "
        "in sagepy's quantized table."
    )


def parse_delta_mass_peptide(peptide: str) -> tuple[str, str]:
    """Convert a delta-mass peptide string to ``(unmodified, UNIMOD-modified)``.

    Example: ``IIPGFMC[+57.0216]QGGDFTR`` ->
    ``("IIPGFMCQGGDFTR", "IIPGFMC[UNIMOD:4]QGGDFTR")``. A delta mass with no
    UNIMOD match is left as the raw ``[+mass]`` bracket (the caller skips it).
    """
    out_parts: list[str] = []
    last_end = 0
    for m in _BRACKET_MOD.finditer(peptide):
        start = m.start()
        out_parts.append(peptide[last_end:start])
        specificity = "[" if start == 0 else peptide[start - 1]
        try:
            unimod = _mass_to_unimod(float(m.group(1)), specificity=specificity)
        except ValueError:
            unimod = m.group(0)  # unknown mass -> leave the raw bracket
        out_parts.append(unimod)
        last_end = m.end()
    out_parts.append(peptide[last_end:])
    modified = "".join(out_parts).replace("-", "")
    unmodified = _BRACKET_MOD.sub("", peptide).replace("-", "")
    return unmodified, modified
