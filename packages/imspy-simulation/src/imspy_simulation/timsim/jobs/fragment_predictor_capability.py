"""P5d: fragment-intensity predictor capability declarations.

A fragment-intensity model is trained for a specific activation regime and
consumes collision energy in a specific encoding. Every model TimSim ships
(local PROSPECT fine-tune, Prosit timsTOF, AlphaPeptDeep, ms2pip) is a
collisional (CID/HCD) model that takes an absolute eV collision energy and
encodes it for the network by dividing by 100.

Making that contract explicit lets the fragment job (a) own the eV->model-input
encoding instead of a magic ``/100`` literal, and (b) REFUSE an activation
condition the model was not trained for — e.g. a Thermo normalized collision
energy (NCE) fed to a timsTOF eV model would be silently wrong. The guard is a
no-op for the Bruker eV path today; it makes the P6 Thermo path fail loudly
unless an Astral-appropriate model is selected.

(Conceptually this belongs with the predictors in imspy-predictors; it lives
here for now as the sim-side contract and mirrors what those models expect.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class FragmentPredictorCapability:
    """What activation an intensity predictor supports + how it encodes CE."""

    supported_methods: FrozenSet[str]  # e.g. {"cid", "hcd"}
    energy_unit: str                   # e.g. "ev"
    ce_encoding: str                   # e.g. "normalized_div100"

    def accepts(self, activation_method: str, energy_unit: str) -> bool:
        return (
            activation_method.lower() in self.supported_methods
            and energy_unit.lower() == self.energy_unit
        )

    def encode_collision_energy(self, ce_values):
        """Map an absolute collision energy (in ``energy_unit``) to the model's
        network input. Owns the encoding the job used to hardcode as ``/100``."""
        if self.ce_encoding == "normalized_div100":
            return ce_values / 100.0
        raise ValueError(
            f"unsupported collision-energy encoding '{self.ce_encoding}'"
        )


# All currently-shipped intensity models are collisional, eV, /100-normalized.
_TIMSTOF_COLLISIONAL = FragmentPredictorCapability(
    supported_methods=frozenset({"cid", "hcd"}),
    energy_unit="ev",
    ce_encoding="normalized_div100",
)

# Declared per model — registered under BOTH the short key and the full Koina
# alias, so a model passed by its full Koina name is recognised too. An
# UNDECLARED model has no entry: its activation contract is unknown, so the
# guard rejects it rather than silently assuming the timsTOF eV/100 contract
# (which would mis-encode e.g. an Orbitrap NCE model).
FRAGMENT_PREDICTOR_CAPABILITIES = {
    "local": _TIMSTOF_COLLISIONAL,
    "prosit": _TIMSTOF_COLLISIONAL,
    "prosit_2023_intensity_timstof": _TIMSTOF_COLLISIONAL,
    "alphapeptdeep": _TIMSTOF_COLLISIONAL,
    "alphapeptdeep_ms2_generic": _TIMSTOF_COLLISIONAL,
    "ms2pip": _TIMSTOF_COLLISIONAL,
    "ms2pip_timstof2024": _TIMSTOF_COLLISIONAL,
    "ms2pip_2023": _TIMSTOF_COLLISIONAL,
    "ms2pip_timstof2023": _TIMSTOF_COLLISIONAL,
}


def capability_for(model_key: str | None) -> Optional[FragmentPredictorCapability]:
    """Declared capability for a model key (short key or full Koina alias), or
    `None` if the model is undeclared — there is intentionally NO default, so an
    unknown model's CE encoding/activation are never silently assumed."""
    return FRAGMENT_PREDICTOR_CAPABILITIES.get((model_key or "local").lower())


def require_capability(model_key: str | None) -> FragmentPredictorCapability:
    """Capability for a model, raising if it is undeclared."""
    cap = capability_for(model_key)
    if cap is None:
        raise ValueError(
            f"intensity model '{model_key}' has no declared activation capability. "
            f"Register it in FRAGMENT_PREDICTOR_CAPABILITIES with its activation "
            f"method(s), energy unit, and CE encoding so its collision energy is "
            f"not silently mis-encoded (known models: "
            f"{sorted(FRAGMENT_PREDICTOR_CAPABILITIES)})."
        )
    return cap


def assert_predictor_supports(
    model_key: str | None, activation_method: str, energy_unit: str
) -> FragmentPredictorCapability:
    """Raise if the predictor is undeclared, or was not trained for this
    activation/unit. Returns the capability so the caller can use its CE encoding.
    """
    cap = require_capability(model_key)
    if not cap.accepts(activation_method, energy_unit):
        raise ValueError(
            f"intensity model '{model_key or 'local'}' does not support activation "
            f"method '{activation_method}' in unit '{energy_unit}' (it expects one "
            f"of {sorted(cap.supported_methods)} in '{cap.energy_unit}'). The stored "
            f"collision energies are for a different instrument/activation; select a "
            f"model trained for it (e.g. an Astral-appropriate model for Thermo NCE)."
        )
    return cap
