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
from typing import FrozenSet


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

FRAGMENT_PREDICTOR_CAPABILITIES = {
    "local": _TIMSTOF_COLLISIONAL,
    "prosit": _TIMSTOF_COLLISIONAL,
    "alphapeptdeep": _TIMSTOF_COLLISIONAL,
    "ms2pip": _TIMSTOF_COLLISIONAL,
    "ms2pip_2023": _TIMSTOF_COLLISIONAL,
}


def capability_for(model_key: str | None) -> FragmentPredictorCapability:
    """Capability for a model key. Unknown / direct-Koina names default to the
    timsTOF collisional contract (every shipped model shares it)."""
    return FRAGMENT_PREDICTOR_CAPABILITIES.get(
        (model_key or "local").lower(), _TIMSTOF_COLLISIONAL
    )


def assert_predictor_supports(
    model_key: str | None, activation_method: str, energy_unit: str
) -> FragmentPredictorCapability:
    """Raise if the predictor was not trained for this activation/unit.

    Returns the capability so the caller can use its CE encoding.
    """
    cap = capability_for(model_key)
    if not cap.accepts(activation_method, energy_unit):
        raise ValueError(
            f"intensity model '{model_key or 'local'}' does not support activation "
            f"method '{activation_method}' in unit '{energy_unit}' (it expects one "
            f"of {sorted(cap.supported_methods)} in '{cap.energy_unit}'). The stored "
            f"collision energies are for a different instrument/activation; select a "
            f"model trained for it (e.g. an Astral-appropriate model for Thermo NCE)."
        )
    return cap
