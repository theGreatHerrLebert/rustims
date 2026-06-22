"""Regression guard for the Koina fragment-intensity decode order.

``_koina_result_to_prosit_vectors`` builds a ``(29, 2, 3)`` ``(ordinal, ion-type,
charge)`` tensor from Koina's annotated output, then serialises it to the 174-dim
Prosit vector the renderer decodes with ``reshape_prosit_array`` (charge-major).

It must serialise with ``flatten_prosit_array`` (charge-major), NOT numpy's default
``tensor.flatten()`` (C-order / ordinal-major): the latter maps every intensity onto
the WRONG fragment, scrambling every Koina-predicted spectrum while staying
self-consistent (so a self-referential library search still "works", but the data is
unsearchable by any tool predicting real spectra).
"""

import numpy as np
import pandas as pd

from imspy_simulation.timsim.jobs.simulate_fragment_intensities import (
    _koina_result_to_prosit_vectors,
)


def test_koina_decode_is_charge_major():
    # one peptide (row index 0), one dominant fragment b8 at charge 1.
    koina_result = pd.DataFrame(
        {"annotation": [b"b8+1"], "intensities": [1.0]}, index=[0]
    )
    original_data = pd.DataFrame({"sequence": ["PEPTIDEKR"]})

    vec = _koina_result_to_prosit_vectors(koina_result, original_data)

    assert vec.shape == (1, 174)
    # Charge-major layout: [Y1..29 @z1 = 0..28] [B1..29 @z1 = 29..57] [@z2] [@z3].
    # So b8 (ordinal 8) at charge 1 -> 29 + (8 - 1) = 36.
    # (The C-order bug would instead place it at 7*6 + 1*3 + 0 = 45.)
    assert vec[0].argmax() == 36
    assert vec[0, 36] == 1.0


def test_koina_decode_multiple_fragments_land_on_their_indices():
    koina_result = pd.DataFrame(
        {"annotation": [b"y3+2", b"b2+1"], "intensities": [1.0, 0.5]}, index=[0, 0]
    )
    original_data = pd.DataFrame({"sequence": ["PEPTIDEKR"]})

    vec = _koina_result_to_prosit_vectors(koina_result, original_data)

    # y3 @z2 -> Y block of charge 2: 58 + (3 - 1) = 60 (base peak, normalised to 1.0)
    assert vec[0, 60] == 1.0
    # b2 @z1 -> B block of charge 1: 29 + (2 - 1) = 30
    assert abs(vec[0, 30] - 0.5) < 1e-6
