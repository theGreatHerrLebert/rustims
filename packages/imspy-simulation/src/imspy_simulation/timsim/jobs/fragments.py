"""``timsim-fragments`` — predicted fragment-ion intensities, per precursor. MEASUREMENT.

The fragment half of the feature space, as a standalone artifact rather than something buried in the
render: a predicted spectral library. It is a *measurement* (the intensities depend on the collision
energy, an instrument setting), but emitting it as its own table decouples "what fragments, and how
strongly" from "how a given acquisition assembles them into a .d".

# The indexing is the dangerous part, so it is not reasoned about — it is pinned

The model emits a Prosit-layout ``(29, 2, 3)`` tensor (fragment position × ion type × charge). One
transposed axis and every intensity pattern is silently, catastrophically wrong.

There are TWO different flat-174 serialisations in the codebase — ``flatten_prosit_array`` is
charge-major (``[y_c1(29), b_c1(29), y_c2(29)…]``) while a training-target helper is ordinal-major —
so a "flatten then decode a slot" approach is a trap: pick the wrong pair and every spectrum is wrong.
This module therefore does **not** flatten at all. It decodes the ``(29,2,3)`` tensor **directly**,
which is unambiguous, and takes the one axis fact it needs — that axis-2 index 0 is a *y* ion and
index 1 is a *b* ion — from ``flatten_prosit_array``'s own source (``array[:, 0, c]`` is y,
``array[:, 1, c]`` is b), not from reasoning. ``test_fragment_decode.py`` pins that fact against that
function, so an axis mistake fails a test rather than corrupting a benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Authoritative from flatten_prosit_array's source: axis-2 index 0 is a y ion, index 1 is a b ion.
# Pinned by test_fragment_decode.py against that function.
_AXIS2_ION = {0: "y", 1: "b"}


def decode_tensor(pred_3d, floor: float):
    """Yield ``(ion_type, ordinal, charge, intensity)`` for the present, above-floor fragments of one
    precursor's ``(29, 2, 3)`` prediction. Prosit marks structurally-absent slots with -1, so the
    ``> floor`` test drops both those and the sub-floor peaks.
    """
    for k in range(pred_3d.shape[0]):          # position -> ordinal
        for t in range(pred_3d.shape[1]):      # 0 = y, 1 = b
            for c in range(pred_3d.shape[2]):  # charge - 1
                v = float(pred_3d[k, t, c])
                if v > floor:
                    yield _AXIS2_ION[t], k + 1, c + 1, v


def predict_tensors(sequences, charges, collision_energies, model: Optional[str] = None):
    """Predict per-precursor ``(29,2,3)`` intensity tensors with the resolved intensity model.
    Returns ``(array[n,29,2,3], provenance)``."""
    from imspy_simulation.timsim.models import resolve

    kind, name = resolve("fragments", model)
    if kind == "koina":
        raise NotImplementedError(
            f"fragment intensities via Koina ({name!r}) are not wired yet; use the default model."
        )

    from imspy_predictors.intensity.predictors import DeepPeptideIntensityPredictor

    predictor = DeepPeptideIntensityPredictor()
    pred = predictor.predict_intensities(
        sequences=list(sequences),
        charges=[int(c) for c in charges],
        collision_energies=[float(e) for e in collision_energies],
    )
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim != 4 or pred.shape[1:] != (29, 2, 3):
        raise ValueError(f"expected model output (n, 29, 2, 3), got {pred.shape}")
    return pred, name


def predict_fragments(
    precursors: pd.DataFrame,
    collision_energy: float,
    floor: float = 1e-3,
    model: Optional[str] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Return ``(fragment_intensities frame, provenance)``.

    ``precursors`` must have ``precursor_id, sequence, charge`` (the sequence being the modform's
    [UNIMOD]-annotated sequence, so a modified peptide fragments as modified). One row per fragment
    with intensity above ``floor``, normalised to the peptide's base peak; structurally-absent slots
    (Prosit marks them -1) are dropped.
    """
    need = {"precursor_id", "sequence", "charge"}
    missing = need - set(precursors.columns)
    if missing:
        raise ValueError(f"precursors is missing columns {sorted(missing)}")

    # Positional isomers with the same annotated sequence + charge predict identically; dedup.
    keys = precursors[["sequence", "charge"]].drop_duplicates().reset_index(drop=True)
    if verbose:
        print("  timsim-fragments")
        print(f"    precursors        : {len(precursors):,}")
        print(f"    distinct (seq,z)  : {len(keys):,}")
        print(f"    collision energy  : {collision_energy}")

    tensors, prov = predict_tensors(
        keys["sequence"], keys["charge"], [collision_energy] * len(keys), model=model
    )

    # Decode each (29,2,3) tensor directly — no flatten, no slot arithmetic.
    rows = {"sequence": [], "charge": [], "ion_type": [], "ordinal": [], "frag_charge": [], "intensity": []}
    for i in range(len(keys)):
        for it, ordinal, fc, v in decode_tensor(tensors[i], floor):
            rows["sequence"].append(keys["sequence"].iloc[i])
            rows["charge"].append(int(keys["charge"].iloc[i]))
            rows["ion_type"].append(it)
            rows["ordinal"].append(ordinal)
            rows["frag_charge"].append(fc)
            rows["intensity"].append(v)
    decoded = pd.DataFrame(rows)

    out = precursors[["precursor_id", "sequence", "charge"]].merge(
        decoded, on=["sequence", "charge"], how="inner"
    )
    out = out[["precursor_id", "ion_type", "ordinal", "frag_charge", "intensity"]]
    if verbose:
        print(f"    model             : {prov}")
        print(f"    fragment rows      : {len(out):,}  (above floor {floor:g})")
    return out.reset_index(drop=True), prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="timsim-fragments",
        description="precursors -> predicted fragment intensities (measurement; spectral library)",
    )
    ap.add_argument("--precursors", required=True, type=Path,
                    help="a table with precursor_id, sequence (annotated), charge")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--collision-energy", type=float, required=True,
                    help="RAW normalized collision energy (~20-45 NCE), NOT the /100-encoded value "
                         "the .d stores. This artifact is the pre-acquisition model prediction; the "
                         "render applies per-run CE calibration and down-sampling on top.")
    ap.add_argument("--floor", type=float, default=1e-3)
    ap.add_argument("--model", default=None,
                    help="intensity model spec: omit for our default. See ...timsim.models.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    prec = pd.read_parquet(a.precursors)
    frame, prov = predict_fragments(
        prec, a.collision_energy, floor=a.floor, model=a.model, verbose=not a.quiet
    )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("precursor_id", pa.uint64(), nullable=False),
            pa.field("ion_type", pa.string(), nullable=False),
            pa.field("ordinal", pa.uint16(), nullable=False),
            pa.field("frag_charge", pa.uint8(), nullable=False),
            pa.field("intensity", pa.float32(), nullable=False),
        ],
        metadata={
            "timsim.table": "fragment_intensities",
            "timsim.schema_version": "2.0",
            "timsim.axis": "measurement",
            "timsim.producer": "timsim-fragments",
            "timsim.fragments.model": prov,
            "timsim.fragments.collision_energy": repr(float(a.collision_energy)),
        },
    )
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    pq.write_table(table, a.out)
    if not a.quiet:
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
