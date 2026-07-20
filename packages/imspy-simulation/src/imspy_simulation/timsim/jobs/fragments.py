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
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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


_KOINA_ANN = re.compile(rb"([yb])(\d+)\+(\d+)")


def _predict_tensors_koina(sequences, charges, collision_energies, name: str):
    """Predict ``(n,29,2,3)`` fragment intensity tensors via a Koina model (e.g. Prosit_2020_intensity_HCD
    for Orbitrap-HCD Astral).

    Koina returns intensities in **long format** — one row per fragment with an ``annotation`` (e.g.
    ``b'y1+1'``, ``b'b2+1'``), an ``mz``, and a scalar ``intensities`` — NOT a flat 174-vector, so we
    parse each annotation into its ``(position, ion-type, charge)`` slot rather than assuming an order
    (axis-2: y→0, b→1; charge k→index k-1; position p→index p-1). The output DataFrame's index maps back
    to the input row. Absent fragments stay 0 (dropped downstream by the intensity floor).
    """
    import pandas as pd
    from imspy_predictors.koina_models.access_models import ModelFromKoina

    n = len(sequences)
    df = pd.DataFrame(
        {
            "peptide_sequences": [str(s) for s in sequences],
            "precursor_charges": [int(c) for c in charges],
            "collision_energies": [float(e) for e in collision_energies],
        }
    )
    out = ModelFromKoina(model_name=name).predict(df)
    for col in ("annotation", "intensities"):
        if col not in out.columns:
            raise ValueError(f"Koina model {name!r} returned no {col!r} column (got {list(out.columns)})")

    pred = np.zeros((n, 29, 2, 3), dtype=np.float32)
    for idx, grp in out.groupby(level=0):
        ii = int(idx)
        if not (0 <= ii < n):
            raise ValueError(f"Koina output index {ii} outside input range 0..{n - 1}")
        for ann, inten in zip(grp["annotation"].to_numpy(), grp["intensities"].to_numpy()):
            a = ann if isinstance(ann, (bytes, bytearray)) else str(ann).encode()
            mo = _KOINA_ANN.match(a)
            if not mo:
                continue
            it, pos, ch = mo.group(1), int(mo.group(2)), int(mo.group(3))
            if 1 <= pos <= 29 and 1 <= ch <= 3:
                pred[ii, pos - 1, 0 if it == b"y" else 1, ch - 1] = max(0.0, float(inten))
    return pred


def predict_tensors(sequences, charges, collision_energies, model: Optional[str] = None):
    """Predict per-precursor ``(29,2,3)`` intensity tensors with the resolved intensity model.
    Returns ``(array[n,29,2,3], provenance)``."""
    from imspy_simulation.timsim.models import resolve

    kind, name = resolve("fragments", model)
    if kind == "koina":
        return _predict_tensors_koina(sequences, charges, collision_energies, name), f"koina:{name}"

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


def fragment_schema(prov: str, collision_energy: float) -> pa.Schema:
    """The ``fragment_intensities`` (measurement) schema, stamped with model + CE provenance."""
    return pa.schema(
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
            "timsim.fragments.collision_energy": repr(float(collision_energy)),
        },
    )


def predict_fragment_batches(
    precursors: pd.DataFrame,
    collision_energy: float,
    floor: float = 1e-3,
    model: Optional[str] = None,
    chunk: int = 2_000_000,
    verbose: bool = True,
) -> tuple[str, pa.Schema, "Iterator[pa.RecordBatch]"]:
    """Streaming core: ``(provenance, schema, generator of RecordBatch)``.

    ``precursors`` must have ``precursor_id, sequence, charge`` (the sequence being the modform's
    [UNIMOD]-annotated sequence, so a modified peptide fragments as modified). Each distinct
    ``(sequence, charge)`` is decoded ONCE and fanned out over the precursors that share it, emitted in
    row-groups of ``chunk`` fragments. Peak memory is one chunk — not the full ~n_precursors×54-row
    frame, which at scale cost ~17 GB (and a slow per-row ``.iloc`` loop). Rows above ``floor``;
    structurally-absent slots (Prosit marks them -1) are dropped.
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
    schema = fragment_schema(prov, collision_energy)

    # key -> the precursor_ids sharing it, built with numpy .values (no pandas scalar indexing in the
    # hot path — that indexing, not the model, was the bulk of the old runtime).
    key2pids: dict = defaultdict(list)
    for pid, s, c in zip(
        precursors["precursor_id"].values, precursors["sequence"].values, precursors["charge"].values
    ):
        key2pids[(s, int(c))].append(int(pid))
    key_tuples = list(zip(keys["sequence"].values, keys["charge"].astype(int).values))

    def batches():
        bp, bi, bo, bf, bv = [], [], [], [], []
        total = 0
        for i, key in enumerate(key_tuples):
            frags = list(decode_tensor(tensors[i], floor))  # decode this key once
            if not frags:
                continue
            for pid in key2pids.get(key, ()):
                for it, ordinal, fc, v in frags:
                    bp.append(pid); bi.append(it); bo.append(ordinal); bf.append(fc); bv.append(v)
            if len(bp) >= chunk:
                yield pa.record_batch(
                    [pa.array(bp, pa.uint64()), pa.array(bi, pa.string()), pa.array(bo, pa.uint16()),
                     pa.array(bf, pa.uint8()), pa.array(bv, pa.float32())], schema=schema)
                total += len(bp)
                bp, bi, bo, bf, bv = [], [], [], [], []
        if bp:
            yield pa.record_batch(
                [pa.array(bp, pa.uint64()), pa.array(bi, pa.string()), pa.array(bo, pa.uint16()),
                 pa.array(bf, pa.uint8()), pa.array(bv, pa.float32())], schema=schema)
            total += len(bp)
        if verbose:
            print(f"    model             : {prov}")
            print(f"    fragment rows      : {total:,}  (above floor {floor:g})")

    return prov, schema, batches()


def predict_fragments(
    precursors: pd.DataFrame,
    collision_energy: float,
    floor: float = 1e-3,
    model: Optional[str] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Return ``(fragment_intensities frame, provenance)`` — the in-memory convenience wrapper around
    :func:`predict_fragment_batches`. Materialises the whole frame, so for large inputs prefer the
    streaming batch generator (that is what the CLI uses)."""
    prov, schema, batches = predict_fragment_batches(
        precursors, collision_energy, floor=floor, model=model, verbose=verbose
    )
    table = pa.Table.from_batches(list(batches), schema=schema)
    return table.to_pandas(), prov


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
    a.out.parent.mkdir(parents=True, exist_ok=True)

    # Stream row-groups straight to the file: the full fragment frame is never resident.
    _prov, schema, batches = predict_fragment_batches(
        prec, a.collision_energy, floor=a.floor, model=a.model, verbose=not a.quiet
    )
    writer = pq.ParquetWriter(a.out, schema)
    try:
        for batch in batches:
            writer.write_table(pa.Table.from_batches([batch], schema=schema))
    finally:
        writer.close()
    if not a.quiet:
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
