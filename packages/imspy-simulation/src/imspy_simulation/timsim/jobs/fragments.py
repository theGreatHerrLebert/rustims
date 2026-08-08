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

# Collision energy: one number for the run, or one per precursor

By default every precursor is predicted at the single ``--collision-energy``. That is right for a
no-IMS instrument — an Astral/Orbitrap DIA method sets one NCE — but wrong for the timsTOF, where
dda-PASEF ramps the collision energy with the mobility scan, so a compact ion is fragmented ~30 eV
harder than an extended one.

``--collision-energies`` (opt-in) closes that gap. ``timsim-frag-ce`` walks CCS → ``1/K0`` → scan →
CE using the run's own mobility calibration and the same ``ActivationPolicy`` v1's
``dda_selection_scheme`` drives, and this stage lifts that per-precursor table onto the per-key axis
the predictor already accepts. CE is a deterministic function of ``(sequence, charge)`` — via CCS —
so the dedup is untouched and the prediction volume does not change; the collapse measures the
within-key spread and refuses to proceed if the invariant is ever violated. Leaving the flag off
produces byte-identical artifacts, down to the schema metadata.
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
    have_seq = "peptide_sequences" in out.columns
    have_chg = "precursor_charges" in out.columns
    matched = skipped = 0
    for idx, grp in out.groupby(level=0):
        ii = int(idx)
        if not (0 <= ii < n):
            raise ValueError(f"Koina output row index {ii} outside input range 0..{n - 1}")
        # Guard against Koina renumbering/reordering the index (which would silently misassign fragments
        # to the wrong peptide): verify this group's echoed sequence/charge match the input at row ii.
        if have_seq and str(grp["peptide_sequences"].iloc[0]) != str(sequences[ii]):
            raise ValueError(
                f"Koina row {ii} sequence {grp['peptide_sequences'].iloc[0]!r} != input "
                f"{sequences[ii]!r} — the output index does not map to the input row"
            )
        if have_chg and int(grp["precursor_charges"].iloc[0]) != int(charges[ii]):
            raise ValueError(f"Koina row {ii} charge does not match input — index does not map to input row")
        for ann, inten in zip(grp["annotation"].to_numpy(), grp["intensities"].to_numpy()):
            a = ann if isinstance(ann, (bytes, bytearray)) else str(ann).encode()
            # fullmatch: a neutral-loss annotation like b"y1+1-NH3" must NOT be placed in the plain y1,+1
            # slot. Prosit HCD emits only bare b/y here; anything else is skipped and counted.
            mo = _KOINA_ANN.fullmatch(a)
            if mo is None:
                skipped += 1
                continue
            it, pos, ch = mo.group(1), int(mo.group(2)), int(mo.group(3))
            if 1 <= pos <= 29 and 1 <= ch <= 3:
                pred[ii, pos - 1, 0 if it == b"y" else 1, ch - 1] = max(0.0, float(inten))
                matched += 1
            else:
                skipped += 1
    if matched == 0 and skipped > 0:
        raise ValueError(
            f"Koina model {name!r}: all {skipped} fragment annotations unparsed — output schema changed?"
        )
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


def fragment_schema(prov: str, collision_energy: float, ce_source: Optional[str] = None) -> pa.Schema:
    """The ``fragment_intensities`` (measurement) schema, stamped with model + CE provenance.

    ``ce_source`` is the per-precursor collision-energy table the run was driven with, if any. When
    it is ``None`` — the default, flat-CE behaviour — the metadata is EXACTLY what it has always
    been, so a default run stays byte-identical to every artifact already on disk. The extra keys
    only appear when the mobility-derived CE capability is actually used.
    """
    meta = {
        "timsim.table": "fragment_intensities",
        "timsim.schema_version": "2.0",
        "timsim.axis": "measurement",
        "timsim.producer": "timsim-fragments",
        "timsim.fragments.model": prov,
        "timsim.fragments.collision_energy": repr(float(collision_energy)),
    }
    if ce_source is not None:
        # The flat `collision_energy` above is retained (it is what `--collision-energy` was set to)
        # but it is NOT what was predicted at; say so, loudly, in the artifact itself.
        meta["timsim.fragments.collision_energy_mode"] = "per-precursor-mobility"
        meta["timsim.fragments.collision_energies"] = ce_source
    return pa.schema(
        [
            pa.field("precursor_id", pa.uint64(), nullable=False),
            pa.field("ion_type", pa.string(), nullable=False),
            pa.field("ordinal", pa.uint16(), nullable=False),
            pa.field("frag_charge", pa.uint8(), nullable=False),
            pa.field("intensity", pa.float32(), nullable=False),
        ],
        metadata=meta,
    )


def load_precursor_collision_energies(path) -> dict:
    """``precursor_id -> collision energy (eV)`` from a ``timsim-frag-ce`` artifact.

    Built off numpy arrays rather than pandas scalar indexing: at PTM scale this table has one row
    per precursor (100M+), and a `.iloc` loop over it would cost more than the model call.
    """
    t = pd.read_parquet(path, columns=["precursor_id", "collision_energy"])
    return dict(zip(t["precursor_id"].to_numpy(), t["collision_energy"].to_numpy()))


def collision_energy_per_key(key_tuples, key2pids, ce_by_precursor: dict, tolerance: float):
    """Collapse a per-PRECURSOR collision energy onto the per-KEY axis the predictor dedups on.

    Returns ``(list of CE aligned to key_tuples, worst within-key spread)``.

    # Why this is allowed to collapse at all

    CE here is mobility-derived: CCS→1/K0→scan→CE. CCS is predicted from ``(sequence, charge, mz)``
    and ``mz`` is fixed by ``(composition, charge)``, so every precursor sharing a ``(sequence,
    charge)`` key — the positional isomers — has the same CCS, the same scan and therefore the same
    CE. The key axis is thus still a complete description of the model input, and the prediction
    count does not move by a single call.

    That is a claim about the whole upstream chain, not an axiom, so it is MEASURED: the worst
    within-key spread is computed and anything above ``tolerance`` raises instead of silently
    picking one isomer's CE for all of them.
    """
    out = []
    worst = 0.0
    for key in key_tuples:
        pids = key2pids.get(key, ())
        ces = [ce_by_precursor[pid] for pid in pids]
        if not ces:
            # A key with no precursors cannot happen (keys come from the precursors), but a silent
            # 0.0 CE would be a catastrophic, invisible mis-prediction. Refuse.
            raise ValueError(f"key {key!r} has no precursors to take a collision energy from")
        lo, hi = min(ces), max(ces)
        worst = max(worst, hi - lo)
        out.append(float(ces[0]))
    if worst > tolerance:
        raise ValueError(
            f"collision energy is not constant within a (sequence, charge) key: worst spread "
            f"{worst:.6g} eV > tolerance {tolerance:g}. The predictor dedups on that key, so one "
            f"isomer's CE would be applied to all of them. Either the CCS/precursor artifacts are "
            f"inconsistent, or raise --ce-key-tolerance deliberately."
        )
    return out, worst


def predict_fragment_batches(
    precursors: pd.DataFrame,
    collision_energy: float,
    floor: float = 1e-3,
    model: Optional[str] = None,
    chunk: int = 2_000_000,
    verbose: bool = True,
    collision_energies: Optional[dict] = None,
    collision_energies_source: Optional[str] = None,
    ce_key_tolerance: float = 1e-9,
) -> tuple[str, pa.Schema, "Iterator[pa.RecordBatch]"]:
    """Streaming core: ``(provenance, schema, generator of RecordBatch)``.

    ``precursors`` must have ``precursor_id, sequence, charge`` (the sequence being the modform's
    [UNIMOD]-annotated sequence, so a modified peptide fragments as modified). Each distinct
    ``(sequence, charge)`` is decoded ONCE and fanned out over the precursors that share it, emitted in
    row-groups of ``chunk`` fragments. Peak memory is one chunk — not the full ~n_precursors×54-row
    frame, which at scale cost ~17 GB (and a slow per-row ``.iloc`` loop). Rows above ``floor``;
    structurally-absent slots (Prosit marks them -1) are dropped.

    ``collision_energies`` is the OPT-IN per-precursor collision energy (``precursor_id -> eV``,
    from ``timsim-frag-ce``). Left ``None`` — the default — every precursor is predicted at the flat
    ``collision_energy``, exactly as before.
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

    # key -> the precursor_ids sharing it, built with numpy .values (no pandas scalar indexing in the
    # hot path — that indexing, not the model, was the bulk of the old runtime).
    key2pids: dict = defaultdict(list)
    for pid, s, c in zip(
        precursors["precursor_id"].values, precursors["sequence"].values, precursors["charge"].values
    ):
        key2pids[(s, int(c))].append(int(pid))
    key_tuples = list(zip(keys["sequence"].values, keys["charge"].astype(int).values))

    # THE collision energy axis.
    #
    # By default: a SINGLE collision energy for every precursor. That is correct for a no-IMS
    # instrument (Astral/Orbitrap DIA sets one NCE for the whole run), which is what the Koina HCD
    # path serves. It is NOT correct for the timsTOF: dda-PASEF collision energy is SCAN-DRIVEN — a
    # function of the ion's mobility (see `handle.get_transmitted_ions`' per-ion collision_energies
    # and `dda_selection_scheme`'s `activation_policy.collision_energy_for_scan`).
    #
    # `--collision-energies` closes that gap: `timsim-frag-ce` walks CCS→1/K0→scan→CE with the run's
    # own mobility calibration and the SAME `ActivationPolicy` v1 uses, and this stage maps that onto
    # the per-key axis the predictor already takes. Because CE is a deterministic function of
    # (sequence, charge) — via CCS — the dedup is untouched and the prediction volume is identical.
    if collision_energies is None:
        ces = [collision_energy] * len(keys)
    else:
        # Coverage check, counted but not materialised: a fully-unmatched run must not build a
        # 100M-entry list just to report it.
        n_unknown, examples = 0, []
        for pids in key2pids.values():
            for pid in pids:
                if pid not in collision_energies:
                    n_unknown += 1
                    if len(examples) < 5:
                        examples.append(pid)
        if n_unknown:
            raise ValueError(
                f"{n_unknown} precursors are absent from the collision-energy table (e.g. "
                f"{examples}). Predicting those at the flat --collision-energy would silently mix "
                f"two CE regimes in one artifact; fix the frag-ce input instead."
            )
        ces, worst_spread = collision_energy_per_key(
            key_tuples, key2pids, collision_energies, ce_key_tolerance
        )
        if verbose:
            arr = np.asarray(ces, dtype=float)
            print(f"    CE mode           : per-precursor mobility-derived ({collision_energies_source})")
            print(
                f"    CE over keys (eV) : min {arr.min():.3f}  median {float(np.median(arr)):.3f}  "
                f"max {arr.max():.3f}"
            )
            print(f"    worst within-key CE spread: {worst_spread:g} eV  (dedup stays valid)")

    tensors, prov = predict_tensors(keys["sequence"], keys["charge"], ces, model=model)
    schema = fragment_schema(prov, collision_energy, ce_source=collision_energies_source)

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
    collision_energies: Optional[dict] = None,
    collision_energies_source: Optional[str] = None,
    ce_key_tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, str]:
    """Return ``(fragment_intensities frame, provenance)`` — the in-memory convenience wrapper around
    :func:`predict_fragment_batches`. Materialises the whole frame, so for large inputs prefer the
    streaming batch generator (that is what the CLI uses)."""
    prov, schema, batches = predict_fragment_batches(
        precursors, collision_energy, floor=floor, model=model, verbose=verbose,
        collision_energies=collision_energies,
        collision_energies_source=collision_energies_source,
        ce_key_tolerance=ce_key_tolerance,
    )
    table = pa.Table.from_batches(list(batches), schema=schema)
    return table.to_pandas(), prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="timsim-fragments",
        description="precursors -> predicted fragment intensities (measurement; spectral library)",
    )
    ap.add_argument("--precursors", required=True, type=Path,
                    help="either a pre-joined (precursor_id, sequence, charge) table, OR the "
                         "timsim-precursors output (precursor_id, peptide_id, charge) when --peptides "
                         "is also given")
    ap.add_argument("--peptides", type=Path, default=None,
                    help="peptides.parquet (peptide_id -> sequence). Legacy convenience: builds the input "
                         "by joining --precursors to the BARE peptide sequence (a modified precursor then "
                         "fragments as unmodified). Prefer `timsim-frag-input`, which freezes the "
                         "[UNIMOD]-annotated modform sequence; then pass its output as --precursors.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--collision-energy", type=float, required=True,
                    help="RAW normalized collision energy (~20-45 NCE), NOT the /100-encoded value "
                         "the .d stores. This artifact is the pre-acquisition model prediction; the "
                         "render applies per-run CE calibration and down-sampling on top.")
    ap.add_argument("--collision-energies", type=Path, default=None,
                    help="OPT-IN: a `timsim-frag-ce` artifact (precursor_id, scan, collision_energy) "
                         "giving each precursor the MOBILITY-DERIVED collision energy a timsTOF "
                         "dda-PASEF ramp would apply to it, instead of one CE for the whole run. "
                         "Omit for the historical flat-CE behaviour (byte-identical output). "
                         "--collision-energy stays required and is recorded in the artifact "
                         "metadata, but is not what the model is driven at.")
    ap.add_argument("--ce-key-tolerance", type=float, default=1e-9,
                    help="Largest within-(sequence,charge) collision-energy spread tolerated, in eV. "
                         "CE is a deterministic function of that key (via CCS), so the default is "
                         "effectively exact agreement; a violation raises rather than letting one "
                         "isomer's CE stand in for its siblings'.")
    ap.add_argument("--floor", type=float, default=1e-3)
    ap.add_argument("--model", default=None,
                    help="intensity model spec: omit for our default. See ...timsim.models.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.peptides is not None:
        # Build the (precursor_id, sequence, charge) fragment-prediction input by joining the precursors
        # table to peptide sequences. MEMORY: a PTM-heavy proteome can have 100M+ precursors, so we never
        # materialise one Python string per row — map peptide_id -> a categorical CODE (int32), then wrap
        # as a Categorical (codes + a small unique-sequence dictionary). 185M rows then cost ~0.7 GB of
        # codes, not ~9 GB of string objects. The predictor dedups (sequence, charge) internally.
        prec = pd.read_parquet(a.precursors, columns=["precursor_id", "peptide_id", "charge"])
        pep = pd.read_parquet(a.peptides, columns=["peptide_id", "sequence"])
        cat = pd.Categorical(pep["sequence"])
        code_of = dict(zip(pep["peptide_id"].to_numpy(), cat.codes.astype("int32")))
        codes = prec["peptide_id"].map(code_of)
        # Drop BOTH unmapped peptide_ids (map -> NaN) AND null sequences (Categorical encodes them as
        # code -1): notna() alone keeps the -1, which from_codes would turn back into a null sequence.
        keep = codes.notna() & (codes >= 0)
        prec = pd.DataFrame({
            "precursor_id": prec.loc[keep, "precursor_id"].to_numpy(),
            "sequence": pd.Categorical.from_codes(
                codes[keep].astype("int32").to_numpy(), categories=cat.categories
            ),
            "charge": prec.loc[keep, "charge"].to_numpy(),
        })
    else:
        prec = pd.read_parquet(a.precursors)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    ce_map = None if a.collision_energies is None else load_precursor_collision_energies(a.collision_energies)

    # Stream row-groups straight to the file: the full fragment frame is never resident.
    _prov, schema, batches = predict_fragment_batches(
        prec, a.collision_energy, floor=a.floor, model=a.model, verbose=not a.quiet,
        collision_energies=ce_map,
        collision_energies_source=None if ce_map is None else str(a.collision_energies),
        ce_key_tolerance=a.ce_key_tolerance,
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
