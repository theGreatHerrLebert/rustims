"""``timsim-rt`` — retention-time index, per peptide. STRUCTURE.

A **deep-predictor exception** (Python), like ``timsim-ccs``. Chronologer is the default model.

# Why an index, not seconds

v1 stored retention time in seconds — but seconds are not a property of the peptide, they are what a
*particular* LC gradient produces. The peptide's property is its **hydrophobicity**: where it elutes
in *order*, and how far from its neighbours. Chronologer predicts exactly that (as minutes on a
reference gradient, which is an affine image of hydrophobicity, hence gradient-independent).

Storing the index here and mapping it to seconds per run (in the simulator, given the run's gradient)
is the same B14 split as CCS→1/K0: the index is structure, the elution time is a measurement. It is
what lets one peptide space be run on a 30-minute gradient and a 2-hour gradient and stay comparable.

# The reference range travels with the artifact

To map an index to seconds you need the range the gradient spans. v1 used each *sample's* own
min/max, so a peptide's RT depended on what else was in the tube. Instead this tool records the index
range over the **whole peptide space** (``timsim.rt.index_min`` / ``index_max`` in the artifact
metadata), so a peptide always lands at the same gradient *fraction*, independent of the sample. That
is both more portable and more physical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def predict_rt_index(
    peptides: pd.DataFrame,
    backend: str = "chronologer",
    verbose: bool = True,
) -> tuple[pd.DataFrame, float, float]:
    """Return ``(peptide_rt frame, index_min, index_max)``.

    ``peptides`` needs ``peptide_id, sequence``. The frame has ``peptide_id, rt_index`` with NaN for
    peptides the model rejects. ``index_min``/``index_max`` span the *predicted* (non-NaN) peptides
    and become the reference range for the per-run seconds mapping.
    """
    from imspy_predictors.rt.predictors import load_deep_retention_time_predictor

    model = load_deep_retention_time_predictor(backend=backend)
    name = type(model).__name__

    seqs = peptides["sequence"].astype(str).tolist()
    idx = np.asarray(model.simulate_separation_times(seqs), dtype=float).flatten()

    n_rejected = int(np.isnan(idx).sum())
    valid = idx[~np.isnan(idx)]
    if valid.size == 0:
        raise ValueError(
            f"{name} produced no usable RT index — every peptide was rejected. Check the model and "
            f"that the peptide sequences are in a form it accepts."
        )
    index_min, index_max = float(valid.min()), float(valid.max())

    if verbose:
        print(f"  timsim-rt")
        print(f"    model            : {name}")
        print(f"    peptides         : {len(seqs):,}")
        if n_rejected:
            print(f"    rejected         : {n_rejected:,}  (unsupported by the model → null index, "
                  f"not fabricated)")
        print(f"    index range      : {index_min:.2f} – {index_max:.2f}  (reference for the "
              f"per-run seconds map)")

    out = pd.DataFrame({"peptide_id": peptides["peptide_id"].to_numpy(), "rt_index": idx})
    return out, index_min, index_max


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="timsim-rt",
        description="peptides -> retention-time index (structure; deep model, Chronologer by default)",
    )
    ap.add_argument("--peptides", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--backend", default="chronologer",
                    help="RT model backend (default: chronologer; falls back to the transformer if "
                         "Chronologer is unavailable)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    pep = pd.read_parquet(a.peptides, columns=["peptide_id", "sequence"])
    frame, idx_min, idx_max = predict_rt_index(pep, backend=a.backend, verbose=not a.quiet)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("peptide_id", pa.uint64(), nullable=False),
            pa.field("rt_index", pa.float64(), nullable=True),
        ],
        metadata={
            "timsim.table": "peptide_rt",
            "timsim.schema_version": "2.0",
            "timsim.axis": "structure",
            "timsim.producer": "timsim-rt",
            # The reference range for index -> seconds. Carried so the mapping is stable across
            # samples and no consumer has to re-derive it (B13).
            "timsim.rt.index_min": repr(idx_min),
            "timsim.rt.index_max": repr(idx_max),
        },
    )
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    pq.write_table(table, a.out)
    if not a.quiet:
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
