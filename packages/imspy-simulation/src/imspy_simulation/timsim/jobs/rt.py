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
import hashlib
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# The elution-peak Beta distributions, matching v1's defaults exactly (sigma ~ Beta(4,4) scaled into
# the gradient band; k ~ Beta(1,20) scaled to the tailing range). We store the UNIT draw sigma_hat /
# k_hat in [0,1] — the gradient band is applied at render, not here — so the shape is portable and
# reproducible.
_SIGMA_ALPHA, _SIGMA_BETA = 4.0, 4.0
_K_ALPHA, _K_BETA = 1.0, 20.0


def _identity_uniform(sequence: str, salt: str) -> float:
    """A deterministic uniform in [0,1) keyed on (sequence, salt). Identity-keyed (B8): a peptide's
    peak shape depends on the peptide, not on draw order or which run it is — reproducible across
    runs and stable when other peptides are added or removed."""
    h = hashlib.blake2b(f"{sequence}#{salt}".encode(), digest_size=8).digest()
    return (int.from_bytes(h, "big") >> 11) / float(1 << 53)


def _beta_hats(sequences):
    """Per-peptide (sigma_hat, k_hat) unit Beta draws, identity-keyed on the sequence."""
    from scipy.stats import beta as _beta

    us = np.array([_identity_uniform(s, "rt_sigma") for s in sequences])
    uk = np.array([_identity_uniform(s, "rt_k") for s in sequences])
    return _beta.ppf(us, _SIGMA_ALPHA, _SIGMA_BETA), _beta.ppf(uk, _K_ALPHA, _K_BETA)


def predict_rt_index(
    peptides: pd.DataFrame,
    model: Optional[str] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, float, float, str]:
    """Return ``(peptide_rt frame, index_min, index_max, provenance)``.

    ``peptides`` needs ``peptide_id, sequence``. The frame has ``peptide_id, rt_index`` with NaN for
    peptides the model rejects. ``index_min``/``index_max`` span the *predicted* (non-NaN) peptides
    and become the reference range for the per-run seconds mapping. ``model`` is a model spec (see
    :mod:`imspy_simulation.timsim.models`): ``None`` for our default (Chronologer), or
    ``"koina:<name>"`` for a Koina model.
    """
    from imspy_simulation.timsim.models import resolve

    kind, name = resolve("rt", model)
    seqs = peptides["sequence"].astype(str).tolist()

    if kind == "koina":
        from imspy_predictors.rt.predictors import predict_retention_time_with_koina

        out = predict_retention_time_with_koina(model_name=name, data=peptides[["sequence"]].copy())
        # The Koina wrapper fills a prediction column; take the last numeric column as the index.
        idx = np.asarray(out.select_dtypes("number").iloc[:, -1], dtype=float).flatten()
    else:
        from imspy_predictors.rt.predictors import load_deep_retention_time_predictor

        predictor = load_deep_retention_time_predictor(backend=name)
        idx = np.asarray(predictor.simulate_separation_times(seqs), dtype=float).flatten()

    n_rejected = int(np.isnan(idx).sum())
    valid = idx[~np.isnan(idx)]
    if valid.size == 0:
        raise ValueError(
            f"{name} produced no usable RT index — every peptide was rejected. Check the model and "
            f"that the peptide sequences are in a form it accepts."
        )
    index_min, index_max = float(valid.min()), float(valid.max())

    prov = f"koina:{name}" if kind == "koina" else name
    if verbose:
        print(f"  timsim-rt")
        print(f"    model            : {prov}")
        print(f"    peptides         : {len(seqs):,}")
        if n_rejected:
            print(f"    rejected         : {n_rejected:,}  (unsupported by the model → null index, "
                  f"not fabricated)")
        print(f"    index range      : {index_min:.2f} – {index_max:.2f}  (reference for the "
              f"per-run seconds map)")

    sigma_hat, k_hat = _beta_hats(seqs)
    frame = pd.DataFrame({
        "peptide_id": peptides["peptide_id"].to_numpy(),
        "rt_index": idx,
        "rt_sigma_hat": sigma_hat,
        "rt_k_hat": k_hat,
    })
    return frame, index_min, index_max, prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="timsim-rt",
        description="peptides -> retention-time index (structure; deep model, Chronologer by default)",
    )
    ap.add_argument("--peptides", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default=None,
                    help="RT model spec: omit for our default (Chronologer), or 'koina:<name>' for "
                         "a Koina model. See imspy_simulation.timsim.models.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    pep = pd.read_parquet(a.peptides, columns=["peptide_id", "sequence"])
    frame, idx_min, idx_max, prov = predict_rt_index(pep, model=a.model, verbose=not a.quiet)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("peptide_id", pa.uint64(), nullable=False),
            pa.field("rt_index", pa.float64(), nullable=True),
            pa.field("rt_sigma_hat", pa.float64(), nullable=False),
            pa.field("rt_k_hat", pa.float64(), nullable=False),
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
            # Provenance: which model produced this index (B13 — a result is reproducible only if
            # you know what made it).
            "timsim.rt.model": prov,
        },
    )
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    pq.write_table(table, a.out)
    if not a.quiet:
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
