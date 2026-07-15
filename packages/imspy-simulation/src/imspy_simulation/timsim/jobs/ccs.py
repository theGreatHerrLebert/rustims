"""``timsim-ccs`` — collision cross section, per precursor. STRUCTURE.

The **deep-predictor exception**. Every other structure tool is Rust; this one is Python because CCS
comes from a trained model, and re-implementing that in Rust buys nothing (SPEC §0).

# Why this stage stores CCS, not 1/K0

The instrument reports ``1/K0`` (inverse reduced mobility), and v1 stored exactly that — but ``1/K0``
is not a property of the ion. It is what a *particular* drift tube measures, and it depends on the
drift gas, its temperature and its pressure. **CCS is the ion**: a geometric property of the molecule
and its charge, and Mason-Schamp turns it into ``1/K0`` *given an instrument*.

So CCS belongs on the structure axis (predicted once, shared by every run) and ``1/K0`` is a
measurement computed per run. That split is what makes cross-instrument simulation possible: one CCS
artifact, measured on instrument A and instrument B, differs only in the gas parameters — the
experiment v1 could not express because it kept only N2-at-305K ``1/K0`` and discarded the CCS.

# How the CCS is obtained

CCS and its uncertainty are read **straight from the deep model in Å²** (``return_ccs=True``). CCS is
what the model actually predicts; ``1/K0`` is a drift-gas-dependent measurement derived from it. So a
tool whose whole job is the instrument-*independent* quantity should never touch a gas — and this one
does not. (An earlier version round-tripped through ``1/K0`` and back, which was lossless but coupled
this tool to the predictor's internal gas defaults for no reason.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def predict_ccs(
    precursors: pd.DataFrame,
    peptides: pd.DataFrame,
    batch_size: int = 2048,
    model: Optional[str] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Return ``(precursor_ccs frame, provenance)``: ``precursor_id, ccs, ccs_std``.

    ``precursors`` must have ``precursor_id, peptide_id, charge, mz``; ``peptides`` must have
    ``peptide_id, sequence``. ``model`` is a model spec (see
    :mod:`imspy_simulation.timsim.models`); ``None`` uses our default CCS model.
    """
    from imspy_simulation.timsim.models import resolve
    from imspy_predictors.ccs import DeepPeptideIonMobilityApex

    kind, model_name = resolve("ccs", model)
    if kind == "koina":
        # The Koina CCS models (AlphaPeptDeep, IM2Deep) return 1/K0 and need a `calcmass` column;
        # converting that back to instrument-independent CCS is the same inversion the local path
        # avoids. It is deliberately NOT wired blind — this environment has no network to verify it
        # against, and shipping an unverified prediction path is exactly the mistake this project
        # keeps catching. Wire and test it where Koina is reachable.
        raise NotImplementedError(
            f"CCS via Koina ({model_name!r}) is not wired yet. Use the default CCS model, or add and "
            f"test the Koina path against a reachable Koina server."
        )

    need = {"precursor_id", "peptide_id", "charge", "mz"}
    missing = need - set(precursors.columns)
    if missing:
        raise ValueError(f"precursors is missing columns {sorted(missing)}")

    df = precursors.merge(peptides[["peptide_id", "sequence"]], on="peptide_id", how="left")
    if df["sequence"].isna().any():
        n = int(df["sequence"].isna().sum())
        raise ValueError(
            f"{n} precursors reference a peptide_id absent from the peptides table. The two "
            f"artifacts were built from different digests."
        )

    # Positional isomers share (sequence, charge, mz) exactly, and the model sees only those three —
    # so they predict identically. Dedup before the (expensive) model call and scatter back after.
    # On a phospho-enriched run this is most of the table.
    keys = df[["sequence", "charge", "mz"]].drop_duplicates().reset_index(drop=True)
    if verbose:
        print(f"  timsim-ccs")
        print(f"    precursors        : {len(df):,}")
        print(f"    distinct (seq,z,mz): {len(keys):,}  ({len(keys) / len(df) * 100:.1f}% — the rest are isomers)")

    predictor = DeepPeptideIonMobilityApex(verbose=False)
    # Read CCS and its Å² uncertainty straight from the model — no gas, no conversion.
    ccs, ccs_std = predictor.simulate_ion_mobilities(
        keys["sequence"].tolist(),
        keys["charge"].tolist(),
        keys["mz"].tolist(),
        batch_size=batch_size,
        return_uncertainty=True,
        return_ccs=True,
    )
    ccs = np.asarray(ccs, dtype=float).flatten()
    ccs_std = (
        np.asarray(ccs_std, dtype=float).flatten()
        if ccs_std is not None
        else np.full(len(keys), np.nan)
    )

    keys = keys.assign(ccs=ccs, ccs_std=ccs_std)
    out = df.merge(keys, on=["sequence", "charge", "mz"], how="left")

    if verbose:
        print(f"    model             : {model_name}")
        print(f"    CCS range         : {ccs.min():.1f} – {ccs.max():.1f} Å²   (instrument-independent)")
    return out[["precursor_id", "ccs", "ccs_std"]].reset_index(drop=True), model_name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="timsim-ccs",
        description="precursors -> collision cross section (structure; deep model)",
    )
    ap.add_argument("--precursors", required=True, type=Path)
    ap.add_argument("--peptides", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--model", default=None,
                    help="CCS model spec: omit for our default. See imspy_simulation.timsim.models.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    prec = pd.read_parquet(a.precursors, columns=["precursor_id", "peptide_id", "charge", "mz"])
    pep = pd.read_parquet(a.peptides, columns=["peptide_id", "sequence"])
    ccs, prov = predict_ccs(prec, pep, batch_size=a.batch_size, model=a.model, verbose=not a.quiet)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    # Build the Arrow schema EXPLICITLY, matching timsim-schema's precursor_ccs spec — including
    # nullability. pyarrow defaults every column to nullable, but the reader validates nullability
    # (a required column arriving nullable is exactly the class of bug that motivated the schema),
    # so precursor_id and ccs must be declared non-null here or the Rust reader refuses the file.
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("precursor_id", pa.uint64(), nullable=False),
            pa.field("ccs", pa.float64(), nullable=False),
            pa.field("ccs_std", pa.float64(), nullable=True),
        ],
        metadata={
            "timsim.table": "precursor_ccs",
            "timsim.schema_version": "2.0",
            "timsim.axis": "structure",
            "timsim.producer": "timsim-ccs",
            # Provenance: which model produced this CCS.
            "timsim.ccs.model": prov,
        },
    )
    table = pa.Table.from_pandas(ccs, schema=schema, preserve_index=False)
    pq.write_table(table, a.out)
    if not a.quiet:
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
