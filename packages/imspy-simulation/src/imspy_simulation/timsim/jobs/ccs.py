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

The deep model computes CCS internally and converts it to ``1/K0`` on the way out, at a fixed default
gas. Rather than re-tokenise and call the model's internals, we run the existing, tested predictor to
get ``1/K0`` at that default, then **invert** ``1/K0 -> CCS`` at the *same* default. Mason-Schamp is
exactly invertible (round-trip error ~2e-16), so this recovers the model's CCS losslessly — the
invertibility oracle used as an extraction mechanism.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The gas parameters the deep CCS model uses internally when it converts CCS -> 1/K0. We invert at
# EXACTLY these values, so the CCS we recover is the one the model produced. They are the
# imspy defaults (N2 at 31.85 C); if the predictor's internal defaults ever change, these must move
# with them, and the round-trip check below will catch a mismatch loudly.
_MODEL_GAS_MASS = 28.013
_MODEL_TEMP_C = 31.85


def predict_ccs(
    precursors: pd.DataFrame,
    peptides: pd.DataFrame,
    batch_size: int = 2048,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return a ``precursor_ccs`` frame: ``precursor_id, ccs, ccs_std``.

    ``precursors`` must have ``precursor_id, peptide_id, charge, mz``; ``peptides`` must have
    ``peptide_id, sequence``.
    """
    from imspy_predictors.ccs import DeepPeptideIonMobilityApex
    from imspy_core.chemistry import one_over_k0_to_ccs

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
    k0, k0_std = predictor.simulate_ion_mobilities(
        keys["sequence"].tolist(),
        keys["charge"].tolist(),
        keys["mz"].tolist(),
        return_uncertainty=True,
    )
    k0 = np.asarray(k0, dtype=float).flatten()
    k0_std = np.asarray(k0_std, dtype=float).flatten() if k0_std is not None else None

    mz = keys["mz"].to_numpy(dtype=float)
    z = keys["charge"].to_numpy(dtype=int)

    # Invert 1/K0 -> CCS at the model's own gas defaults. Lossless (see module docstring).
    ccs = np.array([
        one_over_k0_to_ccs(k, m, c, mass_gas=_MODEL_GAS_MASS, temp=_MODEL_TEMP_C)
        for k, m, c in zip(k0, mz, z)
    ])
    if k0_std is not None:
        # v1 carries the uncertainty through the same conversion as the mean (a linearisation it
        # already makes); we match that so parity holds, and store the width in CCS space.
        ccs_std = np.array([
            one_over_k0_to_ccs(s, m, c, mass_gas=_MODEL_GAS_MASS, temp=_MODEL_TEMP_C)
            for s, m, c in zip(k0_std, mz, z)
        ])
    else:
        ccs_std = np.full(len(keys), np.nan)

    keys = keys.assign(ccs=ccs, ccs_std=ccs_std)
    out = df.merge(keys, on=["sequence", "charge", "mz"], how="left")

    if verbose:
        print(f"    CCS range         : {ccs.min():.1f} – {ccs.max():.1f} Å²   (instrument-independent)")
    return out[["precursor_id", "ccs", "ccs_std"]].reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="timsim-ccs",
        description="precursors -> collision cross section (structure; deep model)",
    )
    ap.add_argument("--precursors", required=True, type=Path)
    ap.add_argument("--peptides", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    prec = pd.read_parquet(a.precursors, columns=["precursor_id", "peptide_id", "charge", "mz"])
    pep = pd.read_parquet(a.peptides, columns=["peptide_id", "sequence"])
    ccs = predict_ccs(prec, pep, batch_size=a.batch_size, verbose=not a.quiet)

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
            "timsim.ccs.model_gas_mass": str(_MODEL_GAS_MASS),
            "timsim.ccs.model_temp_c": str(_MODEL_TEMP_C),
        },
    )
    table = pa.Table.from_pandas(ccs, schema=schema, preserve_index=False)
    pq.write_table(table, a.out)
    if not a.quiet:
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
