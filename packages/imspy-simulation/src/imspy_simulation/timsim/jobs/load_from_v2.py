"""Drive the v1 simulator from timsim v2 artifacts.

The strangler seam. v2 owns the **sample** (proteome → digest → design → yield); v1 keeps the
entire **measurement** axis (RT, charge, CCS, fragments, frame assembly, the ``.d`` writer). This
module converts v2's Parquet tables into exactly the two DataFrames v1's ``main()`` holds just
before ``simulate_retention_times``, so v1 continues from there unchanged.

Nothing in v1 is removed. Absent ``--from-v2``, v1 behaves precisely as it does today.

Two bridges have to be crossed honestly, and both are recorded rather than assumed:

**Identity.** v1 uses integer ``protein_id`` / ``peptide_id``; v2 uses a string accession and a
content hash. The adapter emits ``id_map.parquet`` so v2's ground truth can be joined back to v1's
output. Without it the round trip loses the answer key, which is the entire point of the exercise.

**Amount.** v1's ``events`` is an abundance count; v2's ``amount_amol`` is a molar quantity. The
bridge **preserves relative amounts exactly** — so fold changes, dynamic range, and the mixture all
survive untouched — and declares only the absolute scale, which is the one number v1's noise and
detection thresholds were tuned against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class V2Handoff:
    """The tables v1 expects, plus the map back to v2's ground truth."""

    proteins: pd.DataFrame
    peptides: pd.DataFrame
    id_map: pd.DataFrame
    #: events := amount_amol * event_scale. Declared, not guessed. Record it in provenance.
    event_scale: float
    sample_id: str
    #: `(peptide_id, charge, relative_abundance, mz)` from ``precursors.parquet``, or ``None`` if v2
    #: did not produce one — in which case v1 computes its own charge states as before.
    #:
    #: This replaces v1's ``simulate_charge_states``. Two differences, both deliberate and both on
    #: the intended-changes list:
    #:
    #: 1. **Histidine is not a lysine.** v1's binomial gives every protonatable site p=0.8, so a
    #:    His-containing peptide is charged like an Arg-containing one. v2 uses per-site gas-phase
    #:    basicity (R > K >> H), which puts a fully-tryptic peptide at 82% doubly charged instead of
    #:    v1's 56% (with an absurd 26% singly charged).
    #: 2. **Dropped charge states are not redistributed.** v1 discards any charge contributing under
    #:    ``min_charge_contrib`` (0.15) and then *renormalises the survivors*
    #:    (``predictors.py:121``) — so a 3-site peptide's real 1+ ions vanish and their abundance is
    #:    handed to 2+/3+, inflating them. v2 deletes the mass and reports it.
    ions: Optional[pd.DataFrame] = None


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Produce it with the v2 tools:\n"
            f"  timsim-proteome --spec ... --out proteome.parquet\n"
            f"  timsim-digest   --proteome proteome.parquet ...\n"
            f"  timsim-design   --proteome proteome.parquet --spec design.toml ...\n"
            f"  timsim-yield    --proteome proteome.parquet ... --out peptide_quantities.parquet"
        )
    return path


def load_from_v2(
    v2_dir: str | Path,
    sample_id: Optional[str] = None,
    total_events: float = 1e5 * 25_000,
    max_peptides: Optional[int] = None,
    max_peptide_length: int = 30,
    flyability: str = "lognormal",
    verbose: bool = True,
) -> V2Handoff:
    """Build v1's ``proteins`` and ``peptides`` DataFrames from a v2 artifact directory.

    Args:
        v2_dir: directory holding ``proteome.parquet``, ``peptides.parquet``,
            ``peptide_occurrences.parquet`` and ``peptide_quantities.parquet``.
        sample_id: which sample of the design to render. Defaults to the first, sorted.
            **One v1 run is one sample** — a design with six samples is six v1 runs, which is
            exactly the fan-out the structure/quantity split exists to make cheap.
        total_events: the absolute abundance scale handed to v1. Relative amounts are preserved
            exactly; this fixes only the overall signal level, which is what v1's noise model and
            detection thresholds were calibrated against.
        max_peptides: keep only the ``max_peptides`` most abundant peptides.

            **This is a stand-in for a detection model we have not built yet.**

            v2 emits the complete peptide space — the full human+yeast+E.coli digest is ~4.8M
            peptides — because the structure axis is deliberately unpruned. In reality the vast
            majority sit far below the detection limit and are never observed. That cut belongs in a
            ``design_eligibility`` artifact derived from amount x response x instrument
            (SPEC §1), and until that exists, taking the top-N by amount is an honest, blunt proxy:
            it keeps exactly the peptides that a detection model would keep first.

            It is a **numerical truncation, and its cost is reported** — the same discipline as
            ``--max-missed-cleavages``. ``None`` hands v1 everything, which for a real proteome it
            cannot chew.
        max_peptide_length: drop peptides longer than this before handing them to v1.

            **This is a MEASUREMENT constraint, not a property of the sample.** The Prosit-style
            fragment-intensity model is trained on peptides up to ~30 residues and indexes a fixed
            29-position ion array — hand it a 45-mer and it panics
            (``mscore/src/data/peptide.rs:394: range end index 32 out of range for slice of length
            29``). v1's own ``simulate_peptides`` therefore defaults to ``max_length=30``.

            The structure axis is entitled to contain long peptides; the *predictor* is not. So the
            cap belongs here, at the seam — and it is reported rather than applied silently.
        flyability: ionisation efficiency model — **used only as a FALLBACK, when
            ``precursors.parquet`` is absent.** When it is present, its ``ionization_propensity``
            column is the ground truth and is read rather than redrawn (see B13, below).

            ``"lognormal"`` reproduces v1's ``generate_normal_efficiency`` — a normal draw in log10
            space (median 1e-2, sigma 1), **rejection-truncated to [1e-4, 1]**, i.e. clipped at ±2
            sigma. ``"none"`` sets it to 1 (and does so on both paths).

            Unlike v1, the fallback draw is **identity-keyed** (``hash(sequence)``) rather than a
            bulk ``np.random.normal(size=n)`` assigned by row order — so adding a peptide does not
            reshuffle every other peptide's flyability, and A/B samples agree by construction rather
            than by accident.

            > Note v1 folds efficiency straight into ``events``
            > (``simulate_peptides.py:156``: ``events = (events * efficiency).astype(np.int32)``),
            > so abundance and flyability are **conflated and quantised** — only the product
            > survives. v2 keeps them apart; this bridge is where they collapse.
        verbose: print the bridge that was crossed.
    """
    d = Path(v2_dir)
    proteome = pd.read_parquet(_require(d / "proteome.parquet"))
    peptides = pd.read_parquet(_require(d / "peptides.parquet"))
    occurrences = pd.read_parquet(_require(d / "peptide_occurrences.parquet"))
    quantities = pd.read_parquet(_require(d / "peptide_quantities.parquet"))

    # ── pick the sample ───────────────────────────────────────────────────────
    samples = sorted(quantities["sample_id"].unique())
    if not samples:
        raise ValueError(
            f"{d / 'peptide_quantities.parquet'} contains no samples. Every peptide may have been "
            f"filtered out (check the digest's length bounds), or timsim-yield was given a design "
            f"in which no protein carries material."
        )
    if sample_id is None:
        sample_id = samples[0]
    elif sample_id not in samples:
        raise ValueError(f"sample {sample_id!r} not in the design; have {samples}")
    q = quantities[quantities["sample_id"] == sample_id]

    # ── integer ids, and the map back to v2's ground truth ────────────────────
    accessions = proteome["protein_id"].tolist()
    prot_int = {acc: i for i, acc in enumerate(accessions)}

    # Amounts are summed over OCCURRENCES, so a peptide has one row here regardless of how many
    # proteins (or how many positions within one protein) produced it.
    # Only peptides that are actually IN this sample.
    #
    # `timsim-yield` emits a row for every occurrence, including peptides of proteins that carry
    # amount 0 — because protein subsetting is a QUANTITY operation and excluded proteins stay in the
    # answer key (that is what keeps protein-level FDR answerable). But zero material is not a
    # molecule: handing those peptides to v1 would push non-existent species through RT, charge, CCS
    # and fragment prediction, and then WRITE THEM into the .d as simulated ions.
    q = q[q["amount_amol"] > 0.0]
    if q.empty:
        raise ValueError(f"no peptide carries a positive amount in sample {sample_id!r}")
    amounts = dict(zip(q["peptide_id"], q["amount_amol"]))
    pep = peptides[peptides["peptide_id"].isin(amounts)].reset_index(drop=True)
    if pep.empty:
        raise ValueError(f"no peptides carry an amount in sample {sample_id!r}")

    # ── the measurement-model constraint ─────────────────────────────────────
    n_all = len(pep)
    too_long = int((pep["length"] > max_peptide_length).sum())
    if too_long:
        pep = pep[pep["length"] <= max_peptide_length].reset_index(drop=True)
        if pep.empty:
            raise ValueError(
                f"every peptide exceeds max_peptide_length={max_peptide_length}; "
                f"digest with --max-length {max_peptide_length} or raise the cap"
            )

    # ── the detection stand-in ────────────────────────────────────────────────
    n_before = len(pep)
    amol_before = pep["peptide_id"].map(amounts).to_numpy(dtype=float)
    mass_before = float(amol_before.sum())
    if max_peptides is not None and n_before > max_peptides:
        pep = (
            pep.assign(_amt=pep["peptide_id"].map(amounts))
            .nlargest(max_peptides, "_amt")
            .drop(columns="_amt")
            .sort_values("peptide_id")
            .reset_index(drop=True)
        )
    pep_int = {pid: i for i, pid in enumerate(pep["peptide_id"])}

    # ── flyability (ionisation efficiency) ───────────────────────────────────
    #
    # **B13 — never re-enter, or infer, a fact the artifact already knows.**
    #
    # When `timsim-precursors` has run, it has ALREADY drawn each peptide's ionisation propensity
    # and written it to `precursors.parquet` as `ionization_propensity`. That column is the ground
    # truth: the artifact declares `ion_amol = peptide_amol × ionization_propensity ×
    # charge_fraction`, and the whole point of keeping each factor in its own column is that the
    # chain can be walked backwards.
    #
    # Redrawing flyability here — from the same distribution but a *different* key (blake2b on the
    # sequence, versus the Rust side's FNV/SplitMix identity key) — produced two independent values
    # for one physical quantity. Because both draws share a marginal distribution, every histogram
    # and every summary statistic looked correct; measured per-peptide, the correlation was
    # **r = +0.008** and **43.6% of peptides differed by more than 10×** between the propensity used
    # to build `events` and the propensity recorded as their ground truth. The chain silently stopped
    # being invertible. Read the artifact.
    prec_path = d / "precursors.parquet"
    prec_all = pd.read_parquet(prec_path) if prec_path.exists() else None

    if prec_all is not None:
        propensity = prec_all.groupby("peptide_id")["ionization_propensity"].first()
        fly = pep["peptide_id"].map(propensity).to_numpy(dtype=float)
        missing = int(np.isnan(fly).sum())
        if missing:
            raise ValueError(
                f"{missing:,} peptides carry an amount but have no row in precursors.parquet, so "
                f"their ionisation propensity is unknown. Inventing one here would put a value into "
                f"`events` that contradicts the ion layer's ground truth. Re-run timsim-precursors "
                f"against the same peptides.parquet."
            )
        if flyability == "none":
            fly = np.ones(len(pep))
    elif flyability == "lognormal":
        # v1's distribution exactly: N(-2, 1) in log10, clipped to [1e-4, 1] — which is ±2 sigma,
        # so the truncation is not cosmetic. Identity-keyed, unlike v1's bulk draw.
        import hashlib

        from statistics import NormalDist

        _nd = NormalDist()

        def _fly(seq: str) -> float:
            # REJECTION-sampled, not clipped.
            #
            # v1 REDRAWS values outside [1e-4, 1]; clipping would instead pile the rejected ~4.6%
            # into point masses AT the bounds — 2.3% of the proteome at exactly 1e-4 and 2.3% at
            # exactly 1.0 — which is a different distribution and shows up as two spikes in any
            # flyability histogram. (The Rust side had the same bug; both are fixed.)
            #
            # Identity-keyed: the attempt counter is folded into the hash, so the redraw stays
            # deterministic and independent of row order.
            for attempt in range(64):
                h = hashlib.blake2b(f"{seq}#{attempt}".encode(), digest_size=8).digest()
                u = (int.from_bytes(h, "big") >> 11) / float(1 << 53)
                u = min(max(u, 1e-12), 1.0 - 1e-12)
                v = 10.0 ** (-2.0 + 1.0 * _nd.inv_cdf(u))
                if 1e-4 <= v <= 1.0:
                    return float(v)
            return float(np.clip(v, 1e-4, 1.0))  # unreachable for these parameters

        fly = np.array([_fly(sq) for sq in pep["sequence"]])
    elif flyability == "none":
        fly = np.ones(len(pep))
    else:
        raise ValueError(f"unknown flyability model {flyability!r}; use 'lognormal' or 'none'")

    # ── the amount bridge: preserve ratios exactly, declare the scale ─────────
    #
    # The scale is the ONE number in this bridge that is chosen rather than derived, so it is the one
    # that must be checked. A zero, negative, or non-finite scale converts straight into zero,
    # negative or NaN `events` and sails into v1's simulation as if it were a real intensity.
    if not np.isfinite(total_events) or total_events <= 0.0:
        raise ValueError(
            f"total_events must be finite and strictly positive, got {total_events!r}. "
            f"It is the absolute signal scale handed to v1; relative amounts are preserved exactly, "
            f"and this fixes only the overall level."
        )
    amol = pep["peptide_id"].map(amounts).to_numpy(dtype=float)
    signal = amol * fly  # v1's `events` is the PRODUCT — abundance and flyability, conflated
    total_signal = float(signal.sum())
    if not np.isfinite(total_signal) or total_signal <= 0.0:
        raise ValueError(f"total signal for sample {sample_id!r} is {total_signal}")
    event_scale = float(total_events) / total_signal
    events = signal * event_scale

    # ── protein assignment: a peptide can arise from several proteins ─────────
    #
    # v1's `peptides` table is one row per peptide with ONE protein. We keep the full degeneracy in
    # `id_map` (it is the protein-inference answer key) and hand v1 the first protein by accession,
    # deterministically. v1 never had this information at all, so nothing is lost relative to it —
    # but nothing of ours is thrown away either.
    occ = occurrences[occurrences["peptide_id"].isin(pep_int)]
    first_prot = (
        occ.sort_values(["peptide_id", "protein_id", "start"])
        .groupby("peptide_id", as_index=False)
        .first()
    )
    fp = first_prot.set_index("peptide_id")

    prot_len = dict(zip(proteome["protein_id"], proteome["length"]))
    name_of = dict(zip(proteome["protein_id"], proteome["protein_id"]))

    sel = fp.loc[pep["peptide_id"]]
    prot_acc = sel["protein_id"].to_numpy()
    starts = sel["start"].to_numpy()
    ends = sel["end"].to_numpy()
    lengths = np.array([prot_len[a] for a in prot_acc])

    peptides_v1 = pd.DataFrame(
        {
            "protein_id": [prot_int[a] for a in prot_acc],
            "peptide_id": np.arange(len(pep), dtype=int),
            "sequence": pep["sequence"].to_numpy(),
            "protein": [name_of[a] for a in prot_acc],
            # A sample contains no decoys. Decoys are search-engine apparatus.
            #
            # Singular `decoy`, not `decoys`: v1's own `simulate_peptides` builds the column as
            # `decoys` and renames it on the way out (`simulate_peptides.py:175`). Since we replace
            # that job, we must hand over the POST-rename name everything downstream reads.
            "decoy": np.zeros(len(pep), dtype=bool),
            "missed_cleavages": sel["n_missed_cleavages"].to_numpy().astype(int),
            # Protein-terminal peptides, derived from protein-residue coordinates.
            "n_term": (starts == 0),
            "c_term": (ends == lengths),
            "monoisotopic-mass": pep["mass_monoisotopic"].to_numpy(),
            "events": events,
        }
    )

    # v1 wants protein-level events too; sum the peptide amounts back up per protein.
    prot_events = (
        peptides_v1.groupby("protein_id", as_index=False)["events"]
        .sum()
        .set_index("protein_id")["events"]
    )
    used = sorted(prot_events.index)
    inv = {i: acc for acc, i in prot_int.items()}
    seq_of = dict(zip(proteome["protein_id"], proteome["sequence"]))
    proteins_v1 = pd.DataFrame(
        {
            "protein_id": used,
            "protein": [inv[i] for i in used],
            "sequence": [seq_of[inv[i]] for i in used],
            "events": [float(prot_events[i]) for i in used],
        }
    )

    # ── the answer key, preserved across the id bridge ────────────────────────
    id_map = pd.DataFrame(
        {
            "v1_peptide_id": np.arange(len(pep), dtype=int),
            "v2_peptide_id": pep["peptide_id"].to_numpy(),
            "sequence": pep["sequence"].to_numpy(),
            "amount_amol": amol,
            # Kept SEPARATE here even though v1 conflates them into `events` — so the answer key
            # can still tell "how much peptide" apart from "how well it flies".
            "flyability": fly,
            "events": events,
            "sample_id": sample_id,
        }
    )

    # ── the ion layer, if timsim-precursors produced one ─────────────────────
    ions_v1 = None
    if prec_all is not None:
        prec = prec_all[prec_all["peptide_id"].isin(pep_int)]
        if not prec.empty:
            ions_v1 = pd.DataFrame(
                {
                    "peptide_id": prec["peptide_id"].map(pep_int).to_numpy(),
                    "charge": prec["charge"].to_numpy().astype(int),
                    # v1 calls this `relative_abundance` and uses it as
                    # `intensity = events * relative_abundance`. `events` already carries the
                    # flyability — and now carries the SAME `ionization_propensity` this table
                    # declares — so this must be the CHARGE FRACTION alone. Multiplying the
                    # propensity in here as well would double-count it.
                    "relative_abundance": prec["charge_fraction"].to_numpy().astype(float),
                    "mz": prec["mz"].to_numpy().astype(float),
                }
            ).sort_values(["peptide_id", "charge"]).reset_index(drop=True)

    if verbose:
        degenerate = int((occ.groupby("peptide_id")["protein_id"].nunique() > 1).sum())
        print(f"  v2 → v1 handoff")
        if too_long:
            print(f"    length cap      : dropped {too_long:,} of {n_all:,} peptides longer than "
                  f"{max_peptide_length} residues")
            print(f"                      ⚠ a MEASUREMENT constraint (the fragment model indexes a "
                  f"29-position ion array), not a property of the sample")
        if max_peptides is not None and n_before > max_peptides:
            kept = float(amol.sum()) / mass_before
            print(f"    detection cut   : top {max_peptides:,} of {n_before:,} peptides by amount")
            print(f"                      retains {kept * 100:.3f}% of the peptide molar mass")
            print(f"                      ⚠ a STAND-IN for the detection model (design_eligibility);"
                  f" not physics")
        print(f"    sample          : {sample_id}   (of {len(samples)}: {', '.join(samples)})")
        print(f"    proteins        : {len(proteins_v1):,}")
        print(f"    peptides        : {len(peptides_v1):,}")
        print(f"    degenerate      : {degenerate:,} peptides map to >1 protein "
              f"(kept in id_map; v1's table has room for one)")
        print(f"    flyability      : {flyability}  "
              f"({fly.min():.2e} … {fly.max():.2e}, median {np.median(fly):.2e})")
        print(f"    amount bridge   : events = amount_amol × flyability × {event_scale:.6g}")
        print(f"                      ⚠ v1 conflates abundance × flyability into `events`; "
              f"id_map keeps them apart")
        print(f"      amount_amol   : {amol.min():.3e} … {amol.max():.3e}  "
              f"({np.log10(amol.max() / amol.min()):.2f} orders)")
        print(f"      events        : {events.min():.3e} … {events.max():.3e}   "
              f"(ratios preserved exactly; only the scale is declared)")

    if verbose:
        if ions_v1 is not None:
            frac = ions_v1.groupby("charge")["relative_abundance"].sum()
            tot = frac.sum()
            dist = "  ".join(f"{int(z)}+ {v / tot * 100:.1f}%" for z, v in frac.items())
            kept = ions_v1.groupby("peptide_id")["relative_abundance"].sum()
            print(f"    ion layer       : {len(ions_v1):,} precursors from timsim-precursors")
            print(f"                      {dist}")
            print(f"                      charge fractions sum to {kept.mean():.4f} per peptide "
                  f"(<1 = ions above the charge cap, DELETED not redistributed)")
        else:
            print(f"    ion layer       : none (no precursors.parquet) — v1 will compute its own "
                  f"charge states")

    return V2Handoff(
        proteins=proteins_v1,
        peptides=peptides_v1,
        id_map=id_map,
        event_scale=event_scale,
        sample_id=sample_id,
        ions=ions_v1,
    )
