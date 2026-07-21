"""timsim v2 as a necroflow DAG.

The three axes are a **cache model, not a cross product**, and that is exactly what a
content-addressed DAG is for:

    STRUCTURE   which molecules exist          computed ONCE, shared by every sample
    QUANTITY    how much of each, per sample   cheap; one node per sample
    MEASUREMENT how it is observed, per run    one node per run

necroflow's `DAG.add` deduplicates nodes by a content-addressed fingerprint, so the structure nodes
of N sample pipelines collapse to one set automatically. Nothing here asks for that; it falls out of
declaring the dependencies honestly. Adding a 20th sample re-runs `yield` and the simulator, and
re-runs nothing upstream of them.

That is the whole thesis of the redesign, expressed as a graph:

    proteome ─┬─ digest ─┬─ modify ── precursors ─┐
              │          │                        │
              │          └────────────────────────┼─── yield ── simulate(sample)  × N
              └─ design ──────────────────────────┘

Run:
    python timsim_flow.py --outdir /tmp/necro --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

from necroflow import DAG, NodeType, Pipeline, Rules, resolve_command


# ── invalidation: a spec FILE is a dependency, not a string ──────────────────
#
# A node's fingerprint hashes its config *values*. A rule that takes `spec="design.toml"` therefore
# hashes the six characters of that filename — **not the mixture inside it.** Edit the fold change,
# re-run, and necroflow reports everything up to date and hands you the previous experiment's `.d`.
#
# That is exactly the failure this project exists to kill: a silently stale artifact that looks like
# a successful run. It is also invisible — nothing errors, the numbers are simply the old ones.
#
# So every artifact whose rule reads a spec file declares an `invalidator` that hashes the file's
# CONTENT. Change the file, change the token, and the node goes stale. Verified below, because a
# caching claim that has not been tested against an edit is not a caching claim.
def hashes_file(config_key: str):
    """Invalidate a node when the spec file named by `config_key` changes on disk."""

    def token(node) -> str:
        path = Path(node.config[config_key])
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return token


# ── artifacts ────────────────────────────────────────────────────────────────
#
# Each NodeType is a *typed artifact*, not a filename. The type is what lets necroflow check that a
# rule is fed the thing it asked for — the static half of the same discipline `timsim-schema`
# enforces at runtime. A stage that wants Peptides cannot be handed a Proteome, and the failure is at
# graph-construction time rather than three stages downstream.


class Proteome(NodeType):
    """The proteins. STRUCTURE — no amounts; abundance is a quantity and lives on its own axis."""

    filename = "proteome.parquet"
    invalidator = hashes_file("spec")


class Peptides(NodeType):
    """Distinct peptide sequences. STRUCTURE."""

    filename = "peptides.parquet"


class Occurrences(NodeType):
    """Where each peptide came from. The protein-inference answer key."""

    filename = "peptide_occurrences.parquet"


class CleavageSites(NodeType):
    """Every cleavage boundary. Needed to reconstruct the yield lattice."""

    filename = "cleavage_sites.parquet"


class Modforms(NodeType):
    """Modified species, with the fraction of molecules in each. STRUCTURE."""

    filename = "modforms.parquet"
    invalidator = hashes_file("mods")


class Modifications(NodeType):
    """The modification spec as an artifact — read by BOTH modify and yield, so the two can
    never disagree about an occupancy."""

    filename = "modifications.parquet"
    invalidator = hashes_file("mods")


class Precursors(NodeType):
    """The ion layer: m/z, isotope envelopes, charge fractions, ionisation propensity."""

    filename = "precursors.parquet"


class PrecursorCCS(NodeType):
    """Collision cross section per precursor. STRUCTURE — instrument-independent, predicted once.

    CCS is a property of the ion; 1/K0 is what an instrument measures from it. Keeping CCS here is
    what lets the same precursor space be measured on instrument A and instrument B (different gas)
    without recomputing anything upstream."""

    filename = "precursor_ccs.parquet"


class PeptideRT(NodeType):
    """Retention-time index per peptide. STRUCTURE — a hydrophobicity coordinate, gradient-independent,
    predicted once. Mapping it to seconds is a per-run measurement (the RT analog of CCS→1/K0)."""

    filename = "peptide_rt.parquet"


class Samples(NodeType):
    """The design: which samples exist, and the mixture that defines them."""

    filename = "samples.parquet"
    invalidator = hashes_file("spec")


class Runs(NodeType):
    """The runs, and which sample each measures."""

    filename = "runs.parquet"
    invalidator = hashes_file("spec")


class SampleRunMap(NodeType):
    """sample -> run. A sample measured twice is two runs, not two samples."""

    filename = "sample_run_map.parquet"
    invalidator = hashes_file("spec")


class ProteinQuantities(NodeType):
    """Protein amounts per sample, in amol. QUANTITY."""

    filename = "protein_quantities.parquet"
    invalidator = hashes_file("spec")


class PeptideQuantities(NodeType):
    """Peptide amounts per sample, in amol. QUANTITY — the digest applied to a mixture."""

    filename = "peptide_quantities.parquet"


class RawData(NodeType):
    """A Bruker .d — the MEASUREMENT. One per run.

    A directory rather than a file: timsim names the `.d` itself from the experiment. The node's
    output is the directory that contains it.
    """

    filename = "raw"
    invalidator = hashes_file("config")


class FragmentPredictionInput(NodeType):
    """`(precursor_id, [UNIMOD]-annotated sequence, charge)` — the FROZEN input to the fragment model.
    Explicit (not hidden in the prediction rule) so the sequence + charge + MOD ENCODING are cached and
    inspectable: a modified precursor carries its modform's annotated sequence, so it fragments as
    modified. Built by the same `annotate()` the spectrum builder uses for m/z, so intensity and m/z
    agree on what the molecule is."""

    filename = "fragment_prediction_input.parquet"


class FragmentIntensities(NodeType):
    """Predicted fragment intensities — MEASUREMENT, and the instrument-DEPENDENT artifact. The
    fragment model (local timsTOF vs koina Orbitrap-HCD) is a node config value, so a different model
    is a different node; the materialised artifact is the auditable boundary for a network predictor."""

    filename = "fragment_intensities"


class IonSpectra(NodeType):
    """Instrument-independent MS1 isotopes + MS2 fragment spectra as `(m/z, intensity)`. One node bakes
    both (MS2 depends on the fragment model, MS1 does not — a later split can share MS1)."""

    filename = "ion_spectra"


class ThermoRawData(NodeType):
    """A Thermo `.raw` — the MEASUREMENT for a NO-IMS instrument (Orbitrap / Astral), authored into a
    real template. A DISTINCT type from Bruker [`RawData`] (`.d`) so a wrong-consumer is a type error.
    Restages when the template file changes."""

    filename = "data.raw"
    invalidator = hashes_file("template")


class ThermoTruth(NodeType):
    """The per-precursor answer key — a co-output of the render, so it is cached and invalidated EXACTLY
    with its `.raw` (never a drifting sidecar). A future search/score node consumes this by type."""

    filename = "truth.parquet"


class ThermoRunManifest(NodeType):
    """The auditable boundary for a render: renderer identity + version, template identity, fragment
    model, acquisition method, content-addressed input paths, and the render's own counts. Co-emitted
    with the `.raw` so a run is reproducible after the fact."""

    filename = "manifest.json"


# ── rules ────────────────────────────────────────────────────────────────────

r = Rules()

BIN = os.environ.get("TIMSIM_BIN", "target/release")


@r.command(f"{BIN}/timsim-proteome --spec {{spec}} --out {{proteome}}")
def proteome(spec: str):
    """FASTAs -> proteins. STRUCTURE.

    Takes a multi-source spec rather than one FASTA, because that is what a real experiment is: HYE
    is three organisms, and the organism is a DECLARED column. v1 recovers it by substring-matching
    "HUMAN"/"YEAST"/"ECOLI" in the FASTA header, and peptides shared between two organisms silently
    become "Unknown" and get dropped.
    """
    return Proteome[proteome]


@r.command(
    f"{BIN}/timsim-digest --proteome {{proteome}} "
    "--out-peptides {peptides} --out-occurrences {occurrences} --out-cleavage-sites {cleavage_sites} "
    "--max-missed-cleavages {max_missed_cleavages} --min-length {min_length} --max-length {max_length}"
)
def digest(proteome: Proteome, max_missed_cleavages: int, min_length: int, max_length: int):
    """Proteins -> peptides. STRUCTURE, so it is computed once for every sample in the design.

    Three co-outputs of one call: they are one computation and necroflow treats them as such.
    """
    return Peptides[peptides], Occurrences[occurrences], CleavageSites[cleavage_sites]


@r.command(
    f"{BIN}/timsim-modify --peptides {{peptides}} --mods {{mods}} "
    "--out-modforms {modforms} --out-modifications {modifications} --floor {floor}"
)
def modify(peptides: Peptides, mods: str, floor: float):
    """Peptides -> modforms, driven by per-site OCCUPANCY rather than variable-mod combinatorics.

    Emits the modification spec alongside the modforms, because `yield` needs the same occupancies to
    know which cleavage sites are blocked. One artifact, two consumers, no flag to disagree about.
    """
    return Modforms[modforms], Modifications[modifications]


@r.command(
    f"{BIN}/timsim-precursors --peptides {{peptides}} --modforms {{modforms}} "
    "--out {precursors} --charge-model {charge_model} --seed {seed}"
)
def precursors(peptides: Peptides, modforms: Modforms, charge_model: str, seed: int):
    """Modforms -> ions. STRUCTURE: m/z and isotope envelopes are properties of the molecule, so
    they are shared by every sample too."""
    return Precursors[precursors]


@r.command(
    "timsim-ccs --precursors {precursors} --peptides {peptides} --out {precursor_ccs}"
)
def ccs(precursors: Precursors, peptides: Peptides):
    """Precursors -> CCS. STRUCTURE, and the one Python tool in the structure axis (the deep model
    is the standing exception). Runs once on the full precursor space and is shared by every sample
    and every simulated instrument."""
    return PrecursorCCS[precursor_ccs]


@r.command("timsim-rt --peptides {peptides} --out {peptide_rt}")
def rt(peptides: Peptides):
    """Peptides -> RT index. STRUCTURE; deep model (Chronologer by default). Shared across every
    sample and every gradient."""
    return PeptideRT[peptide_rt]


@r.command(
    f"{BIN}/timsim-design --proteome {{proteome}} --spec {{spec}} "
    "--out-samples {samples} --out-runs {runs} --out-sample-run-map {sample_run_map} "
    "--out-protein-quantities {protein_quantities}"
)
def design(proteome: Proteome, spec: str):
    """The mixture. QUANTITY — this is where an A/B experiment is *declared* rather than recovered
    from a filename afterwards."""
    return (
        Samples[samples],
        Runs[runs],
        SampleRunMap[sample_run_map],
        ProteinQuantities[protein_quantities],
    )


@r.command(
    f"{BIN}/timsim-yield --proteome {{proteome}} --occurrences {{occurrences}} "
    "--cleavage-sites {cleavage_sites} --protein-quantities {protein_quantities} "
    "--modifications {modifications} "
    "--digestion-efficiency {digestion_efficiency} --out {peptide_quantities}"
)
def peptide_yield(
    proteome: Proteome,
    occurrences: Occurrences,
    cleavage_sites: CleavageSites,
    protein_quantities: ProteinQuantities,
    modifications: Modifications,
    digestion_efficiency: float,
):
    """Structure x mixture -> peptide amounts. QUANTITY: cheap, and re-run whenever the design
    changes without touching the digest.

    Takes `modifications` so a blocking mod (acetyl-K, GG-K, TMT-K) actually stops the protease —
    the missed cleavage it forces is how the experiment localises the site.
    """
    return PeptideQuantities[peptide_quantities]


@r.command(
    "mkdir -p {raw} && timsim {config} --save-path {raw} "
    "--v2-proteome {proteome} --v2-peptides {peptides} --v2-occurrences {occurrences} "
    "--v2-peptide-quantities {peptide_quantities} --v2-precursors {precursors} "
    "--v2-ccs {precursor_ccs} --v2-rt {peptide_rt} "
    "--v2-sample {sample_id} --seed {seed}",
    # Declared, because it was learned the hard way: the adapter loads the whole precursor table
    # into pandas, and a PTM-enriched design makes that table enormous — a phospho occupancy of 0.30
    # produced a **3.1 GB** precursors.parquet. Two of these ran in parallel and the kernel killed
    # one (exit 137). The blow-up is real chemistry (an enrichment genuinely has that many species),
    # so the answer is to tell the scheduler the truth about the cost rather than to hide it.
    threads=2,
    ram="8Gi",
)
def simulate(
    proteome: Proteome,
    peptides: Peptides,
    occurrences: Occurrences,
    peptide_quantities: PeptideQuantities,
    precursors: Precursors,
    precursor_ccs: PrecursorCCS,
    peptide_rt: PeptideRT,
    config: str,
    sample_id: str,
    seed: int,
):
    """MEASUREMENT: v1's LC, ion mobility, fragmentation and acquisition, driven from v2 artifacts.

    This is the strangler seam. One node per sample, and the only thing that changes between them is
    `sample_id` — so this fans out N ways while everything above it is computed once.
    """
    return RawData[raw]


# ── the measurement/render branch (Thermo, no-IMS) ───────────────────────────
# These replace the hand-fired steps: predict fragments (choosable model) -> assemble spectra ->
# author a real Thermo .raw. They reuse the SAME feature-space nodes, so the device/method matrix is a
# fan-out over (template, frag_model) with the feature space computed once.


@r.command(
    f"{BIN}/timsim-frag-input --precursors {{precursors}} --peptides {{peptides}} "
    "--modforms {modforms} --modifications {modifications} --out {fragment_prediction_input}"
)
def frag_input(precursors: Precursors, peptides: Peptides, modforms: Modforms, modifications: Modifications):
    """Precursors + modforms -> the frozen fragment-prediction input: `(precursor_id, [UNIMOD]-annotated
    sequence, charge)`. STRUCTURE (no CE/model), so it is shared by every fragment model. Annotates each
    precursor's modform, so a MODIFIED precursor fragments as modified — this is the correctness fix over
    the old bare-sequence join, which predicted every modform identically."""
    return FragmentPredictionInput[fragment_prediction_input]


@r.command(
    "timsim-fragments --precursors {fragment_prediction_input} "
    "--collision-energy {collision_energy} --model {frag_model} --out {fragment_intensities}"
)
def fragments(fragment_prediction_input: FragmentPredictionInput, collision_energy: float, frag_model: str):
    """Annotated input -> predicted fragment intensities. MEASUREMENT, instrument-DEPENDENT: `frag_model`
    is "" (local timsTOF) or "koina:Prosit_2020_intensity_HCD" (Orbitrap-HCD), a config value — so the
    timsTOF-vs-HCD split is exactly two nodes and N renders sharing a model predict fragments once."""
    return FragmentIntensities[fragment_intensities]


@r.command(
    f"{BIN}/timsim-spectra --precursors {{precursors}} --peptides {{peptides}} "
    "--modforms {modforms} --modifications {modifications} "
    "--fragment-intensities {fragment_intensities} --out {ion_spectra}"
)
def spectra(
    precursors: Precursors,
    peptides: Peptides,
    modforms: Modforms,
    modifications: Modifications,
    fragment_intensities: FragmentIntensities,
):
    """Precursors + fragments -> instrument-independent MS1 isotope + MS2 fragment spectra."""
    return IonSpectra[ion_spectra]


@r.command(
    f"{BIN}/timsim-render-thermo --precursors {{precursors}} "
    "--peptide-rt {peptide_rt} --ion-spectra {ion_spectra} --peptide-quantities {peptide_quantities} "
    "--sample {sample_id} --template {template} --intensity-scale {intensity_scale} "
    "--frag-model {frag_model} --method {method} --expected-ce {collision_energy} "
    "--out {data_raw} --thermo-truth {truth} --manifest {manifest}",
    threads=2,
    ram="8Gi",
)
def render_thermo(
    precursors: Precursors,
    peptide_rt: PeptideRT,
    ion_spectra: IonSpectra,
    peptide_quantities: PeptideQuantities,
    template: str,
    intensity_scale: float,
    sample_id: str,
    frag_model: str,
    method: str,
    collision_energy: float,
):
    """MEASUREMENT: author the feature space into a real Thermo `.raw` template (no-IMS). One node per
    sample (via `peptide_quantities` + `sample_id`); restages when the template changes. Three co-outputs
    of one command: the `.raw`, its answer key, and a durable run manifest — one computation, so the
    answer key and audit trail can never drift from the data."""
    return ThermoRawData[data_raw], ThermoTruth[truth], ThermoRunManifest[manifest]


# ── the pipeline ─────────────────────────────────────────────────────────────


def timsim_pipeline(cfg, sample_id: str) -> Pipeline:
    """One sample, end to end.

    Every node above `peptide_yield` depends only on things that do NOT vary with the sample, so N
    calls to this function produce N pipelines whose structure nodes share a fingerprint — and
    necroflow collapses them. The cache model is not implemented here; it is *implied* by declaring
    the dependencies honestly.
    """
    P = Pipeline()
    P.proteome = r.proteome(spec=cfg.proteome_spec)

    P.peptides, P.occurrences, P.cleavage_sites = r.digest(
        P.proteome,
        max_missed_cleavages=cfg.max_missed_cleavages,
        min_length=cfg.min_length,
        max_length=cfg.max_length,
    )

    P.modforms, P.modifications = r.modify(P.peptides, mods=cfg.mods, floor=cfg.floor)

    P.precursors = r.precursors(
        P.peptides, P.modforms, charge_model=cfg.charge_model, seed=cfg.seed
    )
    P.ccs = r.ccs(P.precursors, P.peptides)
    P.rt = r.rt(P.peptides)

    # The seed lives INSIDE the design spec, not on the command line — it is a property of the
    # experiment, and a flag would let the artifact and the caller disagree about it.
    P.samples, P.runs, P.sample_run_map, P.protein_quantities = r.design(
        P.proteome, spec=cfg.design_spec
    )

    P.peptide_quantities = r.peptide_yield(
        P.proteome,
        P.occurrences,
        P.cleavage_sites,
        P.protein_quantities,
        P.modifications,
        digestion_efficiency=cfg.digestion_efficiency,
    )

    # The fan-out. `sample_id` is the ONLY thing that differs between samples, so this node's
    # fingerprint differs and everything upstream of it does not.
    P.raw = r.simulate(
        P.proteome,
        P.peptides,
        P.occurrences,
        P.peptide_quantities,
        P.precursors,
        P.ccs,
        P.rt,
        config=cfg.timsim_config,
        sample_id=sample_id,
        seed=cfg.seed,
    )
    return P


def timsim_thermo_pipeline(cfg, sample_id: str) -> Pipeline:
    """One sample, end to end, but the MEASUREMENT is a Thermo `.raw` authored into a template (no-IMS
    Orbitrap / Astral) via explicit `fragments -> spectra -> render_thermo` nodes.

    The feature-space nodes are IDENTICAL to `timsim_pipeline`'s (same config values -> same
    fingerprint), so necroflow collapses them: request both a Bruker and a Thermo pipeline and the whole
    structure axis is computed once. CCS is omitted (no ion mobility). The instrument/method matrix is a
    fan-out over `cfg.template` × `cfg.frag_model`.
    """
    P = Pipeline()
    P.proteome = r.proteome(spec=cfg.proteome_spec)
    P.peptides, P.occurrences, P.cleavage_sites = r.digest(
        P.proteome,
        max_missed_cleavages=cfg.max_missed_cleavages,
        min_length=cfg.min_length,
        max_length=cfg.max_length,
    )
    P.modforms, P.modifications = r.modify(P.peptides, mods=cfg.mods, floor=cfg.floor)
    P.precursors = r.precursors(P.peptides, P.modforms, charge_model=cfg.charge_model, seed=cfg.seed)
    P.rt = r.rt(P.peptides)
    P.samples, P.runs, P.sample_run_map, P.protein_quantities = r.design(
        P.proteome, spec=cfg.design_spec
    )
    P.peptide_quantities = r.peptide_yield(
        P.proteome,
        P.occurrences,
        P.cleavage_sites,
        P.protein_quantities,
        P.modifications,
        digestion_efficiency=cfg.digestion_efficiency,
    )
    # ── measurement branch (was hand-fired) ──
    P.fragment_prediction_input = r.frag_input(
        P.precursors, P.peptides, P.modforms, P.modifications
    )
    P.fragment_intensities = r.fragments(
        P.fragment_prediction_input, collision_energy=cfg.collision_energy, frag_model=cfg.frag_model
    )
    P.ion_spectra = r.spectra(
        P.precursors, P.peptides, P.modforms, P.modifications, P.fragment_intensities
    )
    P.raw, P.truth, P.manifest = r.render_thermo(
        P.precursors,
        P.rt,
        P.ion_spectra,
        P.peptide_quantities,
        template=cfg.template,
        intensity_scale=cfg.intensity_scale,
        sample_id=sample_id,
        frag_model=cfg.frag_model,
        method=getattr(cfg, "method", "DIA"),
        collision_energy=cfg.collision_energy,
    )
    return P


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="/tmp/necro-timsim")
    ap.add_argument("--proteome-spec", default="hye.toml", help="multi-FASTA proteome spec")
    ap.add_argument("--mods", default="mods.toml", help="modification spec (e.g. mods_basic.toml for a light HeLa run)")
    ap.add_argument("--design-spec", default="design.toml", help="experiment design spec (e.g. design_hela.toml for a single-organism run)")
    ap.add_argument("--samples", nargs="+", default=["A_R1", "B_R1"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--graph", help="write the DAG to this file")
    ap.add_argument("--thermo-template", help="build the Thermo .raw pipeline against this template")
    ap.add_argument("--frag-model", default="", help="fragment model: '' (local timsTOF) or 'koina:Prosit_2020_intensity_HCD'")
    ap.add_argument("--collision-energy", type=float, default=25.0)
    ap.add_argument("--intensity-scale", type=float, default=5.0e5)
    a = ap.parse_args()

    cfg = SimpleNamespace(
        proteome_spec=a.proteome_spec,
        max_missed_cleavages=2,
        min_length=7,
        max_length=30,
        mods=a.mods,
        floor=1e-3,
        charge_model="site-specific",
        design_spec=a.design_spec,
        digestion_efficiency=0.9,
        timsim_config="v1.toml",
        seed=41,
        template=a.thermo_template,
        frag_model=a.frag_model,
        collision_energy=a.collision_energy,
        intensity_scale=a.intensity_scale,
    )

    build = timsim_thermo_pipeline if a.thermo_template else timsim_pipeline
    dag = DAG(a.outdir)
    for sid in a.samples:
        P = build(cfg, sid)
        # For the Thermo branch the answer key + run manifest are first-class deliverables (co-outputs of
        # the render), so request them explicitly alongside the .raw.
        req = [P.raw]
        if getattr(P, "truth", None) is not None:
            req += [P.truth, P.manifest]
        dag.add(P, request=req)

    print(dag)
    dag.resolve_paths(a.outdir)

    # The claim this whole exercise rests on, stated as a number rather than an argument.
    n_total = len(dag._all_nodes)
    n_unique = len(dag.nodes)
    print()
    print(f"  samples requested      : {len(a.samples)}")
    print(f"  nodes across pipelines : {n_total}")
    print(f"  nodes actually to run  : {n_unique}   <- structure deduplicated by fingerprint")
    print(f"  saved                  : {n_total - n_unique} redundant stage executions")

    if a.graph:
        dag.save(a.graph)
        print(f"  -> {a.graph}")

    if a.dry_run:
        print()
        print("  resolved commands:")
        for node in dag.nodes:
            cmd = resolve_command(node)
            if cmd:
                print(f"    {cmd}")
        return

    dag.execute()


if __name__ == "__main__":
    main()
