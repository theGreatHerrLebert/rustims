# NECROFLOW_WIRING — wire the measurement/render stage into the timsim2 DAG

Status: DRAFT for claudex. Goal: everything we built this session (Koina HCD fragments, `timsim-spectra`,
`timsim-render-thermo`, per-instrument templates, the device×method matrix) becomes reproducible
**necroflow nodes**, so we stop hand-firing steps and the whole matrix is content-addressed.

## Current state (the plot)

`flow/timsim_flow.py` wires the instrument-agnostic **feature space** as real DAG nodes
(`proteome → digest → modify → precursors → ccs, rt → design → peptide_quantities`), then collapses the
entire measurement/render into ONE monolithic node:

```
@r.command("mkdir -p {raw} && timsim {config} --save-path {raw} ...")
def simulate(proteome, peptides, occurrences, peptide_quantities, precursors, ccs, rt, config, sample_id, seed) -> RawData
```

So `simulate` = "run the whole legacy `timsim` CLI from a config → one `.d`". Fragments, spectra, and the
render are hidden inside it; the Thermo `.raw` path and the fragment-model choice are not expressible.

## Rule mechanism (how to add nodes)

- A rule is a function decorated `@r.command("<cli> {input_artifact} ... {output_artifact}")`; params are
  typed `NodeType` inputs + scalar config; the return annotation is the output artifact type.
- Rust bins are `{BIN}/timsim-x`; Python console-scripts are bare (`timsim-ccs`, `timsim-rt`).
- A node's fingerprint hashes its **config values**. A rule reading an external file declares
  `invalidator = hashes_file("<config_key>")` so changing the file's CONTENT makes the node stale.

## The changes

### A. New typed artifacts
`FragmentIntensities`, `IonSpectra` (today implicit inside `simulate`).

### B. New rules (each = a bin/job I ran by hand)
```
@r.command("timsim-fragments --precursors {frag_input} --collision-energy {ce} --model {frag_model} --out {fragment_intensities}")
def fragments(frag_input, ce, frag_model) -> FragmentIntensities        # model = "" (local timsTOF) | "koina:Prosit_2020_intensity_HCD"

@r.command(f"{BIN}/timsim-spectra --precursors {{precursors}} --peptides {{peptides}} --modforms {{modforms}} "
           "--modifications {{modifications}} --fragment-intensities {{fragment_intensities}} --out {{ion_spectra}}")
def spectra(precursors, peptides, modforms, modifications, fragment_intensities) -> IonSpectra

@r.command(f"{BIN}/timsim-render-thermo --precursors {{precursors}} --peptide-rt {{peptide_rt}} "
           "--ion-spectra {{ion_spectra}} --peptide-quantities {{peptide_quantities}} --template {{template}} "
           "--intensity-scale {{intensity_scale}} --out {{raw}} --thermo-truth {{truth}}")
def render_thermo(precursors, peptide_rt, ion_spectra, peptide_quantities, template, intensity_scale, sample_id) -> RawData
```
`render_thermo` declares `invalidator = hashes_file("template")` (the `.raw` template is an external input;
changing it must restage). Open: `timsim-fragments` console-script may not exist — today the job is
`python -m ...jobs.fragments`; decide the invocation (add a console-script, or a `python -m` command).

### C. Decompose the render, don't fork the feature space
Add a `timsim_thermo_pipeline(cfg, sample_id, template, frag_model, ...)` that reuses the SAME upstream
feature-space nodes and appends `fragments → spectra → render_thermo`. The Bruker `.d` path stays
(`simulate`) for now — do NOT rip it out in the same change (it's the working reference). A later step can
also decompose `simulate` into `fragments → spectra → render_bruker` so both share `fragments`/`spectra`.

### D. The device×method matrix as a fan-out
The matrix is `render_*` parameterized by `(instrument, template, frag_model, method)`:

| instrument | template | frag model | render rule |
|---|---|---|---|
| timsTOF | ref `.d` | local (timsTOF) | `simulate`/`render_bruker` |
| Astral | astral DIA `.raw` | koina HCD | `render_thermo` |
| Orbitrap | orbi DIA `.raw` | koina HCD | `render_thermo` |
| (DDA cells) | DDA templates | per instrument | future `render_*_dda` |

Because `fragments(model=X)` and the feature space are content-addressed, N renders that share a model
compute fragments ONCE; the timsTOF-vs-HCD split is exactly two `fragments` nodes. The multi-device idea
falls out of the DAG.

### E. CE + instrument-appropriate defaults
The fragment CE must match the template's method (Astral/Orbi ≈ NCE 25–27). Thread `ce` as a config value
per render target; Thermo defaults to the HCD model, Bruker to the local timsTOF model. Open: read CE from
the template automatically, or set it per matrix cell in config?

## Where the pipeline stops
This milestone stops at the artifact `.raw`/`.d` (+ the answer key). Search (DiaNN/Sage) + scoring
(`v2_eval`/`v2_thermo_eval`/`v2_dda_eval`) as DAG nodes is a natural PHASE 2 (a `search`→`score` tail),
not in scope here — keep it explicit so we don't half-wire it.

## Risks / open questions (for claudex)

1. **Decompose vs parallel branch** — is appending a Thermo branch (leaving `simulate` intact) the right
   call, or should we decompose `simulate` now so Bruker and Thermo share `fragments`/`spectra`? (Risk of
   breaking the working `.d` path vs duplicated fragment/spectra logic.)
2. **Template as config-path+invalidator vs input node** — the `.raw` template is a big external file;
   `hashes_file` restages on content change. Is hashing a multi-GB `.raw` acceptable, or should the
   template be a first-class input artifact / a pinned hash?
3. **`timsim-fragments` invocation** — no console-script today (`python -m ...jobs.fragments`). Add one,
   or use a `python -m` command in the rule? And the `_frag_input` artifact (precursor_id, sequence,
   charge) — is it a distinct node or derived inside `fragments`?
4. **Model as a fingerprinted param** — `frag_model="koina:..."` must be in the node config so a different
   model is a different node (it is, since config is hashed) — confirm Koina network calls don't break
   content-addressing (same inputs+model → same output? Koina is deterministic per model, so yes).
5. **`IonSpectra` MS1 vs MS2 provenance** — MS1 isotopes are instrument-agnostic; MS2 fragments are
   model-specific. One `spectra` node bakes both. For multi-device, is that fine (MS1 identical across
   models → dedup by fingerprint only if the fragment_intensities input differs → it won't dedup). Do we
   split MS1/MS2 spectra nodes to share the MS1?
6. **Where `peptide_rt`/`_frag_input` come from** — the flow has `rt` and `precursors`; the fragments job
   needs `(precursor_id, sequence, charge)`. Is that `precursors` + `peptides`, or a distinct artifact?
7. **The RawData type** — currently one `RawData` for `.d`. Do Thermo `.raw` and Bruker `.d` share the type
   (a format field) or become distinct types so the type checker catches a wrong-consumer?

## Claudex-revised decisions (fold into the build)

The review confirmed the sequencing (append Thermo branch, keep `simulate` as a reference oracle, decompose
later once a `render_bruker` is scientifically validated) and sharpened the contracts. Accepted changes:

1. **Type the formats separately.** `ThermoRawData` and `BrukerTimsData` are distinct output types (both
   may satisfy an abstract `RawData`) — a generic `RawData` invites wrong-consumer bugs. Do NOT assume
   Bruker/Thermo share `spectra`; they share `fragment_intensities`, but isotope modelling / serialization
   / acquisition semantics differ.
2. **`AcquisitionMethod` is a typed artifact/config, not just a param.** DIA/DDA drives windows, scan
   timing, selection, duty cycle, CE. Model it explicitly; the `.raw` template is ONE input to it.
3. **Explicit `FragmentPredictionInput` node** (was `_frag_input`) derived from precursors + peptide/mod
   state — freezes sequence, charge, mod encoding, predictor charge-filtering and input normalization.
   Don't hide this in the prediction rule.
4. **Truth is a declared output.** `render_thermo` returns a bundle `RenderedThermoRun { raw, truth }` (or
   two outputs) so the answer key is cached + invalidated exactly with the `.raw`.
5. **Koina is a reproducibility boundary, not a model string.** The `koina:...` config pins neither
   weights, service version, request schema, nor response precision; cache stabilises AFTER the first run
   but a post-change cache miss isn't reproducible. So: record model revision/endpoint/client versions in
   the manifest, and **treat the materialised `FragmentIntensities` artifact as the auditable boundary**
   (retain it; publication-grade runs pin a local/containerised predictor). Also **fingerprint executable/
   package versions** (a changed `timsim-spectra`/render binary MUST invalidate) — not only CLI args.
6. **Template identity = path locator + content digest + metadata**, with an optional pinned expected
   digest (fail on mismatch). Cache the hash by robust file identity (size+mtime+inode) with a forced
   full-verify mode for CI/releases/shared FS — never size/mtime alone for correctness.
7. **Validate template compatibility BEFORE rendering** — instrument family, method type, RT range, scan/
   cycle structure, polarity, m/z coverage — fail fast, not after an expensive render.
8. **CE: explicit per-cell config**, validated against extracted template metadata (fail/warn on
   mismatch). Auto-extraction is a later convenience, not the source of truth.
9. **Keep ONE `IonSpectra` node for now** (don't split MS1/MS2 yet — MS1 is small vs render cost), but
   design it as a versioned container with separable MS1/MS2 sections; split only if profiling shows
   redundant work across many models/devices.
10. **Emit a durable run manifest now** (run id, renderer version, template digest, method metadata,
    feature/fragment artifact ids, format-specific truth schema) so phase-2 search/score wires cleanly and
    the answer key can't drift.
11. **Invocation:** use the tested `python -m ...jobs.fragments` now; add a console-script only when
    packaging a public interface. Fingerprint the module/package version.

## Non-goals
Search/score nodes (phase 2, but emit the manifest/truth schema now); decomposing/removing legacy
`simulate` (later, gated on a validated `render_bruker`); the DDA render rules (need the Thermo DDA
milestone first); GUI/CLI changes beyond the flow.
