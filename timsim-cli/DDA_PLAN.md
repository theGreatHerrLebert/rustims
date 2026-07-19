# DDA-PASEF render — plan (revised after domain review)

Goal: a first, minimal-but-real **DDA-PASEF** render path for timsim v2, producing a Bruker `.d` a DDA
search engine (Sage/FragPipe native ddaPASEF) can search, scorable by the eval harness against v2 answer
keys. Revised 2026-07-19 after a Codex review that hardened the **identity-preservation** design (the
tricky part of DDA). Iron rule inherited from the calibration thread: **biological abundance untouched**.

## What a DDA-PASEF `.d` is (from a real ref, `dda/blanks/G230913…d`)

- **Frames**: `ScanMode=8`; MS1 survey `MsMsType=0`, PASEF MS2 `MsMsType=8`.
- **Precursors**: `Id, LargestPeakMz, AverageMz, MonoisotopicMz, Charge, ScanNumber (mobility, REAL),
  Intensity, Parent (survey MS1 frame it was picked from)`.
- **PasefFrameMsMsInfo**: `Frame, ScanNumBegin, ScanNumEnd, IsolationMz, IsolationWidth, CollisionEnergy,
  Precursor`. One MS2 frame packs SEVERAL precursors, each in a non-overlapping mobility band; a
  precursor's fragments live only in its band.

## Pieces

1. **Writer (DDA path in `TdfWriter`)** — DIA-only today. Add `scan_mode=8` and a finalize that writes
   `Precursors` + `PasefFrameMsMsInfo` from records the render supplies.
2. **Selection scheme** — port v1's `simulate_dda_pasef_selection_scheme`: survey every `precursors_every`
   frames; per cycle top-N by MS1 intensity above `intensity_threshold`, `max_precursors`, dynamic
   exclusion (`exclusion_width`).
3. **PASEF packing + band-limited MS2 render** — greedy-pack a cycle's selected precursors into MS2 frames
   by non-overlapping mobility bands; deposit each precursor's fragments only in its band. MS1 = full survey.
4. **CLI**: `--dda`, `--precursors-every`, `--max-precursors`, `--intensity-threshold`, `--exclusion-width`.

## Identity preservation (revised — the load-bearing design)

Identity lives ONLY in a sidecar answer key (do **not** put peptide IDs in `.d` tables; the point is that
normal tools can search it). But it must survive the search engine, which does **not** carry our
`Precursors.Id` — it re-detects the precursor from the spectrum. So:

- **One acquisition EVENT = one unique `Precursors.Id`.** Never reuse an Id for a biological precursor
  selected twice; two selections of one peptide are two events, two Ids, two spectra. (The reader keys a
  `BTreeMap` on this Id — a duplicate silently drops bands.) Invariant: exactly one PASEF row per Id.
- **Sidecar `dda_selected_precursors` (Parquet), one row per written event:** `selection_id` (stable),
  `our_precursor_id`, `sequence`, `charge`, plus the **engine-visible spectrum locator** —
  `ms2_frame`, `scan_begin/end`, native spectrum/title id (after any export), `rt`, `isolation_mz/width`,
  declared `mono_mz`, `intensity`, `parent_ms1_frame`.
- **Scoring match order:** (1) map a PSM to a selection by **spectrum locator**; (2) then require
  `sequence` + `charge` + precursor-mass agreement under explicit tolerances. `(sequence,charge)` alone is
  NOT the key. Keep a separate peptide-level "identified ≥ once" metric as a secondary view.
- **Golden test:** two selections of one precursor at different RT round-trip as two distinct events.

## `.d` self-consistency invariants (must hold or the file is unreadable / mis-searched)

- `Frames.Id` contiguous, 1-based, **write order == Id** (writer/reader assume `frame_id-1` indexing);
  `TimsId` points to each compressed block after the 64-byte prefix.
- `ScanMode=8`; MS1 `MsMsType=0`, MS2 `MsMsType=8`. Every PASEF `Frame` exists and is an MS2 frame; every
  `Parent` exists and is the actual prior survey MS1 used for selection.
- Each PASEF `Precursor` appears **exactly once** in `Precursors`; its `ScanNumber` lies within its band;
  bands are within `0..NumScans-1`, non-empty, and **non-overlapping** (reader extracts by band ± a 5%
  margin, so edge leakage contaminates neighbours).
- The precursor's **MS1 isotope envelope is actually present** in the parent survey frame at its declared
  RT/mobility; its MS2 ions lie in its band.
- `MonoisotopicMz / LargestPeakMz / AverageMz` are computed from the **deposited, calibrated** envelope
  (not theoretical) — the reader prefers `MonoisotopicMz`; declared `Charge` must agree with isotope
  spacing (easy to break after TOF quantisation). `ScanNumber` is a **real-valued** apex estimate.
- `NumPeaks / MaxIntensity / SummedIntensities` describe the deduplicated encoded bins (the writer already
  computes these — do not hand-write inconsistent summaries).

## Metrics (revised — engine-independent denominator)

- **eligible truth** = truth events detectable by an **explicit, versioned criterion** (e.g. rendered MS1
  intensity ≥ threshold, mobility in range) — defined WITHOUT reference to the search engine (else circular).
- **selection recall** = selected ÷ eligible truth — reported as a function of the **versioned intensity
  model + threshold** (selection depends on the currently-uncalibrated intensity → a sim-property confound,
  flag it).
- **conditional ID recall** = correctly identified ÷ selected.
- **end-to-end recall** = correctly identified ÷ eligible truth.
- FDP = engine calls not matching any truth selection under the locator+mass criteria.

## Milestones & scope

- **M1 (de-risk writer):** DDA `.d` writer + trivial hand-built schedule. Gates: SQLite relational
  validation (all FKs/uniqueness above), raw-band validation, MS1-envelope + Precursor-metadata
  validation, and **round-trip through `TimsDatasetDDA`**.
- **M2 (render):** selection + packing + band-limited render → a searchable `.d`; a native ddaPASEF
  engine (Sage/FragPipe) finds our peptides, and a **`.d` → engine/export → spectrum-locator** mapping
  test proves the identity chain end-to-end (without this the identity claim is unproven).
- **M3 (harness):** Sage report parser (reuse v1 `parse_sage_results`) + `dda_selected_precursors` truth
  → selection / conditional / end-to-end recall + FDP.
- **Scope for the stab — "oracle-isolation" baseline (labelled as such):** clean single target per band,
  BUT still render the **full MS1 isotope envelope** and allow **unrelated precursor/isotope peaks within
  the quadrupole isolation window** (so isolation isn't magically clean). True chimeric co-fragmentation,
  charge mis-assignment, and MS1 feature-detection realism are explicit next milestones — flagged so the
  inflated ID recall of the oracle baseline is never read as real performance.

## Open questions resolved by review
Identity in sidecar (yes); match by spectrum-locator+mass not `(seq,charge)`; unique Id per event;
denominator engine-independent; co-isolation deferral only as a labelled oracle baseline that still shows
in-window contaminants. Remaining unknown to verify in M2: exactly how the chosen engine exposes the
native-`.d` spectrum locator we join on (frame/scan vs an exported title) — settle it empirically before
trusting the identity metric.
