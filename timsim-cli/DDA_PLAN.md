# DDA-PASEF render — plan (grounded in the working v1 code + two domain reviews)

Goal: a first minimal-but-real **DDA-PASEF** render path for timsim v2 — a Bruker `.d` a native
ddaPASEF search engine (Sage/FragPipe) can search, scorable by the eval harness against v2 answer keys.
This revision folds in the **actual v1 implementation** (`jobs/dda_selection_scheme.py`, `tdf.py`
`write_precursor_table`/`write_pasef_meta_table`) so the plan reflects proven code, not guesses. Iron
rule inherited: **biological abundance untouched.**

## The v1 algorithm we are porting (read from source, not summarised)

**Schedule** (`simulate_dda_pasef_selection_scheme` + `schedule_precursors`):
- Frame layout: every `precursors_every`-th frame is an MS1 survey (`MsMsType=0`); the `k = precursors_every−1`
  frames between are the cycle's PASEF MS2 frames (`MsMsType=8`).
- Candidate intensity per MS1 frame = `relative_abundance × events × frame_abundance` (the ion's rendered
  MS1 intensity), filtered to ≥ `intensity_threshold`. **Selection is driven by this (currently
  uncalibrated) intensity** — the metric confound both reviews flagged.
- Per MS1 frame: sort candidates by `(intensity ↓, ion_id ↑)` (deterministic tie-break); for each, skip if
  **dynamically excluded** (`current_frame − last_scheduled < exclusion_width`); else compute its mobility
  band `[apex−11, apex+11]` (clamped to `0..scan_max`) and **first-fit pack** it into the earliest of the
  `k` MS2 frames that has `< max_precursors` and no conflict. **Fidelity note:** v1's conflict test compares
  the *new clamped* interval against existing precursors' `apex ± w` (not their stored/clamped endpoints) —
  port that predicate exactly; add a separate interval-overlap invariant/test rather than "fixing" selection
  in M1. Also: the **final MS1 survey is deliberately not scheduled** (`frame < max_frame_id`) — preserve it.
- Per selection: `IsolationMz = most-intense isotope` (kept distinct from mono in matching);
  `IsolationWidth = 3` if the retained-isotope span > 2 Th else `2`; `CollisionEnergy = 54.1984 − 0.0345 ×
  scan` (Bruker PASEF activation policy, IMS-scan-driven). `AverageMz` in v1 is just
  `(first + last retained-isotope m/z)/2` (isotopes < 5% of max are dropped first) — a crude approximation,
  NOT an intensity-weighted centroid; port faithfully but don't describe it as calibrated envelope metadata.

**Write** (`tdf.py`, to mirror in Rust `TdfWriter`):
- `Precursors` (vendor schema: `Id PK, LargestPeakMz, AverageMz, MonoisotopicMz, Charge, ScanNumber REAL,
  Intensity, Parent FK→Frames.Id`) — **deduped to one row per `ion_id`**.
- `PasefFrameMsMsInfo` (`Frame, ScanNumBegin/End INT, IsolationMz, IsolationWidth, CollisionEnergy,
  Precursor`, `PK(Frame, ScanNumBegin)`) — **many rows per precursor** (one per PASEF event); matches the
  real `.d`'s ~4.4 PASEF rows/precursor.
- Identity: `precursor_id_mapping` table maps `ion_id → sequential tdf_precursor_id` (vendor needs 1..N
  ids; internal sim keeps `ion_id` for fragment lookup). This IS the answer-key linkage.

So the M1 writer is a **port of proven Python** to `TdfWriter`, not a new design; the selection is a
port of `schedule_precursors`.

## Identity preservation (v1 model × codex rigor — synthesis)

- **Adopt v1's ID model:** `Precursors.Id` is per-**ion** (one MS1 feature), with **multiple PASEF
  bands** per precursor — what real Bruker `.d` looks like, so a search engine sees a realistic file.
  `precursor_id_mapping` (`ion_id → tdf_precursor_id`) is the vendor-side id remap.
- **`Precursors` is LOSSY — do not treat it as event truth.** v1 dedups `Precursors` to the *first*
  selection per ion, so a re-fragmented ion's later `Intensity/Parent/ScanNumber` are dropped. Therefore
  the **per-event Parent/Intensity/ScanNumber must live in the sidecar answer key**, not be read back from
  `Precursors`. (Corrects the earlier invariant "every PASEF `Parent` is its actual prior MS1" — that holds
  per event only via the sidecar.)
- **Answer key `dda_selected_precursors`, one row per PASEF EVENT, keyed on `(ms2_frame, scan_begin)`**
  (the vendor PK — our internal `selection_id` is NOT in the `.d`): `ms2_frame, scan_begin, scan_end,
  ion_id, our_precursor_id, tdf_precursor_id, sequence, charge, isolation_mz, mono_mz, parent_ms1_frame,
  event_intensity, rt`. Same ion in several cycles → several rows, all sharing ion/sequence truth.
- **Match a PSM by spectrum locator first** — every exported spectrum must map to a `(ms2_frame,
  scan_begin)`; **M2 must demonstrate that export→key mapping empirically** — then require `sequence +
  charge + precursor-mass` under tolerances (`(sequence,charge)` alone is insufficient; keep mono distinct
  from `IsolationMz`). If an exporter *merges* re-fragmented bands, per-event conditional-ID recall is
  undefined for that output → score an explicit "aggregated-precursor-spectrum" view instead. Peptide-level
  "identified ≥ once" is the secondary view.

## `.d` self-consistency invariants (must hold or unreadable / mis-searched)
- `Frames.Id` contiguous, 1-based, **write order == Id**; `TimsId` = each compressed block after the
  64-byte prefix. `ScanMode=8`; MS1 `MsMsType=0`, MS2 `MsMsType=8`. Every PASEF `Parent` is the actual
  prior survey MS1; every PASEF `Frame` is an MS2 frame.
- Each `Precursors.Id` present once; each PASEF `Precursor` references an existing `Id`; `ScanNumber`
  within its band; bands `0..NumScans-1`, non-empty, non-overlapping (reader adds a 5% margin → edge
  leakage contaminates neighbours).
- Precursor MS1 isotope envelope actually present in the parent survey frame at declared RT/mobility; MS2
  ions land in the band. `MonoisotopicMz`/`LargestPeakMz` from the deposited, calibrated envelope (reader
  prefers `MonoisotopicMz`); `AverageMz` is v1's crude `(first+last retained isotope)/2` (kept for
  fidelity, not treated as a real centroid). `Charge` agrees with isotope spacing; `ScanNumber` real-valued.
- Per-event `Parent/Intensity/ScanNumber` are authoritative in the **sidecar answer key**, not in
  `Precursors` (which keeps only the first selection per ion).
- `NumPeaks/MaxIntensity/SummedIntensities` from the deduplicated encoded bins (writer already computes).

## Metrics (engine-independent denominator)
- **eligible truth** = truth events detectable by an explicit, **versioned** criterion (rendered MS1
  intensity ≥ threshold, mobility in range) — defined WITHOUT the engine.
- **selection recall** = selected ÷ eligible (report vs the versioned intensity model — a sim confound).
- **conditional ID recall** = identified ÷ selected. **end-to-end** = identified ÷ eligible.
- FDP = engine calls not matching any truth selection under locator + mass.

## Milestones & scope
- **M1 (port + de-risk writer):** `TdfWriter` DDA path (Precursors + PasefFrameMsMsInfo, `scan_mode=8`) +
  trivial hand-built schedule. Fixture: a **real DDA `.d`** from `/media/hd02/data/raw/dda/…` for schema +
  row-count comparison (the bundled `NATIVE.d` has no DDA tables — unusable). Gates: SQLite relational
  validation (FKs/uniqueness/invariants above), band validation, MS1-envelope + Precursor-metadata
  validation. **Round-trip caveat (verified in `dda.rs`):** `get_pasef_frame_ms_ms_info` preserves every
  raw band, but `get_selected_precursors` (a `BTreeMap<precursor_id,…>`) keeps only the LAST band and
  `get_preprocessed_pasef_fragments` MERGES bands per precursor. So the M1 identity gate uses the **raw-row
  API + a two-selections-of-one-ion fixture** asserting both PASEF rows survive; the lossy/aggregate APIs
  are documented, not relied on for identity.
- **M2 (render):** port `schedule_precursors` + band-limited MS2 render → searchable `.d`; native engine
  finds our peptides; **`.d` → engine/export → spectrum-locator** mapping test proves the identity chain.
- **M3 (harness):** Sage parser (reuse v1 `parse_sage_results`) + `dda_selected_precursors` truth →
  selection / conditional / end-to-end recall + FDP.
- **Scope — labelled "oracle-isolation" baseline:** one clean target per band, BUT still render the full
  MS1 isotope envelope and allow unrelated precursor/isotope peaks within the quadrupole window (isolation
  not magically clean). True chimeric co-fragmentation, charge mis-assignment, MS1 feature-detection
  realism, and noise (sampled-from-blank, per the calibration thread) are explicit next milestones — so
  the oracle baseline's inflated ID recall is never read as real performance.
