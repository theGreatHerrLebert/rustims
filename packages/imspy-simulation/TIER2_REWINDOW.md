# TIER2_REWINDOW — re-window a real DIA template (Tier 2, increment 1)

> Status: **design / ready to implement** (next session), **Codex-reviewed** (see
> `TIER2_REWINDOW.codex-review.md`). Scope chosen 2026-06-24: *re-window an existing
> template* (vs full-synthetic-layout or insert/remove-scans). Builds on the
> variable-length scan-event parser (thermorawfile main `69ab4c0`) and the Tier-1
> section-graph rebuild (`repack_many`).
>
> **Review correction:** 3a is **same-cardinality** re-windowing (shift / widen / narrow /
> change overlap, or remap an N-window cycle onto another N-window scheme) — *not*
> arbitrary schemes. True finer tiling (8 Th → 4 Th = more windows/cycle) is **3b**. And
> the window is re-encoded in **several** places, not just the reaction — the redundancy
> audit (§5) is the crux, so the first step is a **provenance inspector** (§7), not a writer.

## 1. Goal

Take a **real DIA template `.raw`** and rewrite its MS2 **isolation windows** to a *new*
scheme, keeping everything else real (preamble, calibration, logs, method, data grid).
Unlocks "any DIA window scheme from one seed template" — e.g. turn the Fusion 8 Th
narrow-window file into a 4 Th tiling, retile the Astral 20 Th windows, shift centers,
change widths/overlap — **without** authoring a `.raw` from scratch.

This is the cheapest Tier-2 capability with real payoff: it gives the simulator
arbitrary DIA window schemes (the thing template-mutation alone can't do), reusing
machinery we already have.

## 2. Foundation already in place

- **We can read the full scan-event grammar** (`walk_variable_scan_events`, both fixed-
  and variable-length): `preamble[136] + nprec + nprec·56(reaction) + nranges +
  nranges·16(range) + calib(44+4·nparam)`. Authoring is the inverse — emit the same blocks.
- **In-place window patch exists**: `set_isolation(scan, center, width, ce)` writes
  `EV_ISO_CENTER@+140 / EV_ISO_WIDTH@+148 / EV_COLLISION_ENERGY@+156` within the reaction.
  It now reads the per-scan offset table, so it **already works on variable-length
  (Fusion) events** — confirmed: those fields decode/patch at the table offsets.
- **Section-graph rebuild**: `repack_many` / `splice_packet_and_relocate` already resize
  the data section + relocate the trailing graph. The same pattern resizes the
  *scan-event* section when window **count** changes.

## 3. Two sub-cases (different effort)

### 3a. Same window COUNT — in-place reassign (small, but NOT "reaction-only")
Reassign each existing MS2 scan's window (center/width) to the new scheme; scan count
and cadence unchanged. **No event resize** — the reaction record is fixed-size, so
center/width are f64 overwrites. But the window is **re-encoded in several places**; 3a
must update *every* copy, not just the reaction (this is the whole point of 3a — to map
that redundancy on a constrained surface). See the audit in §5. At minimum:
- reaction `EV_ISO_CENTER/WIDTH` (and CE if the scheme changes it) — the `set_isolation`
  primitive, already works on variable-length events;
- the scan-event **range block** (`nranges`/range) **iff** it carries the isolation
  window (vs a fixed fragment scan range — decode first, §5);
- **trailer-extra** precursor/isolation/selected-ion fields (RawFileReader and downstream
  tools read these directly);
- any **cached filter string** (byte-search for the old `[lo-hi]` text);
- recompute the Adler-32 once at `save` (existing).

Scope (honest): 3a is **same-cardinality retargeting** — shift, widen, narrow, change
overlap, or remap one N-window cycle onto another N-window scheme. It does **not** unlock
arbitrary schemes; true finer tiling (8 Th → 4 Th = 2× windows) is 3b. Its real value is
**diagnostic**: same scans/packets/offsets except the patched fields, so it's the clean
place to learn exactly which fields RawFileReader uses to synthesize the filter.

### 3b. Different window COUNT — resize the event section (medium-large)
Changing windows-per-cycle changes the **MS2 scan count** → must resize the scan-event
array, scan-params stream, scan-index, and the data section (one packet per new scan),
then relocate the section graph. This is where the real retiling lives. Reuses the
Tier-1 relocate machinery, but applied to the **event/index/params** arrays (not just
data). Author each new scan event from a **real-event skeleton** — and choose the
skeleton by **cycle position / window group**, not just MS level (per Codex). Hazards
beyond what `repack_many` handles:
- **Dense scan numbering** — Thermo APIs assume contiguous scan numbers with consistent
  offsets across event / index / params / trailer / status / data; all must stay aligned.
- **Per-scan metadata for new scans** — plausible RT, scan time, injection time,
  TIC/base-peak, charge — not zero packets (zero packets unproven-readable).
- **GenericRecord scan-params stream** — may be variable-length records with counts/offsets
  mirrored elsewhere, not a simple fixed array.
- **Trailer-extra** — master-scan-number, dependent-scan info, precursor/isolation fields
  per scan must be cloned/patched.
- **Scan-event segment / dependent-scan flags** — a copied DDA-style event can make a DIA
  MS2 look like a dependent scan; clean these.
- **MS1/MS2 cadence** — changing windows-per-cycle shifts cycle time, scan-number gaps
  between MS1s, controller event numbering.
- **Window-group / SWATH metadata + chromatogram indexes** — likely the hardest hidden
  dependency on newer instruments; precomputed TIC/BPC/precursor chromatograms go stale.
- **Per-section / per-record length tables** — update all, not just the global Adler-32.

## 4. API sketch

```
// thermorawfile
RawFile::set_scan_window(scan, center, width)         // ≈ set_isolation, case 3a primitive
RawFile::rewindow_in_place(assign: Fn(scan)->Window)  // 3a: reassign all MS2 windows + fix filter/index stats, validate
RawFile::rebuild_with_windows(scheme: &WindowScheme)  // 3b: resize event/index/params/data, relocate (later)

// rustdf / sim
WindowScheme { mz_lo, mz_hi, width, overlap, ms1_every }   // declarative DIA tiling
// timsim config: [dia] window_scheme = {...}  → re-window the template before authoring
```

## 5. The redundancy audit (the crux)

Separate **reaction metadata** (what `GetReaction` reads) from **scan acquisition / filter
metadata** (what `GetFilterForScanNumber`, chromatogram extraction, and downstream indexing
read). `GetReaction` agreement is *necessary*; it is **not sufficient** if a filter, a
window-group table, or a chromatogram index still points at the old window. Audit every
place the window is (or might be) encoded:

| Location | Carries the window? | Action |
|---|---|---|
| Reaction `EV_ISO_CENTER/WIDTH/CE` | yes | overwrite (`set_isolation`) |
| Scan-event `nranges`/range block | **decode first** — isolation window `[c±w/2]`, or a fixed fragment scan range (`150–2000`)? | update **only if** it equals the isolation window; else leave |
| Scan-index `low/high m/z` | maybe — observed-peak bounds vs acquisition vs filter-display | **don't blindly set to the window**; keep peak-derived unless validation proves RawFileReader uses them for the filter `[lo-hi]` |
| Cached filter string | maybe | byte-search the file for the old `[lo-hi]` text; if present, update/invalidate; if absent, confirm RawFileReader synthesizes filters from binary fields |
| Trailer-extra (precursor m/z, isolation width, mono/selected-ion m/z) | likely | patch per scan |
| DIA / SWATH window-group tables (newer instruments) | likely (hardest) | detect + update, or refuse if present and not understood |
| Status log / instrument-method text | low priority (display only) | note, usually skip |
| Chromatogram / precursor-grouped indexes | if grouped by window | stale after re-window — validate, regenerate if RawFileReader/PWiz consult them |

The first deliverable (§7.1) is a **provenance inspector** that dumps all of these per scan
so the table above is filled from *real* Fusion + Astral bytes, not assumed.

## 6. Validation (the gate)

RawFileReader round-trip (open OK, all scans read), plus **positive** DIA assertions
(the simulator's `distinct centers > 2000 && > 10% of MS2` is a DDA *rejection* rule — a
re-windowed DIA must still **pass** it; that's necessary, not the validation):
- per re-windowed MS2 scan, **all** encodings agree on the new window: `GetReaction`,
  filter `[lo-hi]`, trailer precursor/isolation, and any decoded range block;
- the **old** window values/strings are absent where expected (byte-search);
- the **per-cycle window sequence equals the requested scheme**, and every MS2 scan maps to
  exactly **one** declared window; centers repeat across cycles; widths in the DIA range;
- run the simulator on the re-windowed template → valid `.raw`, and **transmission changes
  as expected at old↔new window boundaries** (peptides near a moved edge switch windows);
- cross-reader (ProteoWizard `msaccess` / Sage) as a stretch.

## 7. First increment (next session)

1. **Build a window-provenance inspector FIRST** (before any mutation): for each MS2 scan,
   print reaction center/width, filter `[lo-hi]`, scan-event range block, scan-index
   low/high, trailer precursor/isolation fields, and any cached filter strings. Run it on
   the **Fusion** and **Astral** templates → fill in the §5 table from real bytes.
2. Implement **3a** (`rewindow_in_place`, same-cardinality): reassign MS2 windows to a
   declarative scheme, updating *every* field the inspector showed carries the window.
3. Validate per §6 — reaction **and** filter **and** trailer agree; old values gone;
   boundary transmission in the simulator changes as expected.
4. Wire a `timsim` `[dia] window_scheme` option that re-windows the template pre-authoring.
5. **Only then** design **3b** (resize): start by duplicating/removing **one** MS2 scan in a
   tiny controlled cycle (validate that single mutation in RawFileReader) before full
   retiling, reusing `repack_many`'s relocate.

Estimated: the inspector + 3a ≈ the `set_isolation`/guard work; 3b ≈ the variable-length
parser. Each gets its own Codex review (the established loop).
