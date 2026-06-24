# TIER2_REWINDOW — re-window a real DIA template (Tier 2, increment 1)

> Status: **design / ready to implement** (next session). Scope chosen 2026-06-24:
> *re-window an existing template* (vs full-synthetic-layout or insert/remove-scans).
> Builds on the variable-length scan-event parser (thermorawfile main `69ab4c0`) and the
> Tier-1 section-graph rebuild (`repack_many`).

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

### 3a. Same window COUNT — in-place reassign (small)
Reassign each existing MS2 scan's window (center/width) to the new scheme; the scan
count and cadence are unchanged. **No event resize** — the reaction record is fixed-size,
so center/width are f64 overwrites. Mostly `set_isolation` per scan, plus consistency:
- **Mass-range / filter fields**: the preamble (and possibly a per-event mass-range,
  §scan-event `nranges` block) and the scan-index `low/high m/z` may encode the window
  redundantly — RawFileReader builds the filter string (`… [lo-hi]`) from them. Update
  them or the reader shows a *stale* window in the filter even though `GetReaction` is
  right. **This is the main correctness trap.**
- Recompute the Adler-32 once at `save` (existing).

Limitation: same count means you can shift/resize windows but not change *how many* —
so a true finer retiling (8 Th → 4 Th over the same range = 2× windows) needs 3b.

### 3b. Different window COUNT — resize the event section (medium-large)
Changing windows-per-cycle changes the **MS2 scan count** → must resize the scan-event
array, scan-params stream, scan-index, and the data section (one packet per new scan),
then relocate the section graph. This is where the real retiling lives. Reuses the
Tier-1 relocate machinery, but applied to the **event/index/params** arrays (not just
data). Author each new scan event from a **real-event skeleton** (copy a same-MS-level
event, patch center/width/ranges) — never zero-fill (per the Tier-2 §5 caveat: fields
our reader copies through may be required by RawFileReader).

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

## 5. Risks / open questions

1. **Filter-string consistency (3a)** — the biggest one. RawFileReader's per-scan filter
   `FTMS + p NSI Full ms2 X@hcd35 [lo-hi]` is built from preamble + index fields, not the
   reaction alone. Re-windowing must update every field the filter reads, or `scan_filter`
   / RawFileReader shows the old window. **Verify by reading the filter back, not just the
   reaction.**
2. **Per-event mass-range** (the `nranges`/`range` block) — does it track the isolation
   window, the scan range, or both? Decode it on a real file first (we have the grammar).
3. **Scan-index `low/high m/z`** — recompute per re-windowed scan (the repack path already
   recomputes these from peaks; here they may also bound the window).
4. **Dependent-scan / segment fields** in the preamble — leave as the template's (case 3a
   keeps the same scan structure) unless retiling (3b) changes cadence.

## 6. Validation (the gate)

- RawFileReader round-trip on the re-windowed file: open OK, all scans read.
- For each re-windowed MS2 scan: `GetReaction(0).PrecursorMass/IsolationWidth` **and** the
  **filter string `[lo-hi]`** match the new scheme (catches the §5.1 trap).
- Run the simulator on the re-windowed template → valid `.raw`, peptides now transmit
  through the new windows (sanity: non-empty MS2 count tracks the new tiling).
- Cross-reader (ProteoWizard/Sage) as a stretch.

## 7. First increment (next session)

1. Decode the per-event mass-range block on the Fusion + Astral files; confirm what the
   filter `[lo-hi]` is built from (resolve §5.1/§5.2).
2. Implement **3a** (`rewindow_in_place`): reassign MS2 windows to a declarative scheme,
   fixing filter/index fields; validate with RawFileReader (reaction **and** filter).
3. Wire a `timsim` `[dia] window_scheme` option that re-windows the template pre-authoring.
4. Then **3b** (resize) for true finer/coarser retiling, reusing `repack_many`'s relocate.

Estimated: 3a ≈ the size of the `set_isolation`/guard work; 3b ≈ the size of the
variable-length parser. Each gets its own Codex review (the established loop).
