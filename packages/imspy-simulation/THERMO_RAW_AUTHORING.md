# THERMO_RAW_AUTHORING — lifting the Thermo `.raw` writer from template-overlay to arbitrary authoring

> Status: **design / roadmap.** Captured after a full read of the `thermorawfile`
> crate (pinned `rev 3773aa3`) and the rustims write path, then revised against an
> independent Codex review (see `THERMO_RAW_AUTHORING.codex-review.md`). Goal:
> write `.raw` the way we write `.d` — arbitrary acquisition layout, arbitrary
> peaks per scan — rather than repainting a real template within its limits.

## 1. Where we are today

The Thermo path is **strictly template-mutation**: open a real `.raw`, overwrite
its scans in place, `save`. It cannot change the file's *shape*. Concretely the
limits, and where they bite:

| Limit | Enforced at | Behaviour |
|---|---|---|
| Scan count must equal the template | `rustdf/src/sim/astral_dispatch.rs:184` | hard error ("frame/template mismatch") |
| Scan order / MS-level must match the template | `rustdf/src/sim/acquisition.rs:276` | hard error |
| MS1 must be profile (not centroid) | `rustdf/src/sim/acquisition.rs:191` | hard error at open |
| Peaks per scan ≤ the template packet's **original byte budget** | `thermorawfile author_centroids` | overflow → scan cleared (lossy); counted as `overflow_cleared` at `astral_dispatch.rs:238` |
| Pre-cap 400 MS1 / 120 MS2 peaks | `astral_dispatch.rs:224,234` | top-N by intensity |
| MS1 m/z outside the template grid | `astral_dispatch.rs:217` | silently dropped |
| Every slot authored (Replace mode) | `acquisition.rs:345` | hard error unless `allow_partial` |

The crate write surface, for reference:

- `author_centroids(scan, &[(f64,f32)])` (`lib.rs:1400`) — variable peak count
  **up to the existing packet size**. It fails closed rather than grow:
  `new_len > old_len → "authored centroids exceed the scan's packet budget"`, and
  it explicitly refuses to move data: *"packet would overrun the next scan's
  data."* (max 65535 peaks.)
- `author_profile` (`lib.rs:910`), `overlay_profile` (`lib.rs:1110`),
  `overlay_centroids` (`lib.rs:1326`) — same in-place, same budget ceiling.
- `set_centroid_peaks` (`lib.rs:735`) / `set_profile_intensities` (`lib.rs:833`)
  — the strict same-count primitives.
- `set_isolation(scan, center, width, ce)` (`lib.rs:1671`) — rewrites an existing
  scan-event's precursor window + CE in place.
- `save(path)` (`lib.rs:1556`).

So the writer can repaint values into the existing skeleton, but the skeleton —
scan count, RT schedule, MS1/MS2 cadence, per-scan packet size — is the
template's.

## 2. Why `.d` is easy and `.raw` isn't — the *accurate* version

It is **not** "TDF has no offset table." It does. `tdf.py:210` sets
`r.TimsId = frame_start_pos` — `TimsId` *is* the offset table, one column per
frame pointing into `analysis.tdf_bin`. Nor is it "TDF has no ripple": TDF still
has **semantic** consistency constraints (frame counts, `NumScans`,
calibration/group metadata, MS/MS window tables, bin block lengths must all
agree). The real asymmetry is narrower and purely about *physical* addresses:

> **TDF localizes physical-address ripple to one SQLite column (`TimsId`); RAW
> embeds the physical layout in packed binary sections, so a size change
> back-propagates through the file's section graph.**

| | TDF (`.d`) | Thermo (`.raw`) |
|---|---|---|
| Where the index lives | a SQLite column (`TimsId`), row-addressable | a packed contiguous binary array (ScanIndex) inside the data file |
| Payload structure | append-only, length-prefixed, self-describing blocks in `.tdf_bin` (`tdf.py:297-302, 531-536`) | contiguous packets, immediately followed by the scan-trailer / scan-params / log sections |
| Cost of resizing or adding a block | `tell()` → store the offset in that frame's row. No *physical-address* ripple | repack later packets → rewrite later ScanIndex offsets → shift trailing sections' addresses → fix every RunHeader/controller pointer that crosses the change |

TDF authoring is "append a self-describing block, write its `tell()` into a row" —
the physical offset never back-patches. Thermo's index is packed and its payload
is contiguous with dependents behind it, so any size change ripples through the
tail.

### The checksum is cheap to compute — but it is not the acceptance test

The Adler-32 is computed **once**, at `save()` — `recompute_checksum()` has exactly
one caller (`lib.rs:1557`); no mutator touches it. It covers only
`min(len, 10 MiB)` with the 4-byte field zeroed (`lib.rs:34, 1827-1829`), so for
any real simulated file it's a single cheap pass over the header at finalize. So
the checksum does **not** make authoring harder.

But do not mistake a passing checksum for a valid file. The checksum proves byte
integrity under Thermo's lightweight check; it proves **nothing** about
section-graph consistency. RawFileReader can accept a checksum-valid file and then
fail later while materializing trailers, scan filters, controllers, or
chromatograms. **The acceptance test is a RawFileReader round-trip (plus
ProteoWizard / Sage / DIA-NN), not `checksum_valid()`.**

So the genuine work is the **offset bookkeeping** — the packed ScanIndex plus the
section pointers across the file's controllers. That is bounded and the byte
layouts are already decoded in the crate (§5), but it must be **evidence-driven**,
not assumed-mechanical (see the overconfidence note in §6).

## 3. Tier 0 — the padding bridge (prototype this first)

Before building the full rebuild, there is a cheaper route to **arbitrary peaks**
(not arbitrary layout): **oversized / padded packets.** Reserve `N` centroid slots
per scan, set the actual `count` smaller, leave `data_packet_size` at the padded
size. Because the current `author_centroids` already permits `new_len ≤ old_len`,
a template whose packets are pre-inflated needs **no rebuild at write time** — no
shifted trailing sections, no touched RunHeader pointers.

The nuance: you still need **one** rebuild to inflate a real template's packets, so
padding does not *avoid* Tier 1 — it **amortizes** it. Do the expensive repack
once, offline, to mint a padded template; then every simulation is a cheap in-place
write against it. This cleanly separates the costly step from the hot path.

It is valid only if downstream readers honour the internal peak `count` and ignore
the unused trailing bytes, and if a padded `data_packet_size` doesn't confuse
parsers. **De-risk before committing:** author one scan with a padded packet,
smaller actual count, valid checksum → feed it to RawFileReader, ProteoWizard,
Sage, and DIA-NN. If accepted, padding is a cheap production bridge. It is **not**
a substitute for arbitrary layouts.

## 4. Tier 1 — variable peak counts: rebuild the section-address graph

**What it unlocks:** arbitrary peaks per scan against any template — no more
`overflow_cleared`, no more template-dictated peak ceiling.

**Reframed (per review):** this is **not** "shift three trailing addresses." It is
**rebuild the complete section-address graph after the data region.** Treat every
stored pointer as data: parse all known absolute offsets *before* mutation,
classify each into a region that moves vs. one that doesn't, regenerate, then assert
a full read-back round-trip.

Verified facts that shape the work:

- **`ScanIndexEntry.offset` is data-section-relative, not file-absolute** —
  `author_centroids` computes `pkt = self.data_addr + entry.offset` (`lib.rs:1400`).
  So offsets *within* the data region are stable when the region's start is fixed;
  only entries past a resized packet shift. Still: **assert this** by deriving
  absolute = `data_addr + offset` for every entry and round-trip-checking before
  trusting it (it may be version-dependent).
- **There are multiple controllers.** The reader builds a `runheaders` vec of
  `nctrl` entries and picks the MS one (`lib.rs` controller loop). Any *other*
  controller whose sections sit after the MS data region also shifts — its
  RunHeader pointers must move too. This is the "index of indices" risk made
  concrete.

The work items (fail closed on anything not located):

1. **Repack** the contiguous data section with the new packet sizes.
2. **Rewrite per-entry** `ScanIndexEntry.offset` and `data_packet_size`
   (entry layout `lib.rs:498-520`; stride `scan_index_entry_size(version)`).
3. **Regenerate per-scan summary fields** — explicitly recompute peak count,
   profile/centroid sizes, TIC, base-peak m/z + intensity, low/high m/z. Tools
   trust the index over the packet.
4. **Move every boundary/length field**, not just section *starts*: any stored
   data-start/data-end, packet-region length, scan-index length, or total-packet-
   bytes field must be regenerated.
5. **Shift the trailing-section addresses** — `scantrailer_addr`,
   `scanparams_addr`, `error_log_addr`, plus instrument-method / status-log /
   tune-data / sample-info / chromatogram-index pointers **iff they live after the
   data region** (audit each: before → unmoved, after → moved).
6. **Audit every 32-bit *and* 64-bit mirror** of the above. Fail closed unless each
   mirror is located and updated in lockstep with its 64-bit counterpart.
7. **Update other controllers' RunHeaders** whose sections fall after the change.
8. **Treat padding/alignment as an invariant** — if packets/sections are aligned,
   naive concatenation can shift later addresses by legal-but-unexpected amounts;
   preserve the alignment rule the template used.
9. **Audit trailer `GenericRecord` internals** — fine to relocate if records carry
   only local lengths + typed values; if they hold string/table offsets,
   calibration references, or intra-stream pointers, either keep them byte-identical
   or patch the internal offsets.
10. `recompute_checksum` (already done by `save`), then **RawFileReader round-trip**
    as the real gate.

Land it behind a new `RawFile::repack_with(scan → new payload)` method; keep the
existing in-place fast path for the no-resize case. The scan-event and scan-params
streams are *relocated* but not *restructured* in Tier 1 (same count, same strides).

## 5. Tier 2 — arbitrary scan count / acquisition layout

**What it unlocks:** a `.raw` with a layout no real template provides (novel DIA
window schemes, different MS1/MS2 cadence, different scan count).

**What it is:** grow/shrink the per-scan arrays and author new records:

1. ScanIndex: add/remove entries; set RunHeader `first_scan` / `last_scan`
   (`+8` / `+12` in the run header).
2. Scan-event array (`[scantrailer_addr+4, scanparams_addr)`, fixed stride
   `scan_event_size`): author a **complete** event record per new scan.
3. Scan-params (`GenericRecord` stream, v64+): author per-scan trailer params.
4. Re-run the Tier-1 section-graph rebuild for the resized arrays + data.

**This is materially riskier than a "fill in the fields" tone suggests.** A
scan-event is not just filter-text ingredients. RawFileReader (and ProteoWizard,
the vendor DLL chromatogram path, and downstream search engines) may validate or
derive behaviour from: analyzer type, scan mode, dependent-scan relationships,
precursor metadata, activation, polarity, detector class, resolution,
AGC/injection-time-like fields, centroid/profile flags, mass ranges, method-segment
IDs, trailer-extra keys, and controller/run consistency. **Fields our reader skips
may still be required by theirs** — that's the unknown-stride trap.

**Mitigation:** author against a real template as a **reference skeleton** — copy a
real event of the right MS level, then patch only the known fields — rather than
zero-filling. Consequence to state plainly: "arbitrary layout" is then only safe
for **layouts composable from known real event archetypes**. Prefer **generating a
template with the desired method** over from-scratch scan authoring wherever a
template can express the layout. Gate everything on RawFileReader round-trip tests
in CI.

## 6. Format knowledge already in `thermorawfile` (so Tier 1/2 is rewrite, not RE)

- RunHeader (rev ≥ 64): `first_scan@+8`, `last_scan@+12`, `scan_index_addr`,
  `data_addr`, `error_log_addr`, `scantrailer_addr`, `scanparams_addr`
  (`read_runheader`); reader iterates `nctrl` controllers and selects the MS one.
- ScanIndex entry: `data_packet_size`, RT, TIC, base int, base m/z, low/high m/z,
  `offset` (u64, **data-addr-relative**); stride `scan_index_entry_size(version)`
  (`lib.rs:498-520`).
- Data packet header: `profile_size@+4`, `peaklist_size@+8`, layout`@+12`,
  m/z range f32`@+32/+36`, peak `count@+40`, peaks`@+44`; centroid record width
  8B (narrow, ASTMS/older-FTMS) or 12B (wide, FTMS f64 m/z) via
  `centroid_record_width`.
- Scan-event array region + fixed stride; preamble field codes (OpenTFRaw §22) —
  only a subset of the stride is named (`EV_MS_ORDER`, `EV_ANALYZER`, polarity,
  scan-mode, ionization, `EV_ISO_CENTER/WIDTH`, `EV_COLLISION_ENERGY`, activation).
- Scan-params `GenericRecord` stream; `RawFile.scan_params`.
- Checksum: `CHECKSUM_OFFSET`, Adler-32 seed 0 over `min(len, 10 MiB)`,
  field zeroed (`lib.rs:1823-1829`).

## 7. Calibration on confidence

- **Don't over-claim "bounded, mechanical, like the Bruker compression work."** It
  is the same *kind* of work (rewrite over decoded structures, not new RE), but RAW
  section graphs can carry duplicate directories and version-specific mirrors. Frame
  Tier 1 as **evidence-driven**: parse all known absolute offsets before *and*
  after, classify every pointer into moved/unmoved, regenerate summaries, fail closed
  on anything unlocated, and test against the official RawFileReader plus downstream
  tools — not "shift a few addresses."
- **Don't under-claim either.** With that discipline Tier 1 is very achievable; the
  format facts are already in the crate, the checksum is handled, and offsets are
  data-relative (limiting the blast radius).

## 8. Recommendation

1. **Prototype Tier 0 (padding) first** — one padded scan through
   RawFileReader/ProteoWizard/Sage/DIA-NN. If it passes, it's a cheap production
   bridge to arbitrary peaks while Tier 1 lands.
2. **Build Tier 1** as a section-graph rebuild (§4). High-value, removes the lossy
   overflow + template peak ceiling, and produces the machinery Tier 2 needs anyway.
   It also mints the padded templates Tier 0 relies on.
3. **Defer Tier 2** until a target acquisition genuinely can't be expressed by an
   existing template's method; build it template-skeleton-first and gate on
   RawFileReader round-trip tests.
