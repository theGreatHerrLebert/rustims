# THERMO_RAW_AUTHORING — lifting the Thermo `.raw` writer from template-overlay to arbitrary authoring

> Status: **design / roadmap.** Captured after a full read of the `thermorawfile`
> crate (pinned `rev 3773aa3`) and the rustims write path. Goal: write `.raw`
> the way we write `.d` — arbitrary acquisition layout, arbitrary peaks per scan
> — rather than repainting a real template within its limits.

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
frame pointing into `analysis.tdf_bin`. The real asymmetry is three structural
properties:

| | TDF (`.d`) | Thermo (`.raw`) |
|---|---|---|
| Where the index lives | a SQLite column (`TimsId`), row-addressable | a packed contiguous binary array (ScanIndex) inside the data file |
| Payload structure | append-only, length-prefixed, self-describing blocks in `.tdf_bin` (`tdf.py:297-302, 531-536`) | contiguous packets, immediately followed by the scan-trailer / scan-params / log sections |
| Cost of resizing or adding a block | `tell()` → store the offset in that frame's row. **Zero ripple** | repack every later packet → rewrite every later ScanIndex offset → shift the trailing sections' addresses → fix the RunHeader pointers |

TDF authoring is "append a self-describing block, write its `tell()` into a row" —
no back-patching, ever. Thermo's index is packed and its payload is contiguous
with dependents behind it, so any size change back-propagates through the tail.

### The checksum is a footnote, not a wall

The Adler-32 is computed **once**, at `save()` — `recompute_checksum()` has exactly
one caller (`lib.rs:1557`); no mutator touches it. And it covers only
`min(len, 10 MiB)` with the 4-byte field zeroed (`lib.rs:34, 1827-1829`), so for
any real simulated file it's a single cheap pass over the header region at
finalize. It does not make authoring harder; it's already handled.

So the only genuine work is **the offset bookkeeping**: the packed ScanIndex plus
a handful of section pointers in the RunHeader. That is bounded, mechanical, and
the byte layouts are already decoded in the crate (see §4). It is the same *kind*
of work as the Bruker compression untangle we already shipped — fiddly, not deep.

## 3. The plan: Tier 1, then Tier 2

Sequencing still holds, but right-sized. Tier 1 alone removes the limit that
actually hurts simulation (the per-scan packet ceiling and the lossy overflow);
Tier 2 is only needed for layouts no template provides.

### Tier 1 — variable / larger peak counts within a fixed layout

**What it unlocks:** arbitrary peaks per scan against any template — no more
`overflow_cleared`, no more 120-peak MS2 cap dictated by what the real scan
happened to hold.

**What it is:** manufacture, by hand, the ripple TDF gets for free. When a packet
changes size, repack the data section and update every dependent offset. The crate
already parses all of these, so this is rewrite, not reverse-engineer:

1. Repack the contiguous data section with the new packet sizes.
2. Rewrite each `ScanIndexEntry.offset` and `data_packet_size`
   (entry layout decoded at `lib.rs:498-520`; stride `scan_index_entry_size(version)`).
3. Shift the trailing-section addresses that sit after the data —
   `scantrailer_addr`, `scanparams_addr`, `error_log_addr` — and the RunHeader
   fields (and their 32-bit mirrors) that point at them
   (RunHeader layout decoded at `read_runheader`, `lib.rs` rev≥64 path).
4. `recompute_checksum` (already done by `save`).

The scan-event and scan-params streams are *moved* but not *restructured* in
Tier 1 (same count, same strides), so this is offset arithmetic over regions the
crate already addresses. Land it behind a new `RawFile` method (e.g.
`repack_with(scan → new payload)`), keep the existing in-place fast path for the
no-resize case.

### Tier 2 — arbitrary scan count / acquisition layout

**What it unlocks:** emit a `.raw` with a layout no real template provides
(novel DIA window schemes, different MS1/MS2 cadence, different scan count).

**What it is:** grow/shrink the per-scan arrays and author new records:

1. ScanIndex: add/remove entries; set RunHeader `first_scan` / `last_scan`
   (`+8` / `+12` in the run header).
2. Scan-event array (`[scantrailer_addr+4, scanparams_addr)`, fixed stride
   `scan_event_size`): author a **complete** event record per new scan. Only a
   few offsets in the stride are currently named (`EV_MS_ORDER`, `EV_ANALYZER`,
   polarity, scan-mode, ionization, `EV_ISO_CENTER/WIDTH`, `EV_COLLISION_ENERGY`,
   activation). Filling the whole stride faithfully is the main new work.
3. Scan-params (`GenericRecord` stream, v64+): author the per-scan trailer params.
4. Re-run Tier-1 offset bookkeeping for the resized arrays + data.

**The one real unknown:** fields inside `scan_event_size` that the reverse-engineered
reader reads leniently or skips. The reader builds a correct `scan_filter` from the
named fields, but a brand-new event must satisfy whatever Thermo's own
`RawFileReader` validates. Mitigation: author against a real template as a
**reference skeleton** (copy a real event of the right MS level, then patch the
known fields) rather than zero-filling, and round-trip every output through
`RawFileReader` in CI. Prefer **generating a template with the desired method**
over from-scratch scan authoring wherever a template can express the layout.

## 4. Format knowledge already in `thermorawfile` (so Tier 1/2 is rewrite, not RE)

- RunHeader (rev ≥ 64): `first_scan@+8`, `last_scan@+12`, `scan_index_addr`,
  `data_addr`, `error_log_addr`, `scantrailer_addr`, `scanparams_addr`
  (`read_runheader`).
- ScanIndex entry: `data_packet_size`, RT, TIC, base int, base m/z, low/high m/z,
  `offset` (u64); stride `scan_index_entry_size(version)` (`lib.rs:498-520`).
- Data packet header: `profile_size@+4`, `peaklist_size@+8`, layout`@+12`,
  m/z range f32`@+32/+36`, peak `count@+40`, peaks`@+44`; centroid record width
  8B (narrow, ASTMS/older-FTMS) or 12B (wide, FTMS f64 m/z) via
  `centroid_record_width`.
- Scan-event array region + fixed stride; preamble field codes (OpenTFRaw §22).
- Scan-params `GenericRecord` stream; `RawFile.scan_params`.
- Checksum: `CHECKSUM_OFFSET`, Adler-32 seed 0 over `min(len, 10 MiB)`,
  field zeroed (`lib.rs:1823-1829`).

## 5. Recommendation

Do **Tier 1 first** — it is the high-value, low-risk move and removes the only
limit that degrades simulation fidelity today (lossy overflow + template-dictated
peak ceiling). It is offset bookkeeping over already-decoded structures, with the
checksum already handled; expect effort on the order of the Bruker compression
work, not a new RE effort.

Defer **Tier 2** until a target acquisition genuinely can't be expressed by an
existing template's method. When it's needed, build it template-skeleton-first
and gate it on `RawFileReader` round-trip tests.
