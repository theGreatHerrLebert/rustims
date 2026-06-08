Reading additional input from stdin...
OpenAI Codex v0.136.0
--------
workdir: /scratch/timsim-demo/SUBMISSION/rustims
model: gpt-5.5
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: medium
reasoning summaries: none
session id: 019ea7ef-4649-7ed0-bd1c-e13f500a0128
--------
user
This is v2 of a vendor-neutral AcquisitionScheme design for a DIA mass-spec simulator (TimSim: Bruker timsTOF today, extending to Thermo Orbitrap Astral + SCIEX ZenoTOF). You reviewed v1; v2 applied your feedback: ordered cycle of events (Ms1Event/DiaMs2Event) instead of flat windows; DiaGeometry splits mobility out of the m/z IsolationWindow; CollisionEnergyPolicy enum (incl. Unknown) instead of scalar CE; MS1 promoted to a first-class event; scheme version + Provenance; an explicit to_bruker_tables() adapter + golden tests; and two usage modes (reference-derived layout+noise, and injected/user schemes). Context to trust: the Thermo/SCIEX Rust readers exist and are oracle-verified; thermorawfile::scan_event works per-scan on Astral (uniform 280-byte stride) so from_thermo_raw is NOT blocked; the ThermoRawWriter now authors MS2 isolation/CE via set_isolation (review item #6, fixed).

Review focused: 1) does v2's cycle-of-events model now correctly capture the timsTOF frame↔(mobility window group) relationship AND the linear Astral/SCIEX sequence — any remaining representational gap? 2) is the RepeatPolicy/RT-modeling sufficient or under-specified? 3) the overlay (real⊕sim) writer mode is called the biggest remaining piece — is the design pointing at the right primitive, and what are the concrete hazards (recomputing stats, packet budget, profile vs centroid)? 4) the Bruker to_bruker_tables() + golden-test migration — anything that will still break backward compat? 5) any NEW problems v2 introduced, or v1 issues not fully resolved. Be concrete and specific; cap ~700 words.

<stdin>
# AcquisitionScheme — vendor-neutral DIA acquisition design for TimSim (v2)

**Status:** design draft v2 (revised after Codex review of v1)
**Scope:** the *input/design* side of multi-instrument TimSim — the acquisition
schedule the simulator generates against — complementary to the already-built
`AcquisitionWriter` (output side, `rustdf/src/sim/acquisition.rs`).

> **Changes vs v1** (from review): model an explicit *ordered cycle of events*
> instead of a flat window list; split ion-mobility geometry out of the generic
> isolation window; make collision energy a *policy*, not a scalar; promote MS1
> to a first-class event; add scheme versioning + provenance + validation; spell
> out a Bruker backward-compat *adapter* with golden tests (not a parity claim);
> and resolve several open questions (below). Also corrects v1's assumption that
> Thermo MS2 scan events are hard to locate — they are not (see §2).

---

## 1. Problem & the two usage modes

TimSim must generate scans for the **target instrument's acquisition schedule**
— isolation windows, cycle structure, collision-energy plan, and (timsTOF only)
ion-mobility partitioning. These differ per vendor (timsTOF DIA-PASEF: 2-D m/z×
mobility; Orbitrap Astral: 1-D narrow m/z, no mobility; SCIEX ZenoTOF: 1-D
variable SWATH, no mobility, Zeno).

Two first-class usage modes (both must be supported):

- **Reference-derived (timsim-style):** a real reference run supplies *both* the
  acquisition layout *and* the real background for noise. The reference is used
  twice — extract the scheme from it, and overlay simulated peaks onto its real
  signal (real⊕sim). This is the established TimSim pattern
  (`use_reference_ds_layout` + `add_real_data_noise`), generalized across vendors.
- **Injected (user-authored):** a programmatic or CSV scheme with no reference
  dependence — synthesize a custom window plan from scratch.

## 2. Current state

`TimsTofAcquisitionBuilderDIA` (`packages/imspy-simulation/.../acquisition.py`)
is timsTOF-coupled: it holds a pandas `dia_ms_ms_windows` table
(`window_group, scan_start, scan_end, isolation_mz, isolation_width,
collision_energy`; `scan_start/end` are **mobility scan ranges**), sourced from a
CSV or by copying a real Bruker reference `.d`'s `DiaFrameMsMsWindows`. It also
derives `precursor_every`, builds `frame`/`scan` tables, and persists
`dia_ms_ms_windows` + `dia_ms_ms_info` (frame→group) → the output `.d`.

**Reader inventory (built + oracle-verified this session):**
- **Bruker** — `reference_ds.dia_ms_ms_windows` (Python, today).
- **Thermo** — `thermorawfile::scan_event(scan)` → `ScanEvent { ms_order,
  analyzer, isolation_center, isolation_width, collision_energy }`. **Verified to
  work per-scan on a real Astral file (uniform 280-byte event stride);** scan 2
  reads center 490.473 / width 20.009 / CE 25.0, matching the RawFileReader
  oracle. So walking one MS2 cycle recovers the exact Astral window schedule —
  **not blocked** (corrects v1 open-Q6).
- **SCIEX** — `sciexwiff` → `SWATHMethod` ~60 variable windows + TOF calibration.

**Known existing bug to fix during migration:** in
`TimsTofAcquisitionBuilderDIA`, `num_frames` is computed in the base `__init__`
from the passed `gradient_length/rt_cycle_length` *before* `use_reference_ds_layout`
overwrites `rt_cycle_length`, so the frame count can be stale.

## 3. Model — an ordered cycle of events

Lives in Rust (`rustdf/src/sim/scheme.rs`), exposed via `imspy_connector`.

```rust
pub struct AcquisitionScheme {
    pub version: u16,                 // schema version for forward-compat
    pub instrument: InstrumentKind,   // TimsTofDia | OrbitrapAstral | SciexZenoTof
    pub cycle: Vec<AcquisitionEvent>, // one DIA cycle, in acquisition order
    pub repeat: RepeatPolicy,         // how the cycle tiles the gradient
    pub mz_range: (f64, f64),
    pub provenance: Provenance,       // where each field came from
}

pub enum AcquisitionEvent {
    Ms1(Ms1Event),
    DiaMs2(DiaMs2Event),
}

pub struct Ms1Event {
    pub analyzer: Analyzer,           // FTMS | ASTMS | TOF | ITMS
    pub data_mode: DataMode,          // Profile | Centroid
    pub mz_range: (f64, f64),
    pub duration_s: Option<f64>,      // transient/dwell if known
}

pub struct DiaMs2Event {
    pub window_group: u32,            // events sharing a group co-occur (timsTOF frame)
    pub isolation: IsolationWindow,   // m/z ONLY
    pub collision_energy: CollisionEnergyPolicy,
    pub geometry: DiaGeometry,        // mobility partitioning (or none)
    pub analyzer: Analyzer,
    pub data_mode: DataMode,
    pub duration_s: Option<f64>,
}

pub struct IsolationWindow {          // m/z only; boundary convention is half-open [lo, hi)
    pub center_mz: f64,
    pub width_mz: f64,
}

pub enum DiaGeometry {
    MzOnly,
    TimsMobility { scan_start: u32, scan_end: u32 }, // Bruker-grid coords (need ref calibration)
}

pub enum CollisionEnergyPolicy {
    Fixed(f64),
    PerWindow(f64),
    Linear { intercept: f64, slope_per_mz: f64 }, // CE = intercept + slope*center_mz
    Unknown,                                       // extraction couldn't recover it
}

pub enum RepeatPolicy {
    FixedCycleTime { cycle_time_s: f64, gradient_length_s: f64 },
    // room for staggered / variable-cycle schemes later
}

pub struct Provenance {              // per-source attribution + free-form notes
    pub source: SchemeSource,        // ExtractedBruker | ExtractedThermo | ExtractedSciex | UserCsv | Programmatic
    pub notes: String,
}
```

Why this shape (review points):
- **Ordered events**, not a flat window list, so a timsTOF MS2 *frame* (several
  mobility slices sharing a `window_group`) and a linear Astral/SCIEX MS1→MS2…
  sequence are both representable, plus per-event timing/analyzer.
- **Geometry split out**: `IsolationWindow` is strictly m/z; mobility lives in
  `DiaGeometry::TimsMobility`, so "mobility on Astral" is unrepresentable.
- **CE policy, not scalar**: never silently invent SCIEX rolling CE — extraction
  yields `Unknown` and the caller must supply a model, unless an explicitly
  labeled default-sim mode is chosen.
- **MS1 is first-class** (analyzer/data_mode/mz_range/duration), since it drives
  cadence and the writer's MS1 representation.

## 4. Extractors (populate from a real run)

```rust
impl AcquisitionScheme {
    pub fn from_bruker_d(path: &str) -> io::Result<Self>;      // DiaFrameMsMsWindows (+TimsMobility geometry)
    #[cfg(feature = "thermo")]
    pub fn from_thermo_raw(path: &str) -> io::Result<Self>;    // walk one MS2 cycle via scan_event()
    #[cfg(feature = "sciex")]
    pub fn from_sciex_wiff(path: &str) -> io::Result<Self>;    // SWATHMethod; CE => Unknown (rolling)
    pub fn from_window_table(rows, instrument) -> Self;        // injected / CSV
}
```

Each sets `provenance.source` accordingly. `from_thermo_raw` reads the first
MS1→(MS2…)→nextMS1 span, one `DiaMs2Event` per MS2 scan event, CE = `Fixed`/
`PerWindow` as observed, geometry `MzOnly`.

## 5. Relationship to the writer (output)

- `AcquisitionScheme` (in) and `AcquisitionWriter` (out) are the two halves. A
  reference round trip: `from_thermo_raw(template)` → generate → `ThermoRawWriter`
  (same template) → `.raw`.
- **#6 (fixed):** `ThermoRawWriter` now authors a descriptor's MS2 isolation
  center/width/CE into the scan event (`set_isolation`), not just the peaks.
- **Replace vs overlay (open design):** the writer currently *replaces* template
  signal. The reference-derived noise mode (§1) needs an **overlay** writer mode —
  keep the real template profile/centroids and *add* simulated peaks — to realize
  real⊕sim. This is the main remaining writer feature.
- **Validation:** add `writer.validate_scheme(&scheme)` — for template-mutation
  writers, reject a scheme whose event count/order/types don't fit the template
  (e.g. more MS2 windows than the template cycle provides), with a clear error
  rather than silent truncation.

## 6. Bruker backward-compatibility (explicit adapter + golden tests)

Migration must not change existing `.d` output. The existing code uses
`window_group` to *assign frames* (not merely label windows), with specific
column names/integer types, `scan_start/scan_end` inclusivity, CE rounding, and
`dia_ms_ms_info` frame→group sequencing.

- Add `scheme.to_bruker_tables() -> (dia_ms_ms_windows, dia_ms_ms_info)` that
  reproduces those exact columns/types/ordering.
- Keep the current Python `TimsTofAcquisitionBuilderDIA` constructors
  (CSV + `use_reference_ds_layout`) as a compatibility wrapper over the scheme.
- **Golden tests:** byte-for-value compare both SQLite tables against the current
  path on a reference dataset. Fix the `num_frames`/`rt_cycle_length` ordering bug
  as part of this (or pin it with a regression test).

## 7. Boundary / location

`AcquisitionScheme` + extractors in `rustdf/src/sim`, exposed as
`PyAcquisitionScheme` via `imspy_connector`. Python `TimsTofAcquisitionBuilderDIA`
consumes the scheme through the binding. Bruker extraction may *stay Python*
initially (it works), provided it produces the same versioned scheme DTO and
passes the golden tables (deferrable per review).

## 8. Open questions (revised)

- **Resolved since v1:** MS1 is scheme-level (now an `Ms1Event`); SCIEX rolling CE
  → `CollisionEnergyPolicy::Unknown` + required user model; Thermo MS2 scan-event
  location works (uniform stride) so `from_thermo_raw` is unblocked.
- **Still open (blockers for full fidelity):**
  1. **Overlay writer mode** (real⊕sim) — needed for reference-derived noise;
     biggest remaining piece. (How to add sim peaks to a real profile/centroid
     packet within budget, recompute stats.)
  2. **RT / duty-cycle modeling** — is `RepeatPolicy::FixedCycleTime` enough, or do
     we need per-event durations to model real cycle time / overlap in RT?
  3. **SCIEX CE source** — confirm whether any per-window CE is recoverable from
     the `.wiff` method beyond `SWATHMethod` (else require a user model).
- **Specify-now, implement-later:** overlapping windows need only a documented
  boundary convention (half-open `[lo, hi)`) + preserved acquisition order; no
  special isolation algorithm.

## 9. Test plan

- `from_thermo_raw(astral_template)`: N MS2 windows, monotonic centers, CE policy
  `Fixed(25)`, one `Ms1Event` per cycle, all geometry `MzOnly`.
- `from_sciex_wiff(zenotof)`: ~60 windows covering 399.5–899.9, CE `Unknown`.
- `from_bruker_d(ref)` → `to_bruker_tables()` golden-equal to today's Python path.
- End-to-end: `from_thermo_raw(template)` → generate 1×MS1 + N×MS2 → `ThermoRawWriter`
  → read back via thermorawfile + Thermo oracle (peaks, isolation, CE).
- Overlay mode (once built): authored peaks present *and* template background
  retained.
</stdin>
codex
## Review Findings

**1. The timsTOF grouping is still implicit, not structurally correct.**  
Repeated `window_group` values on serial `DiaMs2Event`s do not express that several mobility windows belong to one physical frame. Consumers could incorrectly assign each event its own duration, timestamp, or output frame.

Make the physical acquisition unit explicit:

```rust
AcquisitionEvent::DiaMs2Frame {
    windows: Vec<DiaWindow>,
    analyzer: Analyzer,
    data_mode: DataMode,
    duration_s: Option<f64>,
}
```

For Astral/SCIEX, each frame/scan contains one window. For timsTOF, one event contains multiple mobility windows. `window_group` can remain as a compatibility identifier, but should not define simultaneity indirectly.

**2. `RepeatPolicy::FixedCycleTime` is insufficiently specified.**  
It does not define:

- Cycle start/phase relative to gradient start.
- Full-cycle count and treatment of a trailing partial cycle.
- Whether `cycle_time_s` is authoritative or derived from event durations.
- Dead time, inter-event gaps, or parallel timsTOF mobility windows.
- Reference runs with cycle-time drift, skipped scans, calibration frames, or staggered cycles.

At minimum, specify `start_time_s`, trailing-cycle policy, and timing precedence. Prefer an explicit event offset/duration schedule plus a repetition rule. Preserve observed timestamps or cycle durations during reference extraction; otherwise layout-derived simulation loses real RT sampling.

**3. Overlay should operate on decoded spectrum/frame data, not raw packet concatenation.**  
The correct primitive is approximately:

```rust
merge(template_signal, simulated_signal, data_mode, merge_policy)
    -> rewritten_signal_and_statistics
```

It must run per matched physical scan/frame after scheme-to-template validation.

Concrete hazards:

- **Profile:** simulated peaks must be evaluated onto the native m/z grid or combined before profile encoding. Appending centroids to a profile packet is invalid.
- **Centroid:** merge nearby peaks using a declared tolerance, sum intensities, sort, and potentially re-centroid. Blind concatenation creates duplicate centroids.
- **Statistics:** recompute TIC, base-peak intensity/m/z, point count, m/z bounds, intensity bounds, and any vendor summary fields.
- **Packet budget:** account for compressed size, offsets, alignment, indexes, and checksums. Define overflow behavior explicitly; silent peak truncation is unacceptable.
- **Template consistency:** retain acquisition metadata but rewrite all size/count pointers atomically.
- **Semantics:** real analyte signal is retained along with noise. Document that overlay is not a noise-only background model and may duplicate analytes represented by the simulation.

**4. Bruker compatibility needs stronger invariants than table golden tests alone.**  
Potential breakpoints remain:

- `scan_end` is documented as half-open globally, while the existing Bruker table may use a different convention. The adapter must encode that conversion explicitly.
- Frame grouping must derive from physical grouped events, not accidental adjacency or numeric equality alone.
- Define deterministic ordering, SQLite affinities, integer widths, floating-point/CE rounding, and group-ID allocation.
- `dia_ms_ms_info` depends on total frame count and cycle phase, so `to_bruker_tables()` needs run timing/frame context, not merely the scheme as currently shown.
- Fixing the stale `num_frames` bug intentionally changes affected historical output. Golden tests should distinguish legacy compatibility from corrected behavior.

Compare ordered rows and declared SQLite types; “byte-for-value” is ambiguous because SQLite physical bytes are not stable.

**5. New and unresolved model issues.**

- `Provenance` is described as per-field attribution but contains only one scheme-level source and notes.
- `Fixed(f64)` and `PerWindow(f64)` are structurally indistinguishable. Use `Fixed` at scheme/group level or make `PerWindow` the event value.
- `TimsMobility` requires calibration, but the scheme carries neither a calibration identity nor a requirement binding it to a reference.
- Extracting only the first Thermo cycle can miss staggered windows or later method segments. Validate repeated cycles and report divergence.
- Versioning needs defined serialization behavior for unknown versions/enum variants; a `u16` alone does not provide forward compatibility.
- Validation should cover finite/positive widths, ranges, scan ordering, overlapping mobility intervals, analyzer/instrument compatibility, and consistency between cycle time and event timing.

The main remaining representational correction is therefore explicit physical-event grouping. Once that is fixed, the same model can cleanly represent timsTOF frames and linear Astral/SCIEX scans.
tokens used
4,961
## Review Findings

**1. The timsTOF grouping is still implicit, not structurally correct.**  
Repeated `window_group` values on serial `DiaMs2Event`s do not express that several mobility windows belong to one physical frame. Consumers could incorrectly assign each event its own duration, timestamp, or output frame.

Make the physical acquisition unit explicit:

```rust
AcquisitionEvent::DiaMs2Frame {
    windows: Vec<DiaWindow>,
    analyzer: Analyzer,
    data_mode: DataMode,
    duration_s: Option<f64>,
}
```

For Astral/SCIEX, each frame/scan contains one window. For timsTOF, one event contains multiple mobility windows. `window_group` can remain as a compatibility identifier, but should not define simultaneity indirectly.

**2. `RepeatPolicy::FixedCycleTime` is insufficiently specified.**  
It does not define:

- Cycle start/phase relative to gradient start.
- Full-cycle count and treatment of a trailing partial cycle.
- Whether `cycle_time_s` is authoritative or derived from event durations.
- Dead time, inter-event gaps, or parallel timsTOF mobility windows.
- Reference runs with cycle-time drift, skipped scans, calibration frames, or staggered cycles.

At minimum, specify `start_time_s`, trailing-cycle policy, and timing precedence. Prefer an explicit event offset/duration schedule plus a repetition rule. Preserve observed timestamps or cycle durations during reference extraction; otherwise layout-derived simulation loses real RT sampling.

**3. Overlay should operate on decoded spectrum/frame data, not raw packet concatenation.**  
The correct primitive is approximately:

```rust
merge(template_signal, simulated_signal, data_mode, merge_policy)
    -> rewritten_signal_and_statistics
```

It must run per matched physical scan/frame after scheme-to-template validation.

Concrete hazards:

- **Profile:** simulated peaks must be evaluated onto the native m/z grid or combined before profile encoding. Appending centroids to a profile packet is invalid.
- **Centroid:** merge nearby peaks using a declared tolerance, sum intensities, sort, and potentially re-centroid. Blind concatenation creates duplicate centroids.
- **Statistics:** recompute TIC, base-peak intensity/m/z, point count, m/z bounds, intensity bounds, and any vendor summary fields.
- **Packet budget:** account for compressed size, offsets, alignment, indexes, and checksums. Define overflow behavior explicitly; silent peak truncation is unacceptable.
- **Template consistency:** retain acquisition metadata but rewrite all size/count pointers atomically.
- **Semantics:** real analyte signal is retained along with noise. Document that overlay is not a noise-only background model and may duplicate analytes represented by the simulation.

**4. Bruker compatibility needs stronger invariants than table golden tests alone.**  
Potential breakpoints remain:

- `scan_end` is documented as half-open globally, while the existing Bruker table may use a different convention. The adapter must encode that conversion explicitly.
- Frame grouping must derive from physical grouped events, not accidental adjacency or numeric equality alone.
- Define deterministic ordering, SQLite affinities, integer widths, floating-point/CE rounding, and group-ID allocation.
- `dia_ms_ms_info` depends on total frame count and cycle phase, so `to_bruker_tables()` needs run timing/frame context, not merely the scheme as currently shown.
- Fixing the stale `num_frames` bug intentionally changes affected historical output. Golden tests should distinguish legacy compatibility from corrected behavior.

Compare ordered rows and declared SQLite types; “byte-for-value” is ambiguous because SQLite physical bytes are not stable.

**5. New and unresolved model issues.**

- `Provenance` is described as per-field attribution but contains only one scheme-level source and notes.
- `Fixed(f64)` and `PerWindow(f64)` are structurally indistinguishable. Use `Fixed` at scheme/group level or make `PerWindow` the event value.
- `TimsMobility` requires calibration, but the scheme carries neither a calibration identity nor a requirement binding it to a reference.
- Extracting only the first Thermo cycle can miss staggered windows or later method segments. Validate repeated cycles and report divergence.
- Versioning needs defined serialization behavior for unknown versions/enum variants; a `u16` alone does not provide forward compatibility.
- Validation should cover finite/positive widths, ranges, scan ordering, overlapping mobility intervals, analyzer/instrument compatibility, and consistency between cycle time and event timing.

The main remaining representational correction is therefore explicit physical-event grouping. Once that is fixed, the same model can cleanly represent timsTOF frames and linear Astral/SCIEX scans.
