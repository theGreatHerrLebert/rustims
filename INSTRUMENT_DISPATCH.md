# Instrument Dispatch — separating physical appearance from device recording

> **v3** — revised after a second Codex pass on v2. v2 folded in the first
> review's 8 structural catches; v3 fixes the contradictions that rewrite
> *introduced* and pins two load-bearing contracts P1 would otherwise freeze:
> mobility ownership (§1/§2.3 resolved — DB stores CCS, profile owns the gas/temp
> conversion), the **identity** and **intensity-conservation** contracts (§3.5),
> the `RenderedEvent` coordinate/intensity stage (§3.2), the geometry/calibration
> de-duplication (§3.1), the stepped projector loop for DDA (§3.4), and a P1 that
> adds *parallel* scalar entities rather than removing the legacy ones (§6).

## 1. Principle — three layers, not two

Three independent layers (v1 fused the middle into the first; too broad):

1. **Analyte chemistry** (intrinsic): sequence, modifications, elemental
   composition, exact isotope m/z, **CCS** (with conformer/model provenance and a
   physical spread).
2. **Appearance under declared conditions** (experiment/source/chromatography):
   retention-time profile, charge-state distribution + ion yield, the
   fragmentation response. **Note: ion mobility is *not* stored here** — see §2.3.
3. **Instrument recording**: acquisition schedule, sampling geometry, m/z
   encoding, detector/peak-shape model, file format.

The DB stores layers 1–2 plus an `experiment_conditions` record. That record is
**provenance for the generated RT/yield/fragmentation only** — it is *not* the
authoritative mobility environment at render time (that belongs to the profile,
§2.3). A DB is reusable across instruments within its declared layer-2 conditions.

## 2. Current state

### 2.1 What is layer-1/2 and roughly the right shape

| Table | Quantity | Layer | Representation |
|-------|----------|-------|----------------|
| `peptides` | RT μ/σ/λ (EMG) | 2 (LC) | scalars ✓ |
| `ions` | charge, m/z | 1–2 | scalars ✓ |
| `ions` | **CCS** + spread | 1 | scalars — **stored as CCS, not 1/K0** (§2.3) |
| `ions` | isotope envelope | 1 (m/z) + recording (intensities) | per-ion vector — §2.3 |
| `fragment_ions` | fragment spectra | 2, **acquisition-leaking today** | per-ion vector — §2.3 |

### 2.2 The three coupling points (device recording fused into the store)

1. `simulate_frame_distributions_emg` → `frame_occurrence[_start/end]`,
   `frame_abundance`: RT EMG **sampled onto Bruker frame indices**.
2. `simulate_scan_distributions_with_variance` → `scan_occurrence`,
   `scan_abundance`: IM Gaussian **sampled onto Bruker scan indices**.
3. `build_acquisition` bootstraps the coordinate system from a Bruker `.d`.

These per-ion **variable-length vectors are JSON-serialized in SQLite** and
dominate DB size + JSON encode/parse cost. They are device-sampled products that
should not be persisted. Moving projection to render-time deletes them — the
architectural cut and the space win are one refactor.

### 2.3 Mobility ownership, isotope intensities, and fragments (resolved)

**Mobility ownership** (resolved, was open-Q3): the contract is split, not stored
once.
- **DB (layer 1):** CCS, CCS provenance/model, and a physical conformer spread.
- **Profile (layer 3):** drift **gas, temperature, charge → 1/K0** conversion
  (the CCS→1/K0 conversion already takes gas mass + temperature,
  `mscore/src/chemistry/formulas.rs:64`). A profile with a different gas
  recomputes mobility correctly.
- `experiment_conditions`: provenance only, **not** the render-time mobility env.

This means `inv_mobility_gru_predictor`/`..._std` are *derived per profile*, not
persisted as the authoritative value. (Migration may cache a 1/K0 under a named
reference environment for the legacy adapter, clearly labelled as derived.)

**Isotope intensities / fragments:** only exact isotope **m/z** is layer-1.
Observed intensities are shaped by transmission, detector response, saturation,
centroiding (layer 3). `fragment_ions` today are generated **only for transmitted
precursors at quantized CE** (`simulate_fragment_intensities.py:270`,
`rustdf/src/sim/dia.rs:570`), so Bruker quad transmission + CE leak in.

Target (reached in **P5**, not P1): a **fragmentation response model** keyed by
`(activation_type, collision_energy)`, generated independently of
acquisition-window transmission. **Until P5, existing fragments stay
Bruker-transmission-conditioned and must carry a `provenance`/legacy marker** —
they must not be mislabeled as the target response model.

## 3. Target architecture

```
  layer 1–2 store (DB)               InstrumentProfile (Rust)              output
 ┌──────────────────────────┐   ┌────────────────────────────────┐   ┌────────────┐
 │ chemistry: seq, mods, CCS │   │ AcquisitionSchedule  [HAVE+ext] │   │ Bruker .d  │
 │ RT μ/σ/λ, charge, m/z     │──►│ AcquisitionController (DDA only)│──►│ Thermo .raw│
 │ isotope composition       │   │ SamplingGeometry (modality+grid)│   │ SCIEX .wiff│
 │ fragmentation response*   │   │ Calibration (mobility + m/z)    │   │ mzML       │
 │ experiment_conditions     │   │ DetectorModel (peak/quantize)   │   └────────────┘
 └──────────────────────────┘   │ Writer (format)                 │
   (*legacy-marked pre-P5)      └────────────────────────────────┘
```

Projector: **store + profile → ordered `RenderedEvent`s → writer.**

### 3.1 `SamplingGeometry` — modality + physical grid (no coordinate conversion)

Coordinate conversion lives **only** in `Calibration.mobility` (§3.2); geometry
holds physical separation + discretization, not native indices.

```rust
pub struct SamplingGeometry {
    modality: MobilityModality,        // None | Tims | DriftTube | Faims
    acceptance: MobilityRange,         // physical range admitted
    grid: MobilityGrid,                // physical bin edges / integration intervals
    overrides: EventOverrides,         // concrete per-event lookup (event_id -> grid/range)
}
```

`MobilityModality::None` (Astral/Orbitrap): the projector **marginalizes the
mobility distribution, conserving total signal** — it does not invent "one scan."
`overrides` is a concrete lookup keyed by `event_id`, not a comment.

### 3.2 `Calibration` + the `RenderedEvent` contract

```rust
pub struct Calibration {
    mobility: MobilityCalib,   // physical mobility <-> native scan index (sole owner)
    mz: MzCalib,               // exact m/z <-> native coord (tof | frequency)
}
```

Each calib exposes `forward`/`inverse` with valid domains, units, monotonicity
checks; per-scan calibration where the format has it. Exact isotope m/z is
upstream (layer 1); m/z error, peak width, quantization, centroid bias live in
`DetectorModel` (layer 3).

**`RenderedEvent` must declare its stage** (else writers can't share it):

```rust
pub enum RenderedEvent {
    Scan(RenderedSpectrum, ScanMeta),
    MobilityFrame(Vec<(ScanIndex, RenderedSpectrum)>, FrameMeta),
}
pub struct RenderedSpectrum {
    peaks: MzSpectrum,
    coords: MzCoordSpace,     // Physical | Native(tof|freq)
    mode: DataMode,           // Profile | Centroid
    detector_applied: bool,   // have DetectorModel effects been applied yet?
    intensity: IntensityUnit, // see §3.5 conservation rule
}
```

The projector emits `Physical` / pre-detector spectra; `DetectorModel` then
produces the native-coordinate, detector-applied spectrum the writer consumes.
Freezing this struct is the gate for the **P2** interface (not P1).

### 3.3 Time projection over event intervals (corrected) + RT-support policy

The existing kernel integrates `[time - cycle_length, time]`
(`mscore/src/algorithm/utility.rs:198`) — wrong for unequal durations / dead
time / overlap. Each schedule event carries an explicit **start/end exposure
interval**; integrate the EMG over that interval. Kernels are **deterministic
CDF integrations** (`utility.rs:407`) — no RNG.

**RT-support truncation policy (must be fixed before P2):** an analyte
contributes to an event iff the EMG CDF mass in the event interval exceeds a
tail tolerance `ε`; the per-analyte RT-support bound is `[F⁻¹(ε/2), F⁻¹(1-ε/2)]`
widened for the EMG tail, so long-tailed / overlapping elutions are never
silently dropped. The replacement RT-support index (§4) is derived from these
bounds, conservatively.

### 3.4 Stepped projector loop (DDA fits without redesign)

The projector API is a **loop**, not a one-shot over a completed schedule, so the
DDA controller (P4) drops in without reworking P2:

```
loop {
    let event = schedule.next(state)?;          // DIA: static; DDA: controller-decided
    let rendered = project(store, profile, event);
    let observed = detector.observe(rendered);  // feeds controller
    controller.update(observed);                 // no-op for DIA
    emit(rendered);
}
```

For DIA, `schedule.next` ignores `state` and `controller.update` is a no-op — so
P2 can ship the static path while the API already supports the P4 feedback
(render MS1 → observe → decide → allocate next `event_id`/interval).

### 3.5 Identity + intensity contracts (new, load-bearing)

**Identity** — seeds and reproducibility require IDs that survive ordering,
batching, schema migration, and controller decisions:
- `analyte_id`: stable across migrations (content-addressed on chemistry, not row
  order).
- `profile_id`: stable hash of the InstrumentProfile.
- `event_id`: deterministic even when DDA *creates* events dynamically — derived
  from `(cycle_index, event_index_within_cycle)` plus a controller-decision
  counter, not from emission order.

Counter-based seeding: `hash(run_seed, profile_id, analyte_id, event_id,
noise_component)` — no traversal-order / thread-local RNG.

**Intensity conservation** — one declared conserved quantity flows end-to-end:
ion **yield** (total ions) → time integration (fraction of yield in the event
interval) → mobility integration (fraction in the scan) → transmission (fraction
passed) → detector response (counts) → centroiding (preserves area). Each stage
is a documented multiplicative/area-preserving transform on this quantity;
`IntensityUnit` (§3.2) names which point of the chain a `RenderedSpectrum` is at.
Mobility marginalization for `None` is then unambiguous: sum the scan fractions.

## 4. DB schema delta + removal scope

**Add:** `experiment_conditions` (LC/source/model provenance — *not* mobility
env); schema **version**; `provenance`/legacy marker on `fragment_ions`.
**Store CCS** (+ spread/provenance) instead of authoritative `1/K0`.
**Remove (P4, after parallel path proven):** `scan_occurrence`,
`scan_abundance` (`ions`); `frame_occurrence[_start/end]`, `frame_abundance`
(`peptides`).

Removal touches more than `assemble_frames`:
- `rustdf/src/sim/containers.rs:64` requires the vectors,
- `rustdf/src/sim/handle.rs:74` deserializes them,
- `rustdf/src/sim/handle.rs:242` lazy-loads by `frame_occurrence_start/end`,
- `dda_selection_scheme.py:82` consumes both projections.

So P1 introduces **parallel scalar-native entities + legacy adapters** (the old
entities keep working); the lazy index is replaced by the §3.3 RT-support index;
column removal is **P4**, after the DDA controller migrates.

## 5. Parity & determinism contract

Byte-identical `.d` is not the first gate. Split:
1. **Numerical parity (gate 1, DIA/static-schedule):** projected
   indices/abundances and ordered `RenderedEvent` spectra equal the legacy path
   within tolerance.
2. **Writer/byte parity (gate 2, separate):** deterministic writer test (SQLite
   ordering/metadata/compression can differ even when content matches; reuse the
   TOF-writer parity harness).

Determinism per §3.5 (counter-based seeding). RNG enters only via optional
scan/RT/m/z noise + fragment assembly — the projection itself is deterministic.

## 6. Phased plan (re-ordered per both reviews)

- **P0 (this doc):** agree the three-layer cut + the §3.5 contracts. ← here
- **P1 — foundations (additive, no behavior change):** schema **versioning** +
  `experiment_conditions`; **CCS** storage + legacy 1/K0 adapter; **identity +
  intensity contracts** (§3.5); **parallel** scalar-native Rust entities +
  legacy adapters (legacy entities/readers untouched).
- **P2 — projection (DIA / non-DDA):** event-**interval** time projection +
  mobility projection as `InstrumentProfile` methods over the existing kernels;
  the **stepped projector loop** (§3.4, static path); freeze `RenderedEvent`
  (§3.2); migrate **non-DDA** Rust paths (`containers.rs`, `handle.rs`, lazy
  index → RT-support); `assemble_frames` reads occurrence from the profile with
  DB fallback. Generation **keeps writing** the legacy vectors through P4.
- **P3 — DIA parity:** numerical parity (gate 1) then deterministic writer parity
  (gate 2) for Bruker `.d`.
- **P4 — DDA controller + removal:** migrate DDA to `AcquisitionController` on the
  §3.4 loop; DDA parity; **then stop writing** the JSON-vector columns and remove
  them (compatibility reader needs a profile + schedule to synthesize).
- **P5 — decouple fragmentation:** activation+CE-keyed response model independent
  of Bruker window transmission (prerequisite for Thermo).
- **P6 — Thermo dispatch:** `SamplingGeometry::None` + `DetectorModel` +
  `ThermoRawWriter` → Astral `.raw` from the same DB.
- **P7 — SCIEX / mzML** behind the same trait.

## 7. Open questions (remaining)

1. **Condition granularity** — one `experiment_conditions` row per run, or
   per-analyte overrides (mixed-gas / mixed-method experiments)? (Load-bearing for
   the P1 schema.)
2. **Residual vector encoding** — isotope composition + fragmentation response
   stay per-ion vectors; compact binary / Arrow / columnar vs JSON? (Deferred —
   not load-bearing for P1.)
