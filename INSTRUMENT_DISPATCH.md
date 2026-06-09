# Instrument Dispatch — separating physical appearance from device recording

> **v2** — revised after Codex review. Changes from v1 are folded in throughout;
> the major shifts are the three-layer model (§1), fragmentation as a response
> model (§2.3), the corrected parity/seeding contract (§5), the wider removal
> scope (§4), the richer `SamplingGeometry`/`Calibration`/`RenderedEvent` (§3),
> event-interval time projection (§3.2), and DDA as a controller (§3.3).

## 1. Principle — three layers, not two

A simulated run has **three** independent layers. v1 fused the middle into the
first; that was too broad — RT, charge yield, mobility, and fragmentation are
not intrinsic peptide properties, they are conditioned on the experiment.

1. **Analyte chemistry** (truly intrinsic): sequence, modifications, elemental
   composition, exact isotope m/z, CCS (with conformer/model provenance).
2. **Appearance under declared conditions** (experiment/source/chromatography):
   retention-time profile (LC method, gradient, column), charge-state
   distribution and ion yield (source), ion mobility `1/K0` (drift gas, temp —
   the CCS→1/K0 conversion already takes gas mass + temperature, see
   `mscore/src/chemistry/formulas.rs:64`), and the fragmentation response.
3. **Instrument recording**: acquisition schedule, sampling geometry, m/z
   encoding, detector/peak-shape model, and file format.

The DB stores layers 1–2 **plus an explicit `experiment_conditions` record** that
declares the gas/temperature/LC/source context the layer-2 quantities are
conditioned on. A given DB is reusable across instruments **only within its
declared conditions**. Layer 3 is the swappable `InstrumentProfile` applied at
render time.

`1/K0` is stored alongside its conditioning (gas, temperature) — or, preferably,
CCS is stored and `1/K0` is derived per-profile from that profile's mobility
environment, so a profile with a different drift gas recomputes mobility
correctly.

## 2. Current state

### 2.1 What is layer-1/2 and roughly the right shape

| Table | Quantity | Layer | Representation |
|-------|----------|-------|----------------|
| `peptides` | RT μ/σ/λ (EMG) | 2 (LC) | scalars ✓ |
| `ions` | charge, m/z | 1–2 | scalars ✓ |
| `ions` | `inv_mobility_gru_predictor` + std | 2 (gas/temp) | scalars — **needs conditioning provenance** |
| `ions` | isotope envelope `simulated_spectrum` | 1 (m/z) + recording (intensities) | per-ion vector — see §2.3 |
| `fragment_ions` | fragment spectra | **2, acquisition-leaking** | per-ion vector — see §2.3 |

### 2.2 The three coupling points (device recording fused into the store)

1. `simulate_frame_distributions_emg` → `frame_occurrence[_start/end]`,
   `frame_abundance`: RT EMG **sampled onto Bruker frame indices**.
2. `simulate_scan_distributions_with_variance` → `scan_occurrence`,
   `scan_abundance`: IM Gaussian **sampled onto Bruker scan indices** via the
   reference `.d` scan→mobility calibration.
3. `build_acquisition` bootstraps the whole coordinate system from a Bruker `.d`.

These per-ion **variable-length vectors are JSON-serialized in SQLite**
(`python_list_to_json_string`) and dominate DB size + cost JSON encode/parse.
They are exactly the device-sampled products that should not be persisted.
**Moving projection to render-time deletes these columns** — the architectural
cut and the space win are one refactor.

### 2.3 Isotope intensities and fragments are not purely physical

Codex catch: `fragment_ions` are generated **only for transmitted precursors at
quantized collision energies** (`simulate_fragment_intensities.py:270`,
`rustdf/src/sim/dia.rs:570` keys by quantized CE), so Bruker quadrupole
transmission + CE already leak in. Likewise, *observed* isotope intensities are
shaped by transmission, detector response, saturation, and centroiding — only
the exact isotope **m/z** is layer-1 chemistry.

Target:
- Store a **fragmentation response model** keyed by `(activation_type,
  collision_energy)` — or fragment spectra under an explicit *neutral*
  activation-condition key — generated **independently of acquisition-window
  transmission**. Transmission/CE selection moves into layer 3.
- Store precursor isotope **composition** (chemistry); apply transmission +
  detector response at render time.

Consequence for ordering: **Thermo dispatch must wait until fragment generation
is decoupled from Bruker transmission/CE** (§6).

## 3. Target architecture

```
  layer 1–2 store (DB)               InstrumentProfile (Rust)              output
 ┌──────────────────────────┐   ┌────────────────────────────────┐   ┌────────────┐
 │ chemistry: seq, mods, CCS │   │ AcquisitionSchedule  [HAVE+ext] │   │ Bruker .d  │
 │ RT μ/σ/λ, charge, m/z     │──►│ AcquisitionController (DDA only)│──►│ Thermo .raw│
 │ IM(1/K0|CCS)+σ +cond.     │   │ SamplingGeometry (modality+grid)│   │ SCIEX .wiff│
 │ isotope composition       │   │ Calibration (sampling+encoding) │   │ mzML       │
 │ fragmentation response    │   │ DetectorModel (peak/quantize)   │   └────────────┘
 │ experiment_conditions     │   │ Writer (format)                 │
 └──────────────────────────┘   └────────────────────────────────┘
```

Projector: **store + profile → ordered `RenderedEvent`s → format writer.**

### 3.1 `SamplingGeometry` — modality + grid + encoding (not a 2-variant enum)

v1's `{IonMobility, None}` conflated separation, calibration, and
discretization, and assumed one global grid. Revised:

```rust
pub struct SamplingGeometry {
    modality: MobilityModality,            // None | Tims | DriftTube | Faims
    acceptance: MobilityRange,             // physical range admitted
    grid: MobilityGrid,                    // bin edges / per-bin integration intervals
    encoding: MobilityEncoding,            // physical mobility <-> native index
    // per-event overrides where geometry/calibration vary by frame/event
}
```

For `MobilityModality::None` (Astral/Orbitrap) the projector **marginalizes the
mobility distribution preserving total signal** — it does not invent "one scan."

Render output is an enum, so flatten-then-rebundle never loses simultaneity:

```rust
pub enum RenderedEvent {
    Scan(MzSpectrum, ScanMeta),                 // non-IMS
    MobilityFrame(Vec<(ScanIndex, MzSpectrum)>, FrameMeta),  // IMS
}
```

### 3.2 `Calibration` — decomposed, forward+inverse, with domains

```rust
pub struct Calibration {
    mobility: MobilityCalib,    // mobility bin <-> physical mobility
    mz: MzCalib,                // exact m/z <-> native coord (tof | frequency)
    // detector concerns are separate (DetectorModel): mass-error / resolution /
    // peak-shape, digitization / centroiding, intensity response.
}
```

Every calib exposes explicit `forward`/`inverse` with valid domains, units, and
monotonicity checks; per-scan calibration where the format has it. Exact isotope
m/z stays upstream (layer 1); m/z error, peak width, quantization, and centroid
bias live in `DetectorModel` (layer 3).

### 3.3 Time projection over event intervals (corrected)

The existing kernel integrates `[time - cycle_length, time]`
(`mscore/src/algorithm/utility.rs:198`) — wrong for unequal event durations,
dead time, or overlapping accumulation. Expand the schedule so each event
carries an explicit **start/end (or exposure) interval**, then integrate the
chromatographic EMG over that interval. Mobility projection (IMS only) then maps
the IM Gaussian onto the `SamplingGeometry` grid; for `None` it marginalizes.

The kernels are **deterministic CDF integrations** (`utility.rs:407`) — no RNG.

### 3.4 DDA is a controller, not a scheme property

DIA is predetermined and *is* a static `AcquisitionSchedule`. DDA is a **stateful
`AcquisitionController`** (dynamic exclusion, intensity thresholds, capacity)
that sits between projection/detection and event generation, consuming
already-"observed" MS1 signal. The current `dda_selection_scheme.py` consumes
device-projected frame/scan coordinates, so DDA must migrate together with
column removal (or be explicitly excluded from that phase).

## 4. DB schema delta + removal scope (wider than v1 thought)

**Add:** `experiment_conditions` (gas, temp, LC, source, model provenance);
schema **version** field; `activation`/CE keying on the fragmentation response.
**Remove (eventually):** `scan_occurrence`, `scan_abundance` from `ions`;
`frame_occurrence[_start/end]`, `frame_abundance` from `peptides`.

Removal touches more than `assemble_frames`:
- `rustdf/src/sim/containers.rs:64` requires the vectors,
- `rustdf/src/sim/handle.rs:74` unconditionally deserializes them,
- `rustdf/src/sim/handle.rs:242` lazy-loads by `frame_occurrence_start/end`,
- `dda_selection_scheme.py:82` consumes both projections.

So: introduce **versioned schema + scalar-native Rust entities first**; replace
frame-range indexing with an **RT-support index derived conservatively from EMG
parameters** (query by RT, project batches). A "compatibility reader" can only
synthesize the old columns if handed a profile + schedule — it is not free.

## 5. Parity & determinism contract (revised)

Byte-identical `.d` is **not** the first acceptance gate. Split it:

1. **Numerical parity** (gate 1): projected indices/abundances and ordered
   `RenderedEvent` spectra equal the legacy path within tolerance.
2. **Writer/byte parity** (gate 2, separate): a deterministic writer test —
   SQLite row ordering, metadata, and compression can differ even when content
   matches, so this is its own harness (reuse the TOF-writer parity harness).

Determinism: replace traversal-order / `thread_rng` noise with **counter-based
seeding** `hash(run_seed, profile_id, analyte_id, event_id, noise_component)`,
so the same DB yields stable output per instrument and noise is reproducible
independent of thread scheduling. (RNG enters only via optional scan/RT/m/z
noise and fragment assembly — the projection itself is deterministic.)

## 6. Phased plan (re-ordered per review)

- **P0 (this doc):** agree the three-layer cut + interfaces. ← we are here
- **P1 — foundations:** schema **versioning** + `experiment_conditions` metadata;
  counter-based **seed semantics**; **scalar-native Rust entities** (no occurrence
  vectors). No pipeline behavior change yet.
- **P2 — projection:** event-**interval** time projection + mobility projection as
  `InstrumentProfile` methods wrapping the existing kernels; PyO3 surface;
  migrate **all** Rust builders/readers (`containers.rs`, `handle.rs`,
  lazy index → RT-support). `assemble_frames` reads occurrence from the profile
  with DB fallback.
- **P3 — parity:** prove **numerical** parity (gate 1), then **deterministic
  writer** parity (gate 2) for Bruker `.d`.
- **P4 — DDA controller:** migrate DDA to `AcquisitionController`; only then stop
  *writing* the JSON-vector columns; later drop them (compatibility reader).
- **P5 — decouple fragmentation:** fragmentation response keyed by activation+CE,
  independent of Bruker window transmission. (Prerequisite for Thermo.)
- **P6 — Thermo dispatch:** `SamplingGeometry::None` + `DetectorModel` +
  `ThermoRawWriter` end-to-end → Astral `.raw` from the same DB.
- **P7 — SCIEX / mzML** writers behind the same trait.

## 7. Open questions (remaining)

1. **Residual vectors** — isotope composition + fragmentation response are still
   per-ion vectors. Compact binary blob / Arrow / columnar store vs. JSON? (The
   user flagged vector space/speed generally; tackle after the dispatch cut.)
2. **Condition granularity** — is one `experiment_conditions` row per run enough,
   or do we need per-analyte overrides (e.g. mixed gas experiments)?
3. **CCS vs stored 1/K0** — store CCS and derive 1/K0 per profile (cleanest, but
   requires every profile to declare its mobility environment), or store 1/K0 +
   conditioning and convert when profiles disagree?
