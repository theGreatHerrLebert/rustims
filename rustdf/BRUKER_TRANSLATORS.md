# Bruker timsTOF axis translators (SDK-free)

Reference for the two axis calibrations that turn raw timsTOF indices into
physical units **without** the proprietary `libtimsdata` SDK:

- **TOF index → m/z**  (mass axis)
- **scan number → 1/K0** (inverse reduced ion-mobility axis)

Both are pure-Rust ports living in [`src/data/calibration.rs`](src/data/calibration.rs)
(`MzCalibrator`, `MobilityCalibrator`), wired into the reader as the
`BrukerFormulaConverter` variant of `TimsIndexConverter`. The algorithms mirror
PAPPSO's GPLv3 `libpappsomspp` (`mzcalibrationmodel1.cpp`, `timsframebase.cpp`)
and are cross-checked against Bruker's published `tims_calibration.py`.

Coefficients come from two SQLite tables in `analysis.tdf`, joined per frame via
`Frames.MzCalibration` and `Frames.TimsCalibration`.

---

## 1. TOF index → m/z

### Formula

```
t        = tof_index * DigitizerTimebase + DigitizerDelay      # flight time (digitizer units)
s        = sqrt(m + C4)                                        # s = sqrt(reduced mass)
t        = C0 + b*s + C2*s^2 + C3*s^3,   b = sqrt(1e12 / (C1 * tc))
m/z      = s^2 - C4        (invert the curve for s, given t)
```

- Forward (`m/z → tof`) is closed form.
- Inverse (`tof → m/z`) inverts for `s`: closed form for the pure base, otherwise
  a Newton refinement.
- **Both ModelTypes use the same quadratic-in-√m curve** `t = C0 + b·s + C2·s²`.
  This was verified empirically: fitting `t` vs `s` recovers `a0→C0`,
  `a1→√(1e12/C1)`, `a2→C2`. The cubic `C3·s³` term is real only for ModelType 1
  (in ModelType 2 the `C3` column duplicates `C0` and is dropped).

**Temperature compensation** (applied to `C1`, and to `c2` for ModelType 1):

```
tc = 1 + ( dC1*(T1 - T1_frame) + dC2*(T2 - T2_frame) ) / 1e6
```

where `T1_frame`, `T2_frame` are the per-frame `Frames.T1`, `Frames.T2`.

### `MzCalibration` table parameters

| Column | Symbol | Meaning |
|---|---|---|
| `ModelType` | — | 1 = classic cubic-in-√m curve (fully modelled here); 2 = modern base curve **+** proprietary correction polynomial (only the base is modelled) |
| `DigitizerTimebase` | `timebase` | ns per digitizer sample; scales the TOF index into flight time |
| `DigitizerDelay` | `delay` | fixed time offset (samples) added before t₀ |
| `T1`, `T2` | `T1_ref`, `T2_ref` | reference temperatures for compensation |
| `dC1`, `dC2` | `dC1`, `dC2` | temperature sensitivities (ppm/°C) → `tc` |
| `C0` | `C0` | constant term of the time↔mass curve (≈ the t-intercept) |
| `C1` | `C1` | governs the dominant √ term: `b = sqrt(1e12 / C1)` |
| `C2` | `C2` | quadratic `s²` term of the curve — **used by both ModelTypes** |
| `C3` | `C3` | cubic `s³` term — **ModelType 1 only** (in ModelType 2 this column duplicates C0 and is dropped) |
| `C4` | `C4` | "reduced mass" shift `m0` (`x = m − m0`); the element claimed by Bruker patent US 7,851,746 |
| `C5` | window low  | ModelType-2 fine correction: **lower m/z bound** of the correction window. A **universal constant** `225.951491` across every dataset seen. |
| `C6` | window high | ModelType-2 fine correction: **upper m/z bound** of the correction window (per-dataset, e.g. 1383.7 / 1519.7). |
| `C7` | degree | ModelType-2 fine correction: polynomial **degree = 7** (matches the fit below). |
| `C8 … C14` | coeffs | ModelType-2 fine correction coefficients — **decode unresolved** (see note). |

**Status of the ModelType-2 C8…C14 correction (partially reverse-engineered).**
After the `C0/C1/C2` quadratic, a smooth residual of ~2.5–3.6 ppm remains.
Established by fitting against the SDK across several datasets:

- The correction is **windowed to `[C5, C6]` in m/z** — outside that interval the
  quadratic base is the whole model.
- Inside the window it is a **smooth degree-7 polynomial** (a degree-7 fit in the
  normalized window coordinate reproduces the SDK to ~5e-8 m/z — effectively
  bit-exact). This matches `C7 = 7`.
- However, the stored `C8…C14` do **not** map onto that polynomial in any tested
  basis (monomial or Chebyshev) / argument (`m/z`, `√m`, normalized window, time,
  tof) / target domain (m/z, √m, flight time, ppm) — every combination leaves
  ~99% residual. The exact encoding of `C8…C14` is still unknown.

Cross-check: **no open-source reader reimplements this** — alphaRaw, alphaDIA,
opentims, and the SDK-lookup tooling all obtain ModelType-2 m/z from Bruker's
`libtimsdata` directly. A fully SDK-free, bit-exact ModelType-2 m/z is therefore
not currently available; the `C0/C1/C2` base here reaches a few ppm.

---

## 2. scan number → 1/K0

### Formula (two steps, both exact)

```
V     = dv_start + slope * (scan - ttrans - ndelay),   slope = (dv_end - dv_start) / ncycles
1/K0  = 1 / ( C0m + C1m / V )
```

`V` (trapping voltage) must lie in `[vmin, vmax]`. The inverse (`1/K0 → scan`)
solves both steps algebraically and rounds to the nearest scan.

### `TimsCalibration` table parameters (ModelType 2 — the only TIMS model)

| Column | Symbol | Meaning |
|---|---|---|
| `ModelType` | — | always 2 (PAPPSO/this port require it) |
| `C0` | `ndelay` | scan offset (delay), subtracted before scaling |
| `C1` | `ncycles` | number of TIMS cycles; sets the voltage-vs-scan `slope` |
| `C2` | `dv_start` | trapping voltage at the **start** of the ramp |
| `C3` | `dv_end` | trapping voltage at the **end** of the ramp |
| `C4` | `ttrans` | transit time in cycles, subtracted before scaling |
| `C5` | — | unused by the mobility formula (polynomial-grade flag, = 1) |
| `C6` | `C0m` | additive constant of the mobility reciprocal |
| `C7` | `C1m` | voltage-scaled term of the mobility reciprocal |
| `C8` | `vmin` | lower voltage validity bound |
| `C9` | `vmax` | upper voltage validity bound |

---

## Accuracy vs the Bruker SDK

Measured by `examples/compare_calibration.rs` (residual of this formula vs
`tims_index_to_mz` / `tims_scannum_to_oneoverk0`), full axis grid, per dataset:

| Dataset | m/z ModelType | TOF → m/z residual | scan → 1/K0 residual |
|---|---|---|---|
| `synchro-hela.d` | **1** | **0.0000 ppm (bit-exact)** | ~4e-16 (machine ε) |
| `G8602.d` | **2** | mean 2.5 ppm / max 11.1 ppm | ~7e-16 (machine ε) |
| `G8027.d` | **2** | mean 3.6 ppm / max 20.8 ppm | ~4e-16 (machine ε) |

(ModelType-2 m/z figures are with the `C0/C1/C2` quadratic; the older base-only
curve without `C2` was ~9 ppm mean.)

**Takeaways**

- **scan → 1/K0 is exactly the SDK computation** on every dataset (agreement at
  the floating-point rounding limit). The mobility formula *is* what we were
  looking for.
- **TOF → m/z is exact for ModelType-1** data (bit-for-bit). For **ModelType-2**
  data (modern instruments) the `C0/C1/C2` quadratic reproduces the SDK to a few
  ppm; the last few ppm come from the `C8…C14` correction whose exact form is not
  resolved (see the ModelType-2 note above).

## Reproduce

```bash
cargo run --release --example compare_calibration -- \
    <path-to-libtimsdata.so> <path-to-.d-folder> [frame_id]
```
