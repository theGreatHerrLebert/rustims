# Calibration sample request — timsTOF DIA (for simulator response/noise model)

**Purpose.** We are calibrating a timsTOF DIA simulator's *measurement* model — how a peptide's abundance
becomes recorded peak intensities, and what the instrument background looks like. To fit that from real
data (rather than guess) we need two things acquired **on the exact method below**: (1) **matched blanks**
to measure the instrument background/noise, and (2) a **dilution series** of one standard to measure how
signal intensity scales with load (the response curve, its linearity, and where it floors/saturates).

This is a request for wet-lab acquisition; no data analysis is needed on your side — just the raw `.d`
files plus the small manifest described at the end.

## Acquisition method — must match exactly

Use the **same method** as the reference run `K240723_*` (method folder `2773.m`), i.e. reuse that method
file unchanged. For the record, its key parameters:

| parameter | value |
|---|---|
| instrument | Bruker **timsTOF Ultra** (timsTOF control 5.1.8) |
| acquisition | **DIA-PASEF**, **24 windows**, MS2 isolation m/z **400–1000** |
| m/z range | 100–1700 |
| 1/K0 (mobility) range | 0.65–1.76 |
| gradient | **44 min** (2640 s) |
| collision energy | ramped **~25.5–46.4 eV** (as in the method) |

**Critical:** keep detector gain / tune / calibration and the LC (column, gradient, flow) **identical
across every run in this set** — the whole point is to vary *only* the peptide load. Do not let any
setting auto-adjust between injections.

## Samples to acquire

### 1. Matched blanks — ≥ 3 injections
Solvent / loading-buffer only (a genuine blank injection, no peptides), run through the full method exactly
as a sample. Purpose: measure the instrument's background peak population (how many peaks, at what
intensities, where in m/z / mobility / retention time) with zero analyte. Space them through the batch
(see order) so we also see any carryover.

### 2. Dilution series — one standard, 6 loads × ≥ 3 replicates
A single well-characterised standard digest (e.g. HeLa tryptic digest, the same one used for `K240723` if
possible), injected at a **geometric load series** spanning the working range:

| point | on-column load (ng) | replicates |
|---|---|---|
| L1 | **1** | ≥3 |
| L2 | **3** | ≥3 |
| L3 | **10** | ≥3 |
| L4 | **30** | ≥3 |
| L5 | **100** | ≥3 |
| L6 | **300** | ≥3 |

(Half-decade spacing over ~2.5 orders. If a load below 1 ng is feasible, an extra L0 ≈ 0.3 ng helps pin
the low-end detection floor. Adjust the absolute ng to whatever the standard's normal load is, keeping the
same ~×3 geometric spacing and the same standard throughout.) Purpose: intensity-vs-load is the response
curve we fit — we need enough points, low enough to see the floor and high enough to see any saturation.

### 3. (Optional, if easy) one known-ratio mix
A two-proteome or spike-in mixture at a known ratio (e.g. HeLa + a spiked standard at a defined amount),
one load, ≥3 replicates — lets us validate quantitative response against a known fold-change. Nice to have,
not required.

## Injection order & controls
- **Bracket with blanks:** a blank at the start, a blank after the highest load (carryover), and blanks
  interspersed when load steps down.
- **Randomise or ascending-load** the dilution points (don't cluster all replicates of one load together
  if avoidable) to separate load effects from drift.
- Same column and a wash between big load changes.

## Deliverables
- The raw **`.d` files** for every run.
- A **manifest** (CSV/sheet) with one row per run: filename, sample type (`blank` / `dilution` / `mix`),
  on-column load (ng), standard used, replicate #, injection order/position, acquisition date, and the
  method/instrument if anything differed. The exact **on-column ng is the single most important field** —
  the response fit depends on it.

## What we do with it (for context)
- **Blanks →** the background/noise model (peak density and intensity distribution per frame type, and its
  structure across m/z / mobility / gradient).
- **Dilution series →** the load→intensity response curve per frame type (linear range, low-end floor,
  any saturation) — fit as the simulator's measurement transfer function.
- **Known-ratio mix →** validation that simulated quantities preserve true ratios.

Roughly **~24–30 injections** total (6 loads × 3 + ~4–6 blanks + optional mix). See
`timsim-cli/RENDER_CALIBRATION.md` for how the fit feeds the render.
