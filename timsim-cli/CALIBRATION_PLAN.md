# Calibration runbook — execute when the blank + dilution samples arrive

Paused until the samples requested in `CALIBRATION_SAMPLE_REQUEST.md` are acquired. This is the exact
procedure to run once they land, so the work resumes without re-deriving anything. The **iron rule**
throughout: the **biological abundance axis is never touched** — we fit a *measurement/observation*
model on top of it (see `RENDER_CALIBRATION.md` for why).

Inputs expected: the `.d` files + a manifest (`filename, type ∈ {blank,dilution,mix}, load_ng, standard,
replicate, injection_order`). Existing tools: `validate/peak_distribution.py` (per-frame-type per-peak
stats), `validate/v2_eval.py` (harness), the K240723 reference for the target distribution.

## Step 0 — Ingest & QC
- Confirm every `.d` matches the K240723 method (24 windows, 44-min gradient, m/z 400–1000, 1/K0 0.65–1.76)
  — same check used in this thread. Reject any that differ.
- Parse the manifest; assert loads are monotically ordered and the `load_ng` column is present per run.
- Run `peak_distribution.py` on each: blanks should have far fewer/dimmer peaks than loads; per-peak
  stats should rise monotonically with load. If not, stop and query the lab (mislabel / carryover / gain drift).

## Step 1 — Background model (from the blanks)
Per frame type (MS1, MS2), from the ≥3 blanks:
- Peak-**intensity** distribution (floor, percentile ladder, density peaks/scan) — reproducibility across blanks.
- Peak **spatial structure**: how background peaks distribute across m/z, mobility (scan), and RT/gradient
  position — uniform, or structured (contaminant lines, mobility bands, gradient-dependent)?
- **Output:** a generative background spec — `background(frame_type, mz, mobility, rt) → (density, intensity dist)`.
  This is the honest, measured answer to "how much of real data is background," which we could not get before.

## Step 2 — Load→response curve (from the dilution series)
- Search each dilution run (DiaNN, same as the harness) → identified precursors + quantities.
- **Subtract background** (Step 1) so response is fit on analyte peaks, not signal+background.
- For precursors identified across ≥3 loads: plot **per-peak intensity (and reported quantity) vs on-column ng**,
  per frame type. Fit the transfer function: linear region, **low-end floor** (where peaks drop below the
  count threshold), and **saturation** at the top **only if the data shows it** (do not assume it).
- **Output:** `response(true_signal) → recorded_intensity` per frame type. Default linear + floor unless the
  curve bends.

## Step 3 — Signal vs background in the reference
- Using Step 1's background and Step 2's response, decompose the K240723 reference per-peak population into
  analyte vs background per frame type. This finalises the density-gap interpretation (how much of the ~30×
  MS1 / ~13× MS2 density is background vs low-level analyte) and sets the density target the model must hit.

## Step 4 — Implement the observation model in the render (abundance held fixed)
Add, after ion generation, per frame type, conditioned on m/z / mobility / gradient where Steps 1–2 show dependence:
1. **Response transfer** (Step 2) mapping the factorised true signal to expected recorded intensity.
2. **Real count floor / censor at ~21** — a physical floor+censor, replacing the current post-quantise drop
   cutoff (`--min-peak-intensity` semantics; `render.rs`).
3. **Ion-count noise** — shot/counting statistics on the per-bin counts.
4. **Background process** (Step 1) — inject background peaks matching the blank's density + intensity + spatial structure.
Parameters come from the fits, not hand-tuning. Keep them in a small config the render reads, so a re-fit is data-driven.

## Step 5 — Validate (against `RENDER_CALIBRATION.md` acceptance)
- **Truth preservation (primary, must hold):** re-run `v2_eval.py` — recall-vs-**unchanged** abundance still
  spans the full range; response for identified/spiked precursors is monotonic and linear where real is; isotope
  ratios preserved. If any of these regress, the model is wrong — do NOT compensate by touching abundance.
- **Hard check:** per-peak floor exactly 21 (MS1 + MS2).
- **Emergent-shape diagnostics (regression only):** `peak_distribution.py` on the rendered output lands within
  tolerance of real, **stratified by frame type + gradient region**, with **analyte and background separated**.
- **End-to-end:** render the standard at the *same* on-column loads as the dilution series; the rendered
  per-peak distributions and DiaNN ID counts should track the real dilution series across loads (the strongest test).

## Step 6 — Iterate & lock
- If diagnostics are off, adjust the observation model (Steps 1–4), never the abundance. Re-run Step 5.
- Commit the fitted parameters, the model, and a short validation report (rendered vs real, per load, per frame type).

## Split of work (for scheduling)
- **Analysis, no render change (Steps 0–3):** Python, can be built/tested against the samples as soon as they arrive.
- **Render change (Step 4):** Rust (`render.rs`) — the one place signal generation changes; gated behind Step 5.
