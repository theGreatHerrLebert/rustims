# SCIEX → NECRO consolidation + HeLa-driven iteration

Status: DRAFT for claudex. Goal: stop debugging SCIEX across three divergent rustims checkouts +
a broken covid-in-hela seed. Move the good SCIEX code into the necroflow v2 branch
(`feat/timsim-v2-structure-axis`) and iterate against our own **dense HeLa** harness — exactly the
approach that made the Thermo `.raw` track tractable.

## Why (the investigation that led here)

Debugging the SCIEX native `.wiff` "0 IDs" bug surfaced a three-checkout mess and a contradiction:

1. **The `.wiff` writer is exonerated** (confirmed the prior crashed session's landing): the
   `ms2_nonempty` count is computed at render time (`dia_render.rs:110`) in code shared with the mzML
   path, before anything wiff-specific — so any emptiness is a render-layer fact.
2. **A known under-scaling bug exists and is documented** (`PhantomBENCH/docs/sim/nontimstof-intensity-bug.md`):
   non-timsTOF renders set `rt_cycle_length` to the median of ALL frame diffs (per-scan spacing) instead
   of the per-cycle interval, under-scaling peaks ~20–40×. Fixed + validated for Astral (commit
   `6c7da880`, MS2 max 20→2594) and — per that commit — for SCIEX/Waters too.
3. **But the covid SCIEX render still gets 0 IDs WITH the fix present.** The render used
   `promotion/rust/rustims` (editable install, fix live since 2026-06-19); the 2026-07-06 covid render is
   still broken. So the fix (as applied to SCIEX) is not sufficient.
4. **The decisive apples-to-apples (which the crashed session lacked):** on the SAME covid `mild` seed
   (2,939-peptide sample), the **fixed Astral path gets 975 precursors / 862 proteins**, while **SCIEX
   gets 0 / 0 — with 643 BILLION MS2 signal (~1,100× Astral's 584M)**. Reconciled with the writer log
   (82,200 MS2, 17,458 non-empty): ~37M per non-empty scan. So SCIEX is **over-concentrated**, not
   under-scaled — the *opposite* of the "sparse render" conclusion. Likely a fourth red herring in that
   thread.

**Prime suspect (REVISED after claudex — the original `rt_cycle_length` hypothesis is downgraded):**
`rt_cycle_length = cycle_time_s` is NOT a meaningful divergence: SCIEX *synthesizes* its schedule using
that same `cycle_time_s`, so MS1→MS1 spacing is 3.5 s by construction and `rt_cycle_length_from_ms1`
would return the same number. And a cycle-time error yields a windows-per-cycle-sized effect (~60×), not
~1,100×. So `rt_cycle_length` is likely NOT the primary cause — keep the MS1-derivation only as an
asserted invariant (`derived == configured`), not as the fix.

The 643B-signal / 0-ID signature more likely indicates a **renderer↔search incompatibility**, not
chromatographic over-concentration: malformed DIA metadata, wrong precursor-m/z units, centroid-vs-profile
mislabelling, extreme cofragmentation, an unnormalized synthetic non-IMS scan axis (measure the
`scan_abundance` sum in the failing artifact — do NOT assume it marginalizes to 1), or a double-applied
`upscale_factor`. **The real Stage-1 work is an intensity-accounting audit + direct mzML-metadata
inspection**, not a cycle-time patch.

## The landscape (all one repo: `theGreatHerrLebert/rustims`)

| checkout | branch | SCIEX state |
|---|---|---|
| `timsim-necro/rustims` (ours) | `feat/timsim-v2-structure-axis` | build-from-`.wiff` → **mzML**, dispatch (`sciex_zenotof`), `.wiff` **reader** dep, `rt_cycle_length` **fix**. NO native writer. |
| `SUBMISSION/rustims` | `feature/sciex-native-writer` (local, **unpushed**, HEAD `ece1c668`, 2026-07-06) | the native `.wiff` **writer** (`dia_render.rs`, per-template profile authoring). Carries the **buggy** `rt_cycle_length`. |
| `promotion/rust/rustims` | `perf/annotated-builder-a-d` | David's active dev; ~= necro (mzML + fix, no native writer). |
| `~/thermo-raw-spike/sciexwiff` | `feature/wiffscan-codec` (local, unpushed) | pure-Rust `.wiff.scan` peak **codec (decode + encode)** — the authoring primitive. rustdf currently pins the reader-only rev `ef7c0cd`. |

So consolidation is a **git merge**, not hand-copying — the native writer is 5-ish commits on a local
branch that must be brought over + reconciled with necro's `rt_cycle_length` fix.

## The plan (staged; validate before merging)

### Stage 1 — DECIDE sound-vs-bugged with a paired micro-run (do FIRST; no merge, no DiaNN yet)
Cheapest deciding experiment (per claudex): a **paired, deterministic micro-run** that isolates render
physics from the search interface.
1. One small FIXED peptide panel (~20–100), predicted ions, fixed RTs + event counts, NO noise.
2. Render it through **SCIEX and Astral** (same panel, same seed), plus a SCIEX variant asserting
   `rt_cycle_length_from_ms1(frame_table) == cycle_time_s` (to close that hypothesis outright).
3. Compare truth↔mzML DIRECTLY (no search): scan count, non-empty fraction, **per-peptide integrated MS2
   intensity**, chromatographic peak width, per-spectrum peak count, and a **known-peptide-recovery** check
   against simulator truth. Also inspect the emitted mzML metadata: precursor-m/z units, isolation windows,
   centroid/profile flag, intensity type; and measure the synthetic `scan_abundance` sum.
   - If SCIEX integrated signal is ~1,100× Astral's WITH equal cycle lengths ⇒ the cause is accounting/
     metadata, not cycle time — chase it in the intensity audit above.
   - Known-peptide recovery separates a bad spectra/search INTERFACE from bad physical RENDERING.

Only AFTER the micro-run verdict: run a **dense HeLa** render as the integration/search test (mzML → DiaNN,
HeLa FASTA DB). Do NOT gate on "~862 proteins" — methods/CE/gradient/search settings differ from Astral;
use truth-recovery + reasonable ID magnitude as the bar.

Deliverable: a one-line verdict (render sound / bugged), the accounting table, and — if bugged — the fix
landed in necro against the micro-run + HeLa.

### Stage 2 — bring the native writer into necro (only if Stage 1 sound / after the fix)
Git (per claudex — do NOT merge the whole unpushed branch; it's a few commits atop an OLDER main while
necro has substantial newer render/flow work):
- First create an **immutable backup ref** for `feature/sciex-native-writer` and **push it** somewhere safe
  (it's currently local-only — one disk failure from gone).
- `git range-diff` the branch vs its merge-base, then **cherry-pick the writer COMMITS in order** (not
  selected files — the writer's Python/Rust/connector/dep changes are coupled) onto a dedicated
  **integration branch off current necro**. Resolve conflicts deliberately in favor of necro's newer render
  interfaces + the cycle-time invariant. Run the existing **Thermo regression** before any SCIEX test.
- Bump the `sciexwiff` dep from the reader-only rev to the codec rev (`feature/wiffscan-codec`) — after
  pushing THAT branch to the sciexwiff remote so the pin is reproducible.
- Add `SciexWiffWriter: AcquisitionWriter` in `rustdf/sim/acquisition.rs` (analogue of `ThermoRawWriter`),
  authoring `.wiff.scan` peaks into a `.wiff` template via the codec.
- **Native-writer tests, independent of DiaNN:** codec round-trip (decode∘encode peaks-stable), fixed
  known-peak injection, reader-visible scan/isolation/RT preservation, and an external tool (pwiz/msconvert)
  opening the produced `.wiff` cleanly. These certify authoring separately from render physics.

### Stage 3 — wire SCIEX into the necroflow flow + HeLa native `.wiff` test
- `timsim-render-sciex` bin (analogue of `timsim-render-thermo`) + a `render_sciex` node in
  `flow/timsim_flow.py` (co-outputs: `.wiff` + truth + manifest; template + CE validation; fail-fast).
- Dense HeLa → native `.wiff` → DiaNN in the flow: the SCIEX analogue of the HeLa Thermo result,
  `.wiff` finally a first-class OUTPUT in the sim inventory (today it is only an input template).

### Gates & invariants (claudex)
- **Revised sequence:** micro-run accounting → dense-HeLa mzML + truth recovery → integration branch /
  cherry-picks + Thermo regression → native-writer validation (non-DiaNN) → native HeLa/DiaNN.
- Gate Stage 2 on a PROVEN renderer AND an explicit product decision that native `.wiff` is worth it
  (mzML is already DiaNN/alphaDIA-readable; mzML success alone does NOT validate native writing).
- K562-template → HeLa-signal is method-valid but not automatically QUANTITATIVELY valid: record the
  template's windows, gradient, cycle time, calibration, and CE model in the manifest; HeLa RT projection
  must map onto the synthesized method gradient.
- **Pin the whole environment** for every result (the three-checkout mess is otherwise reproducible only
  by accident): editable-install path, rustims rev, template checksum, FASTA, seed, DiaNN version+settings,
  and output checksums.

## Open questions / risks (for claudex)

1. **Stage-1 render path.** The SCIEX mzML render is a legacy v1 `simulator.py` path driven by a findings
   CSV / sampled FASTA — NOT the necroflow v2 feature space. Is running v1 `timsim` (instrument=sciex_zenotof,
   sampled HeLa) the right Stage-1 probe, or should Stage 1 render through the v2 feature space we already
   have (which would require wiring SCIEX into the flow first — i.e. doing Stage 3's plumbing before we've
   validated the render)? Trade-off: fastest signal vs testing the actual target path.
2. **Is the over-scaling really `rt_cycle_length`?** 643B vs 584M is ~1,100×, but `cycle_time_s`=3.5 vs a
   plausible true cycle can't be 1,100×. Is there a second factor (double-applied `upscale_factor`, a
   per-window vs per-cycle events count, MS1 vs MS2 asymmetry)? The plan should measure, not assume.
3. **Merge vs cherry-pick vs rebase** for `feature/sciex-native-writer` — it's a local unpushed branch that
   diverged from an older main; how much unrelated drift comes with it, and does merging risk regressing
   necro's newer work (mod-aware fragments, phase-2 nodes)? Cherry-picking only the writer files may be safer.
4. **`.wiff` template requirement.** Native SCIEX authoring needs a real `.wiff` template (unlike Waters
   SONAR which is build-from-parameters). The HeLa test reuses the K562 `.wiff` — is authoring HeLa signal
   into a K562-method template scientifically valid (windows/cal are the method's, only peaks change), the
   same way we author into a foreign Thermo `.raw`? (Believed yes — it worked for Thermo.)
5. **Scope creep.** Stages 2–3 replicate the entire Thermo native track for a second vendor. Should we gate
   Stage 2 on a green Stage 1 AND an explicit decision that native `.wiff` output is worth it vs shipping
   the (already-working-on-HeLa?) mzML path, which DiaNN/alphaDIA read anyway?

## Non-goals
Fixing the PhantomBENCH covid seed itself; the Waters SONAR path; pushing the sciexwiff codec branch
(prerequisite for Stage 2, but a separate action); DDA.

## Stage-1 RESULT (audit executed — root cause found, diagnosis corrected AGAIN)

The audit (not a re-render) found it, exactly as the claudex revision ordered:

1. **The covid SCIEX mzML that got 0 IDs has corrupted fragment m/z up to 594,000,000** (real fragments
   are 140–1829). The working control (`plain_render.mzML`, 766 IDs) has ZERO peaks > 2000 (max 1201). The
   0-ID failure tracks the garbage m/z, NOT peak count (the 0-ID render is *denser* than the 766-ID one) —
   killing the "sparse render" thread. This is the renderer↔search incompatibility, not `rt_cycle_length`.
2. **The 594M is NOT in the render — it is in the `.wiff` writer.** `_mzml/mild_r1.mzML` was produced by
   `pwiz_Reader_ABI` from our `.wiff` (`sourceFile id="WIFF"/"WIFFSCAN"`), i.e. pwiz reads 594M m/z back
   out of OUR `.wiff`. Necro's DIRECT `render_db_to_mzml` on the SAME DB reproduces the exact scan
   structure (83570 scans, 17458 non-empty MS2) but tops out at a clean **m/z 3358** — its > 2000 peaks
   are faint (median int 0.033 vs 0.192), spread, plausibly real large-fragment isotopes, NOT garbage.

**Corrected conclusion:** the prior "writer exonerated" holds only for the sparsity/emptiness (a render
fact in shared code) — it MISSED that the writer **corrupts fragment m/z on round-trip** (its `.wiff.scan`
peak m/z↔TOF/calibration encoding). So:
- **necro + mzML sidesteps the broken writer** (mzML render is clean) — the consolidation plan is validated.
- **native `.wiff` (Stage 3) is gated on fixing the writer's m/z↔TOF encoding**, with the codec round-trip
  test (author known m/z → pwiz readback → assert equality) as the gate — which would have caught this.

### Stage-1 CONFIRMATION (DiaNN, same covid mild_r1 DB)
| path | precursors | protein groups |
|---|---|---|
| our `.wiff` → pwiz readback (594M m/z) | 0 | 0 |
| necro direct mzML render (clean, max m/z 3358) | **524** | **444** |
| Astral reference (same seed) | 975 | 862 |

Render path SOUND (searchable); `.wiff` writer BROKEN (m/z corruption). necro's 444 vs Astral's 862 is a
render QUALITY gap (faint/lossy — tune with dense HeLa), separate from the writer CORRECTNESS bug.

## Decision point
- ~~(a) wire the clean SCIEX **mzML** path into the necro flow as a `render_sciex` node~~ — DONE
  (commit 28c1fd89): `render_sciex` node reuses the shared v2 feature space, Koina HCD, mzML.
  Validated: HeLa v2 feature space -> clean mzML (max m/z 3341) -> DiaNN 14,471 precursors / 534 PGs.
  Needs imspy_connector built --features mzml,sciex. NEXT: OR
- (b) fix the `.wiff` writer m/z↔TOF encoding (codec round-trip gate) for native `.wiff` output.

## Stage-3 progress: .wiff writer bug LOCALIZED (peak encoding sound; Idx/interop suspect)

Added a decode probe (`probe_garbage_mz` in SUBMISSION rustdf `sciex_dispatch.rs`, gated on
`TIMSIM_SCIEX_WIFF_SCAN`) that decodes our own `.wiff.scan` with our own codec + each block's cal.

Result on the corrupt mild_r1 `.wiff.scan`: 95,372 blocks / 3.11M peaks, and our decode recovers ONLY
sane m/z (max **2567** — real large fragments, not corruption); `cal_a` is stable (two clusters 1e-8
apart). **The 594M m/z exists ONLY in pwiz's readback, not our decode.** So:
- The peak m/z→n encoding (`mz_to_n`, per-block cal) is CORRECT — it round-trips to sane m/z.
- The 594M is pwiz mis-decoding a minority (~1.7%) of peaks. A 594M m/z ⇒ absolute n ≈ 243M ≈ 740×
  a normal cut_n (~328K) ⇒ a WRONG SEEK OFFSET, not a wrong calibration.
- **Suspect: the Idx/segment-offset retranslation** (`retranslate_idx`/`rebuild_grow`, commit 03bc4c9):
  a bad recomputed Idx offset for a minority of blocks makes pwiz seek to the wrong byte and read
  garbage as the block seed.

NEXT: localize which blocks pwiz mis-seeks (per-scan diff our decode vs pwiz's mzML by scan index),
then fix the Idx offset for those; gate with a pwiz-readback round-trip test (author known m/z → pwiz
→ assert equal). The peak codec itself is exonerated.
