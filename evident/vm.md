# TimSim validity matrix: internal claims

Internal check of TimSim's validity. Not part of any publication or data deposit.

Plan: `SUBMISSION/revision/VM_EVIDENT_PLAN.md` (v3, two Codex rounds). Step-0 audit: `VM_AUDIT_STEP0.md`.

## How to read the result

Matching distributions is sufficient but not necessary. A conclusion about a search engine can transfer
to real data even if the simulation is easier or harder than real, as long as we know which way it differs.

- The **size** of the gap per feature (discrepancy, marginals, score distributions) is reported as descriptive
  tables. Those claims are retired (see below); a ratio above 3 still marks a stress-test candidate.
- `vm.direction.*` gives the **direction** (easier / harder / comparable). `vm.protocol.transfer-direction`
  states which conclusions transfer: if the sim is never harder, failures transfer; if it is never easier,
  passes transfer; if mixed, only the stress tests count.
- Entrapment (`entrap.*`) compares sim and real at the level of the error rate itself.
- The stress tests (`vm.<row>.fdp-*`) ask whether the conclusion survives a deliberately worse simulation.

## Active claims (22, since 2026-10-13)

| File | Content |
|---|---|
| `claims/protocol.yaml` | primary estimand, truth rule, matched-complexity rule |
| `claims/protein.yaml` | protein-level estimand; protein-level entrapment estimator bias |
| `claims/entrapment.yaml` | entrapment estimator bias vs truth on sims; search-space shift; sim-vs-real band and verdict agreement |
| `claims/direction.yaml` | transfer rule; DIA-NN sim-not-harder / sim-not-easier |
| `claims/scores.yaml` | FragPipe sim-not-harder / sim-not-easier (incl. Percolator target/decoy AUC) |
| `claims/stress.yaml` | baseline FDP bound and nominal; stress rows mass-accuracy, predictor-swap, rt-spread (bound + delta) |

## Retired claims (37)

`retired/` holds the other 37 frozen claims, unchanged, and `retired/RETIRED.md` lists each with its freeze commit,
result and reason. They are out of the active manifest so the report shows only what decides the paper text;
their results are still computed and reported. Retired: pre-registration (now in the manifest header), all 12
discrepancy and 6 marginal claims, 7 score-distribution claims, 2 entrapment and 3 protein side claims, and the
never-run stress rows rt-peakshape, fragmentation and joint.

## What was frozen (history)


| File | Content |
|---|---|
| `claims/protocol.yaml` | pre-registration rule, primary estimand, truth rule, matched-complexity rule |
| `claims/entrapment.yaml` | estimator bias vs truth on sims, search-space shift, sim-vs-real band and verdict agreement, verdicts on real and sim |
| `claims/discrepancy.yaml` | per feature: sim-vs-real distance ≤ 3× the largest real replicate distance, at the matched complexity |
| `claims/direction.yaml` | transfer rule; per-feature easier/harder classification against the real replicate range |
| `claims/marginals.yaml` | Views (i) all MS1 ions from raw and (ii) identified precursors: RT, IM, m/z marginals, ratio ≤ 3 |
| `claims/scores.yaml` | DIA-NN q-value distribution; FragPipe score marginals (hyperscore, SpectralSim, RTScore, IMScore, expectation, q); FragPipe direction incl. Percolator target/decoy AUC |
| `claims/protein.yaml` | protein-level entrapment (FDRBench protein DB, PG.Q.Value ≤ 0.01): estimator bias, verdicts, sim vs real |
| `claims/stress.yaml` | baseline 1.5% bound and 1% nominal, plus per stress row a 1.5% bound and a 0.5 pp paired delta |

## Data (MOGON, `ki-proanagi`)

- **Real**: PXD029836 10 ng LS10/LS11/LS12, job 764353, `searches/vm/recipe-v1/real10/`.
- **Sim**: TimSim built from rustims 85260d52 (imspy-simulation 0.4.5), with LS_10 as reference layout.
  It runs 50k/125k/250k peptides × 3 seeds with 6.5 ppm m/z jitter and no background overlay. Jobs 764355 and 764357.
- **Search recipe**: DIA-NN 2.5.1 with a predicted library from the PXD029836 FASTA, a plain (I→L) and an
  FDRBench 1.1.1 paired entrapment variant (r = 1, seed 2026). MBR off, 15 ppm fixed, runs separate.

## Not yet frozen

- `evident/vm/stress_settings.yaml`, the predictor-swap stress settings, committed before any stress FDP was computed.
- `evident/vm/stress_settings_rt_spread.yaml`, the rt-spread row (sim RT apexes quantile-mapped onto the real RT spread), committed before its replays ran and before the predictor-swap stress FDP was opened.
- Analysis scripts under `evident/vm/`. They implement the frozen rules and must not change them.

## Views without a claim

- View (iii), the simulated ions themselves from `synthetic_data.db`, is plotted next to (i) and (ii). It has no tolerance:
  it differs from identified ions by design.
- DIA-NN reports decoys (`--report-decoys`) only under its reporting cutoff, so the full target/decoy score
  distributions come from FragPipe's kept Percolator tables.

## Known gaps (limitations, not claims)

- FragPipe's MSBooster uses DIA-NN's RT, IM and spectrum predictors. That adds to the predictor-overlap limitation.
- Phospho and HLA peptides are not covered by these claims.

- No background overlay: background fidelity is untested.
- One entrapment shuffle.
- Three runs per side.
- Only the 10 ng / 59 × 12 Th regime.
