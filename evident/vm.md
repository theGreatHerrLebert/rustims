# TimSim validity matrix: internal claims

Internal check for the MCP revision (MCPRO-D-26-00416). Not for the manuscript, SI, Zenodo or the response letter.

Plan: `SUBMISSION/revision/VM_EVIDENT_PLAN.md` (v3, two Codex rounds). Step-0 audit: `VM_AUDIT_STEP0.md`.

## How to read the result

Matching distributions is sufficient but not necessary. A conclusion about a search engine can transfer
to real data even if the simulation is easier or harder than real, as long as we know which way it differs.

- `vm.discrepancy.*` gives the **size** of the gap per feature. It is descriptive, not a validity gate.
  A ratio above 3 marks a stress-test candidate.
- `vm.direction.*` gives the **direction** (easier / harder / comparable). `vm.protocol.transfer-direction`
  states which conclusions transfer: if the sim is never harder, failures transfer; if it is never easier,
  passes transfer; if mixed, only the stress tests count.
- Entrapment (`entrap.*`) compares sim and real at the level of the error rate itself.
- The stress tests (`vm.<row>.fdp-*`) ask whether the conclusion survives a deliberately worse simulation.

## What is frozen here

| File | Content |
|---|---|
| `claims/protocol.yaml` | pre-registration rule, primary estimand, truth rule, matched-complexity rule |
| `claims/entrapment.yaml` | estimator bias vs truth on sims, search-space shift, sim-vs-real band and verdict agreement, verdicts on real and sim |
| `claims/discrepancy.yaml` | per feature: sim-vs-real distance ≤ 3× the largest real replicate distance, at the matched complexity |
| `claims/direction.yaml` | transfer rule; per-feature easier/harder classification against the real replicate range |
| `claims/stress.yaml` | baseline 1.5% bound and 1% nominal, plus per stress row a 1.5% bound and a 0.5 pp paired delta |

## Data (MOGON, `ki-proanagi`)

- **Real**: PXD029836 10 ng LS10/LS11/LS12, job 764353, `searches/vm/recipe-v1/real10/`.
- **Sim**: TimSim built from rustims 85260d52 (imspy-simulation 0.4.5), with LS_10 as reference layout.
  It runs 50k/125k/250k peptides × 3 seeds with 6.5 ppm m/z jitter and no background overlay. Jobs 764355 and 764357.
- **Search recipe**: DIA-NN 2.5.1 with a predicted library from the PXD029836 FASTA, a plain (I→L) and an
  FDRBench 1.1.1 paired entrapment variant (r = 1, seed 2026). MBR off, 15 ppm fixed, runs separate.

## Not yet frozen

- `evident/vm/stress_settings.yaml`, the stress settings. It will be committed from the section-3 results before any stress FDP is computed.
- Analysis scripts under `evident/vm/`. They implement the frozen rules and must not change them.

## Known gaps (limitations, not claims)

- No background overlay: background fidelity is untested.
- One entrapment shuffle.
- Three runs per side.
- Only the 10 ng / 59 × 12 Th regime.
