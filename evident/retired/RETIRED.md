# Retired claims

59 claims registered, **22 active**, **37 retired** (2026-10-13). Nothing was deleted and no tolerance changed.

## What retirement means here
A retired claim is out of the active manifest (`evident.yaml`), so the trust report and any agent reading it see only
the claims that decide what the paper says. It is not exonerated or erased: its frozen block is in this folder, the
analysis scripts still compute its value, and its result is reported below. A retired claim can be restored under the
same id only with its rule unchanged; a changed rule needs a new id.

Integrity check at retirement: the parsed YAML of every claim (active and retired) equals its last active version.
The text is not byte-identical: where a retired block relied on a YAML anchor defined in another block, the anchor's
value is now written out.

## Record
- Decided 2026-10-13 by the author, on a proposal by the analysis assistant, after one independent review round
  (Codex); two claims proposed for retirement were kept active on that review (entrapment search-space shift and the
  sim-real entrapment band).
- **This is a post-result amendment.** At retirement the stage A/B results were known (matched complexity,
  discrepancy, direction, entrapment, baseline FDP; sidecar `analysis-v1/last_verified.json` sha256 1c93d9da51da238d...).
  Not yet seen: stage C (marginals, scores, protein; job running), the predictor-swap stress FDP (searches done,
  unopened) and rt-spread (running). Retiring all 12 failed discrepancy claims therefore carries a selection concern.
  Mitigations: the failed headline claims (baseline FDP, estimator bias, both direction claims) stay active; passed
  claims were retired too; every result stays reported; the discrepancy, marginal and score results will be shown
  as descriptive tables with their frozen thresholds.
- The mass-accuracy stress row stays active and was chosen with knowledge of stage A (the sims are much easier on
  mass error). Its settings are not frozen yet; they will be frozen before any of its data exist.
- Known weakness, kept as frozen: the transfer rule (`vm.protocol.transfer-direction`) assumes that a uniformly easier
  or harder simulation moves FDP in a known direction; that is not shown. With the observed mixed direction the rule
  defers to the stress tests anyway.
- Scope after retirement: no protein-level sim-to-real transfer is claimed; robustness to peak shape and fragmentation
  is not tested.

## Results at retirement (snapshot, do not edit)
Of the 37: 2 passed, 12 failed, 22 not yet assessed, 1 policy.
Stage A/B values, rustims 85260d52.

| Claim | File | Frozen in | Result at retirement | Why retired |
|---|---|---|---|---|
| `vm.discrepancy.ms1_ppm.diann` | discrepancy.yaml | 4359420c | fail (ratio=6.198) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `vm.discrepancy.fr_ppm.diann` | discrepancy.yaml | 4359420c | fail (ratio=7.026) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `vm.discrepancy.fwhm.diann` | discrepancy.yaml | 4359420c | fail (ratio=4.703) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `vm.discrepancy.rt_width.diann` | discrepancy.yaml | 4359420c | fail (ratio=7.809) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `vm.discrepancy.rt_resid.diann` | discrepancy.yaml | 4359420c | fail (ratio=72.61) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `vm.discrepancy.im_resid.diann` | discrepancy.yaml | 4359420c | fail (ratio=56.3) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `vm.discrepancy.log10_ms1_area.diann` | discrepancy.yaml | 4359420c | fail (ratio=39.9) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `vm.discrepancy.evidence.diann` | discrepancy.yaml | 4359420c | fail (ratio=23.58) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `vm.discrepancy.ms1_profile_corr.diann` | discrepancy.yaml | 4359420c | fail (ratio=29.34) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `vm.discrepancy.mass_evidence.diann` | discrepancy.yaml | 4359420c | fail (ratio=25.14) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `vm.discrepancy.pep.diann` | discrepancy.yaml | 4359420c | fail (ratio=8.22) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `vm.discrepancy.charge.diann` | discrepancy.yaml | 4359420c | fail (ratio=157.3) | Per-feature size of the sim-real gap. Reported as a descriptive table; the direction claims carry the conclusion. |
| `entrap.verdict.diann.real10` | entrapment.yaml | d09eeefc | pass (lower_bound_max=0.001221, paired_upper_max=0.002319) | Per-side verdict; an input to the active verdict-agree claim. |
| `entrap.verdict.diann.sim-matched` | entrapment.yaml | d09eeefc | pass (lower_bound_max=0.001631, paired_upper_max=0.003114) | Per-side verdict; an input to the active verdict-agree claim. |
| `vm.marginal.rt.identified.diann` | marginals.yaml | 29572ed1 | not assessed | RT/IM/m/z distribution gap. Reported descriptively; same question as the discrepancy table. |
| `vm.marginal.rt.allions.raw` | marginals.yaml | 29572ed1 | not assessed | RT/IM/m/z distribution gap. Reported descriptively; same question as the discrepancy table. |
| `vm.marginal.im.identified.diann` | marginals.yaml | 29572ed1 | not assessed | RT/IM/m/z distribution gap. Reported descriptively; same question as the discrepancy table. |
| `vm.marginal.im.allions.raw` | marginals.yaml | 29572ed1 | not assessed | RT/IM/m/z distribution gap. Reported descriptively; same question as the discrepancy table. |
| `vm.marginal.mz.identified.diann` | marginals.yaml | 29572ed1 | not assessed | RT/IM/m/z distribution gap. Reported descriptively; same question as the discrepancy table. |
| `vm.marginal.mz.allions.raw` | marginals.yaml | 29572ed1 | not assessed | RT/IM/m/z distribution gap. Reported descriptively; same question as the discrepancy table. |
| `entrap.protein.verdict.diann.real10` | protein.yaml | 29572ed1 | not assessed | Protein-level sim-real bridge. Scope narrowed: no protein-level sim-to-real transfer is claimed; protein bias on sims stays active. |
| `entrap.protein.verdict.diann.sim-matched` | protein.yaml | 29572ed1 | not assessed | Protein-level sim-real bridge. Scope narrowed: no protein-level sim-to-real transfer is claimed; protein bias on sims stays active. |
| `entrap.protein.sim-vs-real.diann` | protein.yaml | 29572ed1 | not assessed | Protein-level sim-real bridge. Scope narrowed: no protein-level sim-to-real transfer is claimed; protein bias on sims stays active. |
| `vm.protocol.preregistration` | protocol.yaml | d09eeefc | policy | Process statement, not a measurement; the manifest header carries it. |
| `vm.scores.qvalue.diann` | scores.yaml | 29572ed1 | not assessed | Score-distribution gap. Reported descriptively; the FragPipe direction claims stay active. |
| `vm.scores.hyperscore.fragpipe` | scores.yaml | 29572ed1 | not assessed | Score-distribution gap. Reported descriptively; the FragPipe direction claims stay active. |
| `vm.scores.spectral_sim.fragpipe` | scores.yaml | 29572ed1 | not assessed | Score-distribution gap. Reported descriptively; the FragPipe direction claims stay active. |
| `vm.scores.rt_score.fragpipe` | scores.yaml | 29572ed1 | not assessed | Score-distribution gap. Reported descriptively; the FragPipe direction claims stay active. |
| `vm.scores.im_score.fragpipe` | scores.yaml | 29572ed1 | not assessed | Score-distribution gap. Reported descriptively; the FragPipe direction claims stay active. |
| `vm.scores.log10_expectation.fragpipe` | scores.yaml | 29572ed1 | not assessed | Score-distribution gap. Reported descriptively; the FragPipe direction claims stay active. |
| `vm.scores.log10_qvalue.fragpipe` | scores.yaml | 29572ed1 | not assessed | Score-distribution gap. Reported descriptively; the FragPipe direction claims stay active. |
| `vm.rt-peakshape.fdp-bound.diann` | stress.yaml | d09eeefc | not assessed | Stress row never designed or run. Scope: robustness to this mismatch is not tested, and the paper states that as a limitation. |
| `vm.rt-peakshape.fdp-delta.diann` | stress.yaml | d09eeefc | not assessed | Stress row never designed or run. Scope: robustness to this mismatch is not tested, and the paper states that as a limitation. |
| `vm.fragmentation.fdp-bound.diann` | stress.yaml | d09eeefc | not assessed | Stress row never designed or run. Scope: robustness to this mismatch is not tested, and the paper states that as a limitation. |
| `vm.fragmentation.fdp-delta.diann` | stress.yaml | d09eeefc | not assessed | Stress row never designed or run. Scope: robustness to this mismatch is not tested, and the paper states that as a limitation. |
| `vm.joint.fdp-bound.diann` | stress.yaml | d09eeefc | not assessed | Never run as its own row; the joint setting is inside the active predictor-swap claims' scope. |
| `vm.joint.fdp-delta.diann` | stress.yaml | d09eeefc | not assessed | Never run as its own row; the joint setting is inside the active predictor-swap claims' scope. |

## Results after retirement (append only)
Stage C values for the retired marginal, score and protein claims are added here when the job finishes.
