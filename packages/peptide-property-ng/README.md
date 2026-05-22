# peptide-property-ng

A next-generation, clean-slate neural network for **peptide property prediction**
on timsTOF proteomics data — fragment intensity, ion mobility / CCS, retention
time and precursor charge from **one shared encoder**.

It is a research-track sibling of `imspy-predictors`, not a replacement for the
production model. See `docs/nn-architecture-exploration/` in the
`claudius-proteomics` project for the design rationale.

## What is different from the production `UnifiedPeptideModel`

| | production `imspy_predictors` | this model |
|---|---|---|
| Modifications | opaque `[UNIMOD:N]` tokens (no chemistry) | **hybrid**: per-mod token **+** atomic-composition feature |
| Unseen mods | embedding stays ~random | generalize via chemistry; open-vocab |
| Conditioning | dormant instrument embedding, never used | instrument / acq-mode / charge / m-z / CE inside the encoder |
| Intensity output | fixed 174-vector → 30-residue cap | **fragment-indexed** → no length cap |
| Backbone | bespoke transformer | built on **Depthcharge** primitives |

## Layout

```
src/peptide_property_ng/
  modifications/   UNIMOD atomic-composition table (from sagepy)
  model/           hybrid embedding, Depthcharge-based encoder, task heads
  data/            Sage-parquet dataset, fragment targets, splits, collate
  train/           multi-task training
  eval/            per-task + OOD + unseen-modification evaluation
```

## Quick start

```bash
# 1. build the modification-composition table (one-off)
python -m peptide_property_ng.modifications.build_table

# 2. train the first-prototype model on processed Sage results
python -m peptide_property_ng.train.train --datasets-glob '/scratch/claudius-proteomics/*'
```
