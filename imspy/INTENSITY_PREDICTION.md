# Fragment Ion Intensity Prediction Integration

This document describes the integration of predicted fragment ion intensities into Sage's database search scoring.

## Overview

Sage supports intensity-weighted scoring where predicted fragment intensities (from external models like Prosit, MS2PIP, etc.) can be used to weight the hyperscore calculation. This enables better discrimination between correct and incorrect PSMs.

## Current Implementation Status

### Completed ✅

**sage-core (Rust)**
- `PeptideFragmentIntensities` struct for per-peptide intensity data
- `PredictedIntensityStore` for loading/creating intensity stores
  - `load(path)` - Load from `.sagi` binary file
  - `uniform(peptide_lengths, max_charge, ion_kinds)` - Create uniform store (all 1.0)
  - `from_predictions(...)` - Create from prediction vectors
  - `write(path)` - Write to `.sagi` file
  - `get_intensity(...)` / `get_intensity_or_default(...)` - Query intensities
- `Scorer` struct extended with `intensity_store: Option<&PredictedIntensityStore>`
- `Score` struct extended with `weighted_summed_b` / `weighted_summed_y` fields
- `ScoreType` enum with `WeightedSageHyperScore` and `WeightedOpenMSHyperScore`
- Scoring loop integration - uses predicted intensities when store is provided

**sagepy-connector (Python bindings)**
- `PyPredictedIntensityStore` class with:
  - `load(path)` - Load from file
  - `uniform(peptide_lengths, max_charge, ion_kinds)` - Create uniform store
  - `get_intensity(...)` / `get_intensity_or_default(...)` - Query intensities
  - Properties: `peptide_count`, `max_charge`, `ion_kinds`
- `PyScorer` scoring methods accept optional `intensity_store` parameter
- `PyScoreType` supports weighted score types

**sagepy (Python wrapper)**
- `PredictedIntensityStore` wrapper class with `uniform()` class method
- `Scorer` methods accept optional `intensity_store` parameter
- `ScoreType` supports `"weightedhyperscore"` and `"weightedopenmshyperscore"`
- `PredictionRequest` / `PredictionResult` dataclasses for imspy interface
- `.sagi` file I/O functions

### TODO ⏳

- Coordinate with imspy on interface contract and integration test
- Add CLI option for intensity file path (if needed)
- Configuration in JSON for weighted scoring (if needed)

---

## Decoy Intensity Predictions

### ⚠️ CRITICAL: Decoys MUST Be Predicted for Production Use

For **fair FDR estimation**, decoy peptides **MUST** have real intensity predictions - not uniform 1.0 values.

**Why this matters:**

| Peptide Type | With uniform decoys (WRONG) | With predicted decoys (CORRECT) |
|--------------|-----------------------------|---------------------------------|
| Target | `weighted = observed × predicted` (0.0-1.0) | `weighted = observed × predicted` (0.0-1.0) |
| Decoy | `weighted = observed × 1.0` (always full) | `weighted = observed × predicted` (0.0-1.0) |

**The problem with uniform decoys:**
- Decoy fragments always contribute at **full weight** (1.0)
- Target fragments may be **down-weighted** by low predicted intensities
- This **inflates decoy scores** relative to targets
- **Breaks FDR estimation** - more false positives pass thresholds

### Required Configuration for Production

```python
# REQUIRED for production weighted scoring
pipeline.predict_intensities(
    charges=[2, 3],
    exclude_decoys=False,  # ← MUST be False for fair scoring
)
```

**Why this works:**
- Decoy sequences are valid amino acid sequences (reversed targets)
- Prosit can predict their fragment intensities
- The intensity distribution will be similar to targets
- This ensures fair target-decoy competition

**Cost:** Doubles prediction workload (2× peptides), but this is necessary for correct FDR.

### When Uniform Decoys Are Acceptable

Uniform 1.0 intensities for decoys are **ONLY** acceptable for:

1. **Testing the code path** - comparing weighted vs unweighted scoring with ALL peptides having uniform intensities should give identical results
2. **Debugging** - isolating issues in the scoring pipeline
3. **NOT for production** - never use uniform decoys for actual searches

### Decoy Identification

Decoys are **NOT** identifiable by checking for "DECOY_" in the peptide sequence string. The decoy tag only appears in the protein header.

**Correct way to identify decoys:**
```python
db = config.generate_indexed_database()
for i, pep in enumerate(db._peptides):
    if pep.decoy:
        # This is a decoy peptide
        pass
```

### Implementation Verification (Complete)

| Component | Status | Details |
|-----------|--------|---------|
| imspy intensity store | ✅ Done | Stores predictions indexed by peptide_idx for all peptides |
| imspy decoy detection | ✅ Fixed | Uses `db._peptides[i].decoy` (not string check) |
| sage-core scoring | ✅ Verified | Looks up intensity for both target AND decoy peptides |
| sagepy-connector | ✅ Verified | Passes intensity_store correctly for all candidates |

**Technical details:**
- `IndexedDatabase.peptides` is a single flat array containing ALL peptides (targets + decoys)
- Scoring uses `score.peptide.0 as usize` to look up from intensity store
- This is the same index used for `db[score.peptide]` - identical for targets and decoys
- `peptides_as_string()` iterates the same array in order, so `peptide_lengths[i]` matches `PeptideIx(i)`

---

## Integration Testing Guide

> ⚠️ **Note:** The uniform intensity tests below are for **code path verification only**.
> For production weighted scoring, you MUST use `exclude_decoys=False` to predict
> intensities for ALL peptides. See "Decoy Intensity Predictions" section above.

### Testing the New Code Path with Uniform Intensities

The weighted scoring code path can be tested without actual predictions by using uniform intensities (all 1.0). This allows comparing the new weighted scoring path against the old unweighted path - **results should be identical**.

**Purpose:** Verify the weighted scoring code path works correctly, NOT for production use.

```python
from sagepy.core import (
  IndexedDatabase,
  Scorer,
  ScoreType,
  PredictedIntensityStore,
  Tolerance,
)

# 1. Build or load your database
indexed_db = ...  # Your IndexedDatabase

# 2. Create uniform intensity store (all predictions = 1.0)
peptide_lengths = [len(seq) for seq in indexed_db.peptides_as_string()]
uniform_store = PredictedIntensityStore.uniform(
  peptide_lengths,
  max_charge=3,
  ion_kinds=[1, 4],  # B and Y ions
)

# 3. Create scorers - one weighted, one unweighted
scorer_unweighted = Scorer(
  score_type=ScoreType("hyperscore"),
  # ... other parameters
)

scorer_weighted = Scorer(
  score_type=ScoreType("weightedhyperscore"),
  # ... same other parameters
)

# 4. Score same spectrum with both
# Unweighted (old path)
features_old = scorer_unweighted.score(indexed_db, spectrum)

# Weighted with uniform store (new path) - should give identical results
features_new = scorer_weighted.score(indexed_db, spectrum, intensity_store=uniform_store)

# 5. Compare results
for old, new in zip(features_old, features_new):
  assert abs(old.hyperscore - new.hyperscore) < 1e-6,
    f"Score mismatch: {old.hyperscore} vs {new.hyperscore}"
  assert old.peptide_idx == new.peptide_idx
  assert old.matched_peaks == new.matched_peaks
```

### Full Integration Test Example

```python
import numpy as np
from sagepy.core import (
  SageSearchConfiguration,
  EnzymeBuilder,
  SpectrumProcessor,
  Scorer,
  ScoreType,
  PredictedIntensityStore,
  Tolerance,
)

def test_weighted_scoring_with_uniform_intensities():
  """
  Test that weighted scoring with uniform (1.0) intensities
  produces identical results to unweighted scoring.
  """
  # Setup database
  config = SageSearchConfiguration(
    fasta=fasta_string,
    static_mods={"C": "[UNIMOD:4]"},
    variable_mods={"M": ["[UNIMOD:35]"]},
    enzyme_builder=EnzymeBuilder(
      missed_cleavages=2,
      min_len=7,
      max_len=30,
      cleave_at="KR",
      restrict="P",
      c_terminal=True,
    ),
    generate_decoys=True,
  )
  indexed_db = config.generate_indexed_database()

  # Create uniform intensity store
  peptide_lengths = [len(seq) for seq in indexed_db.peptides_as_string()]
  uniform_store = PredictedIntensityStore.uniform(peptide_lengths)

  # Process spectra
  processor = SpectrumProcessor(...)
  spectra = [processor.process(raw) for raw in raw_spectra]

  # Score with both methods
  scorer_params = dict(
    precursor_tolerance=Tolerance(ppm=(-10, 10)),
    fragment_tolerance=Tolerance(ppm=(-20, 20)),
    min_matched_peaks=6,
    report_psms=5,
    annotate_matches=True,
    variable_mods={"M": ["[UNIMOD:35]"]},
    static_mods={"C": "[UNIMOD:4]"},
  )

  scorer_old = Scorer(score_type=ScoreType("hyperscore"), **scorer_params)
  scorer_new = Scorer(score_type=ScoreType("weightedhyperscore"), **scorer_params)

  # Compare results for each spectrum
  for spectrum in spectra:
    features_old = scorer_old.score(indexed_db, spectrum)
    features_new = scorer_new.score(indexed_db, spectrum, intensity_store=uniform_store)

    assert len(features_old) == len(features_new), "Different number of PSMs"

    for old, new in zip(features_old, features_new):
      # Scores should be identical with uniform intensities
      assert abs(old.hyperscore - new.hyperscore) < 1e-5,
        f"Hyperscore mismatch: {old.hyperscore} vs {new.hyperscore}"
      assert old.peptide_idx == new.peptide_idx, "Different peptide matched"
      assert old.matched_peaks == new.matched_peaks, "Different peak count"
      assert abs(old.matched_intensity_pct - new.matched_intensity_pct) < 1e-5

  print("✅ All tests passed - weighted path matches unweighted path")
```

### Testing with Batch Scoring

```python
def test_batch_scoring_with_uniform():
  """Test score_collection methods with intensity store."""
  # ... setup as above ...

  # Batch scoring comparison
  features_old = scorer_old.score_collection_top_n(indexed_db, spectra, num_threads=4)
  features_new = scorer_new.score_collection_top_n(
    indexed_db, spectra, num_threads=4, intensity_store=uniform_store
  )

  for old_list, new_list in zip(features_old, features_new):
    assert len(old_list) == len(new_list)
    for old, new in zip(old_list, new_list):
      assert abs(old.hyperscore - new.hyperscore) < 1e-5
```

---

## Score Types

| Score Type | Description | Use Case |
|------------|-------------|----------|
| `hyperscore` | Original X!Tandem-style | Standard scoring |
| `openmshyperscore` | OpenMS variant | Standard scoring |
| `weightedhyperscore` | Uses `observed * predicted` weighted sums | With intensity predictions |
| `weightedopenmshyperscore` | OpenMS variant with weighting | With intensity predictions |

**Important**: When using weighted score types without an intensity store (or with uniform 1.0 intensities), results are mathematically identical to unweighted scoring.

---

## Binary File Format (`.sagi`)

The file has three sections:

### Header (18+ bytes, little-endian)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | magic | `0x49474153` ("SAGI") |
| 4 | 4 | version | Currently `1` |
| 8 | 8 | peptide_count | Number of peptides (u64) |
| 16 | 1 | max_charge | Maximum fragment charge state |
| 17 | 1 | ion_kind_count | Number of ion types |
| 18 | N | ion_kinds | Ion type codes (N = ion_kind_count) |

Ion kind codes: 0=A, 1=B, 2=C, 3=X, 4=Y, 5=Z

### Offsets Section (peptide_count * 8 bytes)

Array of `u64` byte offsets into the data section for each peptide.

### Data Section (variable size)

Concatenated `f32` arrays for each peptide's predicted intensities.
Layout: `[kind_0_pos_0_charge_1, kind_0_pos_0_charge_2, ..., kind_K_pos_(L-2)_charge_C]`

---

## Imspy Interface

### What imspy receives (PredictionRequest)

| Column | Type | Description |
|--------|------|-------------|
| `sequence` | str | Modified peptide with UNIMOD notation |
| `charge` | int32 | Precursor charge state |
| `peptide_idx` | int64 | Index for mapping back |

### What imspy must return (PredictionResult)

| Column | Type | Description |
|--------|------|-------------|
| `peptide_idx` | int64 | Same indices from input |
| `charge` | int32 | Same charges from input |
| `intensities` | np.ndarray | Shape: `[n_ion_kinds, seq_len-1, max_frag_charge]` |

### Intensity Array Specification

- Shape: `[n_ion_kinds, n_positions, max_fragment_charge]`
- Ion kind ordering: Index 0 = B ions, Index 1 = Y ions
- Values: Normalized intensities in `[0.0, 1.0]`
- dtype: `float32`

---

## API Reference

### PredictedIntensityStore

```python
from sagepy.core import PredictedIntensityStore

# Load from file
store = PredictedIntensityStore("predictions.sagi")

# Create uniform store for testing
store = PredictedIntensityStore.uniform(
  peptide_lengths=[10, 15, 12, ...],  # Length of each peptide
  max_charge=3,
  ion_kinds=[1, 4],  # B and Y
)

# Query
intensity = store.get_intensity(peptide_idx, peptide_len, ion_kind, position, charge)
intensity = store.get_intensity_or_default(...)  # Returns 1.0 if not found

# Metadata
store.peptide_count  # Number of peptides
store.max_charge     # Max fragment charge
store.ion_kinds      # List of ion type codes
```

### Scorer with Intensity Store

```python
from sagepy.core import Scorer, ScoreType

# Create weighted scorer
scorer = Scorer(
  score_type=ScoreType("weightedhyperscore"),
  # ... other parameters
)

# Score single spectrum
features = scorer.score(db, spectrum, intensity_store=store)

# Score collection
features = scorer.score_collection_top_n(db, spectra, num_threads=4, intensity_store=store)

# Score to PSMs
psms = scorer.score_collection_psm(db, spectra, num_threads=4, intensity_store=store)
```

---

## File Locations

- `crates/sage/src/intensity_prediction.rs` - Core data structures and file I/O
- `crates/sage/src/scoring.rs` - Score types and scoring logic
- `sagepy-connector/src/py_intensity.rs` - Python bindings
- `sagepy-connector/src/py_scoring.rs` - Scorer bindings with intensity_store
- `sagepy/sagepy/core/intensity.py` - Python interface layer
- `sagepy/sagepy/core/scoring.py` - Python Scorer wrapper
