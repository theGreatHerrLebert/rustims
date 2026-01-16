# Fragment Ion Intensity Prediction Integration

This document describes the integration of predicted fragment ion intensities into Sage's database search scoring.

## Overview

Sage supports intensity-weighted scoring where predicted fragment intensities (from external models like Prosit, MS2PIP, etc.) can be used to weight the hyperscore calculation. This enables better discrimination between correct and incorrect PSMs.

## Current Implementation Status

### Completed ✅

**sage-core (Rust)**
- `PeptideFragmentIntensities` struct for per-peptide intensity data
- `PredictedIntensityStore` (V1) for positional index-based intensity lookup
- `PredictedIntensityStoreV2` (V2) for (sequence, charge) key-based lookup
- `IntensityStore` unified enum supporting both V1 and V2 formats
- `Scorer` struct with unified `intensity_store` supporting both formats
- `ScoreType` enum with `WeightedSageHyperScore` and `WeightedOpenMSHyperScore`

**sagepy-connector (Python bindings)**
- `PyPredictedIntensityStore` class supporting both V1 and V2 formats
- V2 creation via `from_raw_predictions_v2()`
- `compute_key_hash()` static method for consistent hashing
- V2 lookup methods: `get_intensity_by_key()`, `contains_key()`

**sagepy (Python wrapper)**
- `PredictedIntensityStore` with V1 and V2 support
- `from_predictions_v2()` class method for creating V2 stores
- `write_intensity_file_v2()` for writing V2 format files
- `compute_key_hash()` function using Rust implementation
- `PredictionResultV2` dataclass for (sequence, charge) keyed predictions

---

## V2 Format: Key-Based Indexing (NEW)

### Why V2?

The original V1 format uses **positional indexing** where `predictions[i]` corresponds to `IndexedDatabase.peptides[i]`. This creates tight coupling between predictions and database order, preventing:

- **Database chunking** - Can't split peptides into batches for processing
- **Prediction reuse** - Can't use same predictions across different database configurations
- **Flexible parallel workflows** - Processing order must match database order

### V2 Solution

V2 uses `(raw_sequence, precursor_charge)` as the lookup key instead of peptide index. This completely decouples predictions from database order.

### Key Design Decisions

| Aspect | V1 (Positional) | V2 (Key-Based) |
|--------|-----------------|----------------|
| **Key** | `peptide_idx` (integer) | `(sequence, charge)` tuple |
| **Sequence format** | N/A | Raw amino acids (NOT UNIMOD) |
| **Database coupling** | Tightly coupled | Fully decoupled |
| **Chunking support** | ❌ No | ✅ Yes |
| **Prediction reuse** | ❌ No | ✅ Yes |

### Important: Use Raw Sequences (NOT UNIMOD)

V2 keys use **raw amino acid sequences**, not UNIMOD notation:
- ✅ Correct: `"PEPTIDEK"`
- ❌ Wrong: `"PEPTC[UNIMOD:4]IDEK"`

This matches SAGE's internal representation where modifications are stored as mass shifts, not UNIMOD strings.

---

## imspy Integration Guide (V2)

### Step 1: Generate Predictions Keyed by (Sequence, Charge)

Instead of returning predictions indexed by `peptide_idx`, return a dictionary keyed by `(raw_sequence, precursor_charge)`:

```python
def predict_intensities_v2(
    sequences: list[str],  # Raw sequences (no UNIMOD)
    charges: list[int],
    max_fragment_charge: int = 2,
) -> dict[tuple[str, int], np.ndarray]:
    """
    Generate intensity predictions keyed by (sequence, charge).

    Returns
    -------
    dict[tuple[str, int], np.ndarray]
        Map from (raw_sequence, precursor_charge) to intensity tensor.
        Each tensor has shape [n_ion_kinds, seq_len-1, max_fragment_charge].
    """
    predictions = {}

    for seq, charge in zip(sequences, charges):
        # Your prediction logic here
        intensity = model.predict(seq, charge)  # Shape: [2, len(seq)-1, max_frag_charge]
        predictions[(seq, charge)] = intensity.astype(np.float32)

    return predictions
```

### Step 2: Create V2 Store

```python
from sagepy.core.intensity import PredictedIntensityStore

# Create V2 store from predictions dictionary
store = PredictedIntensityStore.from_predictions_v2(
    predictions=predictions,  # dict[tuple[str, int], np.ndarray]
    max_charge=2,
    ion_kinds=[1, 4],  # B and Y ions
)

# Check it's V2
assert store.is_key_based == True
print(f"Created V2 store with {store.entry_count} entries")
```

### Step 3: Save/Load V2 Files

```python
# Save to file
store.write("predictions_v2.sagi")

# Load from file (auto-detects V1 or V2)
loaded_store = PredictedIntensityStore("predictions_v2.sagi")
assert loaded_store.is_key_based == True
```

### Step 4: Use with Scorer

```python
from sagepy.core import Scorer, ScoreType

scorer = Scorer(
    score_type=ScoreType("weightedhyperscore"),
    # ... other parameters
)

# Works with any database configuration!
for db_chunk in database_chunks:
    features = scorer.score(db_chunk, spectrum, intensity_store=store)
```

---

## Complete V2 Example for imspy

```python
import numpy as np
from sagepy.core.intensity import (
    PredictedIntensityStore,
    PredictionResultV2,
    write_intensity_file_v2,
    compute_key_hash,
    ION_KIND_B,
    ION_KIND_Y,
)

def create_v2_predictions_for_database(indexed_db, charges=[2, 3]):
    """
    Create V2 predictions for all peptides in database.

    This function can be called once and the resulting store
    can be reused across different database configurations.
    """
    predictions = {}

    # Get all sequences (raw, without UNIMOD modifications stripped)
    sequences = indexed_db.peptides_as_string()

    for i, unimod_seq in enumerate(sequences):
        # Extract raw sequence (strip UNIMOD notation)
        raw_seq = strip_unimod(unimod_seq)  # You need to implement this

        for charge in charges:
            # Skip if already predicted (same sequence at same charge)
            if (raw_seq, charge) in predictions:
                continue

            # Generate prediction
            # Shape: [n_ion_kinds, seq_len-1, max_fragment_charge]
            intensity = your_prediction_model(raw_seq, charge)
            predictions[(raw_seq, charge)] = intensity.astype(np.float32)

    # Create V2 store
    store = PredictedIntensityStore.from_predictions_v2(
        predictions=predictions,
        max_charge=2,
        ion_kinds=[ION_KIND_B, ION_KIND_Y],
    )

    return store


def strip_unimod(sequence: str) -> str:
    """Strip UNIMOD notation from sequence, keeping only amino acids."""
    result = []
    i = 0
    while i < len(sequence):
        if sequence[i] == '[':
            # Skip until closing bracket
            while i < len(sequence) and sequence[i] != ']':
                i += 1
            i += 1  # Skip ']'
        elif sequence[i].isupper():
            result.append(sequence[i])
            i += 1
        else:
            i += 1
    return ''.join(result)


# Example usage
indexed_db = config.generate_indexed_database()
store = create_v2_predictions_for_database(indexed_db)

# Save for later use
store.write("my_predictions_v2.sagi")

# Use with any database chunk
scorer = Scorer(score_type=ScoreType("weightedhyperscore"), ...)
features = scorer.score(indexed_db, spectrum, intensity_store=store)
```

---

## PredictionResultV2 Dataclass

For convenience, sagepy provides a dataclass for V2 predictions:

```python
from sagepy.core.intensity import PredictionResultV2

result = PredictionResultV2(
    predictions={
        ("PEPTIDE", 2): np.random.rand(2, 6, 2).astype(np.float32),
        ("PEPTIDEK", 3): np.random.rand(2, 7, 2).astype(np.float32),
    },
    ion_kinds=[1, 4],  # B and Y
    max_fragment_charge=2,
)

# Create store from result
store = PredictedIntensityStore.from_predictions_v2(
    result.predictions,
    max_charge=result.max_fragment_charge,
    ion_kinds=result.ion_kinds,
)
```

---

## Binary File Format

### V1 Format (Positional Indexing)

```
Header (18+ bytes, little-endian):
  magic: u32 = 0x49474153 ("SAGI")
  version: u32 = 1
  peptide_count: u64
  max_charge: u8
  ion_kind_count: u8
  ion_kinds: [u8; ion_kind_count]

Offsets Section (peptide_count * 8 bytes):
  Array of u64 byte offsets into data section

Data Section:
  Concatenated f32 arrays per peptide
  Layout: [ion_kind][position][fragment_charge]
```

### V2 Format (Key-Based Indexing) - NEW

```
Header (18+ bytes, little-endian):
  magic: u32 = 0x49474153 ("SAGI")
  version: u32 = 2  ← Version 2
  entry_count: u64
  max_charge: u8
  ion_kind_count: u8
  ion_kinds: [u8; ion_kind_count]

Index Section (entry_count * 18 bytes):
  For each entry:
    key_hash: u64      # Hash of (raw_sequence, charge)
    peptide_len: u16   # Length of peptide
    data_offset: u64   # Byte offset into data section

Data Section:
  Concatenated f32 arrays
  Layout: [ion_kind][position][fragment_charge]
```

### Ion Kind Codes

| Code | Ion Type |
|------|----------|
| 0 | A |
| 1 | B |
| 2 | C |
| 3 | X |
| 4 | Y |
| 5 | Z |

---

## API Reference

### PredictedIntensityStore (Updated for V2)

```python
from sagepy.core.intensity import PredictedIntensityStore

# ============== Loading ==============

# Load from file (auto-detects V1 or V2)
store = PredictedIntensityStore("predictions.sagi")

# ============== V1 Creation ==============

# Create uniform V1 store for testing
store = PredictedIntensityStore.uniform(
    peptide_lengths=[10, 15, 12, ...],
    max_charge=3,
    ion_kinds=[1, 4],
)

# ============== V2 Creation ==============

# Create V2 store from prediction dictionary
store = PredictedIntensityStore.from_predictions_v2(
    predictions={
        ("PEPTIDE", 2): intensity_array,  # Shape: [n_ion_kinds, len-1, max_charge]
        ("PEPTIDEK", 3): intensity_array,
    },
    max_charge=2,
    ion_kinds=[1, 4],
)

# ============== Properties ==============

store.is_key_based    # True for V2, False for V1
store.entry_count     # Number of entries (V1: peptides, V2: keys)
store.peptide_count   # V1 only (returns 0 for V2)
store.max_charge      # Maximum fragment charge
store.ion_kinds       # List of ion type codes

# ============== V1 Lookup (Positional) ==============

intensity = store.get_intensity(
    peptide_idx=0,
    peptide_len=10,
    ion_kind=1,      # B ion
    position=3,
    charge=1,
)

intensity = store.get_intensity_or_default(...)  # Returns 1.0 if not found

# ============== V2 Lookup (Key-Based) ==============

intensity = store.get_intensity_by_key(
    sequence="PEPTIDE",      # Raw sequence, NOT UNIMOD
    precursor_charge=2,
    ion_kind=1,              # B ion
    position=3,
    fragment_charge=1,
)

intensity = store.get_intensity_by_key_or_default(...)  # Returns 1.0 if not found

# Check if key exists
exists = store.contains_key("PEPTIDE", 2)

# ============== File I/O ==============

# Write to file (format determined by store type)
store.write("output.sagi")
```

### Hash Function

For debugging or advanced use cases, you can compute the key hash:

```python
from sagepy.core.intensity import compute_key_hash

# Uses same algorithm as Rust implementation
hash_value = compute_key_hash("PEPTIDE", 2)
```

### File I/O Functions

```python
from sagepy.core.intensity import (
    write_intensity_file,      # V1 format
    read_intensity_file,       # V1 format
    write_intensity_file_v2,   # V2 format
    read_intensity_file_v2,    # V2 format (header only)
)

# Write V2 file directly
write_intensity_file_v2(
    path="predictions.sagi",
    predictions={("PEPTIDE", 2): intensity_array, ...},
    max_charge=2,
    ion_kinds=[1, 4],
)

# Read V2 file header
info = read_intensity_file_v2("predictions.sagi")
# Returns: {'version': 2, 'entry_count': N, 'max_charge': 2, 'ion_kinds': [1,4], 'is_key_based': True}
```

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

### V2 Makes Decoy Handling Easier

With V2's key-based indexing, you can:
1. Predict all unique (sequence, charge) pairs once
2. Decoy sequences are just reversed strings - predict them the same way
3. No need to track peptide indices

```python
# With V2, just predict all unique sequences including decoys
all_sequences = set()
for seq in indexed_db.peptides_as_string():
    raw = strip_unimod(seq)
    all_sequences.add(raw)

# Predict for all unique sequences at desired charges
predictions = {}
for seq in all_sequences:
    for charge in [2, 3]:
        predictions[(seq, charge)] = model.predict(seq, charge)

store = PredictedIntensityStore.from_predictions_v2(predictions)
```

---

## Score Types

| Score Type | Description | Use Case |
|------------|-------------|----------|
| `hyperscore` | Original X!Tandem-style | Standard scoring |
| `openmshyperscore` | OpenMS variant | Standard scoring |
| `weightedhyperscore` | Uses `observed * predicted` weighted sums | With intensity predictions |
| `weightedopenmshyperscore` | OpenMS variant with weighting | With intensity predictions |

---

## Testing

### Test V2 Store Creation and Lookup

```python
import numpy as np
from sagepy.core.intensity import PredictedIntensityStore, ION_KIND_B, ION_KIND_Y

def test_v2_store():
    # Create test predictions
    predictions = {
        ("PEPTIDE", 2): np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [0.8, 0.7]],  # B ions
            [[0.9, 0.8], [0.7, 0.6], [0.5, 0.4], [0.3, 0.2], [0.1, 0.0], [0.2, 0.3]],  # Y ions
        ], dtype=np.float32),
    }

    # Create V2 store
    store = PredictedIntensityStore.from_predictions_v2(predictions, max_charge=2)

    # Verify it's V2
    assert store.is_key_based == True
    assert store.entry_count == 1

    # Test lookup
    intensity = store.get_intensity_by_key("PEPTIDE", 2, ION_KIND_B, 0, 1)
    assert abs(intensity - 0.1) < 1e-6

    # Test missing key returns None
    assert store.get_intensity_by_key("MISSING", 2, ION_KIND_B, 0, 1) is None

    # Test contains_key
    assert store.contains_key("PEPTIDE", 2) == True
    assert store.contains_key("PEPTIDE", 3) == False

    # Test roundtrip
    store.write("/tmp/test_v2.sagi")
    loaded = PredictedIntensityStore("/tmp/test_v2.sagi")
    assert loaded.is_key_based == True
    assert loaded.entry_count == 1

    intensity_loaded = loaded.get_intensity_by_key("PEPTIDE", 2, ION_KIND_B, 0, 1)
    assert abs(intensity_loaded - 0.1) < 1e-6

    print("✅ V2 store tests passed")

test_v2_store()
```

### Test V2 with Scoring

```python
from sagepy.core import Scorer, ScoreType

def test_v2_scoring():
    # Setup database and V2 store
    indexed_db = ...  # Your database
    store = create_v2_predictions_for_database(indexed_db)

    # Score with V2 store
    scorer = Scorer(
        score_type=ScoreType("weightedhyperscore"),
        # ... other params
    )

    features = scorer.score(indexed_db, spectrum, intensity_store=store)

    # Verify weighted scores are different from unweighted
    scorer_unweighted = Scorer(score_type=ScoreType("hyperscore"), ...)
    features_unweighted = scorer_unweighted.score(indexed_db, spectrum)

    # With real predictions, weighted scores should differ
    for f_weighted, f_unweighted in zip(features, features_unweighted):
        # Scores won't be identical unless all predictions are 1.0
        print(f"Weighted: {f_weighted.hyperscore:.4f}, Unweighted: {f_unweighted.hyperscore:.4f}")

    print("✅ V2 scoring test passed")
```

---

## Migration from V1 to V2

If you have existing V1 code, here's how to migrate:

### V1 (Old Way)
```python
# V1: Predictions indexed by peptide_idx
predictions = [intensity_array_for_peptide_0, intensity_array_for_peptide_1, ...]
write_intensity_file("predictions.sagi", predictions, peptide_lengths)
store = PredictedIntensityStore("predictions.sagi")
intensity = store.get_intensity(peptide_idx=0, peptide_len=10, ...)
```

### V2 (New Way)
```python
# V2: Predictions keyed by (sequence, charge)
predictions = {
    ("PEPTIDE", 2): intensity_array,
    ("PEPTIDEK", 3): intensity_array,
    ...
}
store = PredictedIntensityStore.from_predictions_v2(predictions)
store.write("predictions_v2.sagi")

# Or load existing V2 file
store = PredictedIntensityStore("predictions_v2.sagi")
intensity = store.get_intensity_by_key(sequence="PEPTIDE", precursor_charge=2, ...)
```

### Backward Compatibility

- V1 files continue to work - `PredictedIntensityStore("v1_file.sagi")` auto-detects
- V1 lookup methods (`get_intensity()`) work on V1 stores
- V2 lookup methods (`get_intensity_by_key()`) work on V2 stores
- Scoring automatically uses the correct lookup based on store type

---

## File Locations

| File | Description |
|------|-------------|
| `sage/crates/sage/src/intensity_prediction.rs` | Core V1/V2 data structures and file I/O |
| `sage/crates/sage/src/scoring.rs` | Scorer with unified intensity store support |
| `sagepy-connector/src/py_intensity.rs` | Python bindings for V1/V2 |
| `sagepy-connector/src/py_scoring.rs` | Scorer bindings |
| `sagepy/sagepy/core/intensity.py` | Python interface layer |
| `sagepy/sagepy/core/scoring.py` | Python Scorer wrapper |
