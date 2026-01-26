# Code Cleanup Plan - Post-Refactor Bloat Removal

## Overview

This plan addresses code bloat identified after the package split refactor. The goal is to reduce duplication, improve maintainability, and consolidate scattered utilities.

---

## Phase 1: Consolidate Duplicate Functions

### 1.1 `linear_map` function - COMPLETED
- [x] Added numba-optimized `linear_map` to `imspy-core/utility/utilities.py`
- [x] Exported from `imspy-core/utility/__init__.py`
- [x] Updated `imspy-search/utility.py` to import from imspy-core
- [x] Updated `imspy-predictors/rt/predictors.py` to import from imspy-core
- [x] Removed duplicate implementations

**Files modified:**
- `packages/imspy-core/src/imspy_core/utility/utilities.py`
- `packages/imspy-core/src/imspy_core/utility/__init__.py`
- `packages/imspy-search/src/imspy_search/utility.py`
- `packages/imspy-predictors/src/imspy_predictors/rt/predictors.py`

### 1.2 `remove_unimod_annotation` function - COMPLETED
- [x] Added to `imspy-core/utility/sequence.py` (single source)
- [x] Removed from `imspy-predictors/ccs/utility.py`
- [x] Removed from `imspy-predictors/intensity/predictors.py`
- [x] Removed from `imspy-simulation/timsim/validate/parsing.py`
- [x] Updated all imports to use imspy-core

---

## Phase 2: Consolidate Lazy Import Patterns

### 2.1 Create shared lazy imports module - COMPLETED
- [x] Created `imspy-predictors/src/imspy_predictors/lazy_imports.py`
- [x] Consolidated `_get_sagepy_utils()` variants into shared module
- [x] Consolidated `_get_dbsearch_utils()` variants into shared module
- [x] Consolidated `_get_flatten_prosit_array()` into shared module
- [x] Updated imports in:
  - [x] `ccs/predictors.py`
  - [x] `rt/predictors.py`
  - [x] `intensity/predictors.py`

---

## Phase 3: Split Large Files - DEFERRED (Low Priority)

Upon review, these files are cohesive modules rather than random collections.
Splitting would be pure reorganization with minimal benefit. Consider only if
files grow further or become hard to navigate.

### 3.1 `tool_comparison.py` (2054 lines) - REVIEWED
Structure: 9 dataclasses + calculation functions + plotting + entry points
- Cohesive module for multi-tool comparison
- The `generate_comparison_plots()` function (~500 lines) could potentially
  be extracted to `tool_comparison_plots.py` if needed
- **Verdict**: Keep as-is, well-organized

### 3.2 `gui/pages/advanced.py` (1630 lines)
- GUI page code, typically cohesive
- **Verdict**: Keep as-is unless navigation becomes difficult

### 3.3 `clustering/data.py` (1539 lines)
- Data structure definitions
- **Verdict**: Review if adding new structures

### 3.4 `validate/plots.py` (1453 lines)
- Plot generation functions
- **Verdict**: Keep as-is, all plotting related

### 3.5 `validate/html_report.py` (1300 lines)
- Report generation
- **Verdict**: Keep as-is, single responsibility

### 3.6 `training.py` (1265 lines)
- Training orchestration
- **Verdict**: Keep as-is, training is complex enough to warrant this size

---

## Phase 4: Clean Up Dead Code

### 4.1 Remove commented-out code blocks - REVIEWED
Upon inspection, the files flagged mostly contain:
- TODOs for future work (legitimate)
- Section dividers (legitimate)
- Inline documentation comments (legitimate)

No significant dead code blocks found. Files reviewed:
- [x] `imspy-predictors/models/unified.py` - clean
- [x] `imspy-dia/clustering/torch_extractor.py` - clean
- [x] `imspy-simulation/timsim/jobs/simulate_frame_distributions_emg.py` - has TODOs only
- [x] `imspy-simulation/tdf.py` - has TODOs only
- [x] `imspy-core/data/peptide.py` - has TODOs only

### 4.2 Remove unused imports
- [ ] `imspy-dia/pipeline/cluster_pipeline.py` - duplicate logging imports (low priority)

---

## Phase 5: Standardize Utility Organization

### 5.1 Define utility hierarchy
```
imspy-core/utility/
├── __init__.py        # Re-exports common utilities
├── math.py            # linear_map, gaussian, etc.
├── sequence.py        # Sequence processing utilities
└── io.py              # File I/O helpers

imspy-predictors/
├── utility.py         # Predictor-specific utilities only
└── lazy_imports.py    # Lazy loading for optional deps
```

### 5.2 Migration plan
- [ ] Identify truly shared utilities across packages
- [ ] Move to imspy-core
- [ ] Update all imports
- [ ] Keep domain-specific utilities in their packages

---

## Phase 6: Tokenizer Consolidation

### 6.1 Create base tokenizer interface
- [ ] Extract common UNIMOD pattern matching
- [ ] Extract vocabulary building logic
- [ ] Create `BaseTokenizer` ABC
- [ ] Have `ProformaTokenizer` and `SimpleTokenizer` inherit from it

---

## Priority Order

1. **Phase 1.1** - `linear_map` consolidation (quick win, high impact)
2. **Phase 2.1** - Lazy imports consolidation (quick win)
3. **Phase 4.1** - Remove commented code (quick win, improves readability)
4. **Phase 3.1** - Split `tool_comparison.py` (highest LOC reduction)
5. **Phase 5** - Utility organization (foundational)
6. **Phase 3.2-3.6** - Split other large files
7. **Phase 6** - Tokenizer consolidation (lower priority)

---

## Estimated Impact

| Phase | LOC Removed | Files Modified | Risk |
|-------|-------------|----------------|------|
| 1 | ~50 | 4 | Low |
| 2 | ~60 | 4 | Low |
| 3 | 0 (reorganized) | 10+ | Medium |
| 4 | ~200 | 5 | Low |
| 5 | ~100 | 10+ | Medium |
| 6 | ~100 | 3 | Medium |

**Total estimated cleanup: ~500 LOC removed, better organization**

---

## Notes

- All changes should maintain backward compatibility within each package
- Run tests after each phase
- Phase 3 (file splitting) should preserve all functionality, just reorganize
