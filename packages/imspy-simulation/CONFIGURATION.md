# TimSim Configuration Reference

This is the complete reference for the TOML configuration file consumed by the
`timsim` CLI (`timsim config.toml`). It lists **every** available argument, its
type, default value, and meaning.

> **Looking for a quick start instead?** See
> [`SIMULATOR_README.md`](./SIMULATOR_README.md) for a guided introduction and
> ready-to-copy example configs. This document is the exhaustive reference.

---

## How the config file works

A TimSim config is a single TOML file. A few things are worth knowing up front:

- **Section headers (`[paths]`, `[experiment]`, ...) are organizational only.**
  The loader flattens every section into one flat namespace before use, so a key
  works regardless of which `[section]` it lives under (or none at all). The
  sections in this document are a logical grouping for readability and roughly
  track the GUI's panels; because the loader flattens everything, the exact
  section a key sits under has no effect on parsing. See
  `simulator.translate_legacy_config` for the flattening logic.
- **Every key is optional.** Any key you omit falls back to the default listed
  here (the canonical source of truth is `get_default_settings()` in
  `src/imspy_simulation/timsim/simulator.py`).
- **Legacy section names** (`[main_settings]`, `[peptide_digestion]`, ...) are
  auto-detected and translated, and a handful of removed keys are ignored with a
  log message. Old configs keep working.

### Three ways to discover the current arguments

Because the option set evolves between releases, you can always regenerate the
authoritative list from the installed code:

1. **`get_default_settings()`** in `simulator.py` — the canonical dict of every
   key and its default.
2. **The GUI** (`timsim-gui`) — groups all options into labelled panels; its
   dataclasses in `timsim/gui/state.py` define the section→key TOML layout and
   `SimulationConfig.to_toml_dict()` shows the exact structure.
3. **`timsim --help`** — usage, CLI overrides, and a minimal config skeleton.

---

## Command-line overrides

The CLI takes the config path as its only positional argument; a few keys can be
overridden on the command line (the override wins over the file):

| Flag | Overrides | Notes |
|------|-----------|-------|
| `CONFIG_FILE` (positional) | — | Path to the TOML config file (required). |
| `--save-path`, `-s` | `save_path` | Output directory. |
| `--reference-path`, `-r` | `reference_path` | Reference `.d` dataset. |
| `--fasta-path`, `-f` | `fasta_path` | Proteome FASTA. |
| `--findings-path` | `findings_path` | Enables `from_findings` mode. |
| `--intensity-multiplier` | `intensity_multiplier` | Scales all findings intensities. |
| `--projection-mode` | `projection_mode` | `off` / `legacy_compat` / `accurate`. |
| `--resume` | — | Resume from the latest checkpoint (requires `enable_checkpoints = true` on the original run). |

---

## `[paths]` — input/output locations

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save_path` | str | _none_ | Output directory for the simulated dataset. |
| `reference_path` | str | _none_ | Path to a real reference `.d` dataset (used for layout/mobility calibration; Bruker path). |
| `fasta_path` | str | _none_ | Proteome FASTA to digest. |
| `existing_path` | str | _none_ | Path to a previous `synthetic_data.db` to reuse when `from_existing = true`. |
| `findings_path` | str | _none_ | Path to a findings table when `from_findings = true`. |

## `[experiment]` — run-level settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `experiment_name` | str | `TIMSIM-[PLACEHOLDER]-<timestamp>` | Name of the experiment/output dataset. |
| `acquisition_type` | str | `DIA` | Acquisition scheme: `DIA` or `DDA`. |
| `gradient_length` | int (s) | `3600` | LC gradient length in seconds. |
| `use_reference_layout` | bool | `true` | Reuse the reference dataset's frame/scan layout. |
| `reference_in_memory` | bool | `false` | Load the reference dataset fully into memory (faster, more RAM). |
| `use_bruker_sdk` | bool | `true` | Use the Bruker SDK for accurate mass/mobility calibration when available. |
| `apply_fragmentation` | bool | `false` | Generate fragment (MS2) signal in addition to precursors. |
| `silent_mode` | bool | `false` | Suppress most console output. |
| `sample_seed` | int | `41` | Master RNG seed for reproducibility. |
| `from_existing` | bool | `false` | Build from an existing `synthetic_data.db` (see `existing_path`). |
| `from_findings` | bool | `false` | Drive the simulation from a findings table (see `findings_path`). |
| `intensity_multiplier` | float | `1.0` | Global multiplier applied to findings intensities. |
| `findings_reference_median` | float | _none_ | Shared event-scaling denominator across conditions. `none` = per-sample median (legacy). Set the **same** value for every condition (A/B/...) of a multi-sample experiment to preserve cross-sample intensity ratios. |

## `[digestion]` — in-silico digestion

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `digest_proteins` | bool | `true` | Perform in-silico digestion of the FASTA. |
| `n_proteins` | int | `20000` | Number of proteins to draw from the FASTA. |
| `num_peptides_total` | int | `250000` | Total peptide pool size before sampling. |
| `num_sample_peptides` | int | `25000` | Number of peptides actually simulated (when `sample_peptides`). |
| `sample_peptides` | bool | `true` | Subsample the peptide pool down to `num_sample_peptides`. |
| `missed_cleavages` | int | `2` | Allowed missed cleavages. |
| `min_len` | int | `7` | Minimum peptide length. |
| `max_len` | int | `30` | Maximum peptide length. |
| `cleave_at` | str | `KR` | Residues to cleave after (trypsin = `KR`). |
| `restrict` | str | `P` | Residue that blocks cleavage when it follows the cleavage site. |
| `decoys` | bool | `false` | Generate decoy sequences. (Legacy key `add_decoys` is auto-renamed.) |
| `remove_degenerate_peptides` | bool | `false` | Drop peptides shared across proteins. |
| `upscale_factor` | int | `100000` | Abundance up-scaling factor for sampled occurrences. |
| `sample_occurrences` | bool | `true` | Sample per-peptide occurrence counts (abundance). |
| `min_rt_percent` | float | `2.0` | Minimum RT (as % of gradient) before which peptides are excluded. |
| `exclude_accumulated_gradient_start` | bool | `true` | Exclude the artificially crowded gradient start. |

## `[modifications]` — variable/fixed modifications

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `modifications` | str (path) | _none_ | Path to a TOML file specifying variable/static UNIMOD modifications (SAGE-style). **When unset/empty, TimSim loads its bundled default `timsim/configs/modifications.toml`** — variable Oxidation (M) `[UNIMOD:35]` and N-terminal Acetylation `[UNIMOD:1]`, static Carbamidomethylation (C) `[UNIMOD:4]`. So the default run is **not** unmodified; to simulate unmodified peptides, point this at a file with empty `[variable_modifications]`/`[static_modifications]` tables. |

## `[isotopic_pattern]` — isotope envelope

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `isotope_k` | int | `8` | Number of isotope peaks to compute per ion. |
| `isotope_min_intensity` | int | `1` | Minimum isotope-peak intensity to keep. |
| `isotope_centroid` | bool | `true` | Centroid the isotope envelope. |

## `[models]` — prediction-model selection

`none`/`""`/`"local"` selects the bundled local PyTorch model; any other value
selects a remote [Koina](https://koina.wilhelmlab.org/) model by name.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rt_model` | str | _none_ | Retention-time model. Local, or `Deeplc_hela_hf`, `Chronologer_RT`, `AlphaPeptDeep_rt_generic`, `Prosit_2019_irt`. |
| `koina_rt_model` | str | _none_ | **Deprecated** alias for `rt_model`. |
| `ccs_model` | str | _none_ | CCS/ion-mobility model. Local, or `AlphaPeptDeep_ccs_generic`, `IM2Deep`. |
| `intensity_model` | str | _none_ | Fragment-intensity model. `local`/`prosit`, `alphapeptdeep` (supports phospho), `ms2pip`, or a full Koina name. |
| `fragment_intensity_model` | str | _none_ | **Deprecated** — use `intensity_model`. |
| `instrument` | str | `bruker_timstof` | Target instrument. Drives the collision-energy unit, the fragment-model compatibility guard, and provenance. Options: `bruker_timstof`, `orbitrap_astral`, `orbitrap_exploris`, `sciex_zenotof`, `waters_synapt_xs`. |
| `template_path` | str | _none_ | For a Thermo/SCIEX build-from-template run, the vendor template file (`.raw`/`.wiff`) whose real per-scan schedule becomes the acquisition. Required for those instruments. |
| `astral_template_path` | str | _none_ | Historical alias for `template_path` (both accepted). |

## `[retention_time]` — RT distribution shape

EMG-like RT peak-shape parameters (sigma controls width, k controls tailing).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sigma_lower_rt` | float | _none_ | Lower bound for sampled peak width. |
| `sigma_upper_rt` | float | _none_ | Upper bound for sampled peak width. |
| `sigma_alpha_rt` | int | `4` | Alpha of the sigma (width) distribution. |
| `sigma_beta_rt` | int | `4` | Beta of the sigma (width) distribution. |
| `k_lower_rt` | int | `0` | Lower bound for the tailing factor. |
| `k_upper_rt` | int | `10` | Upper bound for the tailing factor. |
| `k_alpha_rt` | int | `1` | Alpha of the tailing-factor distribution. |
| `k_beta_rt` | int | `20` | Beta of the tailing-factor distribution. |
| `target_p` | float | `0.999` | Probability mass the sampled distribution must cover. |
| `sampling_step_size` | float | `0.001` | Step size for distribution sampling. |
| `n_steps` | int | `1000` | Number of sampling steps. |
| `remove_epsilon` | float | `1e-4` | Drop distribution tails below this density. |
| `projection_mode` | str | `off` | Instrument-dispatch projector for frame/scan distributions. `off` keeps legacy columns; `legacy_compat` regenerates them via the Rust projector; `accurate` uses the improved projection (event-interval time + per-scan mobility bins). |

## `[ion_mobility]` — IM distribution

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_inverse_mobility_std_mean` | bool | `true` | Use a fixed std for the inverse-mobility peak. |
| `inverse_mobility_std_mean` | float | `0.009` | The inverse-mobility standard deviation used. |

## `[charge_states]` — precursor charge model

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `p_charge` | float | `0.8` | Success probability of the binomial charge model. |
| `binomial_charge_model` | bool | `true` | Use the binomial charge-state model (the legacy non-binomial model is broken; keep `true`). |
| `max_charge` | int | `4` | Maximum charge state. |
| `min_charge_contrib` | float | `0.005` | Minimum relative contribution for a charge state to be kept. |
| `normalize_charge_states` | bool | `true` | Normalize charge-state probabilities to sum to 1. |
| `charge_state_one_probability` | float | `0.0` | Extra probability mass assigned to the +1 charge state. |

## `[acquisition]` — collision energy & windowing

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `round_collision_energy` | bool | `true` | Round collision energy values. |
| `collision_energy_decimals` | int | `0` | Decimal places when rounding CE. |
| `collision_energy_nce` | float | _none_ | **Optional** override for Thermo build-from-template runs (`orbitrap_astral`, `orbitrap_exploris`). The template already supplies a genuine per-window NCE, used by default; if set, this single NCE (must be positive) overrides every window. Ignored for Bruker. |
| `dia_rewindow_isolation_width` | float (Th) | _none_ | Thermo build-from-template DIA only: re-window the template to this isolation width before authoring (centers + cadence + window count preserved). `none`/`0` = use the template's real windows. Ignored off the Thermo path. |

## `[noise]` — noise model

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `noise_frame_abundance` | bool | `false` | Add frame-level abundance noise. |
| `noise_scan_abundance` | bool | `false` | Add scan-level abundance noise. |
| `mz_noise_precursor` | bool | `false` | Add m/z jitter to precursor peaks. |
| `precursor_noise_ppm` | float | `5.0` | Precursor m/z noise magnitude (ppm). |
| `mz_noise_fragment` | bool | `false` | Add m/z jitter to fragment peaks. |
| `fragment_noise_ppm` | float | `5.0` | Fragment m/z noise magnitude (ppm). |
| `mz_noise_uniform` | bool | `false` | Use uniform (vs. Gaussian) m/z noise. |
| `add_real_data_noise` | bool | `false` | Sample real noise frames from the reference dataset and add them. |
| `reference_noise_intensity_max` | int | `30` | Max intensity of sampled reference noise. |
| `down_sample_factor` | float | `0.5` | Down-sampling factor applied to fragment signal. |
| `superimpose_on_reference` | bool | `false` | Superimpose simulated signal onto full reference frames (overlay mode). |
| `superimpose_merge_ppm` | float | `15.0` | MS2 centroid merge tolerance (ppm) when overlaying onto a Thermo `.raw` template. Bruker/no-overlay: ignored. |
| `precursor_sample_fraction` | float | `0.2` | Fraction of precursor frames sampled for real-data noise. |
| `fragment_sample_fraction` | float | `0.2` | Fraction of fragment frames sampled for real-data noise. |
| `num_precursor_noise_frames` | int | `5` | Number of real precursor noise frames to draw. |
| `num_fragment_noise_frames` | int | `5` | Number of real fragment noise frames to draw. |

## `[variation]` — run-to-run property variation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `re_scale_rt` | bool | `false` | Re-scale retention times across the run. |
| `rt_variation_std` | float | _none_ | Std of per-peptide RT variation. |
| `ion_mobility_variation_std` | float | _none_ | Std of per-peptide ion-mobility variation. |
| `intensity_variation_std` | float | _none_ | Std of per-peptide intensity variation. |

## `[dda]` — DDA-specific settings

Used only when `acquisition_type = DDA`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `precursors_every` | int | `10` | Acquire a precursor (MS1) frame every N frames. |
| `precursor_intensity_threshold` | int | `500` | Minimum precursor intensity to be selectable. |
| `max_precursors` | int | `25` | Max precursors selected per MS1 (TopN). |
| `exclusion_width` | int | `25` | Dynamic exclusion width. |
| `selection_mode` | str | `topN` | Precursor selection strategy. |

## `[quad_transmission]` — quadrupole isotope transmission

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quad_isotope_transmission_mode` | str | `none` | `none` (standard isotope patterns), `precursor_scaling` (fast — uniform scaling by precursor transmission), or `per_fragment` (accurate — per-fragment recalculation). |
| `quad_transmission_min_probability` | float | `0.5` | Minimum transmission probability threshold. |
| `quad_transmission_max_isotopes` | int | `10` | Max isotope peaks considered for transmission. |
| `precursor_survival_min` | float | `0.0` | Min fraction of precursor ions surviving fragmentation intact. Both min/max = `0.0` → no unfragmented precursors (backward compatible). Realistic: ~`0.05`. |
| `precursor_survival_max` | float | `0.0` | Max fraction of surviving precursors. Realistic: ~`0.15`. |

## `[proteome_mix]` — multi-species (e.g. HYE) mixtures

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `proteome_mix` | bool | `false` | Enable multi-FASTA proteome mixing. |
| `multi_fasta_dilution` | str | _none_ | Per-FASTA dilution specification. |

## `[phosphorylation]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `phospho_mode` | bool | `false` | Enable phosphoproteomics simulation. Use an intensity model that supports phospho (e.g. `alphapeptdeep`). |

## `[performance]` — threading, batching, GPU

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_threads` | int | `-1` | Worker threads. `-1` = auto-detect. |
| `batch_size` | int | `256` | Peptide/ion processing batch size. |
| `frame_batch_size` | int | `500` | Frames assembled per batch. |
| `lazy_frame_assembly` | bool | `false` | Assemble frames lazily (lower peak memory for large runs). |
| `use_gpu` | bool | `true` | Use GPU for prediction models when available. |
| `gpu_memory_limit_gb` | int | `4` | Soft GPU memory cap (GB). |

## `[provenance]` — mzPROV self-disclosure

TimSim emits an Ed25519-signed provenance record declaring the output is
simulated and binding it to the config + signing key. Requires the optional
`mzprov` package; import-guarded (a missing package logs a warning, never fails).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `emit_provenance` | bool | `true` | Emit the signed provenance record. |
| `provenance_embed` | bool | `true` | Embed the envelope into the output (`.d` provenance table / mzML `fileContent`). `false` writes a sibling `<name>.provenance.json` sidecar. (Vendor `.raw` always falls back to a sidecar.) |
| `provenance_key_path` | str | _none_ | Signing key path. `none` = default `~/.config/timsim/keys/` (auto-generated). |

## `[logging]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `log_level` | str | `INFO` | Logging level. |
| `log_file` | str | _none_ | Optional log-file path (in addition to console). |

## `[video_settings]` — preview animation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `generate_preview_video` | bool | `false` | Render a preview animation of the simulated run. |
| `preview_video_max_frames` | int | `100` | Max frames in the preview. |
| `preview_video_fps` | int | `10` | Frames per second. |
| `preview_video_dpi` | int | `80` | Render DPI. |
| `preview_video_annotate` | bool | `true` | Overlay annotations on the preview frames. |

## `[checkpoints]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_checkpoints` | bool | `false` | Write checkpoints so a run can be resumed with `--resume`. |

---

## Instrument-specific blocks

The target instrument is set by `models.instrument`. Some instruments need extra
parameters because their acquisition schedule is synthesized rather than taken
from a Bruker reference.

### SCIEX ZenoTOF SWATH (`instrument = sciex_zenotof`)

Build-from-`.wiff` (`template_path` = the `.wiff`). The method has no per-scan
timing, so the SWATH schedule is synthesized from these plus `gradient_length`.
Rolling CE = `ce_intercept + ce_slope_per_mz * precursor_mz`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sciex_cycle_time_s` | float | `3.5` | SWATH cycle time (s). |
| `sciex_ce_intercept` | float | `5.0` | Rolling-CE intercept. |
| `sciex_ce_slope_per_mz` | float | `0.045` | Rolling-CE slope per m/z. |

### Waters SONAR (`instrument = waters_synapt_xs`)

Build-from-parameters (no template file). SONAR is a scanning-quadrupole DIA
fully described by these plus `gradient_length`. `waters_window_step = none` →
contiguous windows; set it `< waters_window_width` for the faithful overlapping
quad scan. Rolling CE = `ce_intercept + ce_slope_per_mz * window_center_mz`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `waters_mz_start` | float | `400.0` | Start of the precursor m/z range. |
| `waters_mz_end` | float | `900.0` | End of the precursor m/z range. |
| `waters_window_width` | float | `20.0` | Quadrupole window width (Th). |
| `waters_window_step` | float | _none_ | Window step (Th). `none` = contiguous. |
| `waters_cycle_time_s` | float | `0.5` | Scan cycle time (s). |
| `waters_ce_intercept` | float | `5.0` | Rolling-CE intercept. |
| `waters_ce_slope_per_mz` | float | `0.04` | Rolling-CE slope per m/z. |

### Thermo Orbitrap Astral / Exploris (`instrument = orbitrap_astral` / `orbitrap_exploris`)

Build-from-`.raw`-template: set `models.template_path` (or
`astral_template_path`) to a real Thermo `.raw`; its per-scan schedule + windows
(including a genuine per-window NCE) become the acquisition. Optionally override
that NCE for every window with a single value via
`acquisition.collision_energy_nce`, and/or re-window via
`acquisition.dia_rewindow_isolation_width`.

### Bruker timsTOF (`instrument = bruker_timstof`, default)

Uses a real reference `.d` (`paths.reference_path`) for the frame/scan layout and
mobility calibration. No extra block required.

---

## Complete example (annotated)

```toml
[paths]
save_path      = "/data/sim/out"
reference_path = "/data/raw/reference.d"
fasta_path     = "/data/fasta/human.fasta"

[experiment]
experiment_name    = "HeLa-DIA-1h"
acquisition_type   = "DIA"
gradient_length    = 3600
apply_fragmentation = true
sample_seed        = 41

[digestion]
n_proteins          = 20000
num_sample_peptides = 25000
missed_cleavages    = 2
min_len             = 7
max_len             = 30

[models]
instrument      = "bruker_timstof"
rt_model        = ""          # local PyTorch
ccs_model       = ""          # local PyTorch
intensity_model = "prosit"    # Prosit via Koina

[charge_states]
p_charge  = 0.8
max_charge = 4

[noise]
add_real_data_noise = true
mz_noise_precursor  = true
precursor_noise_ppm = 5.0

[performance]
num_threads = -1
batch_size  = 256
use_gpu     = true
```

---

## Output and ground truth

Every run writes the simulated dataset (a Bruker `.d`, or vendor `mzML`/`.raw`
depending on `instrument`) **plus** ground truth. Annotation comes in two
distinct flavors:

### 1. Layout ground truth — `synthetic_data.db` (always written)

A SQLite database alongside the output holding the molecule-level blueprint:
`proteins`, `peptides`, `ions`, `fragments`, `frame_occurrences`, and
(DDA) `pasef_meta` / `precursors`. This is the standard ground truth for
evaluation; see the "Ground Truth Database" section of
[`SIMULATOR_README.md`](./SIMULATOR_README.md) and the EVAL pipeline
(`VALIDATION_README.md`).

### 2. Peak-level annotation — the annotated frame builder (Python API)

For machine-learning use cases (e.g. feature/isotope-pattern segmentation) you
can build **annotated frames** that carry per-peak labels — peptide id, charge
state, and isotope-peak index — down to individual peak overlays. This is **not
written into the raw output file**; there is no annotated-raw writer. Instead you
**stream** annotated frames in Python, on demand, straight from
`synthetic_data.db`.

> **Package layout note (latest `main`).** The monolithic `imspy` package has
> been split into modular packages, so imports differ from older notebooks/the
> paper's examples. In particular `from imspy.simulation…` → `from
> imspy_simulation…` and `from imspy.timstof…` → `from imspy_core.timstof…`. The
> snippets below use the current layout.

**Where do the labels come from?** Two annotated builders:

- **Precursor-only** — `TimsTofSyntheticPrecursorFrameBuilder`
  (`imspy_simulation.experiment`). A lightweight path that renders **MS1 frames
  only**; it cannot produce fragment frames. This is what the feature-finder
  example below uses.
- **Full builder (precursor + fragment)** — the DIA/DDA frame builder from
  `imspy_simulation.builders.create_frame_builder(..., with_annotations=True)`.
  This is **not** a fragment-only builder: its `build_frames_annotated(frame_ids)`
  inspects each frame id's `ms_type` and renders precursor (MS1) *or* fragment
  (MS2) accordingly, so it produces both. The per-peak fragment labels come from
  the annotated fragment-ion build path in the Rust backend. Use this whenever
  you need fragment annotations.

> **Heads-up — the full annotated builder is heavy.** Because annotation requires
> the standard (non-lazy) loading strategy, `create_frame_builder(...,
> with_annotations=True)` loads the full fragment-ion annotation data up front:
> expect a **long construction time and high RAM usage** on large simulations.
> The precursor-only builder is much lighter. Optimization of the annotated
> fragment path is pending.

```python
import numpy as np
from imspy_simulation.experiment import TimsTofSyntheticPrecursorFrameBuilder

# --- Precursor (MS1) annotations -----------------------------------------
fb = TimsTofSyntheticPrecursorFrameBuilder("out/EXP/synthetic_data.db")

# Per-peak labels as a DataFrame (peptide_id / charge_state / isotope_peak)
frame = fb.build_precursor_frame_annotated(frame_id=100)
df = frame.df

# Or dense, model-ready windows with label channels
scan_idx, win_idx, mz, mobility, X, iso_labels, charge_labels, pep_labels = \
    frame.to_dense_windows_with_labels(
        window_length=10, resolution=2, min_num_peaks=3, min_intensity=5.0,
        overlapping=False,
    )

# --- Fragment (MS2) annotations, or full-frame streaming -----------------
from imspy_simulation.builders import create_frame_builder, AcquisitionMode

builder = create_frame_builder(
    "out/EXP/synthetic_data.db", AcquisitionMode.DIA, with_annotations=True,
)
fragment_frames = builder.build_frames_annotated(frame_ids=[...])
```

To stream batches end-to-end, the `iter_frame_batches` helper in
`imspy_simulation.utility` consumes any annotation-capable builder and yields
labelled `frame.df` rows; pass `level="precursor"` or `level="fragment"` to pick
the frame type. Note: annotation is **only available with the standard
(non-lazy) loading strategy** (lazy + annotations is rejected).

> **Worked DL example.** A complete training + inference walkthrough — streaming
> annotated frames into a PyTorch segmentation model — lives in the companion
> **`timsim-bench`** repository under
> `timsim_bench/notebooks/examples/` (`AnnotatedFrameExample - Training.ipynb`
> and `AnnotatedFrameExample - Inference.ipynb`).
