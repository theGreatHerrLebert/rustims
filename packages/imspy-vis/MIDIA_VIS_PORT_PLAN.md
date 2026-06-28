# MIDIA Voila visualizer — faithful-port plan

Goal: bring the `imspy_vis.midia` Voila dashboard as close as possible to the original
`proteolizard-midia` tool, while keeping the improvements the rebuild already gained.

- **Original** (private repos): `theGreatHerrLebert/proteolizard-midia`
  (`python/proteolizardmidia/vis/*`, clustering, data) + `theGreatHerrLebert/proteolizard-vis`
  (`proteolizardvis/{cluster,point,filter,data,utility}.py`, the base UI classes).
  Entry point: `MidiaVis()` in `vis/midia_vis.py`, used from `notebook/MidiaVis.ipynb`.
- **Port**: `packages/imspy-vis/src/imspy_vis/midia/{data,clustering,transforms,widgets}.py`.

## Where the port already matches (do not regress)

The data layer, clustering math, and the two HDBSCAN panels are faithful:
- `peak_width_preserving_mz_transform`, cycle/scan `/2**scaling`, MIDIA `mc` recovery.
- `cluster_midia_hdbscan` 4D `(rt, mc, dt, mz)` / 3D fallback; precursor HDBSCAN.
- Summary table + 4 nunique histograms (cycle/scan/mz/size); probability-sized colored scatter.

Improvements over the original to KEEP:
- Extraction windows derived from the run's own `dia_ms_ms_windows` — no external
  `extraction_windows.h5`.
- Window-less fragments dropped (`valid = isfinite(mc)`) instead of folded to 0
  (the noise-sampler bug; see `docs/midia-noise-sampler-bug.md`).
- `max_points` downsample cap; Voila-robust redraw (go.Figure into an Output, not a 3D
  FigureWidget).

## Tab parity

| Original (7 tabs) | Port (5 tabs) | Status |
|---|---|---|
| Data (`MIDIADataLoader`) | Data (`DataPanel`) | merged load+filter; RT FloatText not sliders |
| MS-I filter (`MidiaPrecursorScanFilter`) | (folded into Data) | separate-filter tab dropped |
| MS-I points (`MidiaPrecursorPointCloudVis`) | Precursor cloud (`PointCloudPanel`) | faithful |
| MS-I HDBSCAN (`HDBSCANVisualizer`) | Precursor HDBSCAN (`PrecursorHDBSCANPanel`) | controls trimmed |
| MIDIA filter (`MidiaFragmentFilter`) | — | **missing** |
| MIDIA points (`MidiaFragmentPointCloudVis`) | Fragment cloud (`PointCloudPanel`) | **degraded** (no dim selector) |
| MIDIA HDBSCAN (`MIDIAHDBSCANVisualizer`) | MIDIA HDBSCAN (`MidiaHDBSCANPanel`) | controls trimmed |

## Gaps to close (ordered)

### 1. Restore "Save settings" + Name field  *(DONE)*
Original: both cluster panels have a `person` Text field and a **Save settings** button that
writes a timestamped CSV of every filter + HDBSCAN parameter
(`PRECURSOR-HDBSCAN-<time>.csv` / `MIDIA-HDBSCAN-<time>.csv`; see
`proteolizardvis/cluster.py::HDBSCANVisualizer.on_save_clicked` and
`vis/midia_clustervis.py::on_save_clicked`).
- Add a `Name` text + `Save settings` button to `_ClusterPanel`.
- Collect filter settings (from `DataPanel`) + cluster settings into a one-row DataFrame,
  `to_csv(f"{LEVEL}-HDBSCAN-{timestamp}.csv")`.
- Each panel supplies its own param dict (`get_save_settings()`); MIDIA adds the extra knobs.

### 2. Rebuild the MIDIA fragment point-cloud view  *(DONE)*
Original `MidiaFragmentPointCloudVis` exposes:
- `exclude dim` dropdown: cycle / extraction window / scan / m-z — drop one, plot the other 3.
- `color by` dropdown: cycle / extraction window / scan / m-z / log-intensity.
- `quad scaling` (extraction_scaling) and renders the **extraction-window (mc) axis**.

Port's fragment tab currently plots only cycle/scan/mz colored by log-intensity and ignores
`step`/extraction-window. Plan:
- Give the fragment `PointCloudPanel` (or a new `FragmentPointCloudPanel`) the `exclude dim`
  / `color by` / `quad scaling` controls.
- Compute `mc` per point via `MidiaExperiment.isolation_left_bound(step, scan)` (already
  exists in `data.py`), scaled by `quad scaling`; use it as the chosen axis/color.
- Keep the `max_points` cap and Voila-robust redraw.

### 3. Re-expose the dropped HDBSCAN controls  *(DONE)*
Add the controls the original had but the port hardcodes:
- Shared: `algorithm` (best/eom/leaf — note: precursor uses `best`, MIDIA uses eom/leaf as
  `cluster_selection_method`), `alpha`, `leaf_size`, `approx_min_span_tree`,
  `gen_min_span_tree`, `use_probability` toggle, `mz_scaling`.
- MIDIA-only: `cluster_selection_method` and `cluster_selection_epsilon` (currently hardcoded
  eom / 1.0 inside `cluster_midia_hdbscan`).
- Widen `metric` to the full hdbscan metric set (original used
  `hdbscan.dist_metrics.METRIC_MAPPING.keys()`).
- Honor `use_probability` in point sizing: original
  `size = 2 if noise else bps*p*use_prob + bps*(1-use_prob)` with `bps=3.5`; port currently
  always does `2 + 3*p`.
- Wire `algorithm/alpha/leaf_size/approx/gen/mz_scaling` through the clustering functions
  (signatures already accept most of them).

### 4. MIDIA filter tab  *(DONE — rebuilt, decision: full fidelity)*
Decision (David): **rebuild the full filter tab.** Implemented as `MidiaFilterPanel` +
`MidiaExperiment.fragment_scan_ranges()` (ports `get_scan_ranges`) +
`filter_fragments_by_scan_ranges()`:
- Controls: `query start`, `query length`, `quad width` (`target_length`), `% overlap`,
  `scan min/max`, Filter + Reset.
- For a precursor query it selects, per quadrupole step, the scan range whose isolation window
  overlaps the query (same `allowed_diff` overlap criterion as the original).
- Acts as the **fragment source** for the MIDIA cloud + HDBSCAN tabs (proxies
  `experiment`/`fragment_df`/`filter_settings`), so they consume the filtered set transparently;
  passes everything through until Filter is pressed (Reset restores).
- `filter_settings()` now adds `query_start/length`, `quad_width`, `percent_overlap`,
  `midia_scan_min/max` to the saved CSV when a filter is active (also closes item-1 codex
  finding #4). Dashboard is now 6 tabs in the original order.
- Deviation: overlap math uses the run's real window left bounds with the `quad_width` slider as
  the window width (the original only had the `.h5` left bounds + the same slider). The original's
  `last_used=450` scan crop is intentionally dropped (a fixed-grid `.h5` artifact; our windows are
  the real used ranges, and `scan min/max` covers scan limiting).
- Codex round (originals staged in-repo for the diff): overlap math + proxy wiring confirmed
  correct. Fixed: stale `_filtered` after a slice reload (source-identity invalidation) and
  `filter_settings()` now snapshots the params that produced the filter (no save/data desync).

### 5. Smaller structural / default fixes  *(low priority)*
- *(DONE)* **Run browser** on the Data tab — a dependency-free directory navigator
  (`DataPanel._refresh_browser`/`_on_browse_select`): click folders to navigate, click a `.d`
  to select it (sets the path field; load still on "Load slice"). `.d` runs are dirs so it
  picks them as leaves rather than descending in. Codex-reviewed (re-select-same-`.d`, root
  `..`, unreadable-dir rollback, suppression `finally`). Symlink-following kept (data mounts).
- Optionally split Data-load from an MS-I scan filter tab (re-filter without reload); the
  slice is already cached so this is mostly UX.
- RT as scrubbable sliders (0–46 min) instead of FloatText.
- ~~Align defaults: original precursor HDBSCAN `min_samples=1`~~ *(DONE — precursor
  min_cluster_size=13/min_samples=1; MIDIA min_cluster_size=7/min_samples=7/metric=manhattan,
  all matching the originals)*.

## Acceptance
- [x] Save-settings writes a CSV with Name + all filter/cluster params for both panels.
- [x] Fragment cloud has exclude-dim / color-by / quad-scaling and shows the mc axis.
- [x] All original HDBSCAN knobs are on the UI and wired through (incl. use_probability,
      cluster_selection_method/epsilon, broad metric list).
- [x] MIDIA filter tab decision recorded (rebuilt for full fidelity).
- [x] Improvements retained: metadata-derived windows, window-less-fragment drop, max_points.
