# tims-viewer web — Flow restructure PLAN

Status: **proposed, codex-reviewed (revisions incorporated)**
Scope: `tims-viewer/web/index.html` + `tims-viewer/web/src/lib.rs`. One small server addition
(frame count in `/meta`) is **optional** — see §4.

## 1. Goal

Make the GUI embody the natural working flow so it reads as a guided sequence, not a
pile of controls:

> **① select a dataset → ② select a subsection → ③ iterate on clustering → ④ save a config + results**

…without losing free-form **interactive exploration** (orbit/zoom, colormap, transfer,
volume, etc.).

## 2. Core design decision: layers, not modes

Exploration is **not a mode you switch into — it is always on** (the canvas is live the
whole time, and display tuning is one click away). The flow is a **task spine laid over**
that ambient exploration. Therefore **no hard "Flow/Explore mode" toggle** (it would split a
shared control set, add a hidden "which mode" variable, and double the layout). Combine via
**relocation + progressive disclosure**:

- the numbered tabs carry the **flow** (lean, only flow-essential controls),
- a separate **⚙ Display** tab is the **exploration toolbox** (all render/display knobs),
- a per-tab **`Advanced ▾`** disclosure reveals the few genuinely-deep knobs in context.

A global `Simple/Advanced` toggle is **out of scope for v1** (add later only if missed).
*(Codex concurred this is sound; a hard mode would only be warranted if two canvas drag-gestures
ever conflicted — e.g. box-select vs lasso vs measure. Not the case today.)*

## 3. Target structure

```
tims·view                              ⚙
┌─────────────────────────────────────────┐
│ ① Data   ② Region   ③ Cluster   ④ Save  │   ⚙ = Display (right-aligned, un-numbered)
├─────────────────────────────────────────┤
│ ② REGION                                 │
│ Drag on the cloud to box-select          │
│ [⊕ Focus]   [← Back]                      │
│ m/z 612–680 · 1/K₀ 0.94–1.10 · RT 31–44s │
│ Floor ▁▂▃▅▇  1.2k                         │
└─────────────────────────────────────────┘
   numbers signal order; tabs are NOT gated (re-focus / re-cluster freely)
```

## 4. What goes in each tab (mapping from today)

| Tab | New content | Comes from |
|-----|-------------|-----------|
| **① Data** | Dataset dropdown + a read-only **meta summary** (see scope below) | dataset picker **moves out of `.head`**; summary is **new** |
| **② Region** | Focus/Back, MS show, DIA-windows toggle + legend (`#show-windows`, `#win-groups`), the 3 axis crop sliders, the 3 box-select **maps** (must stay visible — they are layout-measured, never behind a collapsed `<details>`), intensity **Floor** (`#floor`, `#floor-n`, `#floor-src`) | current **`tab-focus`** |
| **③ Cluster** | Algorithm (`#cl-algo` + `#cl-algo-status`, `#opt-sklearn`, `#opt-hdbscan`) + its params, Run/Stop (`#cl-run`), progress, `#cl-color` / `#hide-noise`, result readout. **`Advanced ▾`** holds RT-cycles (`#cl-rtc`), m/z-peak-widths (`#cl-mzw`), HDBSCAN selection-ε (`#cl-cse`) | current **`tab-cluster`**, levers moved under `<details>` |
| **④ Save** | **Export results** + **Export config** (two buttons — see §5) | results button relocates from the right rail (`#cl-export`); config button (`#cfg-export`) is **new** |
| **⚙ Display** | view/reset/roll, Points/Volume + vol style/steps, display fraction (`#disp`), **load** (`#load-n` + `#load-cap`), density/solid, colormap, point size, opacity, transfer, exposure | current **`tab-view`** (renamed) |

The **right rail** (cluster stats: summary table, per-cluster histograms, cluster list, and its
own `#railtgl-r` / `.railbtn-r` toggle) is **unchanged** except the Export button leaves it.

**Data-summary scope (revised — codex [must-fix]).** `MetaInfo` today parses bounds, intensity,
histograms, projections, `cycle_duration` — but **not** `n_points`, dataset name, or frame count
([lib.rs:1121]). `/meta` exposes `n_points` + `cycle_duration` but **not** name/frame count
([serve.rs:609]); the name lives only in `/datasets`. So the summary will show **only what we can
source cheaply**:
- **name** — from the selected `/datasets` entry (stash it in `populate_datasets`),
- **points** — add `n_points` to `MetaInfo` parsing (already in the `/meta` JSON),
- **m/z · 1/K₀ · RT ranges** — from `MetaInfo.bounds`,
- **cycle duration** — from `MetaInfo.cycle_duration`.
- **frame count** — **dropped** in v1 (not in `/meta`). *Optional:* add `frames` to the `/meta`
  JSON in `serve.rs` if we want it. Plan assumes dropped.

## 5. Save: export + config (revised — codex [must-fix])

Two distinct artifacts, two buttons, distinct enable rules:

- **Export config** (`#cfg-export`, **always enabled**): a **live** serializer (new) — the
  reproducible *recipe*, available before any run: `dataset` (id + name), `region`
  (`axis_bounds`), `ms_mask`, `intensity_floor`, `algorithm` + its params + the metric levers.
  **Does not reuse `cluster_params_json`** (that only exists post-run and carries result stats and
  no dataset id — [lib.rs:2876]). Writes `config.json`.
- **Export results** (`#cl-export`, **disabled until a result exists**): the existing dual
  download — `clusters.csv` + the post-run `cluster_params.json` (params + counts that produced
  *these* results). Today `export_clusters` silently no-ops with no result ([lib.rs:2927]); in Save
  it must instead be **disabled** (greyed) until `cluster_stats` is set, toggled where the result
  is finished/invalidated.

So the post-run `cluster_params.json` stays tied to results; the new live `config.json` is the
anytime recipe. (`cluster_params_json` gains a `dataset` field so both agree.)

## 6. Implementation steps (revised)

1. **Tabs bar** (`index.html`): 5 buttons, `data-tab` = `data|region|cluster|save|display`; gear
   right-aligned (`margin-left:auto`), four numbered `①…④`.
2. **`switch_tab` is NOT generic today** — it hard-codes `["view","focus","cluster"]`
   ([lib.rs:2800]). **Must** update that list to `["data","region","cluster","save","display"]`
   (or derive it from `document.querySelectorAll(".tab[data-tab]")`). Update the **default active
   tab** in both HTML (`.active`) and this list.
3. **① Data pane** (`#tab-data`): move `#dspick` here; add summary nodes
   (`#data-name`, `#data-points`, `#data-mz`, `#data-im`, `#data-rt`, `#data-cycle`).
4. **② Region**: rename `#tab-focus` → `#tab-region`, `data-tab="region"`. Content as-is (maps stay
   visible, not behind `<details>`).
5. **③ Cluster**: wrap `#cl-rtc`/`#cl-mzw`/`#cl-cse` rows in `<details class="adv"><summary>` …
   keep their ids (collapsed bound inputs still sync — codex confirmed). Do **not** put any
   button inside an `.ey` header (the section-collapse delegate keys on `.ey`+parent).
6. **④ Save pane** (`#tab-save`): two `.btn` buttons (`#cl-export` relocated → restyle from
   `.cs-export` to `.btn`; `#cfg-export` new). Neither inside an `.ey` header.
7. **⚙ Display**: rename `#tab-view` → `#tab-display`, `data-tab="display"`.
8. **CSS**: 5-tab row, numbered spacing, gear right-aligned, `.adv` disclosure, Data summary list,
   the relocated export buttons.
9. **Rust (`lib.rs`) — the bulk of the work:**
   - `switch_tab` tab-key list + default tab (step 2).
   - Parse `n_points` into `MetaInfo`; stash dataset **name** from `/datasets`; **populate the Data
     summary** (new fn, called after `/meta` + after dataset name is known).
   - **Split export:** `#cl-export` → results (csv + post-run params json), **disabled** until
     `cluster_stats` exists (toggle on finish/invalidate). `#cfg-export` → new **live config
     serializer** (dataset id+name, region, ms, floor, algorithm+params+levers).
   - Add a `dataset` field to `cluster_params_json` so results-json and config agree.
   - `on_cluster_tab` keys on `#tabbtn-cluster` ([lib.rs:2833]) — unchanged (Cluster keeps that id).
   - `populate_datasets` already targets `#dataset-sel`/`#dspick` (now in `#tab-data`) — no change.

## 7. Safety / non-breaking notes (revised — codex [risk])

- **Most** bindings are by id (`by_id`, `set_checked`, id pairs), so moving `#disp`, `#floor`,
  `#cl-eps`, `#dataset-sel`, `#cl-export`, etc. between panes is safe.
- **Structure-delegated listeners are the exception** — keep these intact:
  - `.tab` click via `closest(".tab")` ([lib.rs:2813]) — the new tabs must keep `class="tab"` +
    `data-tab`.
  - **Section collapse** keys on `.ey` + `parent_element()` ([lib.rs:2674]) → **never place a
    button inside an `.ey`**. (The old `#cl-export` lived in an `.ey` and stopped propagation; in
    Save it becomes a normal row, so the issue disappears.)
  - cluster-list `.cs-row` and window-legend `.wg` delegates ([lib.rs:2714]) — both stay in the
    right rail / Region, untouched.
- `.points-only` / `.vol-only` are keyed on `body.volmode`, **not** tab names — they survive the
  `view→display` rename.
- The right stats rail shows iff `#tabbtn-cluster` is active; **Save export reads `gfx`
  state, not the rail DOM**, so it works even though the rail is hidden on the Save tab.
- Carry these **companion ids** with their controls so markup moves don't orphan them:
  `#load-cap`+`#load-n`; `#cl-algo-status`/`#opt-sklearn`/`#opt-hdbscan`+`#cl-algo`;
  `#floor-src`+Floor; `#railtgl-r`/`.railbtn-r` stay with the (unchanged) cluster panel.

## 8. Out of scope (future)

- Global `Simple/Advanced` toggle. — **Loading** a config to restore a flow (v1 only *exports*).
- Per-step "done" cues (② shows the active region, ③ the cluster count). — Frame count in `/meta`.

## 9. Decisions (open questions resolved)

1. **Default landing tab → ② Region.** The page already loads a dataset; the first *action* is
   selecting a subsection. (① Data is one click away to orient/switch.)
2. **Export lives solely in ④ Save** (removed from the right rail) — no duplication.
3. **Floor stays in ② Region** — it's part of selecting *what* to cluster (prior decision).
4. **`#cl-color` / `#hide-noise` stay primary** in ③ Cluster (not Advanced).
