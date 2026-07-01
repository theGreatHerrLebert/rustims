# Feature idea — correlated quad filter (4-point select on the maps)

Status: IDEA (pre-design, for codex feasibility review). Extends the focus-lens (FOCUS_LENS_PLAN.md)
and the 2D box-select maps (focus-lens D).

## Motivation

Today's filter is **separable**: independent per-axis windows (the crop sliders) + an intensity
floor. The box-select on a projection map drags an *axis-aligned rectangle* — two independent
ranges. But real features in timsTOF data follow **correlated, diagonal bands**: a charge state is a
slanted streak in m/z × 1/K₀; an eluting species is a diagonal in m/z × RT. An axis-aligned box
can't isolate a diagonal band without dragging in a lot of neighbouring junk.

A **convex quadrilateral** select expresses the filter as **4 linear functions** (half-planes),
coupling the two projected axes — a *correlated* filter, which we don't have yet. Drag 4 corners
around a diagonal streak and keep only that streak.

## Core idea: 4 half-planes

A convex quad with corners `P0..P3` (consistent winding, say CCW) defines, per edge
`(Pᵢ → Pᵢ₊₁)`, an inside half-plane. A point `(u, v)` is inside iff for all 4 edges
`aᵢ·u + bᵢ·v + cᵢ ≥ 0`, where for edge `(dx, dy) = Pᵢ₊₁ − Pᵢ`:

```
a = −dy,  b = dx,  c = dy·Pᵢ.x − dx·Pᵢ.y
```

So the whole selection is just `[vec3(a,b,c); 4]` evaluated against a point's two projected
coordinates. This is the same lens, made correlated — it slots into the existing filter, it does not
add a new subsystem.

The quad constrains **2 of the 4 axes** (the projection's pair); the third spatial axis and the
intensity floor keep their existing separable filters. The three projections map to axis pairs:
`mz_im → (mz, im)`, `mz_rt → (mz, rt)`, `im_rt → (im, rt)`.

## Where it plugs in (two layers, same as the lens)

### 1. Live preview — shader cull (instant, on the loaded cloud)

Add a polygon to `ParamsUniform`:

```
poly_edges : [vec4; 4]   // (a, b, c, _) per edge, in NORMALIZED cube coords of the two axes
poly_axes  : u32         // which pair: 0=(mz,im)=axes 0,1 · 1=(mz,rt)=0,2 · 2=(im,rt)=1,2
poly_active: u32         // 0/1
```

The point shader (`point_cloud.wgsl`) and the compaction shader (`compact.wgsl`) both pick the two
`pos` components from `poly_axes` and cull the point if any of the 4 half-planes is negative —
alongside the existing window / MS / intensity culls. The CPU displayed-count recount (focus-lens B)
evaluates the same 4 lines so the HUD stays correct.

*(Alignment, confirmed by codex: `ParamsUniform` is currently 80 B (three `vec4` at 0/16/32 + 8
scalar slots at 48..76) with **no spare pad**. Appending `poly_edges:[vec4;4]` + the two `u32`s
lands at a clean **160 B** with an explicit `_poly_pad:[u32;2]`. This must be changed in lockstep
across the Rust struct (`uniforms.rs`), **both** WGSL `Params` structs (`point_cloud.wgsl`,
`compact.wgsl`), and **every** struct literal — `state.rs` and `web/src/lib.rs` — or the buffer
silently desyncs.)*

### 2. Commit — source cull on Focus (densify the band)

`RegionFilter` gains an optional polygon in **real units** of the two axes:

```
poly: Option<{ axes: (u8, u8), edges: [[f64; 3]; 4] }>
```

`handle_point!` already has `mz`, `im` (per point) and `rt` (per frame), so all three projection
pairs are checkable per point at the source. Focus then spends the budget only on points inside the
quad → the diagonal band is rendered densely.

On Focus the cube **re-normalizes to the quad's axis-aligned bounding box** (we always render an
axis-aligned cube); the source cull has already removed the points outside the quad, so the focused
view shows the band filling the bbox with empty corners. The live `poly_active` resets to off (the
loaded set is already inside the quad).

**Stride estimate must use the quad area, not the bbox** (codex blocker): the loader strides *before*
`collect`'s reservoir sees points, so an over-estimate skips points the reservoir can never recover →
budget under-spent. The bbox over-estimates a quad by ~2×, so multiply the bbox-fraction estimate by
`quad_area / bbox_area` (shoelace) to size the stride to the actual poly survivors.

**The committed quad must live in `Region`** (codex blocker): otherwise a later Load-budget reload of
a focused quad re-queries from the stack and silently degrades to the quad's bbox. So the poly must
be part of `Region`, the focus-stack entry, the `/points` + `/meta` query, the server `source_region`,
and the `LoadKey`.

## UI: 4-corner drag on a map

- A per-map mode: **box** (today, 2-corner drag) and **quad** (4 draggable corner handles). A box is
  a special quad, so quad mode can initialize from the current box.
- Render the quad as an SVG `<polygon>` overlay on the map-wrap + 4 round handles; dragging a handle
  updates the live poly (preview) immediately.
- **Focus** commits the current quad (source cull + re-normalize). **Back** pops as usual.
- Keep the quad **convex**: order the 4 handles by angle around the centroid, then *validate* —
  angle-sort alone is not a convexity guarantee (a point inside the others' triangle yields a concave
  ordered polygon). Require consistent non-zero edge cross-product signs + a minimum area + minimum
  edge length + no near-collinear adjacent triple (like the degenerate-box guard in D). Be explicit
  about the map y-flip (heatmaps render low-bin-at-bottom; the box mapping already flips it back).

## Recommended shape: rotated rectangle first (codex nit, elevated)

A **free quad has 8 DOF**; a **rotated rectangle has 5** (center, width, height, angle) and is still
exactly 4 half-planes — but with far less query/cache entropy, trivial convexity (always convex), and
a simpler UI (drag/resize/rotate one rect vs. juggle 4 independent corners). It already captures the
common case: a *straight slanted band*. So:

- **Start with a rotated rectangle.** Free quad (tapered/non-parallel bands) becomes P3 if needed.

## Phasing

- **P1 — live preview, explicitly preview-only.** `ParamsUniform` poly (rotated-rect → 4 half-planes)
  + the cull in the **point shader** and the **CPU recount** (compaction too, for count/perf parity)
  + a **distinct map UI** (rotate/resize handles) that updates the shader params + recount live.
  Crucially this is a *separate path* from the current "drag→commit-focus-on-mouseup": Focus is
  disabled or clearly labelled until P2, so users never think a band was densified when it was only
  previewed on the resident sample.
- **P2 — Focus commit.** Add the poly to `Region` (+ stack/query/`source_region`/`LoadKey`), the
  `RegionFilter` source cull in `handle_point!`, the **quad-area** stride estimate, demo RT policy
  for testing, and re-normalize to the bbox. Densifies the band.
- **P3 — polish (optional).** Free quad (8 DOF), >4-point convex polygons, per-map quads composing
  across projections, named/persisted selections.

## Open questions / risks (for codex)

- **Uniform layout.** Exact placement/alignment of `poly_edges`/`poly_axes`/`poly_active` in
  `ParamsUniform` so the Rust struct and both WGSL shaders agree (std140). Does the current struct
  have room, or does it need a version bump / careful repacking?
- **Normalized vs real units.** Preview keeps the quad in normalized cube coords (cheap shader eval);
  commit needs it in real units for the loader. Two representations of the same quad — derive both
  from the corner positions to avoid drift.
- **Server query encoding.** How to pass a quad (8 floats + axis pair) in the `/points?…` query, and
  how it factors into the `LoadKey` cache key + snapping.
- **Composition.** Poly AND the existing per-axis crops AND the intensity floor all compose; confirm
  the shader/loader apply them together and the displayed-count recount mirrors all of them.
- **Convexity + degeneracy.** Enforce convex ordering; reject zero-area / collinear quads (like the
  degenerate-box guard in D).
- **Focus bbox vs quad.** The focused view re-normalizes to the quad's *bbox* (empty corners are
  expected); is that the desired UX, or should the band be re-fit some other way?
- **Scope.** P1 alone is a meaningful, self-contained increment (shader + UI + recount). P2 is the
  loader/server half. Worth doing P1, verifying in-browser, then P2.

## Codex feasibility review (incorporated)

Verdict: feasible; the half-plane math and `handle_point!` data are confirmed. Three blockers, all
addressable, folded into the design above:

- **[blocker] uniform layout** — no spare pad; grow `ParamsUniform` to a clean 160 B with an explicit
  `_poly_pad`, in lockstep across `uniforms.rs` + both WGSL `Params` + every literal (`state.rs`,
  `web/src/lib.rs`). → folded into "Live preview".
- **[blocker] committed quad must persist in `Region`** — else a Load-budget reload silently degrades
  a focused quad to its bbox. → folded into "Commit".
- **[blocker] stride estimate** — "reservoir makes over-estimate safe" was **wrong**: the loader
  strides before the reservoir, so a bbox over-estimate under-spends the budget. Size the stride to
  the quad area (`quad_area/bbox_area`). → folded into "Commit".

Confirmed/refined risks:
- `handle_point!` has per-point m/z·1/K₀·intensity and per-frame RT, and runs before peak/hist/
  `global_i` — the right place. But the demo forces RT to full in `parse_query`, so testing m/z×RT /
  1/K₀×RT *source* poly on the demo needs a demo-RT policy change.
- Convexity: angle-sort is **not** sufficient — validate cross-product signs + min area/edge.
- Derive normalized (preview) and real-unit f64 (commit) edges both from the same UI corner
  *fractions*, never one from the other (avoids f32 drift across m/z vs 1/K₀ vs RT scales).
- Composition is currently **uneven**: point shader culls spatial+intensity+MS, compaction only
  spatial+MS, CPU recount spatial+floor+MS. P1 must add the poly to the point shader **and** the CPU
  recount at minimum (compaction for parity); decide the intensity-max story.
- Today's box-select **resets** the unselected axes to full and carries only the floor; if the poly
  is meant to compose with the crop sliders, intersect with the active crops + carry the third axis.
- Server cache: canonicalize (snap) the poly before keying or 8 floats of entropy blow up the cache;
  reject non-finite/degenerate; include the committed poly in `/meta`.

**Recommendation:** ship **P1 with a rotated rectangle** (preview-only, Focus gated) as the
self-contained, satisfying core; then P2 for the source-cull densify. Reassess the free quad (P3)
only if straight slanted bands prove insufficient.
