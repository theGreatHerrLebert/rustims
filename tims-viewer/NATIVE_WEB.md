# tims-viewer on the web — embedding the native GPU renderer in a browser

Goal: run the `tims-viewer` GPU renderer **inside a web application** (a `<canvas>`), so the
web tool gets the native viewer's look and large-data efficiency instead of Plotly. The Bruker
`.d` loader stays server-side; the browser renders points on the **client GPU** via WebGPU
(WebGL2 fallback).

## Why this is feasible (verified against the code + codex review)

- **It's `wgpu` + WGSL.** `wgpu` targets Vulkan/Metal/DX12 natively **and WebGPU/WebGL2 in the
  browser via wasm32**; shaders are `src/shaders/*.wgsl` and WGSL *is* the WebGPU shading
  language — so on **WebGPU** there's no shader translation. On the **WebGL2** fallback, `wgpu`'s
  `naga` translates WGSL→GLSL and must validate against GL limits (an acceptance item, not free).
  (`Cargo.toml`: `wgpu=22.1`, `egui=0.29.1`, `egui-wgpu`, `winit=0.30.5`, all wasm-capable.)
- **The point renderer is window-independent (but the canvas path is new work).**
  `PointCloudRenderer::new(device, queue, color_format, depth_format, capacity,
  supports_compaction)` (`render/point_cloud.rs`) needs only a wgpu device/queue, and
  `offscreen.rs::render_png` builds `Instance → adapter → device` and renders to a texture with
  no winit window — so the **renderer** detaches cleanly. But `offscreen.rs` still uses
  `pollster`, the native loader, `Plan`, file output, and **no surface**, so it does *not* prove
  the browser canvas/surface/event-loop path is trivial — that shell is net-new (see Phase 1).
- **Points are a flat POD buffer — necessary but not a complete wire contract.** `GpuPoint`
  (`data/point.rs`) is **always 32 bytes** (`pos[3] f32`, `intensity`, `weight`, `flags`,
  `_pad[2]` where `_pad[0]` = mutable cluster id), uploaded via `bytemuck::cast_slice`. Good as a
  payload body, but it's native-endian raw floats normalized to **server-selected bounds**, so
  the protocol needs a framed header (magic/version/endian/stride/layout-id/bounds/chunk-kind).
  Note: the `compact` Cargo feature is currently an **empty stub** — `GpuPoint` is f32 only;
  any f16/16-byte path must be *implemented* (struct + vertex layout + shader + negotiation + tests).
- **The compute/indirect fallback exists but is NOT yet WebGL2-safe.** The renderer branches on
  `supports_compaction` and can cull in the vertex shader instead — but `PointCloudRenderer::new`
  *unconditionally* allocates buffers with `STORAGE` and `INDIRECT` usages
  (`point_cloud.rs:69/77/83`), which WebGL2 can't provide. The fallback needs **capability-split
  buffer creation** (vertex/copy only when compute is off) before it runs on WebGL2.

## The one real blocker: the loader is native-only

`data/loader.rs` cannot run in a browser:
- reads Bruker `.d` via `rustdf` → `rusqlite` (SQLite) + file I/O + zstd/lzf — no FS/SQLite in wasm;
- runs on `std::thread` + `crossbeam_channel` + `rayon` — browser threads need
  SharedArrayBuffer/cross-origin isolation;
- init uses `pollster::block_on` — the web main thread can't block (must be async).

`data/meta.rs` is also native/server-side: it reads `.d` metadata via `rustdf`, and `Plan`
carries a `MetaIndex` into app init (`app.rs:29`). The browser can't build that — so the server
must also emit a **metadata DTO** (axis bounds, frame/window info, point-count estimate, sensible
default ranges) that seeds the client before any points arrive.

⇒ **Renderer → browser; loader + metadata → server.** The server runs the existing `rustdf` read
+ downsample + `GpuPoint` packing (what `loader.rs` already does) and streams a **message
protocol** to the browser (not just a point buffer — see Phase 2), which uploads point chunks
into the instance buffer and renders. `loader.rs`'s packing becomes the server encoder.

## Chosen approach

**A. Renderer-in-browser (WASM + WebGPU).** Compile the render half to `wasm32`, host in a
`<canvas>`, stream `GpuPoint` chunks from the server. Local rotate/zoom on the client GPU,
millions of points, native look. This is the target.

**B. Server-side render + stream pixels (documented fallback).** Run the native renderer on a
server GPU (the `offscreen.rs` path nearly does this already) and stream frames/WebRTC. Almost
no renderer changes, client needs no GPU — but every interaction is a server round-trip and you
ship pixels. Use only for weak clients / data too large to ship, or as a stopgap.

## Work breakdown

### Phase 0 — crate split (render lib vs native binary)  *(DONE — feature-gate approach)*
**Implemented** via a `native` feature + `src/lib.rs` (rather than a separate crate dir, which
stays an optional later refactor): native deps (`rustdf`, `mscore`, `winit`, `egui-winit`,
`clap`, `env_logger`, `rayon`, `crossbeam-channel`, `png`) are now `optional = true` under
`native` (default on); `[[bin]]` has `required-features = ["native"]`; `app`/`offscreen` and
`data::{loader,meta}` are `#[cfg(feature = "native")]`; `main.rs` consumes the lib. The one real
code coupling — `group_color` in the native loader, used by `ui.rs` — was moved to the
render-safe `render::colormap`. **Verified:** `cargo check` (native) green + 20 unit tests pass;
`cargo check --no-default-features --lib` compiles the render half with **no** rustdf/winit/
rayon/etc.; and `cargo check --no-default-features --lib --target wasm32-unknown-unknown`
**already compiles** (wgpu 22 pulls wasm-bindgen/web-sys automatically). The remaining notes
below are the original rationale / the separate-crate option.

Module moves alone are insufficient: `Cargo.toml` pulls `rustdf`, `mscore`, `pollster`, `rayon`,
`crossbeam-channel`, `winit`, and desktop `egui-winit` features **unconditionally** (`Cargo.toml:17`).
- Create a **separate render crate with its own manifest** (e.g. `tims-viewer-render`) whose deps
  are render-safe only (`wgpu`, `glam`, `bytemuck`, `egui`, `egui-wgpu`, `half`, `log`). It holds
  the render-only modules: `render/*`, `data/point`, **`data/demo`** (portable — no loader/thread
  deps), `camera`, `state`, `ticks`, `colormap`, `uniforms`, the egui UI, and the WGSL shaders.
  Must compile for host **and** `wasm32-unknown-unknown`.
- **Split `app.rs`** (currently imports desktop `winit`, `pollster`, loader types, `crossbeam`
  errors, `rayon` — `app.rs:8/17/357/908`) into three parts: (a) renderer state (portable),
  (b) the loader/message-drain logic (shared, transport-agnostic), (c) the **native platform
  shell** (winit window + event loop + pollster init). A parallel **wasm shell** replaces (c).
- Keep the native binary crate (`tims-viewer`) for `data/loader`, `data/meta`, `offscreen`,
  `main`, the `clap` CLI, and the native deps.
- Acceptance: `cargo build` (native) unchanged & tests green; `cargo build --target
  wasm32-unknown-unknown -p tims-viewer-render` compiles (verified free of
  rustdf/thread/crossbeam/rayon/pollster, including via transitive deps).

### Phase 1 — WASM shell (render a buffer in-browser)  *(BUILT — pending in-browser check)*
**Implemented** as a Trunk crate `tims-viewer/web/` (`tims-viewer-web`, a wasm-only cdylib;
empty on native so the workspace build is unaffected). `src/lib.rs`: gets the `<canvas>`, async
wgpu init (`request_adapter/_device().await`, no `pollster`), WebGPU with `webgl` fallback +
WebGL2 downlevel limits, builds `PointCloudRenderer` with a seeded synthetic `demo_cloud()`
(helix + 2 blobs, 250k pts) and runs a `requestAnimationFrame` loop with an auto-orbiting camera.
**Verified here:** `cargo build -p tims-viewer-web --target wasm32-unknown-unknown` is clean;
native workspace build unaffected. **Codex-reviewed + hardened:** proper `SurfaceError` match
(skip on Timeout, reconfigure on Lost/Outdated, stop on OOM); DPR-correct sizing + a `resize`
listener (shared `Gfx` in `Rc<RefCell>`, depth texture tracks resize); `run() -> Result` that
surfaces fatal errors into a `#status` DOM element instead of a blank page; capacity sized to the
demo cloud (not a fixed 1M). **Not yet verified:** actual in-browser render (needs a display) —
run `cargo install trunk` then `cd tims-viewer/web && trunk serve` and open the URL.
**Mouse controls (DONE):** left-drag orbit, wheel zoom, shift/right-drag pan (wired to the
camera's `orbit`/`zoom`/`pan`); auto-orbit until first interaction; `mousemove`/`mouseup` on the
window so drags survive leaving the canvas; context menu suppressed for right-drag.
Deferred (Phase 4, app-embedding): cancellable RAF loop / teardown for unmount.

**Web UI console (DONE — DOM, not egui):** the web's payoff is real DOM/CSS, so instead of
porting egui we built an *instrument console* over the canvas (`index.html` + `wire_controls`):
a glass control rail (View / Render / Intensity / Filters), a monospace telemetry HUD (backend,
device, points, live FPS), and a colormap colorbar. Controls drive live `ParamsUniform` fields —
mode, colormap, point size/opacity, transfer (lin/sqrt/log) + exposure + intensity floor, MS1/MS2,
and per-axis cube crop (m/z · 1/K₀ · RT) — plus auto-orbit/reset/roll. Palette derived from the
data's viridis/inferno colormaps (teal `#35E0C4` + amber `#FFB23E`) on instrument-blue-black;
Space Grotesk / Inter / JetBrains Mono. All Rust-attached listeners (no hand-written JS). Compiles
for wasm; **in-browser visual check pending** (headless dev box). egui-on-web is no longer planned.
**Codex-reviewed + hardened:** wiring confirmed correct (radio handlers, crop/floor semantics, no
double-borrow); fixed the Auto-orbit switch desync (mouse drag + wheel now uncheck it) and made the
colormap read its option value (not DOM order); a11y pass (real `<label for>` / `role=radiogroup` /
`aria-label`s, `:focus-visible` proxies on the hidden inputs, reduced-motion); CSS hardening (HUD
ellipsis for long device names, `backdrop-filter` opaque fallback, mobile toggle placement).

- Web entry: create the canvas surface, **async** wgpu init (replace `pollster::block_on` with
  `request_adapter().await` / `request_device().await`), request WebGPU with a WebGL2 fallback.
- Port the egui panel to web (egui-wgpu/egui-winit support wasm) — or start headless (no UI) to
  de-risk, UI in Phase 3.
- **Capability-split the renderer** *(DONE)*: `compacted`/`draw_args` moved into `ComputeStage`
  (created only when `supports_compaction`), and `master` drops `STORAGE` on the fallback — so the
  WebGL2 path allocates no storage/indirect buffers. Verified by a forced-fallback GPU smoke test
  (`offscreen_render_smoke_webgl2_fallback`, `run_smoke(Some(false))`) — renders clean on Metal.
- Feed a **hardcoded/demo** `Vec<GpuPoint>` (reuse the portable `data/demo`) into
  `PointCloudRenderer` and render + orbit-camera in the canvas.
- Acceptance: a demo cloud renders and rotates in Chrome (WebGPU) **and** in a WebGL2-only
  context (compaction off, vertex-shader cull path, no storage/indirect resources created),
  with the WGSL→GLSL translation validating against GL limits.

### Phase 2 — server → browser streaming  *(increment 1 DONE: raw points over HTTP)*
**Built + verified here:** a native-gated `serve.rs` (the `--serve <port>` flag) drains the loader
into a `Vec<GpuPoint>` (no GPU — loader is pure CPU, points already normalized to the cube) and
serves them over HTTP via `tiny_http`: `GET /points` → raw little-endian `GpuPoint` bytes
(`application/octet-stream`, CORS `*`), `GET /meta` → JSON (stride, count, cube bounds). Works with
a `.d` path or `DEMO`. Curl-verified: `tims-viewer DEMO --serve 8090 --budget 300000` → `/points`
returns exactly `n*32` bytes, `/meta` valid. Client (`tims-viewer-web`): `fetch_points()` GETs the
bytes (URL `http://localhost:<?port=N | 8090>/points`), reinterprets them, and renders — falling
back to the synthetic `demo_cloud()` if no server. **Run:** `tims-viewer DEMO --serve 8090` +
`trunk serve` (the page fetches port 8090). **Not yet (increment 2+):** the framed message protocol
below (stats/histograms/annotations), websocket-progressive streaming, axis-label bounds wired from
`/meta`, and per-request slice params.

**Codex-reviewed + hardened:** binds `127.0.0.1` (not `0.0.0.0`); serves the byte body from an
`Arc<[u8]>` via a `Cursor` (no per-request copy); `collect_points` bounds the buffer at `capacity`
while still draining to `Done`; `/meta` numbers finite-guarded; `--serve` hard-fails on big-endian
targets; respond errors logged. Client rejects a body that isn't a multiple of the 32-byte stride,
and supports `?points=<url>` override (proxied/same-origin deploys). Re-verified by curl.

**Increment 2 (DONE — real-unit context + validation):** `/meta` is now a `serde_json` DTO
(version, point_stride, n_points, cube bounds, intensity p99). The client fetches `/meta` *first*,
**validates** `version == 1` and `point_stride == 32` before trusting `/points` (falls back to the
demo on mismatch/absence), and uses the bounds so the per-axis crop filters read in **real units**
(m/z · 1/K₀ with decimals · RT) instead of percentages, and the intensity **Floor** control scales
to the data's p99. Curl-verified (`/meta` DTO, `/points` n*32). **Codex-reviewed + hardened:**
exact stride validation (no `as usize` truncation); robust bounds parse (best-effort → % labels on
malformed); `intensity_p99` filters non-finite + uses `select_nth`; server routes by path (handles
`/meta?query`); query-aware `meta_url`. Still deferred to increment 3:
multi-threaded serving, websocket-progressive streaming (one HTTP blob today, no progress bar), the
full framed `LoadMsg` protocol (stats/histograms/annotations), and per-request slice params.

Parity needs the **whole `LoadMsg` set**, not only point bytes: `app.rs` also consumes `Stats`
(transfer-fn percentiles), `Histograms` + 2D projections (the filter UI / minimaps), annotations,
`Progress`, and `PeakChunk` enrichment (`loader.rs:34`, `app.rs:589`). Plus the metadata DTO.
- **Define a framed message protocol** mirroring `LoadMsg`: a header per frame
  (magic, version, endian, message kind, generation) then the body. Point bodies carry the
  `GpuPoint` stride/layout-id and the axis bounds they were normalized to; cluster-id semantics
  in `_pad[0]` stay explicit (it's mutable client state).
- Server endpoint (reuse `loader.rs` packing): given run ref + RT/m-z/scan window + budget, read
  via `rustdf`, downsample (stratified, as today), emit the metadata DTO then stream the frames
  over a websocket.
- Client receiver mirrors `app.rs`'s loader-drain for every message kind (points → instance
  buffer; stats/histograms → UI; annotations → overlay; progress → status).
- **Version + validate** the protocol on both ends (length/stride/layout/endian checks); the f16
  layout (if/when implemented) is a distinct layout-id negotiated up front.
- Acceptance: a real slice from a server-resident `.d` streams to the browser and renders with
  point counts, filters, histograms, and look matching the native tool.

### Phase 3 — feature parity + web polish
- Volume raycaster (`render/volume`): it creates an **`R16Float` 3D texture** with filterable
  float sampling and uploads f16 voxels (`volume.rs:185/234/369`). Gate it behind **explicit
  capability checks**: 3D textures, `R16Float` renderable/filterable support, float-texture
  filtering, max-3D-texture-size, and upload row/alignment — not just "off on WebGL2."
- Annotations (DIA windows / DDA crosses), axis frame + tick labels, colorbar, the control panel.
- **Implement a real `compact` f16 `GpuPoint`** (struct + vertex layout + shader + protocol
  layout-id + tests — today the feature is an empty stub) to roughly halve client buffer memory
  (12M pts × 32 B ≈ 384 MB → ~192 MB). Treat the memory win as *conditional on this work*.
- Verify the WebGL2 fallback end-to-end (no compute/indirect, capability-split buffers, volume
  disabled or degraded per the checks above).

### Phase 4 — packaging & integration
- Build with `trunk`/`wasm-pack`; host the wasm + canvas; wire into the web app shell.
- Co-locate the streaming endpoint with the data (matches the "compute near data" decision).

## Gotchas / risks
- **WebGL2 fallback isn't free** — `PointCloudRenderer::new` must be split so the no-compute path
  allocates **no STORAGE/INDIRECT buffers**; otherwise WebGL2 device creation fails. Plus
  WGSL→GLSL (`naga`) translation must validate against GL limits.
- **`compact` f16 is a stub today** — the memory-halving claim is *conditional* on implementing a
  real f16 point struct + vertex layout + shader + protocol layout-id.
- **Volume needs capability gating** — `R16Float` 3D texture + float filtering + 3D-texture-size
  + upload alignment; not a simple on/off.
- **WebGPU availability**: stable in Chrome/Edge and recent Safari/Firefox; older browsers hit the
  WebGL2 fallback (lower throughput; no compute/indirect; volume likely disabled).
- **No client threads/rayon** — fine, all parallelism is server-side in the loader.
- **Async init** — drop `pollster` on web; keep it native.
- **Surface format / sRGB** differences between native and web canvas configuration.
- **egui-winit on wasm** — event/resize/DPI handling differs from native; budget time for it.
- **Wire-format drift** — the framed protocol (`GpuPoint` body + metadata DTO + `LoadMsg` kinds)
  is a network contract: header with magic/version/endian/stride/layout-id/bounds/chunk-kind,
  validated on both ends; cluster-id (`_pad[0]`) semantics kept explicit.

## Acceptance (project)
- [x] Render library compiles for `wasm32` (no rustdf/thread/crossbeam/rayon, transitively);
      native binary + 20 unit tests unchanged. *(Phase 0 — via the `native` feature + `lib.rs`.)*
- [ ] `PointCloudRenderer` is capability-split; demo cloud renders + orbits in-browser on **WebGPU**
      and on a **WebGL2** context (no storage/indirect buffers created).
- [ ] Server emits a metadata DTO + a versioned, framed message protocol (points + stats +
      histograms/projections + annotations + progress + peak chunks), reusing `loader.rs` packing.
- [ ] A server-resident `.d` slice streams to the browser and renders at parity (points, filters,
      histograms) with the native point cloud.
- [ ] Documented WebGPU/WebGL2 capability matrix (compute, indirect, 3D/`R16Float` volume); f16
      compact path either implemented + negotiated or explicitly deferred.
