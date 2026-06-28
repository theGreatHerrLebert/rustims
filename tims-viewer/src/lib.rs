//! tims-viewer library: the GPU renderer and view state.
//!
//! Split into a **render-safe** core (compiles for `wasm32` too: GPU pipelines, point format,
//! camera, colormap, UI panel) and **native-only** modules behind the `native` feature
//! (`app`, `offscreen`, plus `data::loader`/`data::meta`, which pull rustdf/SQLite/threads).
//! See `NATIVE_WEB.md` for the web-embedding plan this split enables.

// Render-safe core.
pub mod camera;
pub mod cluster;
pub mod data;
pub mod render;
pub mod state;
pub mod ticks;
pub mod ui;

// Native-only: window/event-loop orchestration, headless PNG export, point-streaming server.
#[cfg(feature = "native")]
pub mod app;
#[cfg(feature = "native")]
pub mod offscreen;
#[cfg(feature = "native")]
pub mod serve;
