//! Data layer: metadata, streaming loader, GPU point format, and the synthetic source.

pub mod demo;
pub mod point;
// Native-only: the Bruker `.d` reader (rustdf/SQLite) and its metadata index.
#[cfg(feature = "native")]
pub mod loader;
#[cfg(feature = "native")]
pub mod meta;
