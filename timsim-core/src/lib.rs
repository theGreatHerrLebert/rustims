//! The timsim v2 simulation engine: feature-space assembly, projection, and streaming frame
//! render for timsTOF / Thermo / mzML. Extracted from the former `ms-io::sim` (R4).
//!
//! Reads Bruker TDF metadata + writes frames via [`ms_io::data`]; the acquisition-scheme
//! contract types live in the zero-dependency `timsim-types` leaf (added in R4 stage 2).

pub mod acquisition;
#[cfg(feature = "thermo")]
pub mod astral_dispatch;
pub mod containers;
pub mod dda;
pub mod dia;
pub mod handle;
pub mod lazy_builder;
pub mod library;
#[cfg(feature = "mzml")]
pub mod mzml;
pub mod precursor;
pub mod projector;
pub mod scheme;
pub mod utility;
