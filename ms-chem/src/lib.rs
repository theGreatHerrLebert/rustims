//! # ms-chem — mass-spectrometry chemistry primitives
//!
//! The L0 leaf of the rustims crate federation: elements, amino-acid residues, sum formulas,
//! isotopes, modifications, and backbone fragments. Pure Rust, no PyO3 — usable by any Rust MS
//! consumer (the simulator is only the first).
//!
//! ## Provenance
//!
//! ms-chem unifies the three chemistry implementations that lived in the rustims monorepo
//! (`mscore/chem`, `rustms/chem`, `timsim-chem`). It is not a fresh guess: every table and formula
//! was chosen from those impls' *differential parity* evidence (`CHEM_PARITY.md`). The canonical
//! semantics:
//!
//! - **Element masses** — identical across impls; adopted verbatim ([`elements`]).
//! - **Residue masses** — *computed from elements*, a single source of truth ([`residue`]).
//! - **Isotopic abundances** — the newer CIAAW values (N/S differed; resolved) — *(isotope module,
//!   next increment)*.
//! - **Proton** — full CODATA `1.007276466621` ([`mass::PROTON`]).
//! - **Modifications / fragments** — a unified cross-checked table and typed b/y ions — *(next
//!   increments)*.
//!
//! The parity suite that proved the equivalence rides along as ms-chem's regression gate.

pub mod elements;
pub mod formula;
pub mod fragment;
pub mod isotope;
pub mod mass;
pub mod modification;
pub mod residue;

pub use formula::FormulaError;
pub use fragment::{fragment_ions, Fragment, IonType};
pub use isotope::{envelope, EnvelopeError};
pub use mass::{monoisotopic, mz, UnknownResidue, PROTON, WATER};
pub use modification::{by_id as modification_by_id, Modification, BUILTIN as BUILTIN_MODIFICATIONS};
pub use residue::{peptide_composition, residue_composition, residue_monoisotopic_mass, Composition};
