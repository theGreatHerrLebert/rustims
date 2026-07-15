//! Chemistry core for timsim v2.
//!
//! The parts of a proteomics sample model that are *sample-intrinsic*: FASTA, enzymes, and
//! the analytic (quantitative) digest. Nothing here knows about an instrument, an
//! acquisition method, or a gradient.
//!
//! The digest is split along the structure/quantity boundary: [`digest::Enumerator`]
//! answers *which peptides exist* (shared across every sample in a design);
//! [`digest::YieldModel`] answers *how much of each*, per condition.
//!
//! Cleavage rules and FASTA parsing are derived from Sage (MIT, Copyright (c) 2022 Michael
//! Lazear). Its enumeration semantics — `Digest`, decoys, `max_missed_cleavages` as a
//! generative bound — are deliberately not carried over. See [`digest`].

pub mod design;
pub mod digest;
pub mod enzyme;
pub mod fasta;
pub mod fragment;
pub mod ids;
pub mod ionize;
pub mod isotope;
pub mod mass;
pub mod modify;

pub use digest::{
    BlockingOccupancy, Bounds, DigestStats, Enumerator, Occurrence, ProteinDigest, YieldModel,
};
pub use design::{Design, DesignSpec, Share};
pub use enzyme::{Enzyme, Protocol};
pub use fasta::{parse as parse_fasta, Protein};
pub use ids::{modform_id, peptide_id, precursor_id, CollisionCheck, ID_HASH};
pub use ionize::{ChargeModel, Flyability, Ionizer, Precursor};
pub use isotope::{composition, envelope, Composition};
pub use modify::{enumerate_modforms, Modform, Modification, ModformStats, Site, Stage};
