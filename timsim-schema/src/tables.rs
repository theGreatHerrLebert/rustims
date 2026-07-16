//! **The only place a column name is spelled.**
//!
//! Stage code must never write `df["amount_amol"]`. It writes `peptides::AMOUNT_AMOL`. That
//! is the entire point of this module, and it is not a style preference — it is the fix for
//! the failure that motivated the redesign:
//!
//! > `retention_time_gru_predictor` — a column named after a neural network replaced years
//! > ago — is hard-coded in the v1 Rust SQL reader (`rustdf/src/sim/handle.rs:210`) in 66
//! > places across two languages, Python has degraded to searching for it *by substring*
//! > (`simulator.py:1540`), and it has now spread into raw SQL in the published benchmark
//! > repo (`MBRBenchmark.py:845`).
//!
//! Parquet does not fix that. One definition, read by both languages, does.

use crate::{Axis, TableSpec};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use std::sync::Arc;

/// Schema version. Bump the **major** on any breaking change; readers reject a mismatch.
pub const SCHEMA_VERSION: &str = "2.0";

fn spec(name: &'static str, axis: Axis, fields: Vec<Field>) -> TableSpec {
    TableSpec {
        name,
        axis,
        schema: Arc::new(Schema::new(fields)),
    }
}

/// Every table, by name. The single registry.
pub fn all() -> Vec<TableSpec> {
    vec![
        proteome::spec(),
        peptides::spec(),
        peptide_occurrences::spec(),
        modifications::spec(),
        modforms::spec(),
        precursors::spec(),
        precursor_ccs::spec(),
        peptide_rt::spec(),
        localization_sites::spec(),
        fragment_intensities::spec(),
        ion_spectra::spec(),
        cleavage_sites::spec(),
        samples::spec(),
        runs::spec(),
        sample_run_map::spec(),
        protein_quantities::spec(),
        peptide_quantities::spec(),
    ]
}

pub fn by_name(name: &str) -> Option<TableSpec> {
    all().into_iter().find(|t| t.name == name)
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCTURE — what molecules exist. Computed once; shared by every sample.
// ─────────────────────────────────────────────────────────────────────────────

pub mod proteome {
    use super::*;
    pub const TABLE: &str = "proteome";
    pub const PROTEIN_ID: &str = "protein_id";
    pub const SEQUENCE: &str = "sequence";
    /// Residue count. Redundant with `sequence`, and carried anyway: downstream stages
    /// (e.g. `timsim-yield`) need the protein length to reconstruct the cleavage-boundary
    /// lattice, and *inferring* it from the occurrence table is wrong — the length filter
    /// discards C-terminal peptides, so `max(end)` under-reports the true length.
    pub const LENGTH: &str = "length";
    pub const DESCRIPTION: &str = "description";
    pub const ORGANISM: &str = "organism";
    pub const IS_CONTAMINANT: &str = "is_contaminant";

    /// No amounts. Abundance is a *quantity* and lives on its own axis, so that structure
    /// can be shared across samples (SPEC B3).
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Structure,
            vec![
                Field::new(PROTEIN_ID, DataType::Utf8, false),
                Field::new(SEQUENCE, DataType::Utf8, false),
                Field::new(LENGTH, DataType::UInt32, false),
                Field::new(DESCRIPTION, DataType::Utf8, true),
                Field::new(ORGANISM, DataType::Utf8, true),
                Field::new(IS_CONTAMINANT, DataType::Boolean, false),
            ],
        )
    }
}

pub mod peptides {
    use super::*;
    pub const TABLE: &str = "peptides";
    pub const PEPTIDE_ID: &str = "peptide_id";
    pub const SEQUENCE: &str = "sequence";
    pub const LENGTH: &str = "length";
    pub const MASS_MONOISOTOPIC: &str = "mass_monoisotopic";

    /// `mass_monoisotopic`, not `mass`: monoisotopic is for m/z, *average* is for bulk
    /// quantities (ng on column). Conflating them is a real error, so neither gets to be
    /// called just "mass".
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Structure,
            vec![
                Field::new(PEPTIDE_ID, DataType::UInt64, false),
                Field::new(SEQUENCE, DataType::Utf8, false),
                Field::new(LENGTH, DataType::UInt16, false),
                Field::new(MASS_MONOISOTOPIC, DataType::Float64, false),
            ],
        )
    }
}

pub mod peptide_occurrences {
    use super::*;
    pub const TABLE: &str = "peptide_occurrences";
    pub const PEPTIDE_ID: &str = "peptide_id";
    pub const PROTEIN_ID: &str = "protein_id";
    pub const START: &str = "start";
    pub const END: &str = "end";
    pub const N_MISSED_CLEAVAGES: &str = "n_missed_cleavages";

    /// The digest operator, and the protein-inference answer key.
    ///
    /// Note there is **no `p_yield` column**. Yield depends on digestion efficiency and on
    /// the occupancy of cleavage-blocking modifications — and the latter is regulated, so it
    /// differs between conditions. A yield here would make the structure condition-dependent
    /// and silently destroy sharing across samples. Yields are emitted per-sample into
    /// `peptide_quantities`.
    ///
    /// `start`/`end` are **protein-residue coordinates**, which is what site-localisation
    /// ground truth needs — peptide-relative offsets break for shared peptides and repeats.
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Structure,
            vec![
                Field::new(PEPTIDE_ID, DataType::UInt64, false),
                Field::new(PROTEIN_ID, DataType::Utf8, false),
                Field::new(START, DataType::UInt32, false),
                Field::new(END, DataType::UInt32, false),
                Field::new(N_MISSED_CLEAVAGES, DataType::UInt16, false),
            ],
        )
    }
}

pub mod modifications {
    use super::*;
    pub const TABLE: &str = "modifications";
    pub const NAME: &str = "name";
    pub const UNIMOD_ID: &str = "unimod_id";
    pub const TARGETS: &str = "targets";
    pub const SITE: &str = "site";
    pub const OCCUPANCY: &str = "occupancy";
    pub const MASS_DELTA: &str = "mass_delta";
    pub const COMPOSITION: &str = "composition";
    pub const BLOCKS_CLEAVAGE: &str = "blocks_cleavage";
    pub const STAGE: &str = "stage";

    /// **The modification spec, as an artifact.**
    ///
    /// This table exists because the spec has *two* consumers that must never disagree:
    ///
    /// - [`modforms`] needs `occupancy` and `mass_delta` — *how much of a peptide carries the mod*.
    /// - `timsim-yield` needs `occupancy` and `blocks_cleavage` — because acetyl-K, GG-K, trimethyl-K
    ///   and TMT-K physically stop trypsin, so a modification changes **which peptides exist**.
    ///   `p_eff(k) = p · (1 − blocking_occupancy(k))`.
    ///
    /// Those are the same numbers used for two different jobs, in two different tools. Passing them
    /// as flags to both is exactly the shape of every B13 bug found so far — the digest/yield
    /// `--max-missed-cleavages` split, and the flyability that was drawn twice from the same
    /// distribution with different keys. So the spec is written **once**, by `timsim-modify`, and
    /// read by everyone else. The flag that would let two stages disagree does not exist.
    ///
    /// # Occupancy is a marginal, not a count
    ///
    /// `occupancy` is the chemist's number: the fraction of *this site*, across all molecules, that
    /// carries the mod. Phospho ~0.02, Met oxidation ~0.05, carbamidomethyl ~0.98 (which is just the
    /// alkylation efficiency). It is **not** "variable mods, max 3 per peptide" — that is a database
    /// search parameter, and simulating from it is what makes a simulator agree with a search engine
    /// for reasons that have nothing to do with chemistry (SPEC §7).
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Structure,
            vec![
                Field::new(NAME, DataType::Utf8, false),
                Field::new(UNIMOD_ID, DataType::UInt32, false),
                // Residues the mod may sit on, e.g. "STY" for phospho. Empty for a terminal mod.
                Field::new(TARGETS, DataType::Utf8, false),
                // "residue" | "n_term" | "c_term"
                Field::new(SITE, DataType::Utf8, false),
                Field::new(OCCUPANCY, DataType::Float64, false),
                Field::new(MASS_DELTA, DataType::Float64, false),
                // Elemental formula, e.g. "HO3P". A modification RESHAPES the isotope envelope, it
                // does not merely shift the mass — so the composition, not just the delta, is what
                // downstream needs. It is also the cross-check on `mass_delta`.
                Field::new(COMPOSITION, DataType::Utf8, false),
                Field::new(BLOCKS_CLEAVAGE, DataType::Boolean, false),
                // "protein" (before the protease) | "peptide" (after it). Load-bearing: only a
                // pre-digestion protein-level mod can block cleavage, and pyroglutamate forms on a
                // peptide N-terminus, which does not exist until the protease has cut.
                Field::new(STAGE, DataType::Utf8, false),
            ],
        )
    }
}

pub mod modforms {
    use super::*;
    pub const TABLE: &str = "modforms";
    pub const MODFORM_ID: &str = "modform_id";
    pub const PEPTIDE_ID: &str = "peptide_id";
    pub const MOD_POSITIONS: &str = "mod_positions";
    pub const MOD_NAMES: &str = "mod_names";
    pub const MASS_MONOISOTOPIC: &str = "mass_monoisotopic";
    pub const MASS_DELTA: &str = "mass_delta";
    pub const ABUNDANCE_FRACTION: &str = "abundance_fraction";

    /// The modified forms of each peptide, with the fraction of its molecules in each form.
    ///
    /// # A modform is a species, not an annotation
    ///
    /// v1 treats modifications as a *search-engine parameter* — pick up to N variable mods and
    /// generate every combination — which produces a peptide space shaped like a search space rather
    /// than like a sample. Here a modform is a **real population**: `abundance_fraction` is the
    /// fraction of that peptide's molecules carrying exactly this set of mods, and over the complete
    /// enumeration the fractions sum to exactly 1. The unmodified form is a modform like any other,
    /// and for most peptides it is the overwhelmingly dominant one.
    ///
    /// That identity — **Σ abundance_fraction = 1 over the full enumeration** — is the oracle. With
    /// an abundance floor the sum falls short, and the shortfall is *exactly* the probability mass
    /// discarded, which is reported rather than estimated.
    ///
    /// # Why positions are 0-based within the peptide
    ///
    /// Site-localisation ground truth needs protein-residue coordinates, which is a join through
    /// `peptide_occurrences` — the same peptide occurring in two proteins has two sets of protein
    /// coordinates but one set of peptide coordinates. Storing the peptide-local position keeps the
    /// modform independent of where the peptide came from, which is what lets the structure axis be
    /// computed once and shared.
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Structure,
            vec![
                Field::new(MODFORM_ID, DataType::UInt64, false),
                Field::new(PEPTIDE_ID, DataType::UInt64, false),
                // 0-based, within the peptide. Empty list for the unmodified form.
                Field::new(
                    MOD_POSITIONS,
                    DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
                    false,
                ),
                // Parallel to MOD_POSITIONS: the modification name at each position. Joins to
                // `modifications.name`.
                Field::new(
                    MOD_NAMES,
                    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                    false,
                ),
                // The modform's own mass. Written here so no consumer ever re-derives it by adding
                // deltas to the bare peptide mass — that is a fact this artifact knows (B13).
                Field::new(MASS_MONOISOTOPIC, DataType::Float64, false),
                Field::new(MASS_DELTA, DataType::Float64, false),
                Field::new(ABUNDANCE_FRACTION, DataType::Float64, false),
            ],
        )
    }
}

pub mod ion_spectra {
    use super::*;
    pub const TABLE: &str = "ion_spectra";
    pub const PRECURSOR_ID: &str = "precursor_id";
    pub const MS_LEVEL: &str = "ms_level";
    pub const MZ: &str = "mz";
    pub const INTENSITY: &str = "intensity";

    /// The **instrument-independent** spectra of a peptide ion — one MS1 (precursor isotopes) and one
    /// MS2 (fragment isotopes) — as pure `(m/z, intensity)`. MEASUREMENT.
    ///
    /// This is the seam that makes the render a *projector*. The chemistry (which peaks exist, at what
    /// m/z, how intense) is computed once from the peptide ion (mscore isotopic spectra + Prosit
    /// fragment intensities) and carries no instrument geometry. A per-run projector then maps each
    /// peak onto where it lands in the raw data — timsTOF `(frame, scan, tof)` with diagonal quadrupole
    /// transmission, or a non-IMS `(scan, m/z)` layout — so one spectrum artifact drives any instrument.
    ///
    /// **Two rows per ion**, keyed by `ms_level` (1 = precursor, 2 = fragment; extensible to MS3+).
    /// The MS2 row is the FULL fragment spectrum; transmission gating is the projector's job, not the
    /// spectrum's. `mz`/`intensity` are equal-length lists (the sparse peak list, isotope-expanded).
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Measurement,
            vec![
                Field::new(PRECURSOR_ID, DataType::UInt64, false),
                Field::new(MS_LEVEL, DataType::UInt8, false),
                Field::new(MZ, DataType::List(Arc::new(Field::new("item", DataType::Float64, true))), false),
                Field::new(INTENSITY, DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
            ],
        )
    }
}

pub mod fragment_intensities {
    use super::*;
    pub const TABLE: &str = "fragment_intensities";
    pub const PRECURSOR_ID: &str = "precursor_id";
    pub const ION_TYPE: &str = "ion_type";
    pub const ORDINAL: &str = "ordinal";
    pub const FRAG_CHARGE: &str = "frag_charge";
    pub const INTENSITY: &str = "intensity";

    /// Predicted fragment-ion intensities, per precursor — a spectral library. MEASUREMENT.
    ///
    /// The fragment half of the feature space, emitted as its own artifact rather than buried in the
    /// render. It is a measurement, not structure: fragment intensities depend on the **collision
    /// energy** (recorded in metadata as `timsim.fragments.collision_energy`), an instrument setting,
    /// so the same molecules fragment differently on different acquisitions. Keeping it separate
    /// decouples "what fragments, and how strongly" from "how a given acquisition assembles them".
    ///
    /// One **sparse** row per fragment above the abundance floor: b/y ions never observed are simply
    /// absent, as in a real library. `ion_type` is `"b"`/`"y"`, `ordinal` is 1-based (how many
    /// residues the fragment spans), `frag_charge` is the fragment's own charge. `intensity` is
    /// normalised to the precursor's base peak.
    ///
    /// This joins to `localization_sites` on `(ion_type, ordinal)`: a site is *resolvable* iff its
    /// site-determining fragments appear here with intensity — which is the whole point of the
    /// localization answer key.
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Measurement,
            vec![
                Field::new(PRECURSOR_ID, DataType::UInt64, false),
                Field::new(ION_TYPE, DataType::Utf8, false),
                Field::new(ORDINAL, DataType::UInt16, false),
                Field::new(FRAG_CHARGE, DataType::UInt8, false),
                Field::new(INTENSITY, DataType::Float32, false),
            ],
        )
    }
}

pub mod localization_sites {
    use super::*;
    pub const TABLE: &str = "localization_sites";
    pub const MODFORM_ID: &str = "modform_id";
    pub const PEPTIDE_ID: &str = "peptide_id";
    pub const MODIFICATION: &str = "modification";
    pub const TRUE_POSITION: &str = "true_position";
    pub const CANDIDATE_POSITIONS: &str = "candidate_positions";
    pub const SITE_DETERMINING: &str = "site_determining";

    /// The **site-localization answer key**: for a modform whose modification could sit on more than
    /// one residue of the peptide, where it *actually* is, where it *could* be, and which fragments a
    /// search must observe to tell them apart.
    ///
    /// # This is the benchmark's ground truth, not the simulator's problem
    ///
    /// We place the modification, so `true_position` is known by construction. Localisation is the
    /// *search engine's* task — recovering that position from the fragment spectrum — and this table
    /// is what it is scored against. A row exists only for an **ambiguous** modform (the modified
    /// residue occurs ≥2 times, e.g. a phospho on a peptide with two S/T/Y); a mod with a single
    /// candidate is trivially localised and needs no evidence.
    ///
    /// `site_determining` lists the `(ion_type, ordinal)` fragments — as strings like `"b3"`,
    /// `"y5"` — whose m/z depends on which candidate carries the mod. Joining this against the
    /// fragments a run actually rendered turns a search's wrong-site call into a precise verdict:
    /// *unresolvable* (the discriminating fragments were not in the data — no engine could have
    /// localised it) versus *the engine failed* (they were there and it missed). v1's evaluation
    /// asserts the former without checking; this makes it checkable.
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Structure,
            vec![
                Field::new(MODFORM_ID, DataType::UInt64, false),
                Field::new(PEPTIDE_ID, DataType::UInt64, false),
                Field::new(MODIFICATION, DataType::Utf8, false),
                // 0-based position of the modification we actually placed.
                Field::new(TRUE_POSITION, DataType::UInt32, false),
                // Every 0-based position the modification could have occupied (the target residues).
                Field::new(
                    CANDIDATE_POSITIONS,
                    DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
                    false,
                ),
                // The fragments that discriminate the candidates, e.g. ["b3","b4","y5"].
                Field::new(
                    SITE_DETERMINING,
                    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                    false,
                ),
            ],
        )
    }
}

pub mod peptide_rt {
    use super::*;
    pub const TABLE: &str = "peptide_rt";
    pub const PEPTIDE_ID: &str = "peptide_id";
    pub const RT_INDEX: &str = "rt_index";
    pub const SIGMA_HAT: &str = "rt_sigma_hat";
    pub const K_HAT: &str = "rt_k_hat";

    /// A peptide's retention-time index. STRUCTURE.
    ///
    /// # Why an index and not seconds
    ///
    /// v1 stored retention time in **seconds**, which is not a property of the peptide — it is what a
    /// *particular* LC gradient produces. Run the same peptide on a 30-minute gradient and a 2-hour
    /// gradient and the seconds differ, though the molecule is unchanged.
    ///
    /// **The peptide's property is its hydrophobicity** — where it sits in the elution *order*, and
    /// how far from its neighbours. Chronologer predicts exactly that (as minutes on a reference
    /// gradient, an affine image of hydrophobicity). Storing the index and mapping it to seconds per
    /// run is the same B14 split as CCS→1/K₀: the index is structure, the elution time is a
    /// measurement (SPEC §8, change #4).
    ///
    /// This is what makes runs comparable across gradients — the RT analog of cross-instrument
    /// mobility. It is also *more physical* than v1: v1 linearly stretched each sample's index range
    /// to fill `[0, gradient]`, so a peptide's RT depended on what else was in the tube. With a fixed
    /// reference range (carried in this artifact's metadata, `timsim.rt.index_min`/`index_max`,
    /// computed over the whole peptide space), a peptide always lands at the same gradient *fraction*,
    /// independent of the sample.
    ///
    /// A separate table from `peptides`, because the producer differs: `peptides` is Rust chemistry;
    /// the index comes from a deep model (`timsim-rt`, Python — the deep-predictor exception). The
    /// index is **nullable**: Chronologer rejects some peptides (unsupported mods, out-of-range
    /// length), and a rejected peptide has no index rather than a fabricated one.
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Structure,
            vec![
                Field::new(PEPTIDE_ID, DataType::UInt64, false),
                Field::new(RT_INDEX, DataType::Float64, true),
                // The elution PEAK SHAPE, in normalized [0, 1] coordinates — the EMG width and
                // tailing as their Beta-distributed unit draws, before any gradient is applied.
                //
                // v1 sampled the absolute width per RUN, in gradient-tied units, so the same peptide
                // eluted with a different width in different runs (a B8 non-reproducibility) and the
                // shape could not be reused across gradients. Storing the unit draw instead — drawn
                // once, identity-keyed on the peptide — fixes both: the render maps `sigma_hat` into
                // the run's gradient band (`sigma_lower + sigma_hat·(sigma_upper − sigma_lower)`,
                // where the band is a function of gradient length, and its exponent is where LC peak
                // capacity lives) and `k_hat` onto the tailing range. A longer gradient then naturally
                // reshapes the absolute peak while the peptide's *relative* shape is fixed and
                // reproducible.
                Field::new(SIGMA_HAT, DataType::Float64, false),
                Field::new(K_HAT, DataType::Float64, false),
            ],
        )
    }
}

pub mod precursor_ccs {
    use super::*;
    pub const TABLE: &str = "precursor_ccs";
    pub const PRECURSOR_ID: &str = "precursor_id";
    pub const CCS: &str = "ccs";
    pub const CCS_STD: &str = "ccs_std";

    /// Collision cross section, per precursor. STRUCTURE.
    ///
    /// # Why CCS and not 1/K₀
    ///
    /// The instrument reports `1/K₀` (inverse reduced mobility), and that is what v1 stored — but
    /// `1/K₀` is **not a property of the ion**. It is what a *particular* drift-tube measures, and it
    /// depends on the drift gas, its temperature, and its pressure. Change the gas from N₂ to
    /// anything else and every `1/K₀` moves, though not one molecule has changed.
    ///
    /// **CCS is the ion.** It is the rotationally-averaged collision cross section — a geometric
    /// property of the molecule and its charge — and the Mason–Schamp equation turns it into `1/K₀`
    /// *given an instrument*. So CCS belongs on the structure axis (predicted once, shared by every
    /// run) and `1/K₀` is a **measurement**, computed per run from CCS and the run's gas/temperature
    /// (SPEC §8, and the B14 rule: put the fact where the physics lives, not where today's instrument
    /// needs it).
    ///
    /// This is what makes cross-instrument simulation possible: one CCS artifact, measured on
    /// instrument A and instrument B, differs *only* in the gas parameters — which is exactly the
    /// experiment v1 could not express, because it threw CCS away and kept only N₂-at-305 K `1/K₀`.
    ///
    /// A separate table from `precursors`, not a column on it, because it has a **different
    /// producer**: `precursors` is pure chemistry (Rust); CCS comes from a deep model (`timsim-ccs`,
    /// Python — the standing deep-predictor exception). One table, one producer.
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Structure,
            vec![
                Field::new(PRECURSOR_ID, DataType::UInt64, false),
                // Collision cross section, Å². Instrument-independent.
                Field::new(CCS, DataType::Float64, false),
                // Predicted uncertainty in CCS (Å²), or null if the model does not report one.
                Field::new(CCS_STD, DataType::Float64, true),
            ],
        )
    }
}

pub mod precursors {
    use super::*;
    pub const TABLE: &str = "precursors";
    pub const PRECURSOR_ID: &str = "precursor_id";
    pub const PEPTIDE_ID: &str = "peptide_id";
    pub const MODFORM_ID: &str = "modform_id";
    pub const MODFORM_FRACTION: &str = "modform_fraction";
    pub const CHARGE: &str = "charge";
    pub const MZ: &str = "mz";
    pub const ISOTOPE_INTENSITY: &str = "isotope_intensity";
    pub const CHARGE_FRACTION: &str = "charge_fraction";
    pub const IONIZATION_PROPENSITY: &str = "ionization_propensity";

    /// The ion layer. `mz` and the isotope envelope are pure chemistry; `charge_fraction` and
    /// `ionization_propensity` are **propensities**, not realised values — the actual charge
    /// distribution and response depend on the eluent at elution and on what co-elutes, both of
    /// which are LC outputs (SPEC §8.2).
    ///
    /// # The factorisation is the ground truth
    ///
    /// `ion_amol = peptide_amol × modform_fraction × ionization_propensity × charge_fraction`
    ///
    /// Each multiplier is its own column, so the chain is **invertible**: an ion's amount divided by
    /// its charge fraction and its flyability recovers the peptide, and the peptide divided by
    /// `p_yield` recovers the protein.
    ///
    /// v1 destroys this — `events = (amount × efficiency).astype(int32)` collapses abundance and
    /// flyability into one integer, and nothing downstream can tell them apart again. Then you
    /// cannot ask the question that matters: *is this peptide missing because there is little of it,
    /// or because it does not fly?*
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Structure,
            vec![
                Field::new(PRECURSOR_ID, DataType::UInt64, false),
                Field::new(PEPTIDE_ID, DataType::UInt64, false),
                // The modform this ion comes from. The UNMODIFIED form is a modform like any other,
                // so this is never null: a run without `timsim-modify` gives every peptide one
                // modform at fraction 1.0. That uniformity is why there is no special case here.
                Field::new(MODFORM_ID, DataType::UInt64, false),
                Field::new(MODFORM_FRACTION, DataType::Float32, false),
                Field::new(CHARGE, DataType::UInt8, false),
                Field::new(MZ, DataType::Float64, false),
                Field::new(
                    ISOTOPE_INTENSITY,
                    DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                    false,
                ),
                Field::new(CHARGE_FRACTION, DataType::Float32, false),
                Field::new(IONIZATION_PROPENSITY, DataType::Float32, false),
            ],
        )
    }
}

pub mod cleavage_sites {
    use super::*;
    pub const TABLE: &str = "cleavage_sites";
    pub const PROTEIN_ID: &str = "protein_id";
    pub const POSITION: &str = "position";

    /// Where the enzyme *can* cut. Needed downstream to recompute yields per condition
    /// without re-running the enumeration. Protein termini are **not** listed — they are
    /// boundaries, not cleavage events.
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Structure,
            vec![
                Field::new(PROTEIN_ID, DataType::Utf8, false),
                Field::new(POSITION, DataType::UInt32, false),
            ],
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DESIGN — samples, runs, and the mapping between them.
// ─────────────────────────────────────────────────────────────────────────────

pub mod samples {
    use super::*;
    pub const TABLE: &str = "samples";
    pub const SAMPLE_ID: &str = "sample_id";
    pub const CONDITION: &str = "condition";
    pub const REPLICATE: &str = "replicate";

    /// A *biological* unit: distinct material, distinct amounts. Not a file.
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Design,
            vec![
                Field::new(SAMPLE_ID, DataType::Utf8, false),
                Field::new(CONDITION, DataType::Utf8, false),
                Field::new(REPLICATE, DataType::UInt32, false),
            ],
        )
    }
}

pub mod runs {
    use super::*;
    pub const TABLE: &str = "runs";
    pub const RUN_ID: &str = "run_id";
    pub const TECHNICAL_REPLICATE: &str = "technical_replicate";
    pub const INJECTION_ORDER: &str = "injection_order";

    /// One injection, rendering to one file.
    ///
    /// `injection_order` is required, not optional: carryover and batch drift couple a run to
    /// the runs *before* it, so a run is not determined by its coordinates alone.
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Design,
            vec![
                Field::new(RUN_ID, DataType::Utf8, false),
                Field::new(TECHNICAL_REPLICATE, DataType::UInt32, false),
                Field::new(INJECTION_ORDER, DataType::UInt32, false),
            ],
        )
    }
}

pub mod sample_run_map {
    use super::*;
    pub const TABLE: &str = "sample_run_map";
    pub const SAMPLE_ID: &str = "sample_id";
    pub const RUN_ID: &str = "run_id";
    pub const CHANNEL: &str = "channel";
    pub const MIX_FRACTION: &str = "mix_fraction";

    /// **Sample ≠ Run**, and the mapping is many-to-many:
    ///
    /// | topology       | mapping                |
    /// |----------------|------------------------|
    /// | label-free     | 1 sample → 1 run       |
    /// | fractionation  | 1 sample → N runs      |
    /// | TMT / isobaric | **N samples → 1 run**  |
    ///
    /// Conflating the two makes TMT unrepresentable and fractionation a hack — which is
    /// exactly how A/B support ends up bolted on. `channel` does **not** make TMT supported;
    /// isobaric tags alter the chemical entity and need labelling efficiency, isotopic
    /// impurity, and reporter cross-talk. It only means the model does not *preclude* it.
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Design,
            vec![
                Field::new(SAMPLE_ID, DataType::Utf8, false),
                Field::new(RUN_ID, DataType::Utf8, false),
                Field::new(CHANNEL, DataType::Utf8, true),
                Field::new(MIX_FRACTION, DataType::Float32, false),
            ],
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QUANTITY — how much, per sample. Cheap; recomputed per condition.
// ─────────────────────────────────────────────────────────────────────────────

pub mod protein_quantities {
    use super::*;
    pub const TABLE: &str = "protein_quantities";
    pub const PROTEIN_ID: &str = "protein_id";
    pub const SAMPLE_ID: &str = "sample_id";
    pub const AMOUNT_AMOL: &str = "amount_amol";
    pub const TRUE_LOG2FC: &str = "true_log2fc";
    pub const IS_REGULATED: &str = "is_regulated";

    /// The quantitative answer key.
    ///
    /// `amount_amol` — absolute attomoles. Not a relative abundance, not a multiplier. It is
    /// composable (digestion splits it, enrichment reweights it, loading normalises it) and
    /// correctly scaled: 200 ng on column across ~7 orders of dynamic range puts the low end
    /// at single-digit amol, which is where real detection limits sit.
    ///
    /// It is also a **sample-domain** quantity and stops at the ion source. Downstream it
    /// becomes ion flux, then counts, then ADC values — each a transfer function, not a
    /// multiplication. Nothing past the source is called `amol`.
    ///
    /// `true_log2fc` is computed from **final** amounts against the reference condition, so a
    /// large spike-in's compositional dilution of the background is *recorded* rather than
    /// pretended away.
    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Quantity,
            vec![
                Field::new(PROTEIN_ID, DataType::Utf8, false),
                Field::new(SAMPLE_ID, DataType::Utf8, false),
                Field::new(AMOUNT_AMOL, DataType::Float64, false),
                Field::new(TRUE_LOG2FC, DataType::Float32, true),
                Field::new(IS_REGULATED, DataType::Boolean, false),
            ],
        )
    }
}

pub mod peptide_quantities {
    use super::*;
    pub const TABLE: &str = "peptide_quantities";
    pub const PEPTIDE_ID: &str = "peptide_id";
    pub const SAMPLE_ID: &str = "sample_id";
    pub const AMOUNT_AMOL: &str = "amount_amol";

    pub fn spec() -> TableSpec {
        super::spec(
            TABLE,
            Axis::Quantity,
            vec![
                Field::new(PEPTIDE_ID, DataType::UInt64, false),
                Field::new(SAMPLE_ID, DataType::Utf8, false),
                Field::new(AMOUNT_AMOL, DataType::Float64, false),
            ],
        )
    }
}

/// Convenience for constructing record batches: the fields of a table, in order.
pub fn fields(name: &str) -> Option<Fields> {
    by_name(name).map(|t| t.schema.fields().clone())
}
