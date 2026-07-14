//! The analytic digest, split along the structure/quantity boundary.
//!
//! A search engine *enumerates candidates*: it must bound a search space, so it takes a
//! `max_missed_cleavages` budget and emits peptides with no quantities attached. A sample
//! *contains molecules*. We enumerate the same peptides but attach the **expected molar
//! yield** of each, which is what makes the output a sample rather than a search space.
//!
//! # Two stages, not one
//!
//! [`Enumerator`] answers *which peptides exist* — a **structure** question. It depends on
//! the enzyme and on where we truncate, and on nothing else. Its output is shared by every
//! sample in a design.
//!
//! [`YieldModel`] answers *how much of each* — a **quantity** question. It depends on the
//! digestion efficiency and on the occupancy of cleavage-blocking modifications, and the
//! latter is **condition-dependent**: acetylation and ubiquitination are regulated, so an
//! acetylome A/B experiment has *different cleavage probabilities in the two conditions*.
//!
//! Keeping `p_yield` on the structure table would therefore leak a per-condition quantity
//! into a shared artifact — the same class of bug as pruning structure by detectability.
//! The peptide *set* is shared (both conditions produce both the cut and the uncut form;
//! only the proportions differ); only the numbers move.
//!
//! # Model
//!
//! Each cleavage site fires independently with probability
//!
//! ```text
//!     p_eff(k) = p_cleavage · (1 − blocking_occupancy(k))
//! ```
//!
//! and for a peptide **occurrence** spanning boundaries `i..j`:
//!
//! ```text
//!     p_yield = b_i · b_j · Π_{k internal to (i,j)} (1 − p_eff(k))
//!
//!     where b_x = 1         if x is a protein terminus   ← NOT a cleavage event
//!                 p_eff(x)  otherwise
//! ```
//!
//! An **exact expectation**, not a sample: deterministic, seed-free, identical regardless
//! of thread count.
//!
//! # Mass balance — the invariant that makes this self-checking
//!
//! Every realisation of the digest *partitions* the protein, so every residue lands in
//! exactly one peptide with probability 1. Over the **complete** (untruncated, unfiltered)
//! enumeration:
//!
//! ```text
//!     Σ  p_yield(o) · len(o)  =  L        exactly, for a protein of length L
//! ```
//!
//! This is an independent property of the model, not a restatement of the code, so it is a
//! genuine oracle. It also yields the truncation error **by measurement rather than by
//! formula** — whatever residue mass the enumeration fails to account for is exactly what
//! the `max_missed_cleavages` bound discarded.
//!
//! # What this is not
//!
//! An idealised independent-site model. It ignores site-specific kinetics,
//! protease:substrate ratio and digestion time, enzyme autolysis, sequential cleavage of
//! intermediates, missed-site correlation, structural accessibility and denaturation state,
//! sequence context beyond the proline rule, and competition in double digests. A **useful
//! baseline** — never "correct molar amounts".

use crate::enzyme::Protocol;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Structure
// ─────────────────────────────────────────────────────────────────────────────

/// One occurrence of a peptide within one protein.
///
/// An *occurrence*, not a peptide: the same sequence may occur more than once in a single
/// protein, and in more than one protein. Amounts are summed over occurrences, and
/// coordinates are in **protein-residue space** — which is what site-localisation ground
/// truth downstream requires.
///
/// Carries no yield. Yield is a quantity and depends on the condition; see [`YieldModel`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Occurrence {
    pub protein_id: Arc<str>,
    /// 0-based, inclusive.
    pub start: u32,
    /// 0-based, exclusive.
    pub end: u32,
    pub n_missed_cleavages: u16,
}

impl Occurrence {
    #[inline]
    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }

    pub fn sequence<'a>(&self, protein_sequence: &'a str) -> &'a str {
        &protein_sequence[self.start as usize..self.end as usize]
    }
}

/// Structural truncation bounds. Recorded on the artifact so it is self-describing: the
/// yield stage needs them to measure what the enumeration discarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    /// `u32` deliberately: a `u8` bound cannot express "untruncated" safely, because
    /// `j - i - 1` for a protein with >255 internal sites wraps when cast to `u8` and the
    /// bound silently stops being enforced. Real proteins exceed 255 tryptic sites.
    pub max_missed_cleavages: u32,
    pub min_len: u32,
    pub max_len: u32,
}

/// The structural digest of one protein: where it can be cut, and what that yields.
///
/// Contains no probabilities. Shared across every sample in a design.
#[derive(Clone, Debug, PartialEq)]
pub struct ProteinDigest {
    pub protein_id: Arc<str>,
    pub length: u32,
    /// Internal cleavage-site positions. Protein termini are **not** included — they are
    /// boundaries, not cleavage events.
    pub cleavage_sites: Vec<u32>,
    /// Occurrences surviving the length filter.
    pub occurrences: Vec<Occurrence>,
    pub bounds: Bounds,
}

impl ProteinDigest {
    /// Boundaries: N-terminus, every internal cleavage site, C-terminus.
    fn boundaries(&self) -> Vec<u32> {
        let mut b = Vec::with_capacity(self.cleavage_sites.len() + 2);
        b.push(0);
        b.extend_from_slice(&self.cleavage_sites);
        b.push(self.length);
        b
    }

    /// The boundary lattice must be strictly increasing.
    ///
    /// This exists because a `ProteinDigest` can be *reassembled from artifacts* (that is the
    /// whole point of the structure/quantity split), and a reassembly can be wrong. It was:
    /// `timsim-yield` inferred `length` as `max(end)` over the occurrences, but the length
    /// filter discards C-terminal peptides — so the inferred length fell *below* some cleavage
    /// sites, the lattice stopped being monotonic, and `end - start` underflowed with a bare
    /// `attempt to subtract with overflow`.
    ///
    /// A panic with no message, three stages from the mistake, is exactly the failure this
    /// redesign exists to abolish. So: check, and say what is wrong.
    pub fn validate(&self) -> Result<(), String> {
        let mut prev = 0u32;
        for (i, &s) in self.cleavage_sites.iter().enumerate() {
            if s == 0 || s >= self.length {
                return Err(format!(
                    "protein {}: cleavage site {s} is not internal to a protein of length {} \
                     (site {i} of {}). Termini are boundaries, not cleavage events — and a site \
                     beyond the C-terminus means `length` is wrong, most likely inferred rather \
                     than read.",
                    self.protein_id,
                    self.length,
                    self.cleavage_sites.len()
                ));
            }
            if i > 0 && s <= prev {
                return Err(format!(
                    "protein {}: cleavage sites are not strictly increasing ({prev} then {s})",
                    self.protein_id
                ));
            }
            prev = s;
        }
        // Every occurrence must land on the boundary lattice — not merely inside it.
        //
        // `YieldModel` walks boundary PAIRS. An occurrence with coordinates that are in range but
        // are not boundaries is never visited, so it silently receives no yield at all and the
        // peptide vanishes from the amounts with no error. In-range is not the same as valid.
        let boundaries = self.boundaries();
        let index: HashMap<u32, usize> = boundaries
            .iter()
            .enumerate()
            .map(|(i, &b)| (b, i))
            .collect();

        for o in &self.occurrences {
            if o.end > self.length || o.start >= o.end {
                return Err(format!(
                    "protein {}: occurrence {}..{} is not within [0, {})",
                    self.protein_id, o.start, o.end, self.length
                ));
            }
            let (i, j) = match (index.get(&o.start), index.get(&o.end)) {
                (Some(&i), Some(&j)) => (i, j),
                _ => {
                    return Err(format!(
                        "protein {}: occurrence {}..{} does not lie on the cleavage-boundary \
                         lattice — it is in range but its ends are not cleavage sites or termini, \
                         so it could never receive a yield",
                        self.protein_id, o.start, o.end
                    ))
                }
            };
            let expected = (j - i - 1) as u16;
            if o.n_missed_cleavages != expected {
                return Err(format!(
                    "protein {}: occurrence {}..{} spans {expected} internal sites but is \
                     labelled with {} missed cleavages",
                    self.protein_id, o.start, o.end, o.n_missed_cleavages
                ));
            }
        }
        Ok(())
    }
}

/// Enumerates which peptides exist. Structure only — knows nothing about efficiency,
/// modifications, samples, or conditions.
pub struct Enumerator {
    protocol: Protocol,
    bounds: Bounds,
}

impl Enumerator {
    pub fn new(
        protocol: Protocol,
        max_missed_cleavages: u32,
        min_len: u32,
        max_len: u32,
    ) -> Result<Self, String> {
        if min_len > max_len {
            return Err(format!("min_len ({min_len}) exceeds max_len ({max_len})"));
        }
        // THE bound that makes the narrow types downstream provably safe.
        //
        // `peptides.length` and `n_missed_cleavages` are `u16` in the schema. Rather than widen
        // storage in three places, bound the ONE input that makes all three safe: a "peptide"
        // longer than 65,535 residues is not a peptide, it is a protein. And with
        // `max_len <= u16::MAX`, a peptide spans at most 65,535 residues, hence at most 65,534
        // internal cleavage sites — so `n_missed_cleavages` cannot overflow either.
        //
        // This is the fourth bug of the "narrowed a type somewhere other than where the bound is
        // enforced" family (after the u8 missed-cleavage counter and the u16 modform position).
        // Bound once; do not saturate at N call sites.
        if max_len > u16::MAX as u32 {
            return Err(format!(
                "max_len ({max_len}) exceeds {} — a peptide longer than that is a protein, and \
                 the peptide length and missed-cleavage columns are u16",
                u16::MAX
            ));
        }
        Ok(Enumerator {
            protocol,
            bounds: Bounds {
                max_missed_cleavages,
                min_len,
                max_len,
            },
        })
    }

    pub fn enumerate(&self, protein_id: &str, sequence: &str) -> ProteinDigest {
        let protein_id: Arc<str> = Arc::from(protein_id);
        let length = sequence.len() as u32;
        let cleavage_sites: Vec<u32> = self
            .protocol
            .cleavage_positions(sequence)
            .into_iter()
            .map(|p| p as u32)
            .collect();

        let mut digest = ProteinDigest {
            protein_id: protein_id.clone(),
            length,
            cleavage_sites,
            occurrences: Vec::new(),
            bounds: self.bounds,
        };
        if length == 0 {
            return digest;
        }

        let boundaries = digest.boundaries();
        let last = boundaries.len() - 1;

        for i in 0..last {
            for j in (i + 1)..=last {
                // Compare in usize. Casting to a narrow int BEFORE the bound check lets a
                // protein with many sites wrap the counter and escape the bound entirely.
                let n_missed = j - i - 1;
                if n_missed > self.bounds.max_missed_cleavages as usize {
                    break;
                }
                let (start, end) = (boundaries[i], boundaries[j]);
                let plen = end - start;
                if plen >= self.bounds.min_len && plen <= self.bounds.max_len {
                    digest.occurrences.push(Occurrence {
                        protein_id: protein_id.clone(),
                        start,
                        end,
                        // Safe by construction: max_len <= u16::MAX bounds the span, hence the
                        // number of internal sites. Saturating here would silently mislabel.
                        n_missed_cleavages: u16::try_from(n_missed)
                            .expect("bounded by max_len <= u16::MAX; see Enumerator::new"),
                    });
                }
            }
        }
        digest
    }

    /// Enumerate many proteins in parallel. Deterministic: proteins are independent and
    /// results are collected in input order, so output is byte-identical regardless of
    /// thread count.
    pub fn enumerate_all(&self, proteins: &[(String, String)]) -> Vec<ProteinDigest> {
        proteins
            .par_iter()
            .map(|(id, seq)| self.enumerate(id, seq))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantity
// ─────────────────────────────────────────────────────────────────────────────

/// Occupancy of cleavage-**blocking** modifications, by protein-residue position.
///
/// Acetyl-K, the ubiquitin GG remnant, trimethyl-K and TMT-K all block trypsin. These are
/// states of the protein *before* the protease sees it, so they live in protein-residue
/// coordinates and must be applied here — not as a post-hoc decoration of the peptides
/// that result.
///
/// Note what this is **not**: a "variable modification". A search engine has variable mods
/// with a `max_variable_mods` budget because it is enumerating candidates. A sample has
/// *site occupancies*. Only mods that alter cleavage belong in the digest at all;
/// everything else (phospho, Met oxidation, pyroglutamate) is applied downstream, where it
/// cannot retroactively change which bonds were cut.
///
/// There are also no "fixed" and "variable" mods — only occupancy. Carbamidomethyl-C at
/// `0.98` *is* the alkylation efficiency, which is why free-cysteine peptides show up in
/// real data. "Fixed" is simply occupancy ≈ 1.
///
/// # Why this is condition-dependent
///
/// A missed cleavage at site *k* has two causes: the enzyme failed (probability
/// `(1−occ)(1−p)`), or the site was modified and could not be cut (probability `occ`).
/// Both give the same peptide *sequence* but a different *modform* — and that coupling is
/// the signature diGly and acetylome experiments rely on: the missed cleavage at the
/// modified lysine is *how you find it*. Since acetylation is regulated, `occ` differs
/// between conditions, and so therefore does the yield.
#[derive(Clone, Debug, Default)]
pub struct BlockingOccupancy {
    by_position: HashMap<(Arc<str>, u32), f64>,
}

impl BlockingOccupancy {
    /// No blocking modifications — an unmodified proteome.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn from_sites(
        sites: impl IntoIterator<Item = (Arc<str>, u32, f64)>,
    ) -> Result<Self, String> {
        let mut by_position = HashMap::new();
        for (protein, pos, occ) in sites {
            if !(0.0..=1.0).contains(&occ) {
                return Err(format!(
                    "blocking occupancy at {protein}:{pos} must be in [0, 1], got {occ}"
                ));
            }
            by_position.insert((protein, pos), occ);
        }
        Ok(BlockingOccupancy { by_position })
    }

    #[inline]
    pub fn at(&self, protein: &Arc<str>, position: u32) -> f64 {
        self.by_position
            .get(&(protein.clone(), position))
            .copied()
            .unwrap_or(0.0)
    }

    pub fn is_empty(&self) -> bool {
        self.by_position.is_empty()
    }
}

/// Accounting for what the enumeration accounted for, and what it dropped.
///
/// The rule is "measure it, don't quote a formula": the geometric approximation
/// `(1-p)^(n+1)` holds only for an infinite interior-site model and breaks on finite
/// proteins, terminal peptides, and length filters. We enumerate anyway, so we measure.
///
/// Truncation loss and filter loss are reported **separately** because they are different
/// kinds of loss — one is a numerical bound, the other a modelling choice — and collapsing
/// them would hide both.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DigestStats {
    pub residues_total: f64,
    /// Residues accounted for within the missed-cleavage bound, before length filtering.
    pub residues_enumerated: f64,
    /// Residues surviving the length filter.
    pub residues_retained: f64,
    /// Molar yield by missed-cleavage count, index = n_missed. Amount-weighted.
    pub yield_by_missed_cleavages: Vec<f64>,
}

impl DigestStats {
    /// Fraction of residue mass discarded by the `max_missed_cleavages` bound — the
    /// numerical truncation error, measured.
    pub fn truncation_loss(&self) -> f64 {
        if self.residues_total == 0.0 {
            return 0.0;
        }
        1.0 - self.residues_enumerated / self.residues_total
    }

    /// Fraction of residue mass discarded by the length filter — a modelling choice.
    pub fn filter_loss(&self) -> f64 {
        if self.residues_enumerated == 0.0 {
            return 0.0;
        }
        1.0 - self.residues_retained / self.residues_enumerated
    }

    pub fn missed_cleavage_distribution(&self) -> Vec<f64> {
        let total: f64 = self.yield_by_missed_cleavages.iter().sum();
        if total == 0.0 {
            return vec![];
        }
        self.yield_by_missed_cleavages
            .iter()
            .map(|y| y / total)
            .collect()
    }

    fn merge(&mut self, other: &DigestStats) {
        self.residues_total += other.residues_total;
        self.residues_enumerated += other.residues_enumerated;
        self.residues_retained += other.residues_retained;
        if self.yield_by_missed_cleavages.len() < other.yield_by_missed_cleavages.len() {
            self.yield_by_missed_cleavages
                .resize(other.yield_by_missed_cleavages.len(), 0.0);
        }
        for (slot, v) in self
            .yield_by_missed_cleavages
            .iter_mut()
            .zip(&other.yield_by_missed_cleavages)
        {
            *slot += v;
        }
    }
}

/// Turns a structural digest into molar yields, for **one condition**.
///
/// Cheap: pure arithmetic over the occurrence table. A 20-sample design applies N of these
/// to one shared [`ProteinDigest`], which is what makes multi-sample nearly free.
pub struct YieldModel {
    cleavage_p: f64,
    blocking: BlockingOccupancy,
}

impl YieldModel {
    pub fn new(cleavage_p: f64, blocking: BlockingOccupancy) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&cleavage_p) {
            return Err(format!(
                "digestion efficiency must be in [0, 1], got {cleavage_p}"
            ));
        }
        Ok(YieldModel {
            cleavage_p,
            blocking,
        })
    }

    /// Effective firing probability of a cleavage site: the enzyme must cut it *and* it
    /// must not be blocked by a modification.
    #[inline]
    fn p_eff(&self, protein: &Arc<str>, position: u32) -> f64 {
        self.cleavage_p * (1.0 - self.blocking.at(protein, position))
    }

    /// Yields for each occurrence in `digest`, aligned with `digest.occurrences`, plus the
    /// measured accounting.
    ///
    /// Panics if `digest` is internally inconsistent — see [`ProteinDigest::validate`]. Use
    /// [`YieldModel::try_apply`] to handle that as an error.
    pub fn apply(&self, digest: &ProteinDigest) -> (Vec<f64>, DigestStats) {
        self.try_apply(digest)
            .unwrap_or_else(|e| panic!("inconsistent ProteinDigest: {e}"))
    }

    /// As [`YieldModel::apply`], but returns the structural inconsistency rather than panicking.
    pub fn try_apply(&self, digest: &ProteinDigest) -> Result<(Vec<f64>, DigestStats), String> {
        digest.validate()?;
        Ok(self.apply_unchecked(digest))
    }

    fn apply_unchecked(&self, digest: &ProteinDigest) -> (Vec<f64>, DigestStats) {
        // The histogram grows on demand. Sizing it from `max_missed_cleavages` would try to
        // allocate 4 billion buckets in untruncated mode.
        let mut stats = DigestStats {
            residues_total: digest.length as f64,
            ..Default::default()
        };
        if digest.length == 0 {
            return (Vec::new(), stats);
        }

        let boundaries = digest.boundaries();
        let last = boundaries.len() - 1;

        // Index every enumerated (start, end) so we can align yields with the occurrence
        // list, while still walking the *complete* boundary lattice for the accounting —
        // the truncation loss is only meaningful against the unfiltered enumeration.
        let mut index: HashMap<(u32, u32), usize> = HashMap::with_capacity(digest.occurrences.len());
        for (k, o) in digest.occurrences.iter().enumerate() {
            index.insert((o.start, o.end), k);
        }
        let mut yields = vec![0.0; digest.occurrences.len()];

        for i in 0..last {
            let b_left = if i == 0 {
                1.0
            } else {
                self.p_eff(&digest.protein_id, boundaries[i])
            };
            let mut internal_survival = 1.0f64;

            for j in (i + 1)..=last {
                let n_missed = j - i - 1;
                if n_missed > digest.bounds.max_missed_cleavages as usize {
                    break;
                }
                let b_right = if j == last {
                    1.0
                } else {
                    self.p_eff(&digest.protein_id, boundaries[j])
                };
                let p_yield = b_left * b_right * internal_survival;

                let (start, end) = (boundaries[i], boundaries[j]);
                let plen = (end - start) as f64;

                stats.residues_enumerated += p_yield * plen;
                if stats.yield_by_missed_cleavages.len() <= n_missed {
                    stats.yield_by_missed_cleavages.resize(n_missed + 1, 0.0);
                }
                stats.yield_by_missed_cleavages[n_missed] += p_yield;

                if let Some(&k) = index.get(&(start, end)) {
                    stats.residues_retained += p_yield * plen;
                    yields[k] = p_yield;
                }

                internal_survival *= 1.0 - self.p_eff(&digest.protein_id, boundaries[j]);
            }
        }

        (yields, stats)
    }

    /// Apply to many proteins in parallel. Deterministic.
    pub fn apply_all(&self, digests: &[ProteinDigest]) -> (Vec<Vec<f64>>, DigestStats) {
        let per_protein: Vec<(Vec<f64>, DigestStats)> =
            digests.par_iter().map(|d| self.apply(d)).collect();

        let mut stats = DigestStats::default();
        let mut yields = Vec::with_capacity(per_protein.len());
        for (y, s) in per_protein {
            yields.push(y);
            stats.merge(&s);
        }
        (yields, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn enumerator(max_missed: u32) -> Enumerator {
        // min_len 0 / max_len at the representable maximum: the mass-balance invariant holds only
        // over the *complete* enumeration, so tests of it must not filter.
        Enumerator::new(Protocol::parse("trypsin").unwrap(), max_missed, 0, u16::MAX as u32).unwrap()
    }

    fn plain(p: f64) -> YieldModel {
        YieldModel::new(p, BlockingOccupancy::none()).unwrap()
    }

    fn block(protein: &str, sites: &[(u32, f64)]) -> BlockingOccupancy {
        let id: Arc<str> = Arc::from(protein);
        BlockingOccupancy::from_sites(sites.iter().map(|&(p, o)| (id.clone(), p, o))).unwrap()
    }

    /// The core invariant. Every realisation of the digest partitions the protein, so every
    /// residue lands in exactly one peptide with probability 1. Untruncated and unfiltered,
    /// expected residue coverage must equal the protein length exactly.
    ///
    /// An independent property of the model, not a restatement of the implementation — so
    /// it is a genuine oracle, and it is what lets us *measure* truncation instead of
    /// quoting a formula for it.
    #[test]
    fn mass_balance_holds_exactly_when_untruncated() {
        let seq = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVK";
        let d = enumerator(u32::MAX).enumerate("P1", seq);
        for p in [0.0, 0.1, 0.5, 0.9, 0.99, 1.0] {
            let (_, stats) = plain(p).apply(&d);
            assert_relative_eq!(
                stats.residues_enumerated,
                seq.len() as f64,
                epsilon = 1e-9,
                max_relative = 1e-12
            );
            assert_relative_eq!(stats.truncation_loss(), 0.0, epsilon = 1e-12);
        }
    }

    /// The invariant must survive blocking untouched. Blocking changes *which* bonds are
    /// cut, but every realisation still partitions the protein.
    #[test]
    fn mass_balance_survives_blocking_modifications() {
        let seq = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVK";
        let d = enumerator(u32::MAX).enumerate("P1", seq);
        let sites: Vec<(u32, f64)> = d.cleavage_sites.iter().map(|&s| (s, 0.35)).collect();
        let blocking = block("P1", &sites);

        for p in [0.0, 0.5, 0.9, 1.0] {
            let (_, stats) = YieldModel::new(p, blocking.clone()).unwrap().apply(&d);
            assert_relative_eq!(
                stats.residues_enumerated,
                seq.len() as f64,
                epsilon = 1e-9,
                max_relative = 1e-12
            );
        }
    }

    /// Truncation is real, measurable, and shrinks as the bound loosens.
    #[test]
    fn truncation_loss_is_measured_and_decreases_with_a_looser_bound() {
        let seq = "AAAK".repeat(8);
        let mut prev = f64::INFINITY;
        for max_missed in [0u32, 1, 2, 3, 5] {
            let d = enumerator(max_missed).enumerate("P1", &seq);
            let (_, stats) = plain(0.9).apply(&d);
            let loss = stats.truncation_loss();
            assert!(loss >= -1e-12, "loss must be non-negative, got {loss}");
            assert!(loss < prev, "loss must shrink as the bound loosens");
            prev = loss;
        }
    }

    /// THE structure/quantity split: one enumeration, many conditions. The peptide *set* is
    /// identical across conditions — only the yields differ. This is what makes a 20-sample
    /// A/B design nearly free, and it is the reason `p_yield` cannot live on the structure
    /// table.
    #[test]
    fn one_structure_serves_many_conditions() {
        let seq = "AAAKAAAKCCCK";
        let d = enumerator(2).enumerate("P1", seq);

        // Condition A: lysine 4 lightly acetylated. Condition B: heavily.
        let (ya, _) = YieldModel::new(0.9, block("P1", &[(4, 0.01)])).unwrap().apply(&d);
        let (yb, _) = YieldModel::new(0.9, block("P1", &[(4, 0.50)])).unwrap().apply(&d);

        assert_eq!(ya.len(), yb.len(), "same structure, same occurrences");
        assert_ne!(ya, yb, "different condition ⇒ different yields");

        // The peptide spanning the blocked site is far more abundant in B.
        let joined = d
            .occurrences
            .iter()
            .position(|o| o.sequence(seq) == "AAAKAAAK")
            .unwrap();
        assert!(
            yb[joined] > ya[joined] * 5.0,
            "regulated acetylation must raise the missed-cleavage form"
        );
    }

    /// A fully-occupied blocking modification abolishes the site: the protease cannot cut
    /// there, so the flanking peptides are joined and it reads as a missed cleavage — which
    /// is exactly how diGly and acetylome experiments identify the modified lysine.
    #[test]
    fn full_blocking_occupancy_abolishes_a_cleavage_site() {
        let seq = "AAAKAAAKCCC";
        let d = enumerator(3).enumerate("P1", seq);

        // Perfect enzyme: any missed cleavage must come from blocking, not from failure.
        let (y, _) = plain(1.0).apply(&d);
        let live: Vec<_> = d.occurrences.iter().zip(&y).filter(|(_, &v)| v > 0.0).collect();
        assert_eq!(live.len(), 3, "AAAK | AAAK | CCC");

        let (y, _) = YieldModel::new(1.0, block("P1", &[(4, 1.0)])).unwrap().apply(&d);
        let live: Vec<_> = d.occurrences.iter().zip(&y).filter(|(_, &v)| v > 0.0).collect();
        assert_eq!(live.len(), 2, "AAAKAAAK | CCC — the blocked site is not cut");
        assert_eq!(live[0].0.sequence(seq), "AAAKAAAK");
        assert_eq!(live[0].0.n_missed_cleavages, 1);
        assert_relative_eq!(*live[0].1, 1.0, epsilon = 1e-12);
    }

    /// Partial occupancy splits the yield by exactly the occupancy: the chemist's number
    /// appears directly in the output.
    #[test]
    fn partial_blocking_occupancy_splits_yield_by_exactly_the_occupancy() {
        let seq = "AAAKAAAKCCC";
        let d = enumerator(3).enumerate("P1", seq);
        let occ = 0.3;
        let (y, _) = YieldModel::new(1.0, block("P1", &[(4, occ)])).unwrap().apply(&d);

        let at = |s: &str, start: u32| {
            let k = d
                .occurrences
                .iter()
                .position(|o| o.sequence(seq) == s && o.start == start)
                .unwrap();
            y[k]
        };
        assert_relative_eq!(at("AAAKAAAK", 0), occ, epsilon = 1e-12);
        assert_relative_eq!(at("AAAK", 0), 1.0 - occ, epsilon = 1e-12);
    }

    /// Blocking can only reduce cleavage, so it can only increase the missed-cleavage
    /// fraction.
    #[test]
    fn blocking_increases_missed_cleavages() {
        let seq = "AAAK".repeat(40);
        let d = enumerator(3).enumerate("P1", &seq);
        let sites: Vec<(u32, f64)> = d.cleavage_sites.iter().map(|&s| (s, 0.5)).collect();

        let (_, base) = plain(0.9).apply(&d);
        let (_, blocked) = YieldModel::new(0.9, block("P1", &sites)).unwrap().apply(&d);

        assert!(
            blocked.missed_cleavage_distribution()[0] < base.missed_cleavage_distribution()[0],
            "blocking must reduce the fully-cleaved fraction"
        );
    }

    #[test]
    fn perfect_digestion_yields_only_full_cleavage_products() {
        let d = enumerator(3).enumerate("P1", "AAAKAAARCCC");
        let (y, stats) = plain(1.0).apply(&d);

        let live: Vec<_> = d.occurrences.iter().zip(&y).filter(|(_, &v)| v > 0.0).collect();
        assert_eq!(live.len(), 3);
        for (o, &v) in &live {
            assert_eq!(o.n_missed_cleavages, 0);
            assert_relative_eq!(v, 1.0, epsilon = 1e-12);
        }
        assert_relative_eq!(stats.missed_cleavage_distribution()[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn zero_efficiency_yields_the_intact_protein() {
        let seq = "AAAKAAARCCC";
        let d = enumerator(u32::MAX).enumerate("P1", seq);
        let (y, _) = plain(0.0).apply(&d);

        let live: Vec<_> = d.occurrences.iter().zip(&y).filter(|(_, &v)| v > 0.0).collect();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].0.start, 0);
        assert_eq!(live[0].0.end, seq.len() as u32);
        assert_relative_eq!(*live[0].1, 1.0, epsilon = 1e-12);
    }

    /// Protein termini are not cleavage events. For a protein ending in K, the C-terminal
    /// peptide must have boundary probability 1 on its right, not `p`. If termini were
    /// treated as cleavage sites every terminal peptide would be under-counted by a factor
    /// of `p` — and mass balance would fail, which is how we catch it.
    #[test]
    fn protein_termini_have_boundary_probability_one() {
        let seq = "AAAKAAAK";
        let p = 0.8;
        let d = enumerator(0).enumerate("P1", seq);
        let (y, _) = plain(p).apply(&d);

        assert_eq!(d.occurrences.len(), 2);
        // N-term(1.0)×site(p) = p,  and  site(p)×C-term(1.0) = p.
        assert_relative_eq!(y[0], p, epsilon = 1e-12);
        assert_relative_eq!(y[1], p, epsilon = 1e-12);
    }

    /// A peptide occurring twice in one protein must appear twice. A protein→peptide list
    /// column cannot express this; an occurrence table can.
    #[test]
    fn repeated_sequence_produces_two_occurrences() {
        let seq = "AAAKAAAKCCCK";
        let d = enumerator(0).enumerate("P1", seq);
        let repeats: Vec<_> = d
            .occurrences
            .iter()
            .filter(|o| o.sequence(seq) == "AAAK")
            .collect();
        assert_eq!(repeats.len(), 2);
        assert_eq!(repeats[0].start, 0);
        assert_eq!(repeats[1].start, 4);
    }

    #[test]
    fn missed_cleavage_distribution_follows_from_efficiency() {
        let seq = "AAAK".repeat(40);
        let d = enumerator(3).enumerate("P1", &seq);
        let (_, stats) = plain(0.9).apply(&d);
        let dist = stats.missed_cleavage_distribution();

        assert!(dist[0] > dist[1] && dist[1] > dist[2] && dist[2] > dist[3]);
        assert!(dist[0] > 0.85, "p=0.9 ⇒ mostly fully cleaved, got {}", dist[0]);
    }

    /// Determinism: output must not depend on thread count.
    #[test]
    fn parallel_digest_is_order_independent() {
        let proteins: Vec<(String, String)> = (0..64)
            .map(|i| (format!("P{i}"), "AAAKCCCKDDDR".repeat(i % 5 + 1)))
            .collect();
        let e = Enumerator::new(Protocol::parse("trypsin").unwrap(), 2, 0, u16::MAX as u32).unwrap();

        let (d1, d2) = (e.enumerate_all(&proteins), e.enumerate_all(&proteins));
        assert_eq!(d1, d2);

        let (y1, s1) = plain(0.85).apply_all(&d1);
        let (y2, s2) = plain(0.85).apply_all(&d2);
        assert_eq!(y1, y2);
        assert_eq!(s1, s2);
    }

    /// MONTE CARLO ORACLE for `p_yield`.
    ///
    /// The analytic formula is only worth what an *independent* implementation says it is.
    /// So: actually digest. Flip a coin at every cleavage site, build the partition that
    /// results, and count the peptides that come out. Average over many trials and the
    /// empirical yield of each occurrence must converge on the analytic `p_yield`.
    ///
    /// This shares no code with `YieldModel` — no boundary lattice, no survival product, no
    /// closed form. It is a genuine oracle rather than a restatement of the implementation,
    /// which is the whole point (see PLAN §6.1, the anti-self-consistency rule).
    #[test]
    fn monte_carlo_agrees_with_the_analytic_yield() {
        // Deterministic LCG — this test must not be flaky.
        struct Lcg(u64);
        impl Lcg {
            fn next_f64(&mut self) -> f64 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (self.0 >> 11) as f64 / (1u64 << 53) as f64
            }
        }

        let seq = "AAAKCCCKDDDRPEPTIDEKAAAKGGGRMWSSKLLTRNNPKQQVR";
        let p = 0.75;
        // Non-uniform blocking, so the test exercises per-site probabilities rather than a
        // single global p — a uniform p could hide an indexing error.
        let e = enumerator(u32::MAX);
        let d = e.enumerate("P1", seq);
        let blocked: Vec<(u32, f64)> = d
            .cleavage_sites
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, if i % 2 == 0 { 0.4 } else { 0.0 }))
            .collect();
        let ym = YieldModel::new(p, block("P1", &blocked)).unwrap();
        let (analytic, _) = ym.apply(&d);

        // ── the independent simulation ──
        let n_trials = 400_000;
        let mut counts: HashMap<(u32, u32), u64> = HashMap::new();
        let mut rng = Lcg(0x5DEECE66D);

        for _ in 0..n_trials {
            // Flip each site: it fires iff the enzyme cuts AND it is not blocked.
            let mut cuts: Vec<u32> = vec![0];
            for (i, &site) in d.cleavage_sites.iter().enumerate() {
                let occ = if i % 2 == 0 { 0.4 } else { 0.0 };
                let fires = rng.next_f64() < p && rng.next_f64() >= occ;
                if fires {
                    cuts.push(site);
                }
            }
            cuts.push(d.length);
            // The realised digest is the partition between consecutive cuts.
            for w in cuts.windows(2) {
                *counts.entry((w[0], w[1])).or_insert(0) += 1;
            }
        }

        let mut compared = 0;
        for (o, &expected) in d.occurrences.iter().zip(&analytic) {
            let observed = counts.get(&(o.start, o.end)).copied().unwrap_or(0) as f64
                / n_trials as f64;
            // 3-sigma binomial tolerance, floored so tiny-probability peptides don't
            // dominate the assertion.
            let tol = 3.0 * (expected * (1.0 - expected) / n_trials as f64).sqrt() + 2e-3;
            assert!(
                (observed - expected).abs() < tol,
                "occurrence {}..{}: analytic {expected:.5}, simulated {observed:.5} (tol {tol:.5})",
                o.start,
                o.end
            );
            compared += 1;
        }
        assert!(compared > 20, "test must actually compare something; compared {compared}");

        // And the simulation independently reproduces mass balance: every trial partitions
        // the protein, so mean residue coverage is exactly L.
        let mean_coverage: f64 = counts
            .iter()
            .map(|(&(s, e), &c)| (e - s) as f64 * c as f64)
            .sum::<f64>()
            / n_trials as f64;
        assert_relative_eq!(mean_coverage, seq.len() as f64, max_relative = 1e-12);
    }

    /// REGRESSION: the missed-cleavage counter must not wrap.
    ///
    /// `n_missed` was computed as `(j - i - 1) as u8` and compared against a `u8` bound. On a
    /// protein with ≥257 internal cleavage sites, `j - i - 1` reaches 256, **wraps to 0**,
    /// and the bound stops being enforced entirely: the enumeration goes quadratic and every
    /// long span is tagged with a wrong `n_missed_cleavages`, which `YieldModel` then
    /// mis-bins. Real proteins are easily this large — anything over ~2,500 residues has 250+
    /// tryptic sites, and titin has tens of thousands.
    ///
    /// The unit tests never caught it because they use short proteins, and the proteome run
    /// used `max_missed = 2`. Found by review, not by the suite.
    #[test]
    fn missed_cleavage_counter_does_not_wrap_on_large_proteins() {
        // 300 tryptic sites: "AAAK" × 301 → sites after every K except the C-terminal one.
        let seq = "AAAK".repeat(301);
        let d = enumerator(2).enumerate("BIG", &seq);
        assert!(
            d.cleavage_sites.len() > 256,
            "test needs >256 sites to exercise the wrap; got {}",
            d.cleavage_sites.len()
        );

        // A bounded digest is not where the bug lives: with `max_missed = 2` the loop breaks
        // at 3 and never reaches 256. Assert the bound holds anyway, then go where it bites.
        for o in &d.occurrences {
            assert!(o.n_missed_cleavages <= 2, "bound escaped");
        }

        // ── THE ACTUAL BUG: untruncated mode. ──
        //
        // With the old `u8` bound, `n_missed > max_missed` could never fire (a u8 cannot
        // exceed u8::MAX), so nothing broke the loop — and `(j - i - 1) as u8` wrapped.
        // The intact-protein span has 300 internal sites; as a u8 that becomes 300 - 256 = 44.
        let full = enumerator(u32::MAX).enumerate("BIG", &seq);
        let n_sites = full.cleavage_sites.len();

        let intact = full
            .occurrences
            .iter()
            .find(|o| o.start == 0 && o.end == seq.len() as u32)
            .expect("the intact protein must be enumerated when untruncated");

        assert_eq!(
            intact.n_missed_cleavages as usize, n_sites,
            "the intact protein spans every site, so it has exactly {n_sites} missed \
             cleavages — the old u8 counter reported {} (wrapped mod 256)",
            intact.n_missed_cleavages
        );

        // The yield histogram must have a bucket for every attainable count, not 256 of them.
        let (_, stats) = plain(0.9).apply(&full);
        assert_eq!(
            stats.yield_by_missed_cleavages.len(),
            n_sites + 1,
            "histogram must span 0..={n_sites} missed cleavages"
        );

        // And mass balance still holds on a protein this large.
        assert_relative_eq!(
            stats.residues_enumerated,
            seq.len() as f64,
            max_relative = 1e-9
        );
    }

    /// REGRESSION: the length bound that makes the u16 columns safe must be enforced.
    ///
    /// `--max-length` is a u32, while `peptides.length` and `n_missed_cleavages` are u16. Without
    /// this bound, a permissive max-length wraps the peptide length in the artifact and saturates
    /// the missed-cleavage count — the structural table becomes internally inconsistent with its
    /// own sequences. Bounding max_len is what makes both provably safe. Found by review.
    /// REGRESSION: an occurrence that is in range but off the boundary lattice must be refused.
    ///
    /// `YieldModel` walks boundary PAIRS, so an occurrence whose ends are not cleavage sites is
    /// never visited — it silently gets **no yield at all** and the peptide vanishes from the
    /// amounts with no error. "In range" is not "valid". Found by review.
    #[test]
    fn an_occurrence_off_the_boundary_lattice_is_refused() {
        let seq = "AAAKCCCKDDDR";
        let mut d = enumerator(2).enumerate("P1", seq);
        assert!(d.validate().is_ok());

        // In range (0..12), but 2 and 6 are not cleavage sites.
        d.occurrences.push(Occurrence {
            protein_id: Arc::from("P1"),
            start: 2,
            end: 6,
            n_missed_cleavages: 0,
        });
        let err = d.validate().unwrap_err();
        assert!(err.contains("boundary lattice"), "{err}");
    }

    /// REGRESSION: a mislabelled missed-cleavage count must be refused.
    #[test]
    fn a_mislabelled_missed_cleavage_count_is_refused() {
        let seq = "AAAKCCCKDDDR";
        let mut d = enumerator(2).enumerate("P1", seq);
        d.occurrences[0].n_missed_cleavages += 7;
        let err = d.validate().unwrap_err();
        assert!(err.contains("missed cleavages"), "{err}");
    }

    #[test]
    fn an_unrepresentable_length_bound_is_rejected() {
        let t = || Protocol::parse("trypsin").unwrap();
        assert!(Enumerator::new(t(), 2, 7, u16::MAX as u32).is_ok(), "the boundary itself is fine");
        assert!(Enumerator::new(t(), 2, 7, u16::MAX as u32 + 1).is_err());
        assert!(Enumerator::new(t(), 2, 7, u32::MAX).is_err());
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        assert!(YieldModel::new(1.5, BlockingOccupancy::none()).is_err());
        assert!(YieldModel::new(-0.1, BlockingOccupancy::none()).is_err());
        assert!(Enumerator::new(Protocol::parse("trypsin").unwrap(), 2, 50, 6).is_err());
        let id: Arc<str> = Arc::from("P1");
        assert!(BlockingOccupancy::from_sites([(id, 4u32, 1.5)]).is_err());
    }
}
