//! Modforms — the modified forms of a peptide, and how much of each.
//!
//! # There are no "fixed" and "variable" mods. There is only occupancy.
//!
//! A search engine has *variable mods* with a `max_variable_mods` budget, because it is
//! enumerating candidates and must bound a search space. A sample has **site occupancies**.
//!
//! Carbamidomethyl-C is not a "fixed mod" at 100%: alkylation is ~95–99% efficient, which is
//! *why free-cysteine peptides appear in real data*. Set `occupancy = 0.98` and that falls out
//! with no special case. "Fixed" is occupancy ≈ 1; "variable" is occupancy < 1. The chemist's
//! parameter is the **alkylation efficiency**.
//!
//! # Truncate on probability mass, not on a count
//!
//! `max_variable_mods = 3` is the search-engine budget again: arbitrary, and blind to how much
//! material it discards. Instead we truncate on a **minimum abundance floor** and *measure* the
//! omitted mass. A peptide with four S/T/Y at 2% occupancy is ~92% unmodified, ~7.8% singly
//! modified, ~0.24% doubly — a probability floor captures that automatically; a count budget
//! does not.
//!
//! Same discipline as `max_missed_cleavages` in [`crate::digest`]: the chemist sets occupancy,
//! the informatician sets a floor, and the tool reports what it dropped.
//!
//! # The model
//!
//! Each modifiable site independently takes one of its options, or stays unmodified:
//!
//! ```text
//!     P(modform) = Π_sites  p(chosen option at that site)
//! ```
//!
//! Site independence is an assumption, and a real one — priming phosphorylation and ubiquitin
//! chains genuinely correlate. But correlation only survives digestion when the sites land on
//! the **same peptide** (see SPEC §7.4), so this is the right place to add it later, and the
//! wrong place to pretend it away.
//!
//! # Mass balance, again
//!
//! Over the *complete* enumeration, `Σ P(modform) = 1` exactly — every copy of the peptide is
//! in exactly one modform. That identity is the oracle, and the truncation error is whatever it
//! falls short by.

use crate::mass;
use std::collections::HashMap;

/// When a modification happens, relative to the protease. This is the wet-lab order, and it is
/// load-bearing: only pre-digestion protein-level mods can *block cleavage*, and pyroglutamate
/// forms on a **peptide** N-terminus, which does not exist until after digestion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// On the protein, before the protease sees it: phospho, acetyl, ubiquitin-GG,
    /// carbamidomethyl (reduction/alkylation precedes digestion).
    Protein,
    /// On the peptide, after digestion: pyroglutamate, handling artefacts like Met oxidation.
    Peptide,
}

/// Where on the sequence a modification may sit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    /// Any occurrence of the target residues.
    Residue,
    /// The N-terminus (of the peptide, or of the protein, per `stage`).
    NTerm,
    CTerm,
}

#[derive(Clone, Debug)]
pub struct Modification {
    pub name: String,
    pub unimod_id: u32,
    /// Residues this can sit on, e.g. `"STY"` for phospho. Empty means "any residue" for a
    /// terminal modification.
    pub targets: String,
    pub site: Site,
    /// Marginal per-site occupancy in [0, 1]. **This is the chemist's number**: phospho ~0.02,
    /// Met oxidation ~0.05, carbamidomethyl ~0.98 (= alkylation efficiency).
    pub occupancy: f64,
    pub mass_delta: f64,
    /// Does this physically prevent the protease from cutting here? Acetyl-K, ubiquitin-GG,
    /// trimethyl-K and TMT-K all block trypsin. Consumed by [`crate::digest`], **not here** —
    /// blocking alters *which peptides exist*, which is a digest question.
    pub blocks_cleavage: bool,
    pub stage: Stage,
}

impl Modification {
    pub fn validate(&self) -> Result<(), String> {
        if !self.occupancy.is_finite() || !(0.0..=1.0).contains(&self.occupancy) {
            return Err(format!(
                "modification {:?}: occupancy must be finite and in [0, 1], got {}",
                self.name, self.occupancy
            ));
        }
        if !self.mass_delta.is_finite() {
            return Err(format!(
                "modification {:?}: mass_delta must be finite, got {}",
                self.name, self.mass_delta
            ));
        }
        if self.site == Site::Residue && self.targets.is_empty() {
            return Err(format!(
                "modification {:?}: a residue modification needs target residues",
                self.name
            ));
        }
        Ok(())
    }
}

/// One modified form of a peptide.
#[derive(Clone, Debug, PartialEq)]
pub struct Modform {
    /// `(position, modification index)`. Position is **0-based within the peptide**; callers
    /// join through `peptide_occurrences` to get protein-residue coordinates, which is what
    /// site-localisation ground truth actually needs.
    ///
    /// `u32`, not `u16`: peptides are short in practice, but the public API and `--max-length`
    /// accept a `u32` bound, and narrowing here would wrap position 65,536 to 0 — silently
    /// **merging two distinct sites into competing alternatives at one residue**, corrupting
    /// both the coordinates and the probabilities. Never narrow a type anywhere but the place
    /// that enforces the bound.
    pub mods: Vec<(u32, usize)>,
    /// Fraction of this peptide's molecules in this form. Over the complete enumeration these
    /// sum to exactly 1.
    pub abundance_fraction: f64,
    /// Total mass added, in Da.
    pub mass_delta: f64,
}

impl Modform {
    pub fn is_unmodified(&self) -> bool {
        self.mods.is_empty()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModformStats {
    /// Modforms retained above the floor.
    pub retained: usize,
    /// Probability mass retained. `1 - this` is exactly what the floor discarded — **measured,
    /// not estimated**.
    pub mass_retained: f64,
    /// Modifiable sites found on this peptide.
    pub sites: usize,
}

impl ModformStats {
    /// Fraction of the peptide's molecules discarded by the abundance floor.
    pub fn truncation_loss(&self) -> f64 {
        (1.0 - self.mass_retained).max(0.0)
    }
}

/// The options available at one modifiable position: a categorical choice between staying
/// unmodified and each competing modification.
struct SiteOptions {
    position: u32,
    /// `(modification index, probability)`. Mutually exclusive — one residue carries one mod.
    options: Vec<(usize, f64)>,
    /// Probability of staying unmodified.
    p_unmodified: f64,
    /// The largest probability at this site. Used as an admissible bound for pruning.
    p_max: f64,
}

fn sites_for(sequence: &str, mods: &[Modification]) -> Vec<SiteOptions> {
    let bytes = sequence.as_bytes();
    let n = bytes.len();
    // position -> competing options
    let mut by_pos: HashMap<u32, Vec<(usize, f64)>> = HashMap::new();

    for (idx, m) in mods.iter().enumerate() {
        if m.occupancy <= 0.0 {
            continue;
        }
        match m.site {
            Site::NTerm => {
                if n > 0 && (m.targets.is_empty() || m.targets.contains(bytes[0] as char)) {
                    by_pos.entry(0).or_default().push((idx, m.occupancy));
                }
            }
            Site::CTerm => {
                if n > 0
                    && (m.targets.is_empty() || m.targets.contains(bytes[n - 1] as char))
                {
                    by_pos.entry((n - 1) as u32).or_default().push((idx, m.occupancy));
                }
            }
            Site::Residue => {
                for (i, &b) in bytes.iter().enumerate() {
                    if m.targets.contains(b as char) {
                        by_pos.entry(i as u32).or_default().push((idx, m.occupancy));
                    }
                }
            }
        }
    }

    let mut sites: Vec<SiteOptions> = by_pos
        .into_iter()
        .map(|(position, options)| {
            // Competing modifications at one residue are mutually exclusive. If the declared
            // occupancies over-subscribe the site, normalise them — a residue cannot be more
            // than 100% modified.
            let total: f64 = options.iter().map(|(_, p)| p).sum();
            let options: Vec<(usize, f64)> = if total > 1.0 {
                options.into_iter().map(|(i, p)| (i, p / total)).collect()
            } else {
                options
            };
            let used: f64 = options.iter().map(|(_, p)| p).sum();
            let p_unmodified = (1.0 - used).max(0.0);
            let p_max = options
                .iter()
                .map(|(_, p)| *p)
                .fold(p_unmodified, f64::max);
            SiteOptions {
                position,
                options,
                p_unmodified,
                p_max,
            }
        })
        .collect();

    sites.sort_by_key(|s| s.position);
    sites
}

/// The blocking occupancy at each cleavage site of one protein.
///
/// A modification blocks the protease by sitting on the residue the protease cuts **after** — the
/// P1 residue. Trypsin cleaves C-terminal to K and R, so a cleavage site recorded at position `k`
/// has its P1 residue at `sequence[k - 1]`, not at `sequence[k]`. Reading the wrong side of the
/// scissile bond would block on whatever residue happened to follow the cut — a uniformly random
/// amino acid — producing a digest that is wrong in a way that looks entirely plausible: the missed
/// cleavage rate still rises, just at the wrong lysines.
///
/// Only `Stage::Protein` modifications can block: a mod that forms *after* digestion cannot have
/// prevented the protease from cutting. (`Modification::validate` does not enforce this pairing —
/// it is enforced here, where the two facts meet.)
///
/// Competing blockers on one residue are mutually exclusive — a lysine carries one modification —
/// so their occupancies **add**: the site is blocked if any of them is present. The sum is capped at
/// 1 to survive a spec whose occupancies oversubscribe the residue.
///
/// Returns `(position, blocking_occupancy)` for each site with a non-zero one.
pub fn blocking_at_sites(
    sequence: &str,
    cleavage_sites: &[u32],
    mods: &[Modification],
) -> Vec<(u32, f64)> {
    let blockers: Vec<&Modification> = mods
        .iter()
        .filter(|m| m.blocks_cleavage && m.stage == Stage::Protein && m.site == Site::Residue)
        .collect();
    if blockers.is_empty() {
        return Vec::new();
    }
    let bytes = sequence.as_bytes();
    let mut out = Vec::new();
    for &k in cleavage_sites {
        // k == 0 is the protein N-terminus and k > len the C-terminus; neither has a P1 residue.
        if k == 0 || (k as usize) > bytes.len() {
            continue;
        }
        let p1 = bytes[k as usize - 1] as char;
        let occ: f64 = blockers
            .iter()
            .filter(|m| m.targets.contains(p1))
            .map(|m| m.occupancy)
            .sum::<f64>()
            .min(1.0);
        if occ > 0.0 {
            out.push((k, occ));
        }
    }
    out
}

/// Enumerate every modform of `sequence` whose abundance fraction is at least `floor`.
///
/// Exact, not heuristic: the depth-first search prunes with an **admissible** bound (the
/// product of the best remaining option at each site), so nothing above the floor is missed.
///
/// `floor = 0.0` enumerates everything — which is what the mass-balance oracle uses.
pub fn enumerate_modforms(
    sequence: &str,
    mods: &[Modification],
    floor: f64,
) -> Result<(Vec<Modform>, ModformStats), String> {
    for m in mods {
        m.validate()?;
    }
    if !(0.0..1.0).contains(&floor) {
        return Err(format!("abundance floor must be in [0, 1), got {floor}"));
    }

    let sites = sites_for(sequence, mods);
    let mut stats = ModformStats {
        sites: sites.len(),
        ..Default::default()
    };

    // Suffix bound: the best achievable product from site k onwards. Multiplying the running
    // product by this gives an upper bound on anything reachable below — so pruning on it is
    // exact.
    let mut suffix_max = vec![1.0f64; sites.len() + 1];
    for k in (0..sites.len()).rev() {
        suffix_max[k] = suffix_max[k + 1] * sites[k].p_max;
    }

    let mut out = Vec::new();
    let mut chosen: Vec<(u32, usize)> = Vec::new();

    fn walk(
        k: usize,
        p: f64,
        delta: f64,
        sites: &[SiteOptions],
        mods: &[Modification],
        suffix_max: &[f64],
        floor: f64,
        chosen: &mut Vec<(u32, usize)>,
        out: &mut Vec<Modform>,
    ) {
        if p * suffix_max[k] < floor {
            return; // admissible bound: nothing below here can clear the floor
        }
        if k == sites.len() {
            if p >= floor {
                out.push(Modform {
                    mods: chosen.clone(),
                    abundance_fraction: p,
                    mass_delta: delta,
                });
            }
            return;
        }
        let s = &sites[k];
        // unmodified at this site
        if s.p_unmodified > 0.0 {
            walk(k + 1, p * s.p_unmodified, delta, sites, mods, suffix_max, floor, chosen, out);
        }
        // each competing modification
        for &(idx, prob) in &s.options {
            if prob <= 0.0 {
                continue;
            }
            chosen.push((s.position, idx));
            walk(
                k + 1,
                p * prob,
                delta + mods[idx].mass_delta,
                sites,
                mods,
                suffix_max,
                floor,
                chosen,
                out,
            );
            chosen.pop();
        }
    }

    walk(0, 1.0, 0.0, &sites, mods, &suffix_max, floor, &mut chosen, &mut out);

    // Deterministic order: most abundant first, ties broken structurally.
    out.sort_by(|a, b| {
        b.abundance_fraction
            .partial_cmp(&a.abundance_fraction)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.mods.cmp(&b.mods))
    });

    stats.retained = out.len();
    stats.mass_retained = out.iter().map(|m| m.abundance_fraction).sum();
    Ok((out, stats))
}

/// Plausible charge states for a peptide under electrospray.
///
/// A crude proxy for a real charge model: the number of basic sites (K, R, H, plus the free
/// N-terminus) bounds how many protons a peptide can reasonably carry.
///
/// **This is a placeholder for the S0.5 count, not the model.** The real thing is a *charge
/// propensity* predicted from sequence, conditioned by the ion source — `imspy-predictors`
/// already has one. See SPEC §8.1.
pub fn plausible_charges(sequence: &str, max_charge: u8) -> Vec<u8> {
    let basic = sequence
        .bytes()
        .filter(|b| matches!(b, b'K' | b'R' | b'H'))
        .count() as u8
        + 1; // the free N-terminus
    let hi = basic.min(max_charge).max(1);
    (1..=hi).filter(|&z| z >= 1).collect()
}

/// Monoisotopic mass of a modform.
pub fn modform_mass(sequence: &str, modform: &Modform) -> Result<f64, mass::UnknownResidue> {
    Ok(mass::monoisotopic(sequence)? + modform.mass_delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn phospho() -> Modification {
        Modification {
            name: "Phospho".into(),
            unimod_id: 21,
            targets: "STY".into(),
            site: Site::Residue,
            occupancy: 0.02,
            mass_delta: 79.96633,
            blocks_cleavage: false,
            stage: Stage::Protein,
        }
    }
    fn oxidation() -> Modification {
        Modification {
            name: "Oxidation".into(),
            unimod_id: 35,
            targets: "M".into(),
            site: Site::Residue,
            occupancy: 0.05,
            mass_delta: 15.99491,
            blocks_cleavage: false,
            stage: Stage::Peptide,
        }
    }
    fn carbamidomethyl(occ: f64) -> Modification {
        Modification {
            name: "Carbamidomethyl".into(),
            unimod_id: 4,
            targets: "C".into(),
            site: Site::Residue,
            occupancy: occ,
            mass_delta: 57.02146,
            blocks_cleavage: false,
            stage: Stage::Protein,
        }
    }

    /// THE oracle. Every copy of the peptide is in exactly one modform, so over the complete
    /// enumeration the abundance fractions sum to exactly 1. Independent of the implementation
    /// — and the truncation error is precisely whatever this falls short by.
    #[test]
    fn abundance_fractions_sum_to_one_when_untruncated() {
        let mods = vec![phospho(), oxidation(), carbamidomethyl(0.98)];
        for seq in ["PEPTIDESTYK", "MSTMSTYCCK", "AAAAAAAK", "SSSSSSSSTTTTYYYY"] {
            let (forms, stats) = enumerate_modforms(seq, &mods, 0.0).unwrap();
            // 1e-9, not 1e-12: the last sequence has 16 modifiable sites, so the sum runs over
            // 2^16 = 65,536 terms and accumulates ~1e-12 of float drift. That is arithmetic,
            // not a modelling error.
            assert_relative_eq!(stats.mass_retained, 1.0, epsilon = 1e-9);
            assert_relative_eq!(
                forms.iter().map(|f| f.abundance_fraction).sum::<f64>(),
                1.0,
                epsilon = 1e-9
            );
            assert_relative_eq!(stats.truncation_loss(), 0.0, epsilon = 1e-9);
        }
    }

    /// The floor truncates on probability MASS, and reports exactly what it dropped.
    #[test]
    fn the_floor_truncates_on_mass_and_measures_the_loss() {
        let mods = vec![phospho()];
        let seq = "SSSTTTYYY"; // 9 phospho-able sites
        let (all, full) = enumerate_modforms(seq, &mods, 0.0).unwrap();
        let (kept, stats) = enumerate_modforms(seq, &mods, 1e-3).unwrap();

        assert!(kept.len() < all.len(), "the floor must actually drop modforms");
        assert!(stats.retained == kept.len());
        assert_relative_eq!(full.mass_retained, 1.0, epsilon = 1e-12);

        // Everything kept clears the floor; everything dropped is accounted for.
        for f in &kept {
            assert!(f.abundance_fraction >= 1e-3);
        }
        let dropped: f64 = all
            .iter()
            .filter(|f| f.abundance_fraction < 1e-3)
            .map(|f| f.abundance_fraction)
            .sum();
        assert_relative_eq!(stats.truncation_loss(), dropped, epsilon = 1e-12);
    }

    /// A count budget and a mass floor are different things, and the floor is the honest one.
    /// With 9 sites at 2%, the unmodified and singly-modified forms carry ~99.8% of the
    /// material — so a mass floor keeps the mass while a `max_variable_mods = 3` budget would
    /// blindly keep 130 forms regardless of how little they weigh.
    #[test]
    fn a_mass_floor_keeps_the_mass() {
        let mods = vec![phospho()];
        let (kept, stats) = enumerate_modforms("SSSTTTYYY", &mods, 1e-4).unwrap();
        assert!(
            stats.mass_retained > 0.999,
            "a 1e-4 floor should retain >99.9% of the material, got {}",
            stats.mass_retained
        );
        // and it does so with far fewer forms than the full 2^9 = 512.
        assert!(kept.len() < 100, "kept {} forms", kept.len());
    }

    /// Carbamidomethyl at 0.98 is not a "fixed mod" — it is an alkylation efficiency, and the
    /// 2% of unalkylated cysteine is exactly why free-Cys peptides appear in real data.
    #[test]
    fn alkylation_efficiency_produces_free_cysteine() {
        let (forms, _) = enumerate_modforms("PEPTCIDEK", &[carbamidomethyl(0.98)], 0.0).unwrap();
        assert_eq!(forms.len(), 2, "alkylated and not");

        let alkylated = forms.iter().find(|f| !f.is_unmodified()).unwrap();
        let free = forms.iter().find(|f| f.is_unmodified()).unwrap();
        assert_relative_eq!(alkylated.abundance_fraction, 0.98, epsilon = 1e-12);
        assert_relative_eq!(free.abundance_fraction, 0.02, epsilon = 1e-12);

        // At occupancy 1.0 the free form vanishes entirely — "fixed" is just occupancy 1.
        let (forms, _) = enumerate_modforms("PEPTCIDEK", &[carbamidomethyl(1.0)], 0.0).unwrap();
        assert_eq!(forms.len(), 1);
        assert!(!forms[0].is_unmodified());
    }

    /// Positional isomers — same mass, different site — are distinct modforms and coexist.
    /// This is what makes a *localisation* benchmark rather than an identification one, and v1
    /// cannot produce it at all (it ran two separate simulations and recovered the site from
    /// the filename).
    #[test]
    fn positional_isomers_coexist_and_are_distinct() {
        // Exactly two phospho-able residues: S at 0 and S at 8. (An earlier version of this
        // test used "SPEPTIDESK", which has a T at position 4 as well — three sites, not two.)
        let (forms, _) = enumerate_modforms("SPEPAIDESK", &[phospho()], 0.0).unwrap();

        let singles: Vec<_> = forms.iter().filter(|f| f.mods.len() == 1).collect();
        assert_eq!(singles.len(), 2, "two positional isomers");
        assert_ne!(singles[0].mods[0].0, singles[1].mods[0].0, "different sites");
        assert_relative_eq!(
            singles[0].mass_delta,
            singles[1].mass_delta,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            singles[0].abundance_fraction,
            singles[1].abundance_fraction,
            epsilon = 1e-12
        );
    }

    /// Competing modifications at one residue are mutually exclusive — a lysine is acetylated
    /// or trimethylated, not both.
    #[test]
    fn competing_modifications_at_one_site_are_exclusive() {
        let acetyl = Modification {
            name: "Acetyl".into(), unimod_id: 1, targets: "K".into(), site: Site::Residue,
            occupancy: 0.3, mass_delta: 42.01057, blocks_cleavage: true, stage: Stage::Protein,
        };
        let trimethyl = Modification {
            name: "Trimethyl".into(), unimod_id: 37, targets: "K".into(), site: Site::Residue,
            occupancy: 0.2, mass_delta: 42.04695, blocks_cleavage: true, stage: Stage::Protein,
        };
        let (forms, stats) = enumerate_modforms("PEPTIDEK", &[acetyl, trimethyl], 0.0).unwrap();

        assert_eq!(stats.sites, 1);
        assert_eq!(forms.len(), 3, "unmodified | acetyl | trimethyl — never both");
        for f in &forms {
            assert!(f.mods.len() <= 1);
        }
        assert_relative_eq!(stats.mass_retained, 1.0, epsilon = 1e-12);
    }

    /// Over-subscribed occupancies are normalised: a residue cannot be more than 100% modified.
    #[test]
    fn oversubscribed_site_occupancies_are_normalised() {
        let a = Modification {
            name: "A".into(), unimod_id: 1, targets: "K".into(), site: Site::Residue,
            occupancy: 0.8, mass_delta: 10.0, blocks_cleavage: false, stage: Stage::Protein,
        };
        let mut b = a.clone();
        b.name = "B".into();
        b.occupancy = 0.6; // 0.8 + 0.6 = 1.4 > 1
        let (_, stats) = enumerate_modforms("PEPTIDEK", &[a, b], 0.0).unwrap();
        assert_relative_eq!(stats.mass_retained, 1.0, epsilon = 1e-12);
    }

    /// REGRESSION: modification positions must not wrap.
    ///
    /// Positions were cast `i as u16`. Peptides are short in practice, but the public API and
    /// `--max-length` accept a `u32` bound — so a sequence past 65,535 residues wrapped position
    /// 65,536 to 0, **merging two distinct sites into competing alternatives at one residue**.
    /// That corrupts the coordinates AND the probabilities, silently. Found by review — and it is
    /// the same "narrowed a type before the bound that makes it safe" bug as the u8 missed-cleavage
    /// counter.
    #[test]
    fn modification_positions_do_not_wrap_on_long_sequences() {
        // The second S sits at EXACTLY 65,536 so that a u16 cast wraps it to 0 and collides
        // with the first. (A first draft of this test placed it at 65,601 — which wraps to 65,
        // collides with nothing, and would therefore have PASSED against the buggy code. A
        // regression test that cannot fail against its own bug is theatre.)
        let mut seq = String::with_capacity(65_540);
        seq.push('S'); //                     index 0
        seq.push_str(&"A".repeat(65_535));
        seq.push('S'); //                     index 65_536  →  0 under a u16 cast
        let long_pos = 65_536u32;
        assert_eq!(seq.len(), 65_537);

        let (forms, stats) = enumerate_modforms(&seq, &[phospho()], 0.0).unwrap();

        // Under the u16 bug: the two sites merge into ONE, carrying two "competing" options —
        // so `sites` is 1 and only 3 modforms exist instead of 4.
        assert_eq!(stats.sites, 2, "two distinct S residues, not one merged site");
        assert_eq!(forms.len(), 4, "unmodified | S@0 | S@65536 | both");

        let singles: Vec<_> = forms.iter().filter(|f| f.mods.len() == 1).collect();
        assert_eq!(singles.len(), 2, "two positional isomers");
        let positions: Vec<u32> = singles.iter().map(|f| f.mods[0].0).collect();
        assert!(positions.contains(&0), "the S at index 0");
        assert!(
            positions.contains(&long_pos),
            "the S at index {long_pos} must keep its position, not wrap to 0"
        );
        assert_relative_eq!(stats.mass_retained, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn invalid_specs_are_rejected() {
        let mut m = phospho();
        m.occupancy = 1.5;
        assert!(enumerate_modforms("PEPTIDESK", &[m], 0.0).is_err());
        assert!(enumerate_modforms("PEPTIDESK", &[phospho()], 1.0).is_err());

        // A non-finite mass delta would propagate NaN straight into the m/z of every modform
        // carrying it — a wrong mass that never fails. Found by review.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut m = phospho();
            m.mass_delta = bad;
            assert!(
                enumerate_modforms("PEPTIDESK", &[m], 0.0).is_err(),
                "mass_delta = {bad} must be rejected"
            );
        }
    }

    #[test]
    fn charge_states_track_basic_residues() {
        assert_eq!(plausible_charges("AAAA", 5), vec![1]); // N-term only
        assert_eq!(plausible_charges("AAAK", 5), vec![1, 2]); // + one K
        assert_eq!(plausible_charges("KRHAAA", 5), vec![1, 2, 3, 4]);
        assert_eq!(plausible_charges("KRHKRH", 3), vec![1, 2, 3], "capped");
    }

    fn gg(occ: f64) -> Modification {
        Modification {
            name: "GG".into(),
            unimod_id: 121,
            targets: "K".into(),
            site: Site::Residue,
            occupancy: occ,
            mass_delta: 114.042927,
            blocks_cleavage: true,
            stage: Stage::Protein,
        }
    }

    /// **The off-by-one test.**
    ///
    /// The protease cuts *after* the P1 residue, so a cleavage site at `k` has its P1 at
    /// `sequence[k - 1]`. This peptide is built so that the two readings disagree on every site:
    /// each cleavage is after a K or R, and the residue *following* each cut is deliberately never
    /// a K.
    ///
    /// If the implementation read `sequence[k]` instead, it would find no lysine at any site and
    /// block nothing — and the whole diGly mechanism would silently do nothing while every other
    /// number in the run stayed plausible.
    #[test]
    fn blocking_reads_the_p1_residue_not_the_one_after_the_cut() {
        //           0123456789
        let seq = "AAKGGRDDKEE";
        //           ^  ^    ^
        //  cleavage after K@2 -> site 3 ; after R@5 -> site 6 ; after K@8 -> site 9
        let sites = [3u32, 6, 9];
        let got = blocking_at_sites(seq, &sites, &[gg(0.4)]);

        // Sites 3 and 9 follow a K and must be blocked; site 6 follows an R and must not.
        assert_eq!(got, vec![(3, 0.4), (9, 0.4)], "P1 residues are seq[k-1]");

        // Nail down that this is not an accident of the sequence: the residues AT the sites
        // (seq[k]) are G, D, E — no lysine anywhere — so a `seq[k]` reading returns nothing.
        assert_eq!(seq.as_bytes()[3] as char, 'G');
        assert_eq!(seq.as_bytes()[6] as char, 'D');
        assert_eq!(seq.as_bytes()[9] as char, 'E');
    }

    /// A modification that forms after the protease has already cut cannot have stopped it.
    #[test]
    fn a_peptide_stage_modification_cannot_block_cleavage() {
        let mut late = gg(0.9);
        late.stage = Stage::Peptide;
        assert!(
            blocking_at_sites("AAKGGRDDKEE", &[3, 6, 9], &[late]).is_empty(),
            "a post-digestion modification must not block the protease"
        );
    }

    /// Competing blockers on one lysine are mutually exclusive, so their occupancies add — the site
    /// is blocked if *either* is present — and the total is capped at 1.
    #[test]
    fn competing_blockers_add_and_cap_at_one() {
        let mut acetyl = gg(0.3);
        acetyl.name = "Acetyl".into();
        let combined = blocking_at_sites("AAKEE", &[3], &[gg(0.2), acetyl]);
        assert_eq!(combined, vec![(3, 0.5)], "0.2 + 0.3");

        let mut greedy = gg(0.8);
        greedy.name = "Other".into();
        let capped = blocking_at_sites("AAKEE", &[3], &[gg(0.7), greedy]);
        assert_eq!(capped, vec![(3, 1.0)], "1.5 must cap to 1.0, not overflow the probability");
    }

    /// Termini have no P1 residue, and an out-of-range site must not panic or index past the end.
    #[test]
    fn protein_termini_are_not_blockable_sites() {
        assert!(blocking_at_sites("KAAK", &[0], &[gg(0.5)]).is_empty(), "N-terminus has no P1");
        // Site == len is the C-terminus: P1 is the final residue, which IS blockable.
        assert_eq!(blocking_at_sites("AAAK", &[4], &[gg(0.5)]), vec![(4, 0.5)]);
        // Beyond the end is not a site at all.
        assert!(blocking_at_sites("AAAK", &[9], &[gg(0.5)]).is_empty(), "past the C-terminus");
    }
}
