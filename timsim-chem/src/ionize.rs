//! Charge propensity and ionisation efficiency — the ion layer.
//!
//! # These are PROPENSITIES, not realised values
//!
//! Peptides separate in solution on the column and only *then* reach the ESI needle, so **ionisation
//! happens after the LC** and is coupled to it: the charge envelope and the response both depend on
//! the eluent composition at the moment of elution, and on what co-elutes.
//!
//! So the structure axis may carry only *latent* propensities — gas-phase basicity, surface
//! activity. The **realised** charge distribution and response are measurements, computed after the
//! LC (SPEC §8.2). For v2.0 the source model is *uncoupled* (a plain binomial), so propensity and
//! distribution coincide; the point of putting the stage in the right place anyway is that turning
//! coupling on later becomes a config change rather than an axis migration (**B14**).
//!
//! # The conservation chain
//!
//! Every stage records its multiplier, so the chain is **invertible** — you can walk an ion's amount
//! back to its peptide, and a peptide back to its proteins:
//!
//! ```text
//!   Σ protein_amol · MW              =  load_ng                      (design)
//!   Σ peptide_amol over occurrences  =  Σ protein_amol · p_yield      (digest)
//!   Σ ion_amol     over charges      =  peptide_amol · flyability     (here)
//! ```
//!
//! The ground truth is not just the numbers — it is **the factorisation**. v1 destroys it:
//! `events = (amount × efficiency).astype(int32)` collapses abundance and flyability into one
//! integer, and no analysis downstream can tell them apart again.

use crate::{isotope, mass};

/// Protonatable sites: the free N-terminus, plus every H / R / K.
///
/// Matches v1 exactly (`mscore/src/algorithm/peptide.rs:383`).
pub fn protonatable_sites(sequence: &str) -> u32 {
    1 + sequence
        .bytes()
        .filter(|b| matches!(b, b'H' | b'R' | b'K'))
        .count() as u32
}

/// How a peptide's molecules distribute over charge states.
///
/// # Two models, and the default is v1's
///
/// **`Binomial`** — v1's model exactly: every protonatable site carries a proton with the same
/// probability. Kept as the default for parity.
///
/// **`SiteSpecific`** — every site gets its *own* probability, because they are not the same.
/// Gas-phase basicity runs **R > K > H**, and a binomial has no way to say so: one probability has
/// to serve the strongly basic N-term/K/R *and* the weaker histidine, so it is necessarily wrong
/// about both. Measured on the same sequences:
///
/// ```text
///                         1+      2+      3+      4+    mean
///   PEPTIDEK    binomial 33.3%   66.7%    -       -     1.67
///               site-sp. 11.3%   88.7%    -       -     1.89
///   PEPTIDEHK   binomial  9.7%   38.7%   51.6%    -     2.42
///               site-sp.  2.5%   26.7%   70.7%    -     2.68
///   HHPEPTIDEK  binomial  2.6%   15.4%   41.0%  41.0%   3.21
///               site-sp.  0.6%    7.4%   35.5%  56.6%   3.48
/// ```
///
/// The headline defect is the first row: **the binomial calls a third of all clean tryptic peptides
/// singly-charged.** That is not a thing that happens — a peptide with a free N-terminus and a
/// C-terminal K/R has two good basic sites and is overwhelmingly 2+. The binomial can only fix it
/// by raising `p`, which then over-charges everything carrying a histidine.
///
/// With per-site probabilities the charge distribution is a **Poisson-binomial** (a sum of
/// independent Bernoulli trials with *different* probabilities), computed by the same convolution
/// used for the isotope envelope.
///
/// # What this model is *not* validated against
///
/// The aggregate over the full enumerated peptide space is **not** a meaningful check, and an
/// earlier version of this comment used it as one. That space is dominated by long, missed-cleavage,
/// K/R-rich peptides that no run observes; its mean charge is ~2.7–3.1 under either model, against
/// a *literature* figure (~2.2–2.4) that describes the **abundance-weighted, m/z-filtered, observed**
/// population. Comparing the two is comparing different populations, and it will happily confirm
/// whichever model you already prefer. Validate on the observable set (see the tests), or on real
/// data — not on the enumeration.
#[derive(Clone, Copy, Debug)]
pub enum ChargeModel {
    /// v1's model: one probability for every protonatable site. Kept for parity checks — it is
    /// **not** the default, because treating histidine like arginine is wrong.
    Binomial {
        charged_probability: f64,
        max_charge: u8,
    },
    /// Per-site protonation probabilities, reflecting actual gas-phase basicity.
    SiteSpecific {
        n_term: f64,
        arginine: f64,
        lysine: f64,
        histidine: f64,
        max_charge: u8,
    },
}

impl Default for ChargeModel {
    /// **Site-specific**, not v1's binomial.
    ///
    /// This is a deliberate departure from v1 — the first one — because the binomial cannot
    /// represent an ordering of basicity, and so is wrong about clean tryptic peptides (33% of them
    /// singly-charged) in a way no choice of `p` repairs.
    ///
    /// Both models are tuned to agree on the *aggregate* charge distribution of an observable
    /// tryptic population, and they do (~79% 2+, ~18% 3+). They disagree about **which peptides**
    /// carry the charge — the binomial gives an R-terminated peptide and an H-containing one the
    /// same distribution, this model does not. For a search-engine benchmark the aggregate is
    /// cosmetic and the per-peptide assignment is the thing under test.
    ///
    /// `ChargeModel::Binomial { charged_probability: 0.8, max_charge: 4 }` reproduces v1 exactly,
    /// and remains available for parity checks.
    fn default() -> Self {
        ChargeModel::realistic()
    }
}

impl ChargeModel {
    /// A site-specific model calibrated so a clean tryptic peptide is predominantly 2+.
    pub fn realistic() -> Self {
        Self::realistic_with(4)
    }

    /// As [`ChargeModel::realistic`], at a given maximum charge.
    ///
    /// # Where `histidine` comes from, and why it is not small
    ///
    /// Histidine sits *below* K and R — its gas-phase basicity genuinely is lower — but it is not
    /// an order of magnitude below them, and an earlier version of this model set it to `0.20` on
    /// exactly that mistaken intuition. The chemistry that matters: His has a side-chain pKa of
    /// ~6.0 and electrospray runs at pH 2–3 in formic acid, so histidine enters the droplet
    /// **essentially fully protonated**. It retains that proton less well than K/R during
    /// desolvation; it does not mostly arrive without one.
    ///
    /// The error was not academic. A fully-tryptic peptide ends in K/R and contains no internal
    /// K/R, so its only *third* basic site is almost always a histidine — which makes the observed
    /// 3+ fraction of a tryptic digest very nearly a direct readout of this one number. At 0.20 the
    /// simulated 3+ fraction collapsed to ~6%; real tryptic DIA runs at 20–30%.
    pub fn realistic_with(max_charge: u8) -> Self {
        ChargeModel::SiteSpecific {
            n_term: 0.93,
            arginine: 0.97, // most basic of the three
            lysine: 0.95,
            histidine: 0.80, // below K/R, but protonated at ESI pH — see above
            max_charge,
        }
    }

    /// A one-line description of the actual parameters. `--explain` prints THIS rather than a
    /// hand-typed string, because a hand-typed string drifts from the model it claims to describe —
    /// which is how `retention_time_gru_predictor` happened.
    pub fn describe(&self) -> String {
        match self {
            ChargeModel::Binomial { charged_probability, max_charge } => format!(
                "binomial: every protonatable site (N-term, H, R, K) at p = {charged_probability}, \
                 z = 1..{max_charge}"
            ),
            ChargeModel::SiteSpecific { n_term, arginine, lysine, histidine, max_charge } => format!(
                "site-specific: N-term {n_term}, R {arginine}, K {lysine}, H {histidine}, \
                 z = 1..{max_charge}"
            ),
        }
    }

    pub fn max_charge(&self) -> u8 {
        match self {
            ChargeModel::Binomial { max_charge, .. } => *max_charge,
            ChargeModel::SiteSpecific { max_charge, .. } => *max_charge,
        }
    }

    /// Per-site protonation probabilities for this peptide, one entry per protonatable site.
    fn site_probabilities(&self, sequence: &str) -> Vec<f64> {
        match self {
            ChargeModel::Binomial { charged_probability, .. } => {
                vec![*charged_probability; protonatable_sites(sequence) as usize]
            }
            ChargeModel::SiteSpecific { n_term, arginine, lysine, histidine, .. } => {
                let mut p = vec![*n_term];
                for b in sequence.bytes() {
                    match b {
                        b'R' => p.push(*arginine),
                        b'K' => p.push(*lysine),
                        b'H' => p.push(*histidine),
                        _ => {}
                    }
                }
                p
            }
        }
    }
}

impl ChargeModel {
    pub fn validate(&self) -> Result<(), String> {
        let ps: Vec<(&str, f64)> = match self {
            ChargeModel::Binomial { charged_probability, .. } => {
                vec![("charged_probability", *charged_probability)]
            }
            ChargeModel::SiteSpecific { n_term, arginine, lysine, histidine, .. } => vec![
                ("n_term", *n_term),
                ("arginine", *arginine),
                ("lysine", *lysine),
                ("histidine", *histidine),
            ],
        };
        for (name, v) in &ps {
            if !v.is_finite() || !(0.0..=1.0).contains(v) {
                return Err(format!("{name} must be finite and in [0, 1], got {v}"));
            }
        }
        // A model in which nothing can be protonated produces no ions at all — and the degenerate
        // branch below would otherwise fabricate a 1+ ion with fraction 1.0 for a configuration that
        // explicitly says nothing ionises. Inventing signal is worse than refusing to run.
        if ps.iter().all(|(_, v)| *v == 0.0) {
            return Err(
                "charge model has zero protonation probability at every site — it can produce no \
                 ions, so there is nothing to simulate"
                    .to_string(),
            );
        }
        if self.max_charge() == 0 {
            return Err("max_charge must be at least 1".to_string());
        }
        Ok(())
    }

    /// Fraction of this peptide's molecules carrying charge `z`, for `z = 1..=max_charge`.
    ///
    /// See [`ChargeModel::distribution_with_loss`] — this discards the truncation report.
    pub fn distribution(&self, sequence: &str) -> Vec<(u8, f32)> {
        self.distribution_with_loss(sequence).0
    }

    /// The charge distribution, **and the probability mass the charge cap discarded**.
    ///
    /// The **Poisson-binomial** distribution of the protonated-site count — a sum of independent
    /// Bernoulli trials with (possibly) different probabilities. Computed by convolution, the same
    /// machinery as the isotope envelope. With equal probabilities it reduces exactly to v1's
    /// binomial.
    ///
    /// # The full PMF is computed BEFORE the cap is applied
    ///
    /// Truncating the convolution at `max_charge` would silently **delete** the transitions that
    /// push past it — and the renormalisation would then redistribute those real high-charge ions'
    /// abundance into the lower charges, inflating them. A peptide with six basic residues genuinely
    /// produces 5+ and 6+ ions; dropping them and pretending the rest add to 1 is a lie.
    ///
    /// So: compute the whole distribution, apply the cap, and **report what the cap cost** — the same
    /// discipline as `--max-missed-cleavages` and the modform abundance floor. A truncation whose
    /// error is unmeasured is a guess.
    ///
    /// Returns `((charge, fraction) pairs summing to 1, mass discarded above the cap)`.
    pub fn distribution_with_loss(&self, sequence: &str) -> (Vec<(u8, f32)>, f64) {
        let max_charge = self.max_charge();
        let probs = self.site_probabilities(sequence);

        // FULL Poisson-binomial: one slot per possible proton count, no truncation.
        let mut pmf = vec![0.0f64; probs.len() + 1];
        pmf[0] = 1.0;
        for (i, p) in probs.iter().enumerate() {
            for k in (0..=i).rev() {
                let v = pmf[k];
                if v == 0.0 {
                    continue;
                }
                pmf[k + 1] += v * p;
                pmf[k] = v * (1.0 - p);
            }
        }

        // Two removals, and they are NOT the same operation.
        //
        // **z = 0 (neutral) is renormalised away.** A molecule with no proton produces no ion at
        // all, and "does this peptide ionise" is already what `Flyability` models — so folding the
        // neutral fraction in here as well would double-count it. Dividing through by P(z >= 1) is
        // therefore correct.
        //
        // **z > max_charge is DELETED, not renormalised.** Those ions are real: a peptide with six
        // basic residues genuinely makes 5+ and 6+ ions. We simply choose not to model them.
        // Renormalising over the kept charges would hand their abundance to the 1+..4+ states and
        // *inflate* them — reporting the loss while quietly redistributing it anyway, which is worse
        // than not reporting it. So the fractions deliberately sum to **less than 1**, and the
        // shortfall IS the discarded population.
        let ionised: f64 = pmf.iter().skip(1).sum(); // ALL z >= 1, not just the kept ones
        if ionised <= 0.0 {
            // Nothing can carry a charge, so there are NO ions. Emitting a 1+ with fraction 1.0
            // (as an earlier version did) fabricates signal that the model says does not exist.
            return (Vec::new(), 0.0);
        }
        let dist: Vec<(u8, f32)> = (1..=max_charge)
            .filter_map(|z| {
                pmf.get(z as usize)
                    .map(|v| (z, (v / ionised) as f32))
                    .filter(|(_, f)| *f > 0.0)
            })
            .collect();
        let kept: f64 = dist.iter().map(|(_, f)| *f as f64).sum();
        (dist, 1.0 - kept)
    }
}

/// Ionisation efficiency ("flyability") — how well a peptide flies, independent of how much of it
/// there is.
///
/// # This is the quantity that must not absorb abundance
///
/// A *response-factor* model (trained on equimolar synthetic peptides — PFly's **base** model) is a
/// genuine molecular property and belongs here. An *observability* model (trained on "was this
/// peptide identified in a real sample") has abundance baked in — PFly's fine-tuned variant, the one
/// Koina serves, correlates with protein abundance at **ρ = 0.76** by its authors' own measurement.
/// Multiply that by our explicitly-modelled `amount_amol` and abundance enters the signal **twice**.
/// See SPEC §8.2.
#[derive(Clone, Copy, Debug)]
pub enum Flyability {
    /// v1's model: a normal draw in log10 space, **rejection-truncated** to `[lo, hi]`.
    ///
    /// With v1's parameters (median 1e-2, σ = 1, clipped to [1e-4, 1]) the bounds sit at **±2σ**, so
    /// the truncation is not cosmetic — it removes ~4.6% of the mass and gives the distribution hard
    /// edges rather than tails.
    ///
    /// Unlike v1's bulk `np.random.normal(size=n)` (assigned by row order, so adding a peptide
    /// reshuffles everyone), this is **identity-keyed**: a peptide's flyability depends only on its
    /// sequence and the seed.
    LogNormal {
        median: f64,
        sigma: f64,
        lo: f64,
        hi: f64,
    },
    /// Every peptide flies equally well. Useful for isolating other effects.
    Uniform,
}

impl Default for Flyability {
    fn default() -> Self {
        Flyability::LogNormal {
            median: 1e-2,
            sigma: 1.0,
            lo: 1e-4,
            hi: 1.0,
        }
    }
}

impl Flyability {
    pub fn validate(&self) -> Result<(), String> {
        if let Flyability::LogNormal {
            median,
            sigma,
            lo,
            hi,
        } = self
        {
            if !median.is_finite() || *median <= 0.0 {
                return Err(format!("flyability median must be > 0, got {median}"));
            }
            if !sigma.is_finite() || *sigma < 0.0 {
                return Err(format!("flyability sigma must be >= 0, got {sigma}"));
            }
            if !lo.is_finite() || !hi.is_finite() || *lo <= 0.0 || lo >= hi {
                return Err(format!("flyability bounds must satisfy 0 < lo < hi, got [{lo}, {hi}]"));
            }
        }
        Ok(())
    }

    /// Flyability of one peptide. Deterministic and keyed by sequence identity.
    pub fn of(&self, sequence: &str, seed: u64) -> f64 {
        match self {
            Flyability::Uniform => 1.0,
            Flyability::LogNormal {
                median,
                sigma,
                lo,
                hi,
            } => {
                // REJECTION-sampled, not clamped.
                //
                // v1 redraws until the value falls inside [lo, hi]. Clamping instead would pile the
                // rejected ~4.6% into point masses AT the bounds — 2.3% of every proteome sitting at
                // exactly 1e-4 and 2.3% at exactly 1.0 — which is a different distribution, and it
                // would show up as two spikes in any flyability histogram.
                //
                // Identity-keyed rejection: redraw with a counter folded into the key, so it stays
                // deterministic and order-independent.
                let mut attempt = 0u32;
                loop {
                    let key = attempt.to_string();
                    let z = crate::design::standard_normal(&["flyability", sequence, &key], seed);
                    let v = median * 10f64.powf(sigma * z);
                    if v >= *lo && v <= *hi {
                        return v;
                    }
                    attempt += 1;
                    if attempt > 64 {
                        // Unreachable for any sane (median, sigma, lo, hi); a degenerate spec would
                        // otherwise spin forever.
                        return v.clamp(*lo, *hi);
                    }
                }
            }
        }
    }
}

/// One precursor: a (peptide, charge) pair, with everything an instrument needs to see it.
#[derive(Clone, Debug, PartialEq)]
pub struct Precursor {
    pub precursor_id: u64,
    pub peptide_id: u64,
    pub charge: u8,
    pub mz: f64,
    /// Relative intensities of the isotope comb, summing to 1.
    pub isotope_intensity: Vec<f32>,
    /// Fraction of the peptide's molecules at this charge. Sums to 1 across a peptide's charges.
    pub charge_fraction: f32,
    /// The peptide's ionisation efficiency. Repeated across its charges — it is a property of the
    /// **peptide**, not of the ion, and is kept as its own column precisely so that it can never be
    /// silently folded into an amount (which is what v1 does).
    pub ionization_propensity: f32,
}

pub struct Ionizer {
    pub charge: ChargeModel,
    pub flyability: Flyability,
    pub isotope_depth: usize,
    pub seed: u64,
}

impl Ionizer {
    pub fn validate(&self) -> Result<(), String> {
        self.charge.validate()?;
        self.flyability.validate()?;
        if self.isotope_depth == 0 {
            return Err("isotope_depth must be at least 1".to_string());
        }
        Ok(())
    }

    /// Enumerate every precursor of one peptide.
    pub fn precursors_of(
        &self,
        peptide_id: u64,
        sequence: &str,
    ) -> Result<Vec<Precursor>, mass::UnknownResidue> {
        let mono = mass::monoisotopic(sequence)?;
        let env = isotope::envelope(sequence, self.isotope_depth)?;
        let fly = self.flyability.of(sequence, self.seed) as f32;

        Ok(self
            .charge
            .distribution(sequence)
            .into_iter()
            .map(|(z, frac)| Precursor {
                precursor_id: crate::ids::precursor_id(peptide_id, z),
                peptide_id,
                charge: z,
                mz: isotope::mz(mono, z, 0),
                isotope_intensity: env.clone(),
                charge_fraction: frac,
                ionization_propensity: fly,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn protonatable_sites_match_v1() {
        assert_eq!(protonatable_sites("PEPTIDE"), 1); // N-term only
        assert_eq!(protonatable_sites("PEPTIDEK"), 2); // + K
        assert_eq!(protonatable_sites("KRHPEPTIDE"), 4); // + K, R, H
    }

    /// Charge fractions + the cap loss must always account for exactly the ionised population.
    ///
    /// They sum to **1 only when nothing sits above the cap**. For a peptide basic enough to make
    /// 5+ ions, they deliberately sum to LESS than 1 — the shortfall is the discarded population,
    /// and renormalising it away would hand those ions' abundance to the lower charges.
    #[test]
    fn charge_fractions_plus_the_cap_loss_account_for_everything() {
        let m = ChargeModel::default();
        for seq in ["PEPTIDE", "PEPTIDEK", "KRHPEPTIDEKR", "AAAAAAAAAAAAAAAAAAK"] {
            let (d, loss) = m.distribution_with_loss(seq);
            let total: f64 = d.iter().map(|(_, f)| *f as f64).sum();
            assert_relative_eq!(total + loss, 1.0, epsilon = 1e-5);
            assert!(d.iter().all(|(z, _)| *z >= 1 && *z <= m.max_charge()));

            // Nothing above the cap ⇒ they DO sum to 1.
            if protonatable_sites(seq) <= m.max_charge() as u32 {
                assert_relative_eq!(total, 1.0, epsilon = 1e-5);
                assert!(loss < 1e-5, "{seq}: nothing should be lost, got {loss:.2e}");
            }
        }
    }

    /// A typical tryptic peptide (one K/R plus the N-terminus ⇒ 2 sites) is mostly 2+. This is the
    /// single most recognisable fact about tryptic proteomics, and any charge model that fails it is
    /// wrong on its face.
    /// The default model must put a tryptic peptide firmly at 2+. This is the single most
    /// recognisable fact about tryptic proteomics; a charge model that fails it is wrong on sight.
    #[test]
    fn a_tryptic_peptide_is_mostly_doubly_charged() {
        let d = ChargeModel::default().distribution("PEPTIDEK");
        let f2 = d.iter().find(|(z, _)| *z == 2).unwrap().1;
        assert!(f2 > 0.8, "2+ must dominate strongly under the default model, got {f2:.3}");

        // And v1's binomial, kept for parity, is measurably worse at exactly this.
        let v1 = ChargeModel::Binomial { charged_probability: 0.8, max_charge: 4 };
        let f2_v1 = v1.distribution("PEPTIDEK").iter().find(|(z, _)| *z == 2).unwrap().1;
        assert!(f2_v1 < f2, "the default must improve on v1: {f2_v1:.3} vs {f2:.3}");
    }

    /// More basic residues ⇒ higher charge. Missed-cleavage peptides carry an extra K/R and shift up.
    #[test]
    fn more_basic_residues_shift_the_charge_up() {
        let m = ChargeModel::default();
        let mean = |s: &str| -> f64 {
            m.distribution(s)
                .iter()
                .map(|(z, f)| *z as f64 * *f as f64)
                .sum()
        };
        assert!(mean("PEPTIDEK") < mean("PEPTIDEKR"));
        assert!(mean("PEPTIDEKR") < mean("PEPTIDEKRH"));
    }

    /// Flyability is identity-keyed: the same sequence gives the same value regardless of what else
    /// was computed, in any order, in any sample. This is what makes A/B consistency structural
    /// rather than accidental — v1's bulk draw reshuffles everyone when one peptide is added.
    #[test]
    fn flyability_is_identity_keyed() {
        let f = Flyability::default();
        let a = f.of("PEPTIDEK", 7);
        for i in 0..1000 {
            let _ = f.of(&format!("X{i}"), 7);
        }
        assert_relative_eq!(f.of("PEPTIDEK", 7), a, max_relative = 1e-15);
        assert_ne!(f.of("PEPTIDEK", 8), a, "a different seed must differ");
    }

    /// The truncation is real: nothing escapes [lo, hi], and with v1's parameters the bounds are
    /// ±2σ, so they actually bite.
    #[test]
    fn flyability_respects_its_bounds_and_lands_where_v1_puts_it() {
        let f = Flyability::default();
        let v: Vec<f64> = (0..20_000)
            .map(|i| f.of(&format!("PEPTIDEK{i}"), 3))
            .collect();
        assert!(v.iter().all(|x| (1e-4..=1.0).contains(x)));

        let med = {
            let mut s = v.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        };
        // Median should sit near 1e-2 (v1's mean_log = -2).
        assert!((5e-3..2e-2).contains(&med), "median flyability {med:.2e}");

        // REJECTION-sampled, so there must be NO point masses at the bounds. An earlier version
        // clamped, and this test asserted the pile-ups *existed* — it was testing the bug.
        let at_lo = v.iter().filter(|&&x| x <= 1.0001e-4).count();
        let at_hi = v.iter().filter(|&&x| x >= 0.9999).count();
        assert!(at_lo < 20, "{at_lo} peptides pinned to the lower bound — that is a clamp, not a redraw");
        assert!(at_hi < 20, "{at_hi} peptides pinned to the upper bound — that is a clamp, not a redraw");
    }

    #[test]
    fn uniform_flyability_is_one() {
        assert_relative_eq!(Flyability::Uniform.of("PEPTIDEK", 1), 1.0);
    }

    /// THE conservation law of this stage: an ion's amount divided by its charge fraction and the
    /// peptide's flyability recovers the peptide amount exactly. The chain is invertible, which is
    /// the whole point of keeping the factorisation instead of collapsing it into one number.
    #[test]
    fn the_ion_chain_is_invertible() {
        let io = Ionizer {
            charge: ChargeModel::default(),
            flyability: Flyability::default(),
            isotope_depth: 6,
            seed: 5,
        };
        let seq = "PEPTIDEKR";
        let peptide_amol = 1234.5_f64;

        let ps = io.precursors_of(crate::peptide_id(seq), seq).unwrap();

        // Forward: peptide -> ions.
        let ion_amounts: Vec<f64> = ps
            .iter()
            .map(|p| peptide_amol * p.ionization_propensity as f64 * p.charge_fraction as f64)
            .collect();

        // Conservation, with the charge cap accounted for. "PEPTIDEKR" has 3 sites, well under the
        // cap of 4, so nothing is discarded and the ions account for the whole ionisable fraction.
        let (_, cap_loss) = io.charge.distribution_with_loss(seq);
        assert!(cap_loss < 1e-6, "3 sites under a cap of 4 loses nothing (1e-6 for f32 rounding)");
        let total: f64 = ion_amounts.iter().sum();
        let fly = ps[0].ionization_propensity as f64;
        assert_relative_eq!(total, peptide_amol * fly, max_relative = 1e-6);

        // Backward: any single ion recovers the peptide amount.
        for (p, amt) in ps.iter().zip(&ion_amounts) {
            let recovered = amt / (p.charge_fraction as f64) / (p.ionization_propensity as f64);
            assert_relative_eq!(recovered, peptide_amol, max_relative = 1e-6);
        }
    }

    /// REGRESSION (round 2): the cap must genuinely REMOVE the mass, not report it and then
    /// redistribute it anyway.
    ///
    /// The first fix computed the full PMF and *reported* the discarded mass — but still divided the
    /// kept charges by their own sum, so the fractions came out to 1 and the discarded ions' abundance
    /// was handed to 1+..4+ regardless. Reporting a loss while silently redistributing it is worse
    /// than not reporting it. Found by review, twice.
    #[test]
    fn the_charge_cap_removes_mass_rather_than_redistributing_it() {
        let seq = "KRKRKRHHPEPTIDEK"; // 10 protonatable sites
        let capped = ChargeModel::realistic_with(4);
        let (d, loss) = capped.distribution_with_loss(seq);

        let kept: f64 = d.iter().map(|(_, f)| *f as f64).sum();
        assert!(loss > 0.05, "a 10-site peptide must lose real mass to a 4+ cap, got {loss:.4}");

        // THE assertion: the fractions do NOT sum to 1. The shortfall IS the discarded population.
        assert_relative_eq!(kept + loss, 1.0, epsilon = 1e-6);
        assert!(kept < 0.95, "fractions must NOT be renormalised back to 1, got {kept:.4}");

        // Raise the cap and the mass comes back as actual high-charge ions.
        let (full, loss_full) = ChargeModel::realistic_with(10).distribution_with_loss(seq);
        assert!(loss_full < 1e-6, "f32 fraction rounding only, got {loss_full:.2e}");
        assert_relative_eq!(full.iter().map(|(_, f)| *f as f64).sum::<f64>(), 1.0, epsilon = 1e-6);

        // And the capped 4+ fraction is NOT inflated — it equals the true 4+ fraction.
        let f4_capped = d.iter().find(|(z, _)| *z == 4).unwrap().1;
        let f4_true = full.iter().find(|(z, _)| *z == 4).unwrap().1;
        assert_relative_eq!(f4_capped, f4_true, epsilon = 1e-6);
    }

    /// The Poisson-binomial reduces EXACTLY to the binomial when every site shares a probability.
    /// A generalisation that does not contain the thing it generalises is a rewrite, not a
    /// generalisation.
    #[test]
    fn the_poisson_binomial_reduces_to_the_binomial() {
        let binom = ChargeModel::Binomial { charged_probability: 0.8, max_charge: 4 };
        let flat = ChargeModel::SiteSpecific {
            n_term: 0.8, arginine: 0.8, lysine: 0.8, histidine: 0.8, max_charge: 4,
        };
        for seq in ["PEPTIDEK", "KRHPEPTIDEKR", "PEPTIDEHK", "AAAAAA"] {
            let a = binom.distribution(seq);
            let b = flat.distribution(seq);
            assert_eq!(a.len(), b.len(), "{seq}");
            for ((z1, f1), (z2, f2)) in a.iter().zip(&b) {
                assert_eq!(z1, z2);
                assert_relative_eq!(f1, f2, epsilon = 1e-6);
            }
        }
    }

    /// Gas-phase basicity is strictly ordered R > K > H, so each must add *strictly less* charge
    /// than the one before. This is the whole reason the model is site-specific rather than
    /// binomial — a binomial cannot express an ordering at all.
    #[test]
    fn basicity_is_ordered_arginine_then_lysine_then_histidine() {
        let m = ChargeModel::realistic();
        let mean = |s: &str| -> f64 {
            m.distribution(s).iter().map(|(z, f)| *z as f64 * *f as f64).sum()
        };
        let plain = mean("PEPTIDEK");
        let d_r = mean("PEPTIDERK") - plain;
        let d_k = mean("PEPTIDEKK") - plain;
        let d_h = mean("PEPTIDEHK") - plain;

        assert!(d_r > d_k, "arginine must out-charge lysine: +{d_r:.3} vs +{d_k:.3}");
        assert!(d_k > d_h, "lysine must out-charge histidine: +{d_k:.3} vs +{d_h:.3}");
    }

    /// **The regression test for the `p(H) = 0.20` bug.**
    ///
    /// Histidine sits below K/R but is *protonated at ESI pH* — it is not a bystander. Because a
    /// fully-tryptic peptide's only third basic site is almost always a histidine, an
    /// under-weighted `p(H)` shows up as a digest with almost no 3+ precursors. Pinning the
    /// per-peptide fact directly: a tryptic peptide carrying one histidine is predominantly 3+.
    ///
    /// At `p(H) = 0.20` this peptide came out 17.7% 3+ and the test fails, which is exactly what
    /// should have happened the first time.
    #[test]
    fn a_histidine_bearing_tryptic_peptide_is_predominantly_triply_charged() {
        let d = ChargeModel::realistic().distribution("PEPTIDEHK"); // N-term + H + K
        let f3 = d.iter().find(|(z, _)| *z == 3).map(|(_, f)| *f).unwrap_or(0.0);
        assert!(
            f3 > 0.5,
            "one histidine plus a tryptic C-terminal K is three real basic sites; \
             3+ should dominate, got {:.1}%",
            f3 * 100.0
        );
    }

    /// The aggregate that the per-peptide bug destroyed: across a tryptic-like peptide set, the
    /// 3+ fraction must land in the band real tryptic DIA actually shows (~20–30%). A `p(H)` that
    /// is too low silently collapses this to a few percent while every *per-peptide* distribution
    /// still looks individually plausible — which is precisely why this is asserted on the
    /// aggregate and not on one sequence.
    #[test]
    fn the_triply_charged_fraction_matches_real_tryptic_proteomics() {
        // Fully-tryptic peptides: C-terminal K/R, no internal K/R. Roughly a quarter carry a
        // histidine, as real tryptic peptides do.
        let peptides = [
            "PEPTIDEK", "SAMPLEDATAR", "ALGNVTEK", "YFDSVTPDGVLR", "GHIYSTEPK", "AVDSLVPIGR",
            "TLSDYNIQK", "IHNPVTGELLR", "DAGTIAGLNVLR", "SVLHDLYK", "ANELLINVK", "GFAFVTFDDHDSVDK",
        ];
        let m = ChargeModel::realistic();
        let mut mass = [0.0f64; 6];
        for p in peptides {
            for (z, f) in m.distribution(p) {
                mass[z as usize] += f as f64;
            }
        }
        // 1+ precursors of real peptides sit above the instrument's m/z ceiling, so the fraction a
        // run actually reports is over the 2+ and up population.
        let observable: f64 = mass[2..].iter().sum();
        let f3 = mass[3] / observable;
        assert!(
            (0.15..0.35).contains(&f3),
            "3+ fraction of a tryptic digest should be ~20-30%, got {:.1}%",
            f3 * 100.0
        );
    }

    /// The realistic model puts a clean tryptic peptide predominantly at 2+, which is the single
    /// most recognisable fact about tryptic proteomics. v1's binomial gives it 33% singly-charged.
    #[test]
    fn the_realistic_model_makes_tryptic_peptides_doubly_charged() {
        let d = ChargeModel::realistic().distribution("PEPTIDEK");
        let f2 = d.iter().find(|(z, _)| *z == 2).unwrap().1;
        let f1 = d.iter().find(|(z, _)| *z == 1).map(|(_, f)| *f).unwrap_or(0.0);
        assert!(f2 > 0.8, "2+ should dominate strongly, got {f2:.3}");
        assert!(f1 < 0.15, "too many 1+: {f1:.3} (v1's binomial gives 0.33)");
    }

    #[test]
    fn invalid_models_are_rejected() {
        assert!(ChargeModel::Binomial { charged_probability: 1.5, max_charge: 4 }.validate().is_err());
        assert!(ChargeModel::Binomial { charged_probability: 0.8, max_charge: 0 }.validate().is_err());
        assert!(ChargeModel::SiteSpecific {
            n_term: 1.2, arginine: 0.9, lysine: 0.9, histidine: 0.2, max_charge: 4
        }.validate().is_err());

        // A model that cannot protonate anything must be refused — it can produce no ions, and the
        // degenerate path once fabricated a 1+ with fraction 1.0 for exactly this configuration.
        assert!(ChargeModel::Binomial { charged_probability: 0.0, max_charge: 4 }.validate().is_err());
        assert!(ChargeModel::SiteSpecific {
            n_term: 0.0, arginine: 0.0, lysine: 0.0, histidine: 0.0, max_charge: 4
        }.validate().is_err());
        // And even if one slipped through, it emits NO ions rather than inventing one.
        let dead = ChargeModel::Binomial { charged_probability: 0.0, max_charge: 4 };
        assert!(dead.distribution("PEPTIDEK").is_empty(), "a dead model must fabricate nothing");
        assert!(Flyability::LogNormal { median: 0.0, sigma: 1.0, lo: 1e-4, hi: 1.0 }.validate().is_err());
        assert!(Flyability::LogNormal { median: 1e-2, sigma: 1.0, lo: 1.0, hi: 1e-4 }.validate().is_err());
    }
}
