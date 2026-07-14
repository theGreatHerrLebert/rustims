//! The experimental design: mixtures, conditions, replicates, and the quantitative ground
//! truth that follows from them.
//!
//! # The mixture is the spec; the fold change is *derived*
//!
//! At the bench you do not specify a fold change — you specify a **mixture**. "65% human,
//! 30% yeast, 5% E. coli; now hold human fixed and move the other two." The fold changes
//! are a *consequence*. Today's benchmark suite has the expected HYE ratios
//! (1.0 / 0.667 / 3.0) **typed into a plotting cell**, because the simulator never knew
//! them. Here they are computed, and they are computed from the final amounts — so they
//! remain correct even when regulation or spike-ins shift the composition.
//!
//! # Sample ≠ Run, and replicates are where you see why
//!
//! A **biological** replicate is different material: the amounts genuinely differ.
//! A **technical** replicate is *the same tube injected twice*: the amounts are **identical**,
//! and all the variation is in the measurement.
//!
//! So biological variance is applied here, on the quantity axis, and technical variance is
//! *declared* here but *applied* on the measurement axis — where it physically belongs.
//! Two technical replicates are two `Run`s pointing at one `Sample`. That is the whole
//! reason `Sample` and `Run` are separate entities, and conflating them is what makes
//! TMT (N samples → 1 run) unrepresentable.
//!
//! # Mass balance
//!
//! ```text
//!     Σ_proteins  amount_amol × MW_average × 1e-9  ==  load_ng      exactly, per sample
//! ```
//!
//! If a mixture, an abundance profile, and a load cannot reproduce the requested
//! nanograms, that is a bug. Same discipline as the digest's residue conservation.

use crate::mass;
use std::collections::{BTreeMap, HashMap};

/// A fold change is a biological quantity, not an arbitrary float. 2^128 is ~3.4e38 — still
/// finite, but far beyond anything a proteome does; 2^1024 is infinity, and infinity times zero
/// is the NaN that ends up in the answer key.
const MAX_ABS_LOG2FC: f64 = 64.0;

// ─────────────────────────────────────────────────────────────────────────────
// Spec
// ─────────────────────────────────────────────────────────────────────────────

/// A share of the mixture, by **mass**. `Rest` fills whatever is left, so a mixture cannot
/// be mistyped into not summing to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Share {
    Fraction(f64),
    Rest,
}

/// Within-organism relative molar abundance across proteins.
///
/// # Shape versus identity
///
/// A real proteome's abundance distribution is roughly log-normal in *shape* — that part is well
/// supported. What no parametric draw can give you is **identity**: the abundant proteins are
/// reliably *the same proteins* (ribosomal, glycolytic, cytoskeletal; albumin in plasma). Identity
/// is what determines which proteins sit near the detection limit, hence the entire missing-value
/// structure that MBR and quant benchmarks actually measure.
///
/// So `Table` (PaxDb, or a real quantification) is the **honest** source: right shape *and* right
/// identities. `LogNormal` and `HockeyStick` give the right marginal with random identities — fine
/// for abstract benchmarks, and reproducible without external data, but they cannot be compared
/// against a real run.
#[derive(Clone, Debug)]
pub enum AbundanceProfile {
    /// Draw a log-normal abundance per protein. `sigma` in natural-log units; σ≈2 gives roughly the
    /// ~7 orders of dynamic range a real proteome spans.
    LogNormal { sigma: f64 },
    /// A **rank-abundance curve**: exponential decay into a linear tail. This is the shape v1 uses
    /// (`get_tenzer_hokey`), and it is a better-shaped marginal than a log-normal.
    ///
    /// Each protein is assigned a rank — **keyed by identity**, unlike v1's
    /// `np.random.uniform(1, 1e4, n)`, which is draw-order dependent and therefore reshuffles every
    /// protein's abundance whenever one is added (a B8 violation).
    HockeyStick { decay: f64, tail: f64 },
    /// Explicit relative abundances (e.g. from PaxDb). Normalised internally; absolute scale is
    /// irrelevant. **The only source that gets identity right.**
    Table(HashMap<String, f64>),
}

/// The rank-abundance curve, evaluated at a rank in (0, 1].
///
/// `exp(-r/decay)` dominates at the head; `tail` keeps the low-abundance end from collapsing to
/// zero, which is what gives the curve its hockey-stick elbow.
fn hockey_stick(rank01: f64, decay: f64, tail: f64) -> f64 {
    (-rank01 / decay).exp() + tail
}

/// Regulation of individual proteins, on top of (or instead of) an organism mixture.
#[derive(Clone, Debug)]
pub enum Regulation {
    /// A named set moved by a known amount — for reproducing a published benchmark.
    Explicit { proteins: Vec<String>, log2fc: f64 },
    /// A random fraction moved by a draw from `N(0, log2fc_sd)` — for power analysis.
    Generative { fraction: f64, log2fc_sd: f64 },
}

#[derive(Clone, Debug)]
pub struct Condition {
    pub name: String,
    /// organism → mass share.
    pub mix: BTreeMap<String, Share>,
    /// Number of **biological** replicates (distinct material → distinct amounts).
    pub replicates: u32,
    /// Injections per biological replicate. Same material, so **same amounts**.
    pub technical_replicates: u32,
    pub regulate: Option<Regulation>,
}

/// How much each protein varies between biological replicates.
///
/// # Every protein has its OWN variance
///
/// A single global CV makes the model homoscedastic in log space, which is wrong: housekeeping
/// proteins are tightly controlled, regulatory ones swing. So the abundance model is
/// **hierarchical** — each protein carries *both* parameters of its own log-normal:
///
/// ```text
///   log μ_i  ~  Normal(m, σ_pop)         the abundance profile (7 orders of dynamic range)
///        σ_i ~  LogNormal(ln cv, spread) THIS protein's variability
///   amount_i,s = μ_i · exp(σ_i · z_i,s)  the per-sample draw
/// ```
///
/// Both `μ_i` and `σ_i` are **identity-keyed by protein**, so they are the same in every sample and
/// every condition — a protein's *variability* is as much a property of the biology as its *mean*.
///
/// # What must NOT go here
///
/// Observed CV rises sharply at low abundance in real data. It is tempting to encode that as a
/// mean–variance relationship on `cv`. **Do not.** Most of that trend is *counting statistics* —
/// fewer ions, CV ∝ 1/√N — which is **technical**, and technical noise belongs on the measurement
/// axis, where it should *emerge* from the ion-counting physics rather than being declared.
///
/// Declare it in both places and you double-count the same effect; and then "how much of this CV
/// is biology and how much is the instrument?" — the one question this design exists to answer —
/// becomes unanswerable.
#[derive(Clone, Copy, Debug, Default)]
pub struct Variance {
    /// **Mean** CV across proteins, between biological replicates.
    pub biological: f64,
    /// Spread of per-protein CVs, in natural-log units. `0` ⇒ every protein shares the mean CV
    /// (the old, homoscedastic behaviour). `~0.5` gives a realistic mix of tightly-controlled and
    /// highly-variable proteins.
    pub biological_heterogeneity: f64,
    /// CV between injections. Declared here, applied on the **measurement** axis — a technical
    /// replicate is the same tube, so its *amounts* are identical. Its abundance dependence should
    /// emerge from ion counting, not from this number.
    pub technical: f64,
}

/// How many proteins are actually *in* the sample.
///
/// # Subsetting is a QUANTITY operation, never a structure one
///
/// v1 subsets by **removing proteins from the proteome** — and that is precisely what made
/// protein-level FDR unanswerable:
///
/// > *"the proteins were chosen from a subset … a reported peptide might map to a protein present
/// > in the full concatenated fasta but not in the subset used for simulation … protein level FDR
/// > cannot be reported."* — the published benchmark suite, in a code comment
///
/// Here, excluded proteins simply get **`amount_amol = 0`**. The structure (digest, peptides,
/// occurrences, degeneracy) remains the **full proteome** — so it stays shared and cacheable — and
/// the ground truth is unambiguous: a protein is in the sample iff its amount is positive. A
/// reported protein with zero amount is a **false positive**, and protein FDR is well-defined again.
///
/// Same lesson as enrichment and detectability: **never prune structure; reweight quantity.**
///
/// There is deliberately **no peptide subsetting**. Peptides do not randomly vanish — they fall
/// below the detection limit, which follows from load × dynamic range × response. A complexity ramp
/// is expressed physically, with `n_proteins` (a simpler sample) or `load_ng` (less material).
#[derive(Clone, Copy, Debug, Default)]
pub struct Complexity {
    /// Number of proteins with a non-zero amount. `None` ⇒ the whole proteome.
    pub n_proteins: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct DesignSpec {
    /// Condition against which `true_log2fc` is computed.
    pub reference: String,
    /// Total peptide mass on column, per run.
    pub load_ng: f64,
    /// How much of the proteome is actually present (§`Complexity`).
    pub complexity: Complexity,
    pub abundance: BTreeMap<String, AbundanceProfile>,
    pub conditions: Vec<Condition>,
    pub variance: Variance,
    pub seed: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Output
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReplicateKind {
    Biological,
    Technical,
}

/// A biological unit. Distinct material, distinct amounts.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub sample_id: String,
    pub condition: String,
    pub replicate: u32,
}

/// One injection, rendering to one file.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    pub run_id: String,
    pub technical_replicate: u32,
    /// Enables run-order and batch-drift modelling on the measurement axis: carryover and
    /// drift couple a run to the runs before it.
    pub injection_order: u32,
}

/// Which samples are in which run, and in what proportion. Many-to-many: fractionation is
/// 1 sample → N runs; TMT is N samples → 1 run.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleRun {
    pub sample_id: String,
    pub run_id: String,
    /// TMT channel; `None` for label-free.
    pub channel: Option<String>,
    pub mix_fraction: f64,
}

/// The quantitative answer key.
#[derive(Clone, Debug, PartialEq)]
pub struct ProteinQuantity {
    pub protein_id: String,
    pub sample_id: String,
    pub amount_amol: f64,
    /// Computed from **final** amounts against the reference condition, so it stays correct
    /// even when regulation shifts the composition.
    pub true_log2fc: f64,
    pub is_regulated: bool,
}

impl OrganismRow {
    /// `"A:0.650 B:0.650"` — the mixture as specified, echoed back so nothing is implicit.
    pub fn mask_fractions_display(&self) -> Vec<String> {
        self.mass_fraction
            .iter()
            .map(|(c, f)| format!("{c}:{f:.3}"))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct OrganismRow {
    pub organism: String,
    pub mass_fraction: BTreeMap<String, f64>,
    pub mean_mw: f64,
    pub true_log2fc: BTreeMap<String, f64>,
}

#[derive(Clone, Debug)]
pub struct DesignReport {
    pub organisms: Vec<OrganismRow>,
    pub load_ng: f64,
    /// Worst absolute mass-balance error across samples, in ng. Must be ~0.
    pub mass_balance_error_ng: f64,
    /// Proteins skipped because they contain non-standard residues (`X`, `B`, `Z`, …). We
    /// cannot compute their mass, and guessing one would corrupt the mass balance silently.
    pub skipped_proteins: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Design {
    pub samples: Vec<Sample>,
    pub runs: Vec<Run>,
    pub sample_runs: Vec<SampleRun>,
    pub protein_quantities: Vec<ProteinQuantity>,
    pub report: DesignReport,
}

// ─────────────────────────────────────────────────────────────────────────────
// Resolution
// ─────────────────────────────────────────────────────────────────────────────

/// One protein as the design sees it.
pub struct DesignProtein<'a> {
    pub id: &'a str,
    pub sequence: &'a str,
    pub organism: &'a str,
}

/// FNV-1a. A *stable* hash — `DefaultHasher` is explicitly not guaranteed across Rust
/// releases, and a reproducibility guarantee that depends on the compiler version is not a
/// guarantee.
///
/// FNV alone is **not** used to produce random variates: its avalanche is weak, so two FNV
/// hashes of the same input under nearby seeds are correlated. Always pass the output
/// through [`splitmix64`] before turning it into a uniform. See [`gauss`].
fn stable_hash(parts: &[&str], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for p in parts {
        for &b in p.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// SplitMix64 finaliser — a strong bit-mixing function with full avalanche.
///
/// This is what makes two variates drawn from one hash independent. Without it, deriving
/// `u1` and `u2` from FNV hashes of the same input under nearby seeds gives *correlated*
/// uniforms, and Box–Muller fed correlated uniforms does not produce a Gaussian. That would
/// quietly corrupt every protein abundance and every replicate's biological variance while
/// every existing test still passed.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform in **(0, 1]** — open at zero by construction, so `ln(u)` is always finite.
///
/// The `+ 1` matters: mapping to `[0, 1)` and then clamping a zero to `f64::MIN_POSITIVE`
/// (as an earlier version did) yields `sqrt(-2·ln(1e-308)) ≈ 37`, injecting a 37-sigma
/// outlier into a protein's abundance whenever a hash happens to have zero high bits.
#[inline]
fn uniform_01(h: u64) -> f64 {
    ((h >> 11) + 1) as f64 / ((1u64 << 53) + 1) as f64
}

/// Standard normal via Box–Muller. Deterministic and keyed by *identity*, not by draw order
/// — so adding a protein or a replicate does not reshuffle the others, and condition A and
/// condition B rederive the same value for the same entity without storing anything.
///
/// Public so every stage draws from the SAME generator. A second, subtly different RNG elsewhere in
/// the codebase is how reproducibility quietly dies.
pub fn standard_normal(parts: &[&str], seed: u64) -> f64 {
    gauss(parts, seed)
}

fn gauss(parts: &[&str], seed: u64) -> f64 {
    let h = stable_hash(parts, seed);
    let u1 = uniform_01(splitmix64(h));
    let u2 = uniform_01(splitmix64(h ^ 0xD1B5_4A32_D192_ED03));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

pub fn resolve(spec: &DesignSpec, proteins: &[DesignProtein]) -> Result<Design, String> {
    // A zero, negative, NaN or infinite load divides by zero in the normalisation below and
    // writes NaN or negative attomole amounts into the answer key — without an error. There is
    // no such thing as a negative nanogram.
    // All three are coefficients of variation. `technical` is not *applied* here — a technical
    // replicate is the same tube, so its amounts are identical and its variation is a measurement —
    // but it IS stamped into samples.parquet for the measurement stage, so an unusable value must be
    // caught here rather than propagated as metadata.
    for (name, v) in [
        ("variance.biological", spec.variance.biological),
        ("variance.biological_heterogeneity", spec.variance.biological_heterogeneity),
        ("variance.technical", spec.variance.technical),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(format!("{name} must be finite and non-negative, got {v}"));
        }
    }
    if !spec.load_ng.is_finite() || spec.load_ng <= 0.0 {
        return Err(format!(
            "load_ng must be finite and strictly positive, got {}",
            spec.load_ng
        ));
    }

    // ── masses; refuse (and report) proteins we cannot weigh ──────────────────
    let mut skipped = Vec::new();
    let mut mw: HashMap<&str, f64> = HashMap::new();
    for p in proteins {
        match mass::average(p.sequence) {
            Ok(m) => {
                mw.insert(p.id, m);
            }
            Err(_) => skipped.push(p.id.to_string()),
        }
    }
    let usable: Vec<&DesignProtein> = proteins.iter().filter(|p| mw.contains_key(p.id)).collect();
    if usable.is_empty() {
        return Err("no proteins with computable mass".to_string());
    }

    // ── which proteins are actually IN the sample ─────────────────────────────
    //
    // Subsetting is a quantity operation. Excluded proteins stay in the structure and receive
    // amount 0, so the degeneracy map — and therefore protein-level FDR — survives intact.
    //
    // The selection is made PER ORGANISM, and that is not a detail. A global hash draw can exclude
    // every protein of a small organism (E. coli is 4,401 of HYE's 30,881), which would give that
    // organism zero mass — while the load normalisation quietly redistributed its declared share to
    // the others and the report still printed the mixture you asked for. The fold changes would be
    // wrong and nothing would say so. Subsetting within each organism keeps every declared mass
    // share intact.
    let mut org_members: BTreeMap<&str, Vec<&DesignProtein>> = BTreeMap::new();
    for p in &usable {
        org_members.entry(p.organism).or_default().push(p);
    }

    let present: std::collections::HashSet<&str> = match spec.complexity.n_proteins {
        None => usable.iter().map(|p| p.id).collect(),
        Some(n) => {
            if n == 0 {
                return Err("complexity.n_proteins must be at least 1".to_string());
            }
            if n > usable.len() {
                return Err(format!(
                    "complexity.n_proteins ({n}) exceeds the {} proteins with computable mass — the \
                     design would silently contain fewer proteins than it declares",
                    usable.len()
                ));
            }
            if n < org_members.len() {
                return Err(format!(
                    "complexity.n_proteins ({n}) is below the number of organisms ({}) — every \
                     organism with a declared mass share needs at least one protein, or its share \
                     would be silently redistributed to the others",
                    org_members.len()
                ));
            }

            // Allocate proportionally to organism size, at least one each, summing to exactly n.
            let total = usable.len() as f64;
            let mut alloc: Vec<(&str, usize)> = org_members
                .iter()
                .map(|(org, m)| {
                    let want = (n as f64 * m.len() as f64 / total).round() as usize;
                    (*org, want.max(1).min(m.len()))
                })
                .collect();

            // Reconcile rounding against the requested total.
            let n_orgs = alloc.len();
            let caps: Vec<usize> = alloc.iter().map(|(org, _)| org_members[*org].len()).collect();
            let mut got: usize = alloc.iter().map(|(_, k)| *k).sum();
            let mut idx = 0usize;
            while got != n && n_orgs > 0 {
                let i = idx % n_orgs;
                let k = &mut alloc[i].1;
                if got < n && *k < caps[i] {
                    *k += 1;
                    got += 1;
                } else if got > n && *k > 1 {
                    *k -= 1;
                    got -= 1;
                }
                idx += 1;
                if idx > 16 * n_orgs {
                    break; // every organism is capped; take what we can reach
                }
            }

            let mut chosen = std::collections::HashSet::new();
            for (org, k) in alloc {
                // Identity-keyed within the organism: raising n yields a SUPERSET, and adding a
                // protein to the FASTA does not reshuffle which of the others were selected.
                let mut ranked: Vec<(u64, &str)> = org_members[org]
                    .iter()
                    .map(|p| (stable_hash(&["present", p.id], spec.seed), p.id))
                    .collect();
                ranked.sort_unstable();
                chosen.extend(ranked.into_iter().take(k).map(|(_, id)| id));
            }
            chosen
        }
    };

    // ── within-organism molar abundance, normalised over the PRESENT proteins ─
    //
    // Normalising over *all* proteins of an organism and then zeroing the absent ones would leave
    // each organism holding only a random fraction of its declared mass — and a different fraction
    // per organism. The global load renormalisation cannot restore the ratios, so the mixture drifts
    // and the fold changes come out wrong. Normalising over what is actually in the sample keeps
    // every organism's declared mass share exact.
    let mut by_organism: BTreeMap<&str, Vec<&DesignProtein>> = BTreeMap::new();
    for p in &usable {
        if present.contains(p.id) {
            by_organism.entry(p.organism).or_default().push(p);
        }
    }
    if by_organism.is_empty() {
        return Err("no proteins present in the sample".to_string());
    }

    let mut molar_fraction: HashMap<&str, f64> = HashMap::new();
    let mut mean_mw: BTreeMap<String, f64> = BTreeMap::new();
    for (org, members) in &by_organism {
        let profile = spec.abundance.get(*org).ok_or_else(|| {
            format!("no abundance profile for organism {org:?}")
        })?;

        let raw: Vec<f64> = members
            .iter()
            .map(|p| match profile {
                AbundanceProfile::LogNormal { sigma } => {
                    (sigma * gauss(&["abundance", p.id], spec.seed)).exp()
                }
                AbundanceProfile::HockeyStick { decay, tail } => {
                    // Rank keyed by IDENTITY, so adding a protein does not reshuffle the others.
                    let r = uniform_01(splitmix64(stable_hash(&["rank", p.id], spec.seed)));
                    hockey_stick(r, *decay, *tail)
                }
                AbundanceProfile::Table(t) => t.get(p.id).copied().unwrap_or(0.0),
            })
            .collect();

        // Validate BEFORE normalising. A table containing `{A: 2.0, B: -1.0}` has a positive total,
        // so a `total <= 0.0` check passes it — and B then gets a negative molar fraction and a
        // negative molar amount. NaN slips through both comparisons the same way. A negative amount
        // of protein is not a thing.
        for (p, r) in members.iter().zip(&raw) {
            if !r.is_finite() || *r < 0.0 {
                return Err(format!(
                    "abundance for {:?} (organism {org:?}) must be finite and non-negative, got {r}",
                    p.id
                ));
            }
        }

        let total: f64 = raw.iter().sum();
        if total <= 0.0 {
            return Err(format!("organism {org:?} has zero total abundance"));
        }
        for (p, r) in members.iter().zip(&raw) {
            molar_fraction.insert(p.id, r / total);
        }

        // <M>_org — the molar-weighted mean MW *of what is actually in the sample*. This converts a
        // mass share into a molar amount. It cancels out of the fold change (see the module docs),
        // but not out of the absolute amounts: E. coli's smaller proteins mean a given mass
        // contributes more molecules, which shifts what sits above the LOD.
        let m: f64 = members
            .iter()
            .zip(&raw)
            .map(|(p, r)| (r / total) * mw[p.id])
            .sum();
        mean_mw.insert((*org).to_string(), m);
    }

    // ── resolve mass shares per condition ─────────────────────────────────────
    //
    // Condition names key `shares`, `condition_amounts` and `regulated`, so a duplicate name
    // would silently overwrite the first definition — while the sample loop, which iterates
    // the *vector*, still emitted samples and runs for BOTH, attaching the overwritten
    // amounts. Ambiguous artifacts, no error.
    {
        let mut seen = std::collections::HashSet::new();
        for c in &spec.conditions {
            if !seen.insert(c.name.as_str()) {
                return Err(format!("duplicate condition name {:?}", c.name));
            }
            // A cardinality the user typed is a cardinality we must honour or refuse. Zero
            // biological replicates silently produced no samples at all; zero technical
            // replicates was silently rounded up to one by a `.max(1)`. Either way the artifacts
            // disagree with the design that was asked for.
            if c.replicates == 0 {
                return Err(format!(
                    "condition {:?}: replicates must be at least 1",
                    c.name
                ));
            }
            if c.technical_replicates == 0 {
                return Err(format!(
                    "condition {:?}: technical_replicates must be at least 1",
                    c.name
                ));
            }
        }
    }

    // Regulation is part of the answer key, so a typo in it must not silently produce a design
    // with no regulation at all.
    {
        let known: std::collections::HashSet<&str> = usable.iter().map(|p| p.id).collect();
        for c in &spec.conditions {
            match &c.regulate {
                Some(Regulation::Explicit { proteins, log2fc }) => {
                    if !log2fc.is_finite() {
                        return Err(format!("condition {:?}: log2fc must be finite", c.name));
                    }
                    for id in proteins {
                        if !known.contains(id.as_str()) {
                            return Err(format!(
                                "condition {:?}: regulated protein {id:?} is not in the proteome \
                                 — a stale accession would silently apply no regulation at all",
                                c.name
                            ));
                        }
                    }
                }
                Some(Regulation::Generative { fraction, log2fc_sd }) => {
                    if !fraction.is_finite() || !(0.0..=1.0).contains(fraction) {
                        return Err(format!(
                            "condition {:?}: regulated fraction must be finite and in [0, 1], got {fraction}",
                            c.name
                        ));
                    }
                    if !log2fc_sd.is_finite() || *log2fc_sd < 0.0 {
                        return Err(format!(
                            "condition {:?}: log2fc_sd must be finite and non-negative, got {log2fc_sd}",
                            c.name
                        ));
                    }
                }
                None => {}
            }
        }
    }

    let organisms: Vec<String> = by_organism.keys().map(|o| o.to_string()).collect();
    let mut shares: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    for c in &spec.conditions {
        let mut resolved = BTreeMap::new();

        // Validate each share individually first. Summing alone is not enough: a mixture of
        // `Fraction(-0.1)` and `Fraction(1.1)` sums to exactly 1.0 and sails through, then
        // produces a NEGATIVE mass share and negative protein amounts. `Fraction(NaN)` evades
        // both `>` and `<` comparisons entirely and propagates NaN into the answer key.
        for (org, share) in &c.mix {
            if let Share::Fraction(f) = share {
                if !f.is_finite() || !(0.0..=1.0).contains(f) {
                    return Err(format!(
                        "condition {:?}: mass fraction for {org:?} must be finite and in [0, 1], got {f}",
                        c.name
                    ));
                }
            }
        }

        let named: f64 = c
            .mix
            .values()
            .filter_map(|s| match s {
                Share::Fraction(f) => Some(*f),
                Share::Rest => None,
            })
            .sum();
        let n_rest = c.mix.values().filter(|s| matches!(s, Share::Rest)).count();

        if named > 1.0 + 1e-9 {
            return Err(format!(
                "condition {:?}: mass fractions sum to {named} (> 1)",
                c.name
            ));
        }
        if n_rest == 0 && (named - 1.0).abs() > 1e-9 {
            return Err(format!(
                "condition {:?}: mass fractions sum to {named}, not 1 (use `rest` to fill)",
                c.name
            ));
        }
        for (org, share) in &c.mix {
            if !organisms.contains(org) {
                return Err(format!("condition {:?}: unknown organism {org:?}", c.name));
            }
            let f = match share {
                Share::Fraction(f) => *f,
                Share::Rest => (1.0 - named) / n_rest as f64,
            };
            resolved.insert(org.clone(), f);
        }
        shares.insert(c.name.clone(), resolved);
    }
    if !shares.contains_key(&spec.reference) {
        return Err(format!(
            "reference condition {:?} is not among the conditions",
            spec.reference
        ));
    }

    // ── per-condition protein amounts, before replicate variance ──────────────
    //
    // amount_i = total_amol(org) · f_i,  where  total_amol(org) = mass_ng(org) · 1e9 / <M>_org
    let mut condition_amounts: BTreeMap<String, HashMap<&str, f64>> = BTreeMap::new();
    let mut regulated: BTreeMap<String, HashMap<&str, bool>> = BTreeMap::new();

    for c in &spec.conditions {
        // Every usable protein appears in the answer key. Absent ones carry amount 0 — which is what
        // makes a search engine's hit on them a nameable false positive, and protein FDR answerable.
        let mut amounts: HashMap<&str, f64> = HashMap::new();
        let mut reg: HashMap<&str, bool> = HashMap::new();
        for p in &usable {
            if !present.contains(p.id) {
                amounts.insert(p.id, 0.0);
                reg.insert(p.id, false);
            }
        }

        for (org, members) in &by_organism {
            let share = shares[&c.name].get(*org).copied().unwrap_or(0.0);
            let mass_ng = share * spec.load_ng;
            let total_amol = mass::ng_to_amol(mass_ng, mean_mw[*org]);
            for p in members {
                // Absent from the sample ⇒ amount 0. It stays in the structure, and in the answer
                // key, so a search engine reporting it is a false positive we can name.
                let amt = if present.contains(p.id) {
                    total_amol * molar_fraction[p.id]
                } else {
                    0.0
                };
                amounts.insert(p.id, amt);
                reg.insert(p.id, false);
            }
        }

        // Regulation on top of the mixture.
        if let Some(r) = &c.regulate {
            for p in &usable {
                let fc = match r {
                    Regulation::Explicit { proteins, log2fc } => {
                        if proteins.iter().any(|x| x == p.id) {
                            Some(*log2fc)
                        } else {
                            None
                        }
                    }
                    Regulation::Generative {
                        fraction,
                        log2fc_sd,
                    } => {
                        let u = (stable_hash(&["regulate", p.id], spec.seed) >> 11) as f64
                            / (1u64 << 53) as f64;
                        if u < *fraction {
                            Some(log2fc_sd * gauss(&["regulate_fc", p.id, &c.name], spec.seed))
                        } else {
                            None
                        }
                    }
                };
                if let Some(fc) = fc {
                    // `log2fc = 1024` makes 2^fc infinite; the load normalisation then computes an
                    // infinite total and writes NaN amounts (inf * 0) into the answer key. A fold
                    // change is a biological quantity, not an arbitrary float.
                    if !fc.is_finite() || fc.abs() > MAX_ABS_LOG2FC {
                        return Err(format!(
                            "condition {:?}: log2 fold change for {:?} must be finite and within \
                             ±{MAX_ABS_LOG2FC}, got {fc}",
                            c.name, p.id
                        ));
                    }
                    // A protein absent from the sample cannot be regulated. Absent proteins ARE in
                    // `amounts` — with value 0, because subsetting is a quantity operation and they
                    // stay in the answer key — so a bare `get_mut` is not enough: multiplying 0 by a
                    // fold change is still 0, and flagging it `is_regulated` would make the ground
                    // truth claim a molecule was regulated when it is not in the sample at all.
                    if let Some(a) = amounts.get_mut(p.id) {
                        if *a > 0.0 {
                            *a *= 2f64.powf(fc);
                            reg.insert(p.id, true);
                        }
                    }
                }
            }
        }

        // Renormalise to the requested load. This is what happens at the bench — you inject
        // 200 ng whatever the composition — and it is why `true_log2fc` must be computed
        // from FINAL amounts: a large spike-in compositionally dilutes everything else, and
        // that dilution is real and must appear in the answer key.
        let total_ng: f64 = amounts
            .iter()
            .map(|(id, a)| mass::amol_to_ng(*a, mw[*id]))
            .sum();
        if !total_ng.is_finite() || total_ng <= 0.0 {
            return Err(format!(
                "condition {:?}: total mass is {total_ng} ng after regulation — cannot normalise \
                 to the requested load",
                c.name
            ));
        }
        let scale = spec.load_ng / total_ng;
        for a in amounts.values_mut() {
            *a *= scale;
        }

        condition_amounts.insert(c.name.clone(), amounts);
        regulated.insert(c.name.clone(), reg);
    }

    // ── samples, runs, and the mapping ────────────────────────────────────────
    let mut samples = Vec::new();
    let mut runs = Vec::new();
    let mut sample_runs = Vec::new();
    let mut protein_quantities = Vec::new();
    let mut injection_order = 0u32;
    let mut worst_error = 0.0f64;

    let reference = &condition_amounts[&spec.reference];

    for c in &spec.conditions {
        for bio in 1..=c.replicates {
            let sample_id = format!("{}_R{bio}", c.name);
            samples.push(Sample {
                sample_id: sample_id.clone(),
                condition: c.name.clone(),
                replicate: bio,
            });

            // Biological variance: distinct material, so the amounts genuinely differ.
            let base = &condition_amounts[&c.name];
            let mut amounts: HashMap<&str, f64> = HashMap::new();

            let mean_cv = spec.variance.biological;
            let het = spec.variance.biological_heterogeneity;

            for p in &usable {
                let b = base.get(p.id).copied().unwrap_or(0.0);
                if b <= 0.0 {
                    amounts.insert(p.id, 0.0); // absent from the sample; stays in the answer key
                    continue;
                }

                // Each protein's OWN CV, drawn once and keyed by identity — so it is the same in
                // every sample and every condition, exactly like its mean abundance.
                let cv_i = if mean_cv > 0.0 && het > 0.0 {
                    mean_cv * (het * gauss(&["cv", p.id], spec.seed) - 0.5 * het * het).exp()
                } else {
                    mean_cv
                };

                // The user declares a **coefficient of variation**, not a log-space sigma. A
                // log-normal with log-sigma `s` has CV sqrt(exp(s^2) - 1), so feeding the CV in
                // directly gives the wrong spread (a declared 0.5 realises as 0.533).
                //     s = sqrt(ln(1 + CV^2))
                let log_sigma = if cv_i > 0.0 {
                    (1.0 + cv_i * cv_i).ln().sqrt()
                } else {
                    0.0
                };

                let jitter = if log_sigma > 0.0 {
                    (log_sigma * gauss(&["bio", p.id, &sample_id], spec.seed)).exp()
                } else {
                    1.0
                };
                amounts.insert(p.id, b * jitter);
            }

            // Variance perturbs composition, so restore the load. Mass balance is exact.
            let total_ng: f64 = amounts
                .iter()
                .map(|(id, a)| mass::amol_to_ng(*a, mw[*id]))
                .sum();
            let scale = spec.load_ng / total_ng;
            for a in amounts.values_mut() {
                *a *= scale;
            }

            let check_ng: f64 = amounts
                .iter()
                .map(|(id, a)| mass::amol_to_ng(*a, mw[*id]))
                .sum();
            worst_error = worst_error.max((check_ng - spec.load_ng).abs());

            for p in &usable {
                let a = amounts[p.id];
                let r = reference[p.id];
                protein_quantities.push(ProteinQuantity {
                    protein_id: p.id.to_string(),
                    sample_id: sample_id.clone(),
                    amount_amol: a,
                    // Absent from the sample in either condition ⇒ the fold change is undefined,
                    // not zero. NaN is the honest value, and consumers must handle it.
                    true_log2fc: if r > 0.0 && a > 0.0 {
                        (a / r).log2()
                    } else {
                        f64::NAN
                    },
                    is_regulated: regulated[&c.name][p.id],
                });
            }

            // Technical replicates: the SAME tube, injected again. Same sample, new run,
            // identical amounts. All variation lives on the measurement axis.
            for tech in 1..=c.technical_replicates {
                let run_id = format!("{sample_id}_T{tech}");
                runs.push(Run {
                    run_id: run_id.clone(),
                    technical_replicate: tech,
                    injection_order,
                });
                sample_runs.push(SampleRun {
                    sample_id: sample_id.clone(),
                    run_id,
                    channel: None,
                    mix_fraction: 1.0,
                });
                injection_order += 1;
            }
        }
    }

    // ── report ────────────────────────────────────────────────────────────────
    let organisms_report = by_organism
        .keys()
        .map(|org| {
            let mass_fraction: BTreeMap<String, f64> = shares
                .iter()
                .map(|(c, s)| (c.clone(), s.get(*org).copied().unwrap_or(0.0)))
                .collect();
            let ref_share = mass_fraction[&spec.reference];
            let true_log2fc = mass_fraction
                .iter()
                .map(|(c, f)| {
                    (
                        c.clone(),
                        if ref_share > 0.0 && *f > 0.0 {
                            (f / ref_share).log2()
                        } else {
                            f64::NAN
                        },
                    )
                })
                .collect();
            OrganismRow {
                organism: (*org).to_string(),
                mass_fraction,
                mean_mw: mean_mw[*org],
                true_log2fc,
            }
        })
        .collect();

    Ok(Design {
        samples,
        runs,
        sample_runs,
        protein_quantities,
        report: DesignReport {
            organisms: organisms_report,
            load_ng: spec.load_ng,
            mass_balance_error_ng: worst_error,
            skipped_proteins: skipped,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn hye_proteins() -> Vec<(String, String, String)> {
        let mut v = Vec::new();
        // Different mean sizes per organism, deliberately: human proteins larger than
        // E. coli, which is what makes mass-vs-molar non-trivial.
        for i in 0..40 {
            v.push((format!("HUM{i}"), "PEPTIDEKAAAKCCCRDDDK".repeat(6), "HUMAN".into()));
        }
        for i in 0..30 {
            v.push((format!("YST{i}"), "PEPTIDEKAAAKCCCRDDDK".repeat(7), "YEAST".into()));
        }
        for i in 0..20 {
            v.push((format!("ECO{i}"), "PEPTIDEKAAAKCCCRDDDK".repeat(3), "ECOLI".into()));
        }
        v
    }

    fn spec(load_ng: f64, bio_cv: f64) -> DesignSpec {
        let mut abundance = BTreeMap::new();
        for org in ["HUMAN", "YEAST", "ECOLI"] {
            abundance.insert(org.to_string(), AbundanceProfile::LogNormal { sigma: 2.0 });
        }
        DesignSpec {
            reference: "A".into(),
            load_ng,
            complexity: Complexity::default(),
            abundance,
            variance: Variance {
                biological: bio_cv,
                biological_heterogeneity: 0.0, // most tests want the homoscedastic case
                technical: 0.05,
            },
            seed: 42,
            conditions: vec![
                Condition {
                    name: "A".into(),
                    mix: [
                        ("HUMAN".to_string(), Share::Fraction(0.65)),
                        ("YEAST".to_string(), Share::Fraction(0.30)),
                        ("ECOLI".to_string(), Share::Fraction(0.05)),
                    ]
                    .into(),
                    replicates: 3,
                    technical_replicates: 2,
                    regulate: None,
                },
                Condition {
                    name: "B".into(),
                    mix: [
                        ("HUMAN".to_string(), Share::Fraction(0.65)),
                        ("YEAST".to_string(), Share::Fraction(0.15)),
                        ("ECOLI".to_string(), Share::Rest), // → 0.20
                    ]
                    .into(),
                    replicates: 3,
                    technical_replicates: 2,
                    regulate: None,
                },
            ],
        }
    }

    fn run(spec: &DesignSpec, ps: &[(String, String, String)]) -> Design {
        let dp: Vec<DesignProtein> = ps
            .iter()
            .map(|(id, seq, org)| DesignProtein {
                id,
                sequence: seq,
                organism: org,
            })
            .collect();
        resolve(spec, &dp).unwrap()
    }

    /// THE invariant. Every sample must contain exactly the requested nanograms.
    #[test]
    fn mass_balance_is_exact_for_every_sample() {
        let ps = hye_proteins();
        for load in [50.0, 200.0, 1000.0] {
            let d = run(&spec(load, 0.15), &ps);
            assert!(
                d.report.mass_balance_error_ng < 1e-6,
                "load {load} ng: worst mass-balance error was {} ng",
                d.report.mass_balance_error_ng
            );
        }
    }

    /// `rest` fills the mixture so it cannot be mistyped into not summing to 1.
    #[test]
    fn rest_fills_the_mixture() {
        let d = run(&spec(200.0, 0.0), &hye_proteins());
        let ecoli = d.report.organisms.iter().find(|o| o.organism == "ECOLI").unwrap();
        assert_relative_eq!(ecoli.mass_fraction["B"], 0.20, epsilon = 1e-12);
    }

    /// The fold change is DERIVED from the mixture — not typed into a plotting cell.
    /// Classic HYE: human flat, yeast down 2×, E. coli up 4×.
    #[test]
    fn hye_fold_changes_follow_from_the_mixture() {
        let d = run(&spec(200.0, 0.0), &hye_proteins());
        let fc = |org: &str| {
            d.report
                .organisms
                .iter()
                .find(|o| o.organism == org)
                .unwrap()
                .true_log2fc["B"]
        };
        assert_relative_eq!(fc("HUMAN"), 0.0, epsilon = 1e-9); // 0.65 → 0.65
        assert_relative_eq!(fc("YEAST"), -1.0, epsilon = 1e-9); // 0.30 → 0.15
        assert_relative_eq!(fc("ECOLI"), 2.0, epsilon = 1e-9); // 0.05 → 0.20
    }

    /// Mean MW cancels out of the fold change — so the per-protein truth matches the
    /// per-organism mass ratio *exactly*, even though the three organisms have very
    /// different mean protein sizes.
    #[test]
    fn mean_mw_cancels_from_the_fold_change_but_not_from_the_amounts() {
        let ps = hye_proteins();
        let d = run(&spec(200.0, 0.0), &ps); // no biological variance

        let mws: Vec<f64> = d
            .report
            .organisms
            .iter()
            .map(|o| o.mean_mw)
            .collect();
        assert!(
            mws.windows(2).any(|w| (w[0] - w[1]).abs() > 1000.0),
            "test proteins must differ in mean MW, else this proves nothing"
        );

        // Per-protein truth for a yeast protein in a B replicate == the organism mass ratio.
        let q = d
            .protein_quantities
            .iter()
            .find(|q| q.protein_id == "YST0" && q.sample_id == "B_R1")
            .unwrap();
        assert_relative_eq!(q.true_log2fc, -1.0, epsilon = 1e-9);
    }

    /// A technical replicate is the same tube injected twice: one Sample, two Runs,
    /// identical amounts. This is why Sample and Run are separate entities.
    #[test]
    fn technical_replicates_share_a_sample_and_therefore_its_amounts() {
        let d = run(&spec(200.0, 0.15), &hye_proteins());

        assert_eq!(d.samples.len(), 6, "2 conditions × 3 biological replicates");
        assert_eq!(d.runs.len(), 12, "× 2 technical replicates = 12 injections");

        let for_sample: Vec<_> = d
            .sample_runs
            .iter()
            .filter(|sr| sr.sample_id == "A_R1")
            .collect();
        assert_eq!(for_sample.len(), 2, "one sample, two runs");

        // Exactly one amount row per (protein, sample) — never per run.
        let rows = d
            .protein_quantities
            .iter()
            .filter(|q| q.sample_id == "A_R1" && q.protein_id == "HUM0")
            .count();
        assert_eq!(rows, 1);
    }

    /// Biological replicates are different material, so their amounts must differ — but the
    /// load is still exactly conserved.
    #[test]
    fn biological_replicates_differ_but_conserve_the_load() {
        let d = run(&spec(200.0, 0.20), &hye_proteins());
        let amount = |sample: &str| {
            d.protein_quantities
                .iter()
                .find(|q| q.sample_id == sample && q.protein_id == "HUM0")
                .unwrap()
                .amount_amol
        };
        assert!((amount("A_R1") - amount("A_R2")).abs() > 0.0);
        assert!(d.report.mass_balance_error_ng < 1e-6);
    }

    /// Zero biological variance ⇒ replicates are identical. Guards the variance path.
    #[test]
    fn zero_variance_makes_replicates_identical() {
        let d = run(&spec(200.0, 0.0), &hye_proteins());
        let amount = |s: &str| {
            d.protein_quantities
                .iter()
                .find(|q| q.sample_id == s && q.protein_id == "HUM0")
                .unwrap()
                .amount_amol
        };
        assert_relative_eq!(amount("A_R1"), amount("A_R2"), max_relative = 1e-12);
    }

    /// Regulation is applied, then the load is restored — so a large spike-in compositionally
    /// dilutes everything else. That dilution is REAL, and because `true_log2fc` is computed
    /// from final amounts, the answer key records it instead of pretending it away.
    #[test]
    fn regulation_dilutes_the_background_and_the_truth_records_it() {
        let ps = hye_proteins();
        let mut s = spec(200.0, 0.0);
        s.conditions[1].regulate = Some(Regulation::Explicit {
            proteins: (0..20).map(|i| format!("HUM{i}")).collect(),
            log2fc: 3.0, // 8× on half the human proteome — a big compositional shift
        });
        let d = run(&s, &ps);

        let fc = |p: &str| {
            d.protein_quantities
                .iter()
                .find(|q| q.protein_id == p && q.sample_id == "B_R1")
                .unwrap()
        };

        // The spiked proteins go up, but by LESS than the nominal 3.0 — the load is fixed,
        // so they crowd out the rest.
        let up = fc("HUM0");
        assert!(up.is_regulated);
        assert!(up.true_log2fc > 0.0 && up.true_log2fc < 3.0, "got {}", up.true_log2fc);

        // An unregulated background protein is pushed DOWN even though nothing was done to
        // it. This is compositional bias, and it is exactly what happens at the bench.
        let down = fc("YST0");
        assert!(!down.is_regulated);
        assert!(down.true_log2fc < -1.0, "yeast should be diluted below its -1.0 mix ratio, got {}", down.true_log2fc);

        assert!(d.report.mass_balance_error_ng < 1e-6);
    }

    /// Seeds derive from identity, not draw order: adding a fourth replicate must not
    /// perturb the first three.
    #[test]
    fn adding_a_replicate_does_not_reshuffle_the_others() {
        let ps = hye_proteins();
        let d3 = run(&spec(200.0, 0.2), &ps);

        let mut s4 = spec(200.0, 0.2);
        s4.conditions[0].replicates = 4;
        let d4 = run(&s4, &ps);

        let get = |d: &Design, s: &str| {
            d.protein_quantities
                .iter()
                .find(|q| q.sample_id == s && q.protein_id == "HUM0")
                .unwrap()
                .amount_amol
        };
        // NOTE: not bit-identical, because the load renormalisation is over a different
        // protein set only if proteins change — they don't — so this must be exact.
        assert_relative_eq!(get(&d3, "A_R1"), get(&d4, "A_R1"), max_relative = 1e-12);
        assert_relative_eq!(get(&d3, "A_R2"), get(&d4, "A_R2"), max_relative = 1e-12);
    }

    /// Proteins we cannot weigh are reported, never silently guessed.
    #[test]
    fn unweighable_proteins_are_skipped_and_reported() {
        let mut ps = hye_proteins();
        ps.push(("BAD1".into(), "PEPTXDEK".into(), "HUMAN".into()));
        let d = run(&spec(200.0, 0.0), &ps);
        assert_eq!(d.report.skipped_proteins, vec!["BAD1".to_string()]);
        assert!(d.report.mass_balance_error_ng < 1e-6);
    }

    /// STATISTICAL ORACLE for `gauss`.
    ///
    /// A generator that is subtly non-Gaussian passes every other test in this file — the
    /// mass balance still holds, the fold changes are still right, the replicates still
    /// differ. It would only show up as wrong dynamic range and wrong replicate CVs, i.e.
    /// as *bad science that runs cleanly*.
    ///
    /// **Excess kurtosis is the load-bearing assertion.** Box–Muller fed *correlated*
    /// uniforms still produces something centred and roughly unit-variance; what it gets
    /// wrong is the tails. Mean and standard deviation alone would not have caught the FNV
    /// correlation bug that this test was written to guard.
    #[test]
    fn gauss_is_actually_gaussian() {
        let n = 200_000;
        let xs: Vec<f64> = (0..n)
            .map(|i| gauss(&["abundance", &format!("P{i}")], 7))
            .collect();

        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let sd = var.sqrt();
        let m3 = xs.iter().map(|x| ((x - mean) / sd).powi(3)).sum::<f64>() / n as f64;
        let m4 = xs.iter().map(|x| ((x - mean) / sd).powi(4)).sum::<f64>() / n as f64;

        // Thresholds are calibrated to FAIL the implementation this test was written to
        // catch — raw FNV under Box–Muller measures sd 1.0076 / excess kurtosis +0.0912 at
        // this n, versus 0.9995 / +0.0053 for the splitmix-finalised version. A guard that
        // cannot catch its own bug is theatre.
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((sd - 1.0).abs() < 0.004, "sd {sd}");
        assert!(m3.abs() < 0.05, "skew {m3}");
        assert!((m4 - 3.0).abs() < 0.03, "excess kurtosis {}", m4 - 3.0);

        // Empirical coverage against the normal CDF.
        let within = |k: f64| xs.iter().filter(|x| x.abs() < k).count() as f64 / n as f64;
        assert!((within(1.0) - 0.6827).abs() < 0.01, "1σ {}", within(1.0));
        assert!((within(2.0) - 0.9545).abs() < 0.01, "2σ {}", within(2.0));
        assert!((within(3.0) - 0.9973).abs() < 0.005, "3σ {}", within(3.0));

        // No absurd outliers. The old `MIN_POSITIVE` clamp could emit ~37σ.
        let max = xs.iter().fold(0.0f64, |a, x| a.max(x.abs()));
        assert!(max < 6.0, "largest |z| was {max} — a clamp is leaking pathological values");
    }

    /// Identity-keyed, not order-keyed: the value drawn for an entity depends only on its
    /// id and the seed, so conditions A and B rederive the same flyability/abundance for the
    /// same protein without storing anything, and inserting new entities perturbs nothing.
    #[test]
    fn gauss_is_keyed_by_identity_not_draw_order() {
        let a = gauss(&["abundance", "P42"], 7);
        // Draw a thousand unrelated values in between.
        for i in 0..1000 {
            let _ = gauss(&["abundance", &format!("Q{i}")], 7);
        }
        let b = gauss(&["abundance", "P42"], 7);
        assert_relative_eq!(a, b, max_relative = 1e-15);

        // Different seed ⇒ different value; different id ⇒ different value.
        assert!((gauss(&["abundance", "P42"], 8) - a).abs() > 1e-9);
        assert!((gauss(&["abundance", "P43"], 7) - a).abs() > 1e-9);
    }

    /// REGRESSION: a negative or NaN mass share must be rejected.
    ///
    /// Summing alone is not enough. `Fraction(-0.1)` + `Fraction(1.1)` sums to exactly 1.0
    /// and sailed through both bounds checks, producing a **negative mass share** and hence
    /// negative protein amounts. `Fraction(NaN)` evades `>` and `<` entirely and propagates
    /// NaN straight into the answer key. Found by review.
    #[test]
    fn negative_or_nan_mass_shares_are_rejected() {
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps
            .iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org })
            .collect();

        // Sums to exactly 1.0, but one share is negative.
        let mut s = spec(200.0, 0.0);
        s.conditions[0].mix = [
            ("HUMAN".to_string(), Share::Fraction(-0.1)),
            ("YEAST".to_string(), Share::Fraction(1.1)),
            ("ECOLI".to_string(), Share::Fraction(0.0)),
        ]
        .into();
        assert!(resolve(&s, &dp).is_err(), "a negative mass share must not be accepted");

        // NaN evades every comparison.
        let mut s = spec(200.0, 0.0);
        s.conditions[0].mix = [
            ("HUMAN".to_string(), Share::Fraction(f64::NAN)),
            ("YEAST".to_string(), Share::Fraction(0.5)),
            ("ECOLI".to_string(), Share::Rest),
        ]
        .into();
        assert!(resolve(&s, &dp).is_err(), "a NaN mass share must not be accepted");
    }

    /// REGRESSION: a negative or NaN abundance in a table must be rejected.
    ///
    /// `{A: 2.0, B: -1.0}` has a positive total, so the `total <= 0.0` guard passed it — and B
    /// then received a negative molar fraction and a negative molar amount. Found by review.
    #[test]
    fn negative_or_nan_table_abundances_are_rejected() {
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps
            .iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org })
            .collect();

        for bad in [-1.0, f64::NAN] {
            let mut s = spec(200.0, 0.0);
            let mut table: HashMap<String, f64> =
                (0..40).map(|i| (format!("HUM{i}"), 1.0)).collect();
            table.insert("HUM0".into(), bad);
            s.abundance
                .insert("HUMAN".into(), AbundanceProfile::Table(table));
            assert!(
                resolve(&s, &dp).is_err(),
                "abundance {bad} must not be accepted"
            );
        }
    }

    /// REGRESSION: duplicate condition names must be rejected.
    ///
    /// `shares`, `condition_amounts` and `regulated` are keyed by condition name, so a
    /// duplicate silently overwrote the first definition — while the sample loop, which
    /// iterates the *vector*, still emitted samples and runs for BOTH, attached to the
    /// overwritten amounts. Ambiguous artifacts, no error. Found by review.
    #[test]
    fn duplicate_condition_names_are_rejected() {
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps
            .iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org })
            .collect();

        let mut s = spec(200.0, 0.0);
        let mut dup = s.conditions[1].clone();
        dup.name = "A".into(); // collides with conditions[0]
        s.conditions.push(dup);

        let err = resolve(&s, &dp).unwrap_err();
        assert!(err.contains("duplicate condition"), "{err}");
    }

    /// REGRESSION: a zero, negative, or non-finite load must be rejected.
    ///
    /// It divided by zero in the load normalisation and wrote NaN or negative attomole amounts
    /// straight into the answer key, with no error. Found by review.
    #[test]
    fn invalid_loads_are_rejected() {
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps
            .iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org })
            .collect();
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let s = spec(bad, 0.0);
            assert!(resolve(&s, &dp).is_err(), "load_ng = {bad} must be rejected");
        }
        assert!(resolve(&spec(200.0, 0.0), &dp).is_ok());
    }

    /// REGRESSION: a fold change that overflows to infinity must be rejected.
    ///
    /// `log2fc = 1024` makes `2^fc` infinite; the load normalisation then computed an infinite
    /// total and wrote `NaN` amounts (inf * 0) straight into the answer key. Found by review.
    #[test]
    fn a_fold_change_that_overflows_is_rejected() {
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps
            .iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org })
            .collect();
        for bad in [1024.0, f64::INFINITY, f64::NAN, -1024.0] {
            let mut s = spec(200.0, 0.0);
            s.conditions[1].regulate = Some(Regulation::Explicit {
                proteins: vec!["HUM0".into()],
                log2fc: bad,
            });
            assert!(resolve(&s, &dp).is_err(), "log2fc = {bad} must be rejected");
        }
        // A real fold change still works, and the amounts stay finite.
        let mut s = spec(200.0, 0.0);
        s.conditions[1].regulate = Some(Regulation::Explicit {
            proteins: vec!["HUM0".into()],
            log2fc: 3.0,
        });
        let d = resolve(&s, &dp).unwrap();
        assert!(d.protein_quantities.iter().all(|q| q.amount_amol.is_finite()));
    }

    /// REGRESSION: zero replicate counts must be rejected, not silently reinterpreted.
    ///
    /// `replicates = 0` produced no samples at all; `technical_replicates = 0` was rounded up to
    /// one by a `.max(1)`. Both yield artifacts that disagree with the design the user wrote.
    /// Found by review.
    #[test]
    fn zero_replicate_counts_are_rejected() {
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps
            .iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org })
            .collect();

        let mut s = spec(200.0, 0.0);
        s.conditions[0].replicates = 0;
        assert!(resolve(&s, &dp).is_err(), "replicates = 0 must be rejected");

        let mut s = spec(200.0, 0.0);
        s.conditions[0].technical_replicates = 0;
        assert!(resolve(&s, &dp).is_err(), "technical_replicates = 0 must be rejected");
    }

    /// REGRESSION: a protein absent from the sample must never be flagged as regulated.
    ///
    /// Absent proteins keep their row in the answer key with amount 0 (that is what makes protein
    /// FDR answerable). But multiplying 0 by a fold change is still 0 — and flagging it
    /// `is_regulated` would have the ground truth assert that a molecule *not in the sample* was
    /// regulated. Found by review.
    #[test]
    fn an_absent_protein_is_never_marked_regulated() {
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps.iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org }).collect();

        // Subset to 30 proteins, then try to regulate ALL of them — most are not in the sample.
        let mut s = spec(200.0, 0.0);
        s.complexity = Complexity { n_proteins: Some(30) };
        s.conditions[1].regulate = Some(Regulation::Explicit {
            proteins: (0..40).map(|i| format!("HUM{i}")).collect(),
            log2fc: 2.0,
        });
        let d = resolve(&s, &dp).unwrap();

        for q in d.protein_quantities.iter().filter(|q| q.sample_id == "B_R1") {
            if q.amount_amol == 0.0 {
                assert!(!q.is_regulated,
                        "{} has zero material but is flagged regulated", q.protein_id);
            }
        }
        // And the ones that ARE present did get regulated.
        let regulated = d.protein_quantities.iter()
            .filter(|q| q.sample_id == "B_R1" && q.is_regulated).count();
        assert!(regulated > 0, "the present subset must still be regulated");
    }

    /// REGRESSION: a stale accession in an explicit regulation list must be rejected.
    ///
    /// It matched no protein, so the design completed with **no regulation applied at all** — a
    /// silently wrong answer key. Found by review.
    #[test]
    fn an_unknown_regulated_protein_is_rejected() {
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps
            .iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org })
            .collect();
        let mut s = spec(200.0, 0.0);
        s.conditions[1].regulate = Some(Regulation::Explicit {
            proteins: vec!["HUM0".into(), "NOT_A_PROTEIN".into()],
            log2fc: 1.0,
        });
        let err = resolve(&s, &dp).unwrap_err();
        assert!(err.contains("NOT_A_PROTEIN"), "{err}");
    }

    /// REGRESSION: generative regulation controls must be validated before sampling.
    #[test]
    fn invalid_generative_regulation_is_rejected() {
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps
            .iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org })
            .collect();
        for (fraction, sd) in [(1.5, 1.0), (-0.1, 1.0), (f64::NAN, 1.0), (0.1, -1.0), (0.1, f64::NAN)] {
            let mut s = spec(200.0, 0.0);
            s.conditions[1].regulate = Some(Regulation::Generative { fraction, log2fc_sd: sd });
            assert!(resolve(&s, &dp).is_err(), "fraction={fraction} sd={sd} must be rejected");
        }
    }

    /// REGRESSION: the declared biological CV must actually BE the CV.
    ///
    /// It was used directly as the log-space sigma of the log-normal jitter — but a log-normal
    /// with log-sigma `s` has CV `sqrt(exp(s^2) - 1)`, so a declared 0.5 realised as 0.533. The
    /// conversion is `s = sqrt(ln(1 + CV^2))`. Found by review.
    ///
    /// The declared CV is the variation of the **biological material**. After jittering, the
    /// sample is renormalised to the requested load — because at the bench you inject 200 ng
    /// whatever the composition — and that rescaling perturbs the realised CV by a compositional
    /// term. That term is *physical*, not an error (it is the same effect that makes a spike-in
    /// dilute its background). So this test uses a large, flat proteome, where the rescale factor
    /// concentrates on 1 and the conversion can be checked on its own.
    #[test]
    fn the_declared_cv_is_the_realised_cv() {
        // 3,000 equal-abundance, equal-length proteins ⇒ the load rescale factor is ~1.
        let ps: Vec<(String, String, String)> = (0..3000)
            .map(|i| (format!("P{i}"), "PEPTIDEKAAAKCCCRDDDK".repeat(5), "HUMAN".to_string()))
            .collect();
        let dp: Vec<DesignProtein> = ps
            .iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org })
            .collect();

        for declared in [0.15, 0.5] {
            let mut abundance = BTreeMap::new();
            abundance.insert("HUMAN".to_string(), AbundanceProfile::LogNormal { sigma: 0.0 });
            let spec = DesignSpec {
                reference: "A".into(),
                load_ng: 200.0,
                complexity: Complexity::default(),
                abundance,
                variance: Variance { biological: declared, biological_heterogeneity: 0.0, technical: 0.0 },
                seed: 7,
                conditions: vec![Condition {
                    name: "A".into(),
                    mix: [("HUMAN".to_string(), Share::Fraction(1.0))].into(),
                    replicates: 300,
                    technical_replicates: 1,
                    regulate: None,
                }],
            };
            let d = resolve(&spec, &dp).unwrap();

            // Realised CV of one protein across its 300 biological replicates.
            let v: Vec<f64> = d
                .protein_quantities
                .iter()
                .filter(|q| q.protein_id == "P0")
                .map(|q| q.amount_amol)
                .collect();
            assert_eq!(v.len(), 300);
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt();
            let cv = sd / mean;

            // Sampling error on a CV from n=300 is ~CV/sqrt(2n) ≈ 4%; allow 15% relative.
            assert!(
                (cv - declared).abs() < 0.15 * declared,
                "declared CV {declared}, realised {cv:.4} — the declared value must BE the CV"
            );
        }
    }

    /// Each protein gets its OWN variance. A single global CV makes the model homoscedastic in log
    /// space, which is wrong — housekeeping proteins are tightly controlled, regulatory ones swing.
    /// The per-protein CV is identity-keyed, so a protein's *variability* is as fixed as its *mean*.
    #[test]
    fn each_protein_has_its_own_variance() {
        let ps: Vec<(String, String, String)> = (0..600)
            .map(|i| (format!("P{i}"), "PEPTIDEKAAAKCCCRDDDK".repeat(5), "HUMAN".to_string()))
            .collect();
        let dp: Vec<DesignProtein> = ps.iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org }).collect();

        let build = |het: f64| {
            let mut abundance = BTreeMap::new();
            abundance.insert("HUMAN".to_string(), AbundanceProfile::LogNormal { sigma: 0.0 });
            DesignSpec {
                reference: "A".into(), load_ng: 200.0, complexity: Complexity::default(),
                abundance, seed: 3,
                variance: Variance { biological: 0.2, biological_heterogeneity: het, technical: 0.0 },
                conditions: vec![Condition {
                    name: "A".into(),
                    mix: [("HUMAN".to_string(), Share::Fraction(1.0))].into(),
                    replicates: 120, technical_replicates: 1, regulate: None,
                }],
            }
        };
        let realised_cvs = |spec: &DesignSpec| -> Vec<f64> {
            let d = resolve(spec, &dp).unwrap();
            (0..40).map(|i| {
                let id = format!("P{i}");
                let v: Vec<f64> = d.protein_quantities.iter()
                    .filter(|q| q.protein_id == id).map(|q| q.amount_amol).collect();
                let m = v.iter().sum::<f64>() / v.len() as f64;
                (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt() / m
            }).collect()
        };

        // het = 0 ⇒ every protein shares the mean CV (the old, homoscedastic behaviour).
        let flat = realised_cvs(&build(0.0));
        let spread_flat = flat.iter().cloned().fold(f64::MIN, f64::max)
            - flat.iter().cloned().fold(f64::MAX, f64::min);

        // het > 0 ⇒ proteins genuinely differ in how much they vary.
        let het = realised_cvs(&build(0.7));
        let spread_het = het.iter().cloned().fold(f64::MIN, f64::max)
            - het.iter().cloned().fold(f64::MAX, f64::min);

        assert!(
            spread_het > 2.5 * spread_flat,
            "heterogeneous CVs must spread much wider than flat ones: {spread_het:.4} vs {spread_flat:.4}"
        );
        // And the mean CV is still roughly what was declared.
        let mean: f64 = het.iter().sum::<f64>() / het.len() as f64;
        assert!((mean - 0.2).abs() < 0.08, "mean realised CV {mean:.4}, declared 0.2");
    }

    /// Protein subsetting is a QUANTITY operation: excluded proteins stay in the structure and get
    /// amount 0. That is what makes protein-level FDR answerable — a reported protein with zero
    /// amount is unambiguously a false positive. v1 deleted them from the proteome instead, which
    /// destroyed the degeneracy map and made the question unanswerable (the benchmark suite says so
    /// in a code comment).
    #[test]
    fn protein_subsetting_zeroes_amounts_rather_than_deleting_proteins() {
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps.iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org }).collect();

        let mut s = spec(200.0, 0.0);
        s.complexity = Complexity { n_proteins: Some(30) };
        let d = resolve(&s, &dp).unwrap();

        let a: Vec<&ProteinQuantity> = d.protein_quantities.iter()
            .filter(|q| q.sample_id == "A_R1").collect();

        // EVERY protein is still in the answer key — none was deleted.
        assert_eq!(a.len(), 90, "all 90 proteins remain in the ground truth");
        let present = a.iter().filter(|q| q.amount_amol > 0.0).count();
        assert_eq!(present, 30, "exactly 30 are in the sample");
        assert_eq!(a.iter().filter(|q| q.amount_amol == 0.0).count(), 60,
                   "the other 60 are present in the truth with amount 0 — a reported hit on any of \
                    them is a nameable false positive");

        // The load is still exactly conserved across the 30 that are present.
        assert!(d.report.mass_balance_error_ng < 1e-6);
    }

    /// REGRESSION: subsetting must not silently destroy an organism's declared mass share.
    ///
    /// A GLOBAL hash draw can exclude every protein of a small organism (E. coli is 4,401 of HYE's
    /// 30,881). That organism then gets zero mass, the load normalisation redistributes its declared
    /// share to the others — and the report still prints the mixture you asked for. The fold changes
    /// would be wrong and nothing would say so. Found by review.
    #[test]
    fn subsetting_preserves_every_organism_and_its_fold_changes() {
        let ps = hye_proteins(); // 40 HUMAN / 30 YEAST / 20 ECOLI
        let dp: Vec<DesignProtein> = ps.iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org }).collect();

        for n in [3usize, 10, 25, 60] {
            let mut s = spec(200.0, 0.0);
            s.complexity = Complexity { n_proteins: Some(n) };
            let d = resolve(&s, &dp).unwrap();

            // Every organism still has at least one protein in the sample.
            for org in ["HUMAN", "YEAST", "ECOLI"] {
                let prefix = match org { "HUMAN" => "HUM", "YEAST" => "YST", _ => "ECO" };
                let present = d.protein_quantities.iter()
                    .filter(|q| q.sample_id == "A_R1" && q.protein_id.starts_with(prefix)
                                && q.amount_amol > 0.0)
                    .count();
                assert!(present >= 1, "n={n}: organism {org} was wiped out entirely");
            }

            // And the declared HYE fold changes still come out EXACTLY right.
            let fc = |prefix: &str| -> f64 {
                let v: Vec<f64> = d.protein_quantities.iter()
                    .filter(|q| q.sample_id == "B_R1" && q.protein_id.starts_with(prefix)
                                && q.true_log2fc.is_finite())
                    .map(|q| q.true_log2fc).collect();
                v.iter().sum::<f64>() / v.len() as f64
            };
            assert_relative_eq!(fc("HUM"), 0.0, epsilon = 1e-9);
            assert_relative_eq!(fc("YST"), -1.0, epsilon = 1e-9);
            assert_relative_eq!(fc("ECO"), 2.0, epsilon = 1e-9);
        }

        // Fewer proteins than organisms is refused rather than silently dropping one.
        let mut s = spec(200.0, 0.0);
        s.complexity = Complexity { n_proteins: Some(2) };
        assert!(resolve(&s, &dp).is_err(), "n_proteins < n_organisms must be rejected");
    }

    /// Raising `n_proteins` yields a SUPERSET — identity-keyed selection, not a fresh draw.
    #[test]
    fn a_larger_subset_is_a_superset() {
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps.iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org }).collect();
        let ids = |n: usize| -> std::collections::HashSet<String> {
            let mut s = spec(200.0, 0.0);
            s.complexity = Complexity { n_proteins: Some(n) };
            resolve(&s, &dp).unwrap().protein_quantities.iter()
                .filter(|q| q.sample_id == "A_R1" && q.amount_amol > 0.0)
                .map(|q| q.protein_id.clone()).collect()
        };
        let small = ids(20);
        let large = ids(50);
        assert_eq!(small.len(), 20);
        assert_eq!(large.len(), 50);
        assert!(small.is_subset(&large), "raising n_proteins must extend, not reshuffle");
    }

    /// The hockey-stick rank-abundance curve reproduces v1's shape — but keyed by identity, so
    /// adding a protein does not reshuffle everyone else's abundance (v1's
    /// `np.random.uniform(1, 1e4, n)` is draw-order dependent, and does).
    #[test]
    fn the_hockey_stick_curve_gives_a_realistic_dynamic_range() {
        let ps: Vec<(String, String, String)> = (0..5000)
            .map(|i| (format!("P{i}"), "PEPTIDEKAAAKCCCRDDDK".repeat(5), "HUMAN".to_string()))
            .collect();
        let dp: Vec<DesignProtein> = ps.iter()
            .map(|(id, seq, org)| DesignProtein { id, sequence: seq, organism: org }).collect();

        let mut abundance = BTreeMap::new();
        abundance.insert("HUMAN".to_string(),
                         AbundanceProfile::HockeyStick { decay: 0.06, tail: 1e-4 });
        let s = DesignSpec {
            reference: "A".into(), load_ng: 200.0, complexity: Complexity::default(),
            abundance, seed: 11,
            variance: Variance::default(),
            conditions: vec![Condition {
                name: "A".into(),
                mix: [("HUMAN".to_string(), Share::Fraction(1.0))].into(),
                replicates: 1, technical_replicates: 1, regulate: None,
            }],
        };
        let d = resolve(&s, &dp).unwrap();
        let v: Vec<f64> = d.protein_quantities.iter().map(|q| q.amount_amol).collect();
        let (lo, hi) = (v.iter().cloned().fold(f64::MAX, f64::min),
                        v.iter().cloned().fold(f64::MIN, f64::max));
        let orders = (hi / lo).log10();
        assert!((3.0..6.0).contains(&orders),
                "hockey stick should span a few orders, got {orders:.2}");
        assert!(d.report.mass_balance_error_ng < 1e-6);
    }

    /// A mixture that does not sum to 1, with no `rest`, is an error rather than a silent
    /// renormalisation.
    #[test]
    fn a_mixture_that_does_not_sum_is_rejected() {
        let mut s = spec(200.0, 0.0);
        s.conditions[0].mix = [
            ("HUMAN".to_string(), Share::Fraction(0.65)),
            ("YEAST".to_string(), Share::Fraction(0.20)),
            ("ECOLI".to_string(), Share::Fraction(0.05)),
        ]
        .into(); // 0.90
        let ps = hye_proteins();
        let dp: Vec<DesignProtein> = ps
            .iter()
            .map(|(id, seq, org)| DesignProtein {
                id,
                sequence: seq,
                organism: org,
            })
            .collect();
        assert!(resolve(&s, &dp).is_err());
    }
}
