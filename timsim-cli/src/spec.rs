//! The TOML config surface.
//!
//! **Two vocabularies, one physics.** The chemist's parameters are the primary interface;
//! they are transfer functions over a physical parameterisation the informatician may
//! override directly. Both are recorded in the report, so neither audience has to learn the
//! other's language and neither is lied to.

use anyhow::{bail, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use timsim_chem::design;

// ─────────────────────────────────────────────────────────────────────────────
// proteome.toml
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-source, because that is how HYE actually works.
///
/// v1 recovers the organism by substring-matching `"HUMAN"` / `"YEAST"` / `"ECOLI"` in the
/// FASTA header (`table_labeling.py:12`), and peptides shared between organisms silently
/// become `"Unknown"` and get **dropped**. Here the organism is a declared column from the
/// moment the protein enters the model.
#[derive(Debug, Deserialize)]
pub struct ProteomeSpec {
    #[serde(default, rename = "source")]
    pub sources: Vec<Source>,
}

#[derive(Debug, Deserialize)]
pub struct Source {
    pub path: String,
    #[serde(default)]
    pub organism: Option<String>,
    #[serde(default)]
    pub is_contaminant: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// design.toml
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct DesignFile {
    pub design: DesignHeader,
    pub abundance: BTreeMap<String, AbundanceSpec>,
    #[serde(rename = "condition")]
    pub conditions: Vec<ConditionSpec>,
    #[serde(default)]
    pub variance: VarianceSpec,
}

#[derive(Debug, Deserialize)]
pub struct DesignHeader {
    /// Condition against which `true_log2fc` is computed.
    pub reference: String,
    /// Total peptide mass on column, per run. The physically meaningful replacement for
    /// `num_sample_peptides`, which had no physical meaning at all.
    pub load_ng: f64,
    /// How many proteins are actually in the sample. Excluded proteins keep their place in the
    /// structure and receive amount 0 — which is what keeps protein-level FDR answerable. There is
    /// deliberately no peptide-count knob: peptides vanish by falling below the detection limit,
    /// not at random.
    #[serde(default)]
    pub n_proteins: Option<usize>,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_seed() -> u64 {
    42
}

#[derive(Debug, Deserialize)]
#[serde(tag = "source", rename_all = "lowercase")]
pub enum AbundanceSpec {
    /// σ ≈ 2 gives roughly the ~7 orders of dynamic range a real proteome spans.
    Lognormal { sigma: f64 },
    /// The rank-abundance curve v1 uses (`get_tenzer_hokey`) — a better-shaped marginal than a
    /// log-normal, and here the rank is keyed by identity rather than by draw order.
    Hockeystick {
        #[serde(default = "hs_decay")]
        decay: f64,
        #[serde(default = "hs_tail")]
        tail: f64,
    },
    /// From real data — PaxDb, or a real quantification. Two columns: id, abundance.
    /// **The only source that gets protein IDENTITY right**, not merely the shape.
    Table { path: String },
}

fn hs_decay() -> f64 {
    0.06
}
fn hs_tail() -> f64 {
    1e-4
}

#[derive(Debug, Deserialize)]
pub struct ConditionSpec {
    pub name: String,
    /// organism → mass fraction, or the string `"rest"`.
    pub mix: BTreeMap<String, toml::Value>,
    #[serde(default = "one")]
    pub replicates: u32,
    #[serde(default = "one")]
    pub technical_replicates: u32,
    #[serde(default)]
    pub regulate: Option<RegulateSpec>,
}

fn one() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum RegulateSpec {
    /// A named set moved by a known amount — for reproducing a published benchmark.
    Explicit { proteins: Vec<String>, log2fc: f64 },
    /// A random fraction moved by a draw — for power analysis.
    Generative { fraction: f64, log2fc_sd: f64 },
}

#[derive(Debug, Default, Deserialize)]
pub struct VarianceSpec {
    /// CV between biological replicates. Distinct material, so the amounts really differ.
    #[serde(default)]
    pub biological: f64,
    /// Spread of per-protein CVs (natural-log units). 0 ⇒ every protein shares the mean CV.
    #[serde(default)]
    pub biological_heterogeneity: f64,
    /// CV between injections. Declared here, applied on the measurement axis — a technical
    /// replicate is the *same tube*, so its amounts are identical and all the variation is
    /// in the measurement.
    #[serde(default)]
    pub technical: f64,
}

// ─────────────────────────────────────────────────────────────────────────────

pub fn load_proteome_spec(path: &Path) -> Result<ProteomeSpec> {
    let s: ProteomeSpec = toml::from_str(&std::fs::read_to_string(path)?)?;
    if s.sources.is_empty() {
        bail!("{}: no [[source]] entries", path.display());
    }
    Ok(s)
}

pub fn load_design(path: &Path, abundance_dir: &Path) -> Result<design::DesignSpec> {
    let f: DesignFile = toml::from_str(&std::fs::read_to_string(path)?)?;

    let mut abundance = BTreeMap::new();
    for (org, a) in &f.abundance {
        let profile = match a {
            AbundanceSpec::Lognormal { sigma } => {
                if !sigma.is_finite() || *sigma < 0.0 {
                    bail!("abundance.{org}: sigma must be finite and non-negative, got {sigma}");
                }
                design::AbundanceProfile::LogNormal { sigma: *sigma }
            }
            AbundanceSpec::Hockeystick { decay, tail } => {
                if !decay.is_finite() || *decay <= 0.0 || !tail.is_finite() || *tail < 0.0 {
                    bail!("abundance.{org}: hockeystick decay must be > 0 and tail >= 0");
                }
                design::AbundanceProfile::HockeyStick { decay: *decay, tail: *tail }
            }
            AbundanceSpec::Table { path: p } => {
                let full = abundance_dir.join(p);
                let text = std::fs::read_to_string(&full)
                    .map_err(|e| anyhow::anyhow!("abundance.{org}: {}: {e}", full.display()))?;
                let mut t = std::collections::HashMap::new();
                for (lineno, line) in text.lines().enumerate() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    let mut it = line.split_whitespace();
                    match (it.next(), it.next()) {
                        (Some(id), Some(v)) => {
                            let v: f64 = v.parse().map_err(|_| {
                                anyhow::anyhow!("{}:{}: {v:?} is not a number", full.display(), lineno + 1)
                            })?;
                            t.insert(id.to_string(), v);
                        }
                        _ => bail!("{}:{}: expected `id abundance`", full.display(), lineno + 1),
                    }
                }
                design::AbundanceProfile::Table(t)
            }
        };
        abundance.insert(org.clone(), profile);
    }

    let mut conditions = Vec::new();
    for c in &f.conditions {
        let mut mix = BTreeMap::new();
        for (org, v) in &c.mix {
            let share = match v {
                toml::Value::Float(x) => design::Share::Fraction(*x),
                toml::Value::Integer(x) => design::Share::Fraction(*x as f64),
                toml::Value::String(s) if s.eq_ignore_ascii_case("rest") => design::Share::Rest,
                other => bail!(
                    "condition {:?}: mix.{org} must be a number or \"rest\", got {other}",
                    c.name
                ),
            };
            mix.insert(org.clone(), share);
        }
        conditions.push(design::Condition {
            name: c.name.clone(),
            mix,
            replicates: c.replicates,
            technical_replicates: c.technical_replicates,
            regulate: c.regulate.as_ref().map(|r| match r {
                RegulateSpec::Explicit { proteins, log2fc } => design::Regulation::Explicit {
                    proteins: proteins.clone(),
                    log2fc: *log2fc,
                },
                RegulateSpec::Generative {
                    fraction,
                    log2fc_sd,
                } => design::Regulation::Generative {
                    fraction: *fraction,
                    log2fc_sd: *log2fc_sd,
                },
            }),
        });
    }

    Ok(design::DesignSpec {
        reference: f.design.reference,
        load_ng: f.design.load_ng,
        complexity: design::Complexity { n_proteins: f.design.n_proteins },
        abundance,
        conditions,
        variance: design::Variance {
            biological: f.variance.biological,
            biological_heterogeneity: f.variance.biological_heterogeneity,
            technical: f.variance.technical,
        },
        seed: f.design.seed,
    })
}
