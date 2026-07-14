//! `timsim-precursors` — the ion layer. STRUCTURE.
//!
//! Turns peptides into precursors: m/z, the isotope envelope, the charge distribution, and the
//! ionisation propensity. Everything here is a property of the **molecule**, so it is computed once
//! and shared by every sample in the design.
//!
//! # The factorisation is the ground truth
//!
//! ```text
//!   ion_amol = peptide_amol × ionization_propensity × charge_fraction
//! ```
//!
//! Each multiplier is its own column, so the chain is **invertible** — walk an ion back to its
//! peptide, and a peptide back to its proteins. v1 collapses abundance × flyability into a single
//! `int32` and the question *"is this peptide missing because there is little of it, or because it
//! does not fly?"* becomes unanswerable.

use anyhow::Result;
use arrow::array::{
    ArrayRef, Float32Array, Float32Builder, Float64Array, ListBuilder, UInt64Array, UInt8Array,
};
use clap::Parser;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use timsim_chem::ionize::{ChargeModel, Flyability, Ionizer};
use timsim_cli::{batch, print_schema, producer};
use timsim_schema::tables::{peptides as PEP, precursors as PRE};

#[derive(Parser)]
#[command(name = "timsim-precursors", about = "peptides -> precursors (m/z, isotopes, charge, flyability)")]
struct Args {
    #[arg(long)]
    peptides: Option<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,

    /// `site-specific` (default — gas-phase basicity R > K > H; a clean tryptic peptide comes out
    /// 88.7% doubly charged) or `binomial` (v1 parity — every basic site equal, which calls a third
    /// of all clean tryptic peptides singly-charged, and is kept only for comparison).
    #[arg(long, default_value = "site-specific")]
    charge_model: String,
    /// Probability that any one protonatable site (N-term, H, R, K) carries a proton. `binomial` only.
    #[arg(long, default_value_t = 0.8)]
    charged_probability: f64,
    #[arg(long, default_value_t = 4)]
    max_charge: u8,
    /// Isotope peaks to keep, starting at the monoisotopic.
    #[arg(long, default_value_t = 6)]
    isotope_depth: usize,

    /// Ionisation efficiency model: `lognormal` (v1 parity) or `uniform`.
    ///
    /// A response-factor model trained on EQUIMOLAR peptides (PFly's base) belongs here. An
    /// observability model (PFly's fine-tuned variant, the one Koina serves) does NOT — it has
    /// abundance baked in at ρ=0.76 and would square the abundance effect.
    #[arg(long, default_value = "lognormal")]
    flyability: String,
    #[arg(long, default_value_t = 1e-2)]
    flyability_median: f64,
    #[arg(long, default_value_t = 1.0)]
    flyability_sigma: f64,

    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    schema: bool,
    #[arg(long)]
    explain: bool,
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.schema {
        return print_schema(PRE::TABLE);
    }

    let flyability = match a.flyability.as_str() {
        "uniform" => Flyability::Uniform,
        "lognormal" => Flyability::LogNormal {
            median: a.flyability_median,
            sigma: a.flyability_sigma,
            lo: 1e-4,
            hi: 1.0,
        },
        other => anyhow::bail!("unknown flyability model {other:?}; use 'lognormal' or 'uniform'"),
    };
    let charge = match a.charge_model.as_str() {
        "binomial" => ChargeModel::Binomial {
            charged_probability: a.charged_probability,
            max_charge: a.max_charge,
        },
        "site-specific" => ChargeModel::realistic_with(a.max_charge),
        other => anyhow::bail!("unknown charge model {other:?}; use 'binomial' or 'site-specific'"),
    };
    let io = Ionizer {
        charge,
        flyability,
        isotope_depth: a.isotope_depth,
        seed: a.seed,
    };
    io.validate().map_err(anyhow::Error::msg)?;

    if a.explain {
        // Printed FROM the model, not from a hand-typed string that can drift away from it.
        println!("  charge model     : {}", io.charge.describe());
        match io.charge {
            ChargeModel::SiteSpecific { .. } => {
                println!("                     gas-phase basicity R > K > H — a binomial cannot order them");
                println!("                     Poisson-binomial over the sites");
                println!("                     a clean tryptic peptide comes out 88.7% doubly charged");
            }
            ChargeModel::Binomial { .. } => {
                println!("                     v1 parity: every basic site treated alike");
                println!("                     ⚠ calls a THIRD of clean tryptic peptides singly-charged");
                println!("                       — kept for comparison, not for use");
            }
        }
        println!("                     z = 0 (neutral) is never observed, so the distribution is");
        println!("                     renormalised over the charges that can actually be seen");
        println!("  flyability       : {}", a.flyability);
        println!("  isotope envelope : exact from elemental composition (C,H,N,O,S), depth {}", a.isotope_depth);
        println!();
        println!("  ion_amol = peptide_amol × ionization_propensity × charge_fraction");
        println!("             ^ each factor is its own column, so the chain is invertible");
        return Ok(());
    }

    let peptides = a.peptides.ok_or_else(|| anyhow::anyhow!("--peptides is required"))?;
    let out = a.out.ok_or_else(|| anyhow::anyhow!("--out is required"))?;

    // Validate-on-read: a renamed or retyped column fails HERE, not three stages downstream.
    let mut input: Vec<(u64, String)> = Vec::new();
    for b in timsim_schema::read(&peptides, PEP::TABLE)? {
        let id: &UInt64Array = b.column_by_name(PEP::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let sq: &arrow::array::StringArray =
            b.column_by_name(PEP::SEQUENCE).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            input.push((id.value(i), sq.value(i).to_string()));
        }
    }

    let t = std::time::Instant::now();
    let per_peptide: Vec<Vec<timsim_chem::Precursor>> = input
        .par_iter()
        .map(|(id, seq)| io.precursors_of(*id, seq).unwrap_or_default())
        .collect();
    // What the charge cap discarded. A truncation whose error is unmeasured is a guess — the same
    // discipline as --max-missed-cleavages and the modform abundance floor.
    let cap_losses: Vec<f64> = input
        .par_iter()
        .map(|(_, seq)| io.charge.distribution_with_loss(seq).1)
        .collect();
    let elapsed = t.elapsed();

    let mut all: Vec<&timsim_chem::Precursor> = per_peptide.iter().flatten().collect();
    all.sort_unstable_by_key(|p| (p.peptide_id, p.charge)); // deterministic row order
    let skipped = per_peptide.iter().filter(|v| v.is_empty()).count();

    let n = all.len();
    let mut iso = ListBuilder::new(Float32Builder::new());
    for p in &all {
        for v in &p.isotope_intensity {
            iso.values().append_value(*v);
        }
        iso.append(true);
    }

    let cols: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(all.iter().map(|p| p.precursor_id).collect::<Vec<_>>())),
        Arc::new(UInt64Array::from(all.iter().map(|p| p.peptide_id).collect::<Vec<_>>())),
        Arc::new(UInt8Array::from(all.iter().map(|p| p.charge).collect::<Vec<_>>())),
        Arc::new(Float64Array::from(all.iter().map(|p| p.mz).collect::<Vec<_>>())),
        Arc::new(iso.finish()),
        Arc::new(Float32Array::from(all.iter().map(|p| p.charge_fraction).collect::<Vec<_>>())),
        Arc::new(Float32Array::from(all.iter().map(|p| p.ionization_propensity).collect::<Vec<_>>())),
    ];
    timsim_schema::write(&out, PRE::TABLE, &batch(PRE::TABLE, cols)?, &producer("timsim-precursors"), None)?;

    // ── the accounting ────────────────────────────────────────────────────────
    let mean_z: f64 = all.iter().map(|p| p.charge as f64 * p.charge_fraction as f64).sum::<f64>()
        / input.len().max(1) as f64;
    let fly: Vec<f64> = per_peptide.iter().filter_map(|v| v.first()).map(|p| p.ionization_propensity as f64).collect();
    let mut sorted = fly.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_fly = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);

    // Sized from the MODEL, not from the flag: re-entering a fact the model already knows is
    // how two things silently disagree — and here it indexed out of bounds (B13).
    let mut by_charge = vec![0f64; io.charge.max_charge() as usize + 1];
    for p in &all {
        by_charge[p.charge as usize] += p.charge_fraction as f64;
    }
    let tot: f64 = by_charge.iter().sum();

    println!("  peptides             : {}", input.len());
    println!("  precursors           : {n}   ({:.2} charge states per peptide)", n as f64 / input.len().max(1) as f64);
    print!("  charge distribution  :");
    for (z, f) in by_charge.iter().enumerate().skip(1) {
        print!("  {z}+ → {:.1}%", f / tot * 100.0);
    }
    // NOT molar. This averages over the peptide TABLE, which is unweighted by abundance — and at
    // max_missed_cleavages=2, 79% of *distinct* peptides carry a missed cleavage even though only
    // ~10% of the *material* does. The molar distribution needs peptide_quantities, which is a
    // quantity artifact and does not belong on this (structural) stage.
    println!("   (per distinct peptide — NOT molar; abundance weighting needs peptide_quantities)");
    println!("  mean charge          : {mean_z:.2}   (same caveat — and note the MEAN is a poor");
    println!("                         summary here: both charge models give ~1.95 for tryptic");
    println!("                         peptides while differing hugely in SHAPE)");
    println!("  flyability           : {} (median {:.2e}, {:.2e} … {:.2e})",
             a.flyability, median_fly,
             fly.iter().cloned().fold(f64::MAX, f64::min),
             fly.iter().cloned().fold(f64::MIN, f64::max));
    println!("  isotope envelope     : exact from composition, {} peaks", a.isotope_depth);
    {
        let mean_loss = cap_losses.iter().sum::<f64>() / cap_losses.len().max(1) as f64;
        let affected = cap_losses.iter().filter(|&&l| l > 1e-6).count();
        println!(
            "  charge cap z<={}      : discards {:.4}% of the ion population   (measured, not assumed)",
            io.charge.max_charge(),
            mean_loss * 100.0
        );
        println!(
            "                         {affected} peptides ({:.2}%) have real ions above the cap",
            affected as f64 / cap_losses.len().max(1) as f64 * 100.0
        );
    }
    if skipped > 0 {
        eprintln!("  warning: {skipped} peptides skipped (non-standard residues)");
    }
    println!("  wall time            : {elapsed:.2?}");
    println!();
    println!("  ion_amol = peptide_amol × ionization_propensity × charge_fraction");
    println!("             (each factor a column — the chain is invertible)");
    println!("  -> {}", out.display());
    Ok(())
}
