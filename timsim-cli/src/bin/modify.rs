//! `timsim-modify` — post-translational modifications. STRUCTURE.
//!
//! Turns peptides into **modforms**: the actual modified species present in the sample, each with
//! the fraction of that peptide's molecules carrying exactly that set of modifications.
//!
//! # This is the stage where v1 simulates a search engine instead of a sample
//!
//! v1 asks for *variable modifications* — "oxidation on M, phospho on STY, at most 3 per peptide" —
//! and generates every combination. That is a **database search parameter**. It describes the space
//! a search engine is willing to look in, not the population of molecules in a tube. Simulating from
//! it guarantees the simulator and the search engine agree about which species exist, for reasons
//! that have nothing whatsoever to do with chemistry — and a benchmark built on that agreement
//! cannot detect the failure it most needs to detect.
//!
//! The physical parameter is **occupancy**: the fraction of *this site*, across all molecules of the
//! protein, that carries the mod. It is the chemist's number and it is measurable — phospho ~0.02 at
//! a regulated site, Met oxidation ~0.05 (mostly a handling artefact), carbamidomethyl ~0.98, which
//! is simply the alkylation efficiency. From occupancies the modform distribution follows, and it is
//! wildly non-uniform: the unmodified form usually holds >90% of the molecules, and the
//! triply-phosphorylated form that a "max 3 variable mods" search happily looks for is present at
//! 1e-5 and would never be seen.
//!
//! # Occupancy also decides which peptides exist
//!
//! Acetyl-K, ubiquitin-GG, trimethyl-K and TMT-K **physically stop trypsin**. A lysine carrying one
//! of them is not cleaved, so the peptide spanning it is a missed cleavage — and that missed
//! cleavage *is how the experiment finds the site*. A diGly enrichment in which the protease ignores
//! the GG is not a simulation of a diGly experiment.
//!
//! That coupling is why the modification spec is written here as an artifact
//! (`modifications.parquet`) and **read** by `timsim-yield`, rather than re-entered as flags on both
//! tools. Every B13 bug found so far has had this exact shape: two stages handed the same fact
//! separately, and allowed to disagree.

use anyhow::{Context, Result};
use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Float64Builder, ListBuilder, StringArray, StringBuilder,
    UInt32Array, UInt32Builder, UInt64Array,
};
use clap::Parser;
use rayon::prelude::*;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use timsim_chem::ids;
use timsim_chem::mass;
use timsim_chem::modify::{enumerate_modforms, Modification, Site, Stage};
use timsim_cli::{batch, print_schema, producer, set_threads};
use timsim_schema::tables::{modforms as MF, modifications as MODS, peptides as PEP};

#[derive(Parser)]
#[command(
    name = "timsim-modify",
    about = "peptides -> modforms (occupancy-driven, not variable-mod combinatorics)"
)]
struct Args {
    #[arg(long)]
    peptides: Option<PathBuf>,

    /// The modification spec (TOML). See `--example` for the format.
    #[arg(long)]
    mods: Option<PathBuf>,

    /// Directory for `modforms.parquet` and `modifications.parquet`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Discard modforms below this fraction of a peptide's molecules.
    ///
    /// The enumeration is exponential in the number of modifiable sites, and almost all of that mass
    /// is vanishingly rare species. The floor is **exact, not heuristic**: the depth-first search
    /// prunes with an admissible bound, so nothing above the floor is missed, and the probability
    /// mass actually discarded is *measured* and reported — never estimated.
    #[arg(long, default_value_t = 1e-3)]
    floor: f64,

    #[arg(long)]
    threads: Option<usize>,
    #[arg(long)]
    schema: bool,
    #[arg(long)]
    explain: bool,
    /// Print an example modification spec and exit.
    #[arg(long)]
    example: bool,
}

// ── the spec file ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SpecFile {
    #[serde(rename = "modification", default)]
    modifications: Vec<ModSpec>,
}

#[derive(Deserialize)]
struct ModSpec {
    name: String,
    unimod_id: u32,
    #[serde(default)]
    targets: String,
    #[serde(default = "default_site")]
    site: String,
    occupancy: f64,
    mass_delta: f64,
    #[serde(default)]
    blocks_cleavage: bool,
    #[serde(default = "default_stage")]
    stage: String,
}

fn default_site() -> String {
    "residue".into()
}
fn default_stage() -> String {
    "protein".into()
}

const EXAMPLE: &str = r#"# timsim-modify — modification spec.
#
# `occupancy` is the fraction of THIS SITE, across all molecules, that carries the mod.
# It is not "max N variable mods" — that is a search parameter, not a property of a sample.

[[modification]]
name            = "Carbamidomethyl"
unimod_id       = 4
targets         = "C"
occupancy       = 0.98        # this is just the alkylation efficiency
mass_delta      = 57.021464
stage           = "protein"   # reduction/alkylation precedes digestion

[[modification]]
name            = "Oxidation"
unimod_id       = 35
targets         = "M"
occupancy       = 0.05        # largely a handling artefact
mass_delta      = 15.994915
stage           = "peptide"   # forms on the peptide, after the protease

[[modification]]
name            = "Phospho"
unimod_id       = 21
targets         = "STY"
occupancy       = 0.02        # a REGULATED site; most STY are far below this
mass_delta      = 79.966331
stage           = "protein"

# Blocks trypsin: a modified lysine is not cleaved, so the peptide spanning it carries a
# missed cleavage — which is exactly how a diGly experiment localises the site.
[[modification]]
name            = "GG"
unimod_id       = 121
targets         = "K"
occupancy       = 0.001
mass_delta      = 114.042927
blocks_cleavage = true
stage           = "protein"
"#;

fn parse_mods(path: &PathBuf) -> Result<Vec<Modification>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading modification spec {}", path.display()))?;
    let spec: SpecFile = toml::from_str(&text)
        .with_context(|| format!("parsing modification spec {}", path.display()))?;

    if spec.modifications.is_empty() {
        anyhow::bail!(
            "modification spec {} declares no [[modification]] entries. An empty spec would \
             silently produce one unmodified modform per peptide, which looks like success. \
             Run with --example for the format.",
            path.display()
        );
    }

    let mut out = Vec::with_capacity(spec.modifications.len());
    let mut seen = std::collections::HashSet::new();
    for m in spec.modifications {
        if !seen.insert(m.name.clone()) {
            // `modform_id` keys on the name, so duplicates would silently merge two distinct
            // modifications into one id.
            anyhow::bail!("modification {:?} is declared twice; names must be unique", m.name);
        }
        let site = match m.site.as_str() {
            "residue" => Site::Residue,
            "n_term" => Site::NTerm,
            "c_term" => Site::CTerm,
            o => anyhow::bail!(
                "modification {:?}: unknown site {o:?}; use 'residue', 'n_term' or 'c_term'",
                m.name
            ),
        };
        let stage = match m.stage.as_str() {
            "protein" => Stage::Protein,
            "peptide" => Stage::Peptide,
            o => anyhow::bail!(
                "modification {:?}: unknown stage {o:?}; use 'protein' or 'peptide'",
                m.name
            ),
        };
        // Only a mod that is present BEFORE the protease can stop it. Accepting
        // `blocks_cleavage` on a peptide-stage mod would let a config express a physical
        // impossibility and get a plausible-looking digest out of it.
        if m.blocks_cleavage && stage != Stage::Protein {
            anyhow::bail!(
                "modification {:?} sets blocks_cleavage but has stage='peptide'. A modification \
                 that forms AFTER digestion cannot have prevented the protease from cutting.",
                m.name
            );
        }
        let md = Modification {
            name: m.name,
            unimod_id: m.unimod_id,
            targets: m.targets,
            site,
            occupancy: m.occupancy,
            mass_delta: m.mass_delta,
            blocks_cleavage: m.blocks_cleavage,
            stage,
        };
        md.validate().map_err(anyhow::Error::msg)?;
        out.push(md);
    }
    Ok(out)
}

fn site_str(s: Site) -> &'static str {
    match s {
        Site::Residue => "residue",
        Site::NTerm => "n_term",
        Site::CTerm => "c_term",
    }
}
fn stage_str(s: Stage) -> &'static str {
    match s {
        Stage::Protein => "protein",
        Stage::Peptide => "peptide",
    }
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.schema {
        print_schema(MODS::TABLE)?;
        return print_schema(MF::TABLE);
    }
    if a.example {
        print!("{EXAMPLE}");
        return Ok(());
    }
    if a.explain {
        println!("  timsim-modify — occupancy, not variable-mod combinatorics");
        println!();
        println!("  v1 asks for 'oxidation on M, phospho on STY, max 3 per peptide' and generates");
        println!("  every combination. That is a DATABASE SEARCH PARAMETER — the space a search");
        println!("  engine looks in, not the population of molecules in a tube. Simulating from it");
        println!("  makes the simulator and the search engine agree by construction.");
        println!();
        println!("  Here the input is per-site OCCUPANCY (the chemist's number), and the modform");
        println!("  distribution follows from it:");
        println!();
        println!("    P(modform) = Π_modified occupancy(site) · Π_unmodified (1 − Σ occupancy(site))");
        println!();
        println!("  Σ abundance_fraction = 1 over the complete enumeration — that identity is the");
        println!("  oracle, and with a floor the shortfall is exactly the mass discarded (measured).");
        println!();
        println!("  blocks_cleavage is consumed by timsim-yield, NOT here: a modified lysine is not");
        println!("  cleaved, so blocking changes WHICH PEPTIDES EXIST — a digest question. The spec");
        println!("  is written to modifications.parquet so both tools read one source of truth.");
        return Ok(());
    }
    set_threads(a.threads);

    let peptides_path = a.peptides.ok_or_else(|| anyhow::anyhow!("--peptides is required"))?;
    let mods_path = a.mods.ok_or_else(|| anyhow::anyhow!("--mods is required (see --example)"))?;
    let out_dir = a.out.ok_or_else(|| anyhow::anyhow!("--out is required"))?;

    let mods = parse_mods(&mods_path)?;

    // Flatten to (peptide_id, sequence). The enumeration is embarrassingly parallel per peptide, so
    // the batch boundaries carry no meaning here.
    let mut pep_ids: Vec<u64> = Vec::new();
    let mut pep_seqs: Vec<String> = Vec::new();
    for b in timsim_schema::read(&peptides_path, PEP::TABLE)? {
        let ids_col = b
            .column_by_name(PEP::PEPTIDE_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let seq_col = b
            .column_by_name(PEP::SEQUENCE)
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..b.num_rows() {
            pep_ids.push(ids_col.value(i));
            pep_seqs.push(seq_col.value(i).to_string());
        }
    }
    let n_input = pep_ids.len();
    if n_input == 0 {
        anyhow::bail!("{} contains no peptides", peptides_path.display());
    }

    println!("  timsim-modify");
    println!("    peptides         : {}", peptides_path.display());
    println!("    modifications    : {}", mods_path.display());
    for m in &mods {
        let where_ = if m.site == Site::Residue {
            format!("on {}", m.targets)
        } else {
            site_str(m.site).to_string()
        };
        println!(
            "      {:<16} {:>6.3} occupancy  {:>10.5} Da  {:<10} {}{}",
            m.name,
            m.occupancy,
            m.mass_delta,
            where_,
            stage_str(m.stage),
            if m.blocks_cleavage { "  BLOCKS CLEAVAGE" } else { "" },
        );
    }
    println!("    abundance floor  : {:e}", a.floor);
    println!();

    // ── enumerate ────────────────────────────────────────────────────────────
    struct Row {
        modform_id: u64,
        peptide_id: u64,
        positions: Vec<u32>,
        names: Vec<String>,
        mass: f64,
        mass_delta: f64,
        fraction: f64,
    }

    let indices: Vec<usize> = (0..n_input).collect();
    let per_peptide: Vec<Result<(Vec<Row>, f64, usize, bool), String>> = indices
        .par_iter()
        .map(|&i| {
            let pid = pep_ids[i];
            let seq = pep_seqs[i].as_str();
            let base = mass::monoisotopic(seq).map_err(|e| format!("{seq}: {e}"))?;
            let (forms, stats) = enumerate_modforms(seq, &mods, a.floor)?;
            let modified = stats.sites > 0;
            let mut rows = Vec::with_capacity(forms.len());
            for f in forms {
                let names: Vec<String> =
                    f.mods.iter().map(|(_, mi)| mods[*mi].name.clone()).collect();
                let positions: Vec<u32> = f.mods.iter().map(|(p, _)| *p).collect();
                let key: Vec<(u32, &str)> = f
                    .mods
                    .iter()
                    .map(|(p, mi)| (*p, mods[*mi].name.as_str()))
                    .collect();
                rows.push(Row {
                    modform_id: ids::modform_id(pid, &key),
                    peptide_id: pid,
                    positions,
                    names,
                    mass: base + f.mass_delta,
                    mass_delta: f.mass_delta,
                    fraction: f.abundance_fraction,
                });
            }
            Ok((rows, stats.truncation_loss(), stats.sites, modified))
        })
        .collect();

    let mut rows: Vec<Row> = Vec::new();
    let mut worst_loss = 0.0f64;
    let mut total_loss = 0.0f64;
    let mut with_sites = 0usize;
    let mut skipped = 0usize;
    let mut total_sites = 0usize;
    for r in per_peptide {
        match r {
            Ok((rs, loss, sites, has_sites)) => {
                worst_loss = worst_loss.max(loss);
                total_loss += loss;
                total_sites += sites;
                if has_sites {
                    with_sites += 1;
                }
                rows.extend(rs);
            }
            // A non-standard residue (X, B, Z) has no mass. Skipping is right; skipping SILENTLY
            // is not — the count is reported.
            Err(_) => skipped += 1,
        }
    }
    if rows.is_empty() {
        anyhow::bail!("no modforms produced — every peptide was skipped or fell below the floor");
    }

    // ── the oracle: Σ abundance_fraction == 1 per peptide, at floor = 0 ───────
    //
    // Every molecule of a peptide is in exactly one modform, so the fractions partition it. With a
    // floor they fall short, and the shortfall must equal the mass the floor discarded — which is
    // what `truncation_loss` independently measured. Two routes to the same number.
    let n_peptides = n_input - skipped;
    let sum_fractions: f64 = rows.iter().map(|r| r.fraction).sum();
    let expected = n_peptides as f64 - total_loss;
    let residual = (sum_fractions - expected).abs();
    if residual > 1e-6 * n_peptides as f64 {
        anyhow::bail!(
            "mass balance violated: Σ abundance_fraction = {sum_fractions:.9} but the measured \
             truncation loss says it should be {expected:.9} (residual {residual:.3e}). The \
             modform distribution does not partition the peptide's molecules."
        );
    }

    // ── write ────────────────────────────────────────────────────────────────
    std::fs::create_dir_all(&out_dir)?;

    let mut pos_b = ListBuilder::new(UInt32Builder::new());
    let mut nam_b = ListBuilder::new(StringBuilder::new());
    let mut mass_b = Float64Builder::new();
    let mut delta_b = Float64Builder::new();
    let mut frac_b = Float64Builder::new();
    for r in &rows {
        for p in &r.positions {
            pos_b.values().append_value(*p);
        }
        pos_b.append(true);
        for n in &r.names {
            nam_b.values().append_value(n);
        }
        nam_b.append(true);
        mass_b.append_value(r.mass);
        delta_b.append_value(r.mass_delta);
        frac_b.append_value(r.fraction);
    }
    let mf = batch(
        MF::TABLE,
        vec![
            Arc::new(UInt64Array::from(rows.iter().map(|r| r.modform_id).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(UInt64Array::from(rows.iter().map(|r| r.peptide_id).collect::<Vec<_>>())),
            Arc::new(pos_b.finish()),
            Arc::new(nam_b.finish()),
            Arc::new(mass_b.finish()),
            Arc::new(delta_b.finish()),
            Arc::new(frac_b.finish()),
        ],
    )?;

    let ms = batch(
        MODS::TABLE,
        vec![
            Arc::new(StringArray::from(mods.iter().map(|m| m.name.clone()).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(UInt32Array::from(mods.iter().map(|m| m.unimod_id).collect::<Vec<_>>())),
            Arc::new(StringArray::from(mods.iter().map(|m| m.targets.clone()).collect::<Vec<_>>())),
            Arc::new(StringArray::from(mods.iter().map(|m| site_str(m.site)).collect::<Vec<_>>())),
            Arc::new(Float64Array::from(mods.iter().map(|m| m.occupancy).collect::<Vec<_>>())),
            Arc::new(Float64Array::from(mods.iter().map(|m| m.mass_delta).collect::<Vec<_>>())),
            Arc::new(BooleanArray::from(mods.iter().map(|m| m.blocks_cleavage).collect::<Vec<_>>())),
            Arc::new(StringArray::from(mods.iter().map(|m| stage_str(m.stage)).collect::<Vec<_>>())),
        ],
    )?;

    // The floor travels WITH the artifact: a consumer must never have to be told again what
    // fraction of the molecules was discarded, nor re-derive it (B13).
    let meta: Vec<(&str, String)> = vec![
        ("timsim.modify.floor", a.floor.to_string()),
        (
            "timsim.modify.truncation_loss_mean",
            (total_loss / n_peptides.max(1) as f64).to_string(),
        ),
    ];
    let mf_path = out_dir.join("modforms.parquet");
    let ms_path = out_dir.join("modifications.parquet");
    timsim_schema::write_with(&mf_path, MF::TABLE, &mf, &producer("timsim-modify"), None, &meta)?;
    timsim_schema::write(&ms_path, MODS::TABLE, &ms, &producer("timsim-modify"), None)?;

    let unmodified = rows.iter().filter(|r| r.positions.is_empty()).count();
    println!("  peptides           : {:>12}", n_peptides);
    if skipped > 0 {
        println!("  skipped            : {skipped:>12}  (non-standard residues — no mass)");
    }
    println!(
        "    with a site      : {:>12}  ({:.1}%)   {} modifiable sites in total",
        with_sites,
        with_sites as f64 / n_peptides as f64 * 100.0,
        total_sites
    );
    println!("  modforms           : {:>12}", rows.len());
    println!(
        "    unmodified       : {:>12}  ({:.1}% of modforms; they carry most of the mass)",
        unmodified,
        unmodified as f64 / rows.len() as f64 * 100.0
    );
    println!(
        "    per peptide      : {:>12.2}  average",
        rows.len() as f64 / n_peptides as f64
    );
    println!();
    println!(
        "  truncation loss    : mean {:.4}%   worst {:.4}%   (molecules below the floor)",
        total_loss / n_peptides as f64 * 100.0,
        worst_loss * 100.0
    );
    println!("                       measured, not estimated — Σ fractions cross-checks it");
    println!("  mass balance       : Σ abundance_fraction == 1 − loss   (residual {residual:.2e}) OK");
    println!();
    println!("  -> {}", mf_path.display());
    println!("  -> {}   (read by timsim-yield for blocks_cleavage — one source of truth)", ms_path.display());
    Ok(())
}
