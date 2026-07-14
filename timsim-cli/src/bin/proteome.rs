//! `timsim-proteome` — build the sample's protein universe from FASTA. STRUCTURE.
//!
//! No amounts. Abundance is a *quantity* and lives on its own axis, so that the structure can
//! be shared across every sample in a design.

use anyhow::{bail, Result};
use arrow::array::{ArrayRef, BooleanArray, StringArray, UInt32Array};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use timsim_cli::{batch, print_schema, producer, spec};
use timsim_schema::tables::proteome as T;

#[derive(Parser)]
#[command(name = "timsim-proteome", about = "FASTA -> proteome.parquet (structure; no amounts)")]
struct Args {
    /// Multi-source spec: several FASTAs with organism tags and contaminants. This is how HYE works.
    #[arg(long, conflicts_with = "fasta")]
    spec: Option<PathBuf>,
    /// Single FASTA, for the simple case.
    #[arg(long)]
    fasta: Option<PathBuf>,
    /// Organism tag for --fasta.
    #[arg(long, requires = "fasta")]
    organism: Option<String>,
    #[arg(long)]
    out: Option<PathBuf>,
    /// Print the output schema and exit.
    #[arg(long)]
    schema: bool,
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.schema {
        return print_schema(T::TABLE);
    }
    let out = a.out.clone().ok_or_else(|| anyhow::anyhow!("--out is required"))?;

    let sources: Vec<(PathBuf, Option<String>, bool)> = if let Some(p) = &a.spec {
        let dir = p.parent().unwrap_or(std::path::Path::new("."));
        spec::load_proteome_spec(p)?
            .sources
            .into_iter()
            .map(|s| (dir.join(&s.path), s.organism, s.is_contaminant))
            .collect()
    } else if let Some(f) = &a.fasta {
        vec![(f.clone(), a.organism.clone(), false)]
    } else {
        bail!("one of --spec or --fasta is required");
    };

    let (mut ids, mut seqs, mut lens, mut descs, mut orgs, mut contam) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut seen = std::collections::HashSet::new();
    let mut dupes = 0usize;

    for (path, organism, is_contaminant) in &sources {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let proteins = timsim_chem::parse_fasta(&text);
        if proteins.is_empty() {
            bail!("{}: no proteins parsed", path.display());
        }
        for p in proteins {
            // A duplicated accession across sources would silently merge two proteins.
            if !seen.insert(p.id.clone()) {
                dupes += 1;
                continue;
            }
            ids.push(p.id);
            lens.push(p.sequence.len() as u32);
            seqs.push(p.sequence);
            descs.push(Some(p.description));
            orgs.push(organism.clone());
            contam.push(*is_contaminant);
        }
        println!("  {:<50} organism={:<8} contaminant={}", path.display(),
                 organism.as_deref().unwrap_or("-"), is_contaminant);
    }
    if dupes > 0 {
        eprintln!("  warning: {dupes} duplicate accessions skipped (first occurrence kept)");
    }

    let cols: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(ids.clone())),
        Arc::new(StringArray::from(seqs)),
        Arc::new(UInt32Array::from(lens)),
        Arc::new(StringArray::from(descs)),
        Arc::new(StringArray::from(orgs)),
        Arc::new(BooleanArray::from(contam)),
    ];
    timsim_schema::write(&out, T::TABLE, &batch(T::TABLE, cols)?, &producer("timsim-proteome"), None)?;

    println!("\n  proteins : {}", ids.len());
    println!("  -> {}", out.display());
    Ok(())
}
