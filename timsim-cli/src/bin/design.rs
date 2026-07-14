//! `timsim-design` — the mixture is the spec; the fold change is DERIVED.
//!
//! At the bench you do not specify a fold change. You specify a **mixture**: "65% human, 30%
//! yeast, 5% E. coli; hold human fixed and move the other two." The fold changes are a
//! consequence.
//!
//! v1's benchmark suite has the expected HYE ratios (1.0 / 0.667 / 3.0) **typed into a
//! plotting cell**, because the simulator never knew them.

use anyhow::Result;
use arrow::array::{Array, ArrayRef, BooleanArray, Float32Array, Float64Array, StringArray, UInt32Array};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use timsim_chem::design::{resolve, DesignProtein};
use timsim_cli::{batch, print_schema, producer, spec};
use timsim_schema::tables::{
    protein_quantities as PQ, proteome as PROT, runs as RUN, sample_run_map as SRM, samples as SAM,
};

#[derive(Parser)]
#[command(name = "timsim-design", about = "proteome + design.toml -> samples, runs, mapping, protein quantities")]
struct Args {
    #[arg(long)]
    proteome: Option<PathBuf>,
    #[arg(long)]
    spec: Option<PathBuf>,
    #[arg(long)]
    out_samples: Option<PathBuf>,
    #[arg(long)]
    out_runs: Option<PathBuf>,
    #[arg(long)]
    out_sample_run_map: Option<PathBuf>,
    #[arg(long)]
    out_protein_quantities: Option<PathBuf>,
    #[arg(long)]
    schema: bool,
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.schema {
        for t in [SAM::TABLE, RUN::TABLE, SRM::TABLE, PQ::TABLE] {
            print_schema(t)?;
            println!();
        }
        return Ok(());
    }
    let req = |o: Option<PathBuf>, n: &str| o.ok_or_else(|| anyhow::anyhow!("--{n} is required"));
    let proteome = req(a.proteome, "proteome")?;
    let spec_path = req(a.spec, "spec")?;
    let (os, or, om, oq) = (
        req(a.out_samples, "out-samples")?,
        req(a.out_runs, "out-runs")?,
        req(a.out_sample_run_map, "out-sample-run-map")?,
        req(a.out_protein_quantities, "out-protein-quantities")?,
    );

    let spec = spec::load_design(&spec_path, spec_path.parent().unwrap_or(std::path::Path::new(".")))?;

    let batches = timsim_schema::read(&proteome, PROT::TABLE)?;
    let (mut ids, mut seqs, mut orgs) = (Vec::new(), Vec::new(), Vec::new());
    for b in &batches {
        let id = b.column_by_name(PROT::PROTEIN_ID).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let sq = b.column_by_name(PROT::SEQUENCE).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let og = b.column_by_name(PROT::ORGANISM).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            ids.push(id.value(i).to_string());
            seqs.push(sq.value(i).to_string());
            orgs.push(if og.is_null(i) { "UNKNOWN".to_string() } else { og.value(i).to_string() });
        }
    }
    let proteins: Vec<DesignProtein> = (0..ids.len())
        .map(|i| DesignProtein { id: &ids[i], sequence: &seqs[i], organism: &orgs[i] })
        .collect();

    let d = resolve(&spec, &proteins).map_err(anyhow::Error::msg)?;

    // The declared technical CV rides along with the samples. The design cannot apply it — a
    // technical replicate is the same tube, so its amounts are identical — but the measurement stage
    // will need it, and a config value that silently evaporates is worse than one that is refused.
    let variance_meta: Vec<(&str, String)> = vec![(
        timsim_schema::meta::VARIANCE_TECHNICAL,
        spec.variance.technical.to_string(),
    )];

    timsim_schema::write_with(&os, SAM::TABLE, &batch(SAM::TABLE, vec![
        Arc::new(StringArray::from(d.samples.iter().map(|s| s.sample_id.clone()).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(StringArray::from(d.samples.iter().map(|s| s.condition.clone()).collect::<Vec<_>>())),
        Arc::new(UInt32Array::from(d.samples.iter().map(|s| s.replicate).collect::<Vec<_>>())),
    ])?, &producer("timsim-design"), None, &variance_meta)?;

    timsim_schema::write(&or, RUN::TABLE, &batch(RUN::TABLE, vec![
        Arc::new(StringArray::from(d.runs.iter().map(|r| r.run_id.clone()).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(UInt32Array::from(d.runs.iter().map(|r| r.technical_replicate).collect::<Vec<_>>())),
        Arc::new(UInt32Array::from(d.runs.iter().map(|r| r.injection_order).collect::<Vec<_>>())),
    ])?, &producer("timsim-design"), None)?;

    timsim_schema::write(&om, SRM::TABLE, &batch(SRM::TABLE, vec![
        Arc::new(StringArray::from(d.sample_runs.iter().map(|x| x.sample_id.clone()).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(StringArray::from(d.sample_runs.iter().map(|x| x.run_id.clone()).collect::<Vec<_>>())),
        Arc::new(StringArray::from(d.sample_runs.iter().map(|x| x.channel.clone()).collect::<Vec<_>>())),
        Arc::new(Float32Array::from(d.sample_runs.iter().map(|x| x.mix_fraction as f32).collect::<Vec<_>>())),
    ])?, &producer("timsim-design"), None)?;

    timsim_schema::write(&oq, PQ::TABLE, &batch(PQ::TABLE, vec![
        Arc::new(StringArray::from(d.protein_quantities.iter().map(|q| q.protein_id.clone()).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(StringArray::from(d.protein_quantities.iter().map(|q| q.sample_id.clone()).collect::<Vec<_>>())),
        Arc::new(Float64Array::from(d.protein_quantities.iter().map(|q| q.amount_amol).collect::<Vec<_>>())),
        Arc::new(Float32Array::from(d.protein_quantities.iter()
            .map(|q| if q.true_log2fc.is_finite() { Some(q.true_log2fc as f32) } else { None })
            .collect::<Vec<Option<f32>>>())),
        Arc::new(BooleanArray::from(d.protein_quantities.iter().map(|q| q.is_regulated).collect::<Vec<_>>())),
    ])?, &producer("timsim-design"), None)?;

    println!("  organism   {:>10} {:>10}   mean MW    true log2FC", "mass frac", "");
    for o in &d.report.organisms {
        let fracs: Vec<String> = o.mask_fractions_display();
        let fcs: Vec<String> = o.true_log2fc.iter().map(|(c, f)| format!("{c}:{f:+.3}")).collect();
        println!("  {:<10} {:<21}  {:>7.1} kDa   {}", o.organism, fracs.join(" "), o.mean_mw / 1000.0, fcs.join("  "));
    }
    println!();
    println!("  load                 : {:.1} ng per run", d.report.load_ng);
    println!("  mass balance error   : {:.3e} ng   (must be ~0)", d.report.mass_balance_error_ng);
    println!("  samples (biological) : {}", d.samples.len());
    if spec.variance.technical > 0.0 {
        println!("  technical CV         : {} — recorded on samples.parquet for the measurement",
                 spec.variance.technical);
        println!("                         stage; a technical replicate is the SAME tube, so its");
        println!("                         amounts are identical and all its variation is measured");
    }
    println!("  runs    (injections) : {}", d.runs.len());
    if !d.report.skipped_proteins.is_empty() {
        eprintln!("  warning: {} proteins skipped (non-standard residues; we refuse to guess a mass)",
                  d.report.skipped_proteins.len());
    }
    Ok(())
}
