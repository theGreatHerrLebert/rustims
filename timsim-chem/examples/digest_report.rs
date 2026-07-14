//! Digest a real FASTA and report what the enumeration accounted for.
//! usage: cargo run --release -p timsim-chem --example digest_report -- <fasta> [eff] [max_mc]
use std::collections::HashMap;
use timsim_chem::{parse_fasta, BlockingOccupancy, Enumerator, Protocol, YieldModel};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let eff: f64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.90);
    let max_mc: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);

    let proteins = parse_fasta(&std::fs::read_to_string(&a[1]).expect("read fasta"));
    let input: Vec<(String, String)> = proteins.iter().map(|p| (p.id.clone(), p.sequence.clone())).collect();
    let seqs: HashMap<&str, &str> = proteins.iter().map(|p| (p.id.as_str(), p.sequence.as_str())).collect();

    // ── structure: computed once, shared by every sample in a design ──
    let e = Enumerator::new(Protocol::parse("trypsin").unwrap(), max_mc, 7, 50).unwrap();
    let t = std::time::Instant::now();
    let digests = e.enumerate_all(&input);
    let t_struct = t.elapsed();

    // ── quantity: one of these per condition ──
    let y = YieldModel::new(eff, BlockingOccupancy::none()).unwrap();
    let t = std::time::Instant::now();
    let (_yields, stats) = y.apply_all(&digests);
    let t_quant = t.elapsed();

    let mut by_seq: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut n_occ = 0usize;
    for d in &digests {
        for o in &d.occurrences {
            n_occ += 1;
            by_seq.entry(o.sequence(seqs[&*d.protein_id])).or_default().push(&d.protein_id);
        }
    }
    let degenerate = by_seq.values().filter(|v| {
        let mut p: Vec<_> = v.to_vec(); p.sort_unstable(); p.dedup(); p.len() > 1
    }).count();

    println!("  proteins             : {}", proteins.len());
    println!("  digestion efficiency : {eff}  (per-site cleavage probability)");
    let dist = stats.missed_cleavage_distribution();
    print!("  missed cleavages     :");
    for (n, f) in dist.iter().enumerate() { print!("  {n} → {:.1}%", f * 100.0); }
    println!("   (molar, observed)");
    println!("  truncated at n={max_mc}     : omits {:.4}% of residue mass", stats.truncation_loss() * 100.0);
    println!("  length filter 7-50   : omits {:.2}% of enumerated mass", stats.filter_loss() * 100.0);
    println!("  occurrences          : {n_occ}");
    println!("  unique peptides      : {}", by_seq.len());
    println!("  degenerate           : {degenerate} peptides ({:.1}%) mapping to >1 protein",
             degenerate as f64 / by_seq.len().max(1) as f64 * 100.0);
    println!();
    println!("  structure (once)     : {:.2?}", t_struct);
    println!("  quantity  (/sample)  : {:.2?}   ← a 20-sample A/B design pays this 20×, not the above",
             t_quant);
}
