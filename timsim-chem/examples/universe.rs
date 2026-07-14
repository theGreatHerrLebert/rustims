//! S0.5 — the candidate-universe measurement.
//!
//! The architecture says: keep an UNPRUNED structural universe, and never prune it by
//! detectability (that would make structure depend on the design). The open question that
//! decision rests on is whether the unpruned modform x charge universe is tractable to
//! GPU-predict. This measures it. A number, not an argument.
//!
//! usage: cargo run --release -p timsim-chem --example universe -- <fasta>

use rayon::prelude::*;
use timsim_chem::modify::{plausible_charges, Modification, Site, Stage};
use timsim_chem::{enumerate_modforms, parse_fasta, Enumerator, Protocol};

fn m(name: &str, unimod: u32, targets: &str, site: Site, occ: f64, delta: f64, blocks: bool, stage: Stage) -> Modification {
    Modification { name: name.into(), unimod_id: unimod, targets: targets.into(), site,
                   occupancy: occ, mass_delta: delta, blocks_cleavage: blocks, stage }
}

fn scenarios() -> Vec<(&'static str, Vec<Modification>)> {
    let camc = m("Carbamidomethyl", 4, "C", Site::Residue, 0.98, 57.02146, false, Stage::Protein);
    let oxid = m("Oxidation", 35, "M", Site::Residue, 0.05, 15.99491, false, Stage::Peptide);
    let nace = m("Acetyl(N-term)", 1, "", Site::NTerm, 0.01, 42.01057, false, Stage::Protein);
    let phos = m("Phospho", 21, "STY", Site::Residue, 0.02, 79.96633, false, Stage::Protein);
    let phos_hi = m("Phospho(enriched)", 21, "STY", Site::Residue, 0.25, 79.96633, false, Stage::Protein);
    vec![
        ("unmodified (baseline)", vec![]),
        ("standard proteomics", vec![camc.clone(), oxid.clone(), nace.clone()]),
        ("+ phospho (2% occupancy)", vec![camc.clone(), oxid.clone(), nace.clone(), phos]),
        ("phospho-enriched (25%)", vec![camc, oxid, nace, phos_hi]),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let proteins = parse_fasta(&std::fs::read_to_string(&args[1]).expect("read fasta"));
    let input: Vec<(String, String)> = proteins.iter().map(|p| (p.id.clone(), p.sequence.clone())).collect();

    let e = Enumerator::new(Protocol::parse("trypsin/p").unwrap(), 2, 7, 50).unwrap();
    let digests = e.enumerate_all(&input);

    // Unique peptide sequences — the structural base.
    let mut peptides: Vec<String> = Vec::new();
    {
        let seqs: std::collections::HashMap<&str, &str> =
            proteins.iter().map(|p| (p.id.as_str(), p.sequence.as_str())).collect();
        let mut seen = std::collections::HashSet::new();
        for d in &digests {
            let ps = seqs[&*d.protein_id];
            for o in &d.occurrences {
                let s = o.sequence(ps);
                if seen.insert(s) { peptides.push(s.to_string()); }
            }
        }
    }

    println!("proteins : {}", proteins.len());
    println!("peptides : {}   (trypsin/p, <=2 missed cleavages, length 7-50)\n", peptides.len());

    println!("{:<26} {:>7}  {:>14}  {:>16}  {:>11}  {:>9}", "scenario", "floor", "modforms", "precursors", "retained", "GPU@50k/s");
    println!("{}", "-".repeat(94));

    for (name, mods) in scenarios() {
        for &floor in &[1e-2, 1e-3, 1e-4] {
            let out: Vec<(u64, u64, f64)> = peptides
                .par_iter()
                .map(|seq| {
                    let (forms, st) = enumerate_modforms(seq, &mods, floor).unwrap();
                    let charges = plausible_charges(seq, 4).len() as u64;
                    // Only charges >= 1; a precursor is (modform, charge).
                    (forms.len() as u64, forms.len() as u64 * charges, st.mass_retained)
                })
                .collect();

            let modforms: u64 = out.iter().map(|x| x.0).sum();
            let precursors: u64 = out.iter().map(|x| x.1).sum();
            let retained: f64 = out.iter().map(|x| x.2).sum::<f64>() / out.len() as f64;
            let gpu_s = precursors as f64 / 50_000.0;

            println!("{:<26} {:>7.0e}  {:>14}  {:>16}  {:>10.4}%  {:>8.0}s",
                     if floor == 1e-2 { name } else { "" },
                     floor, fmt(modforms), fmt(precursors), retained * 100.0, gpu_s);
        }
        println!();
    }
}

fn fmt(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 { out.push(','); }
        out.push(c);
    }
    out
}
