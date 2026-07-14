//! Independent oracle for the cleavage rules.
//!
//! Our Monte Carlo oracle proves the YIELD maths is right. It says nothing about whether we
//! cut the protein in the right PLACES. If our trypsin is subtly wrong, every yield is a
//! correct number about the wrong peptides, and everything downstream inherits it.
//!
//! So: digest the same FASTA with Sage (an independent, widely-used, battle-tested search
//! engine) and with us, and compare the peptide SETS. Sage dedups within a protein, so we
//! collapse our occurrences to unique (protein, sequence) pairs to compare like with like.

use sage_core::enzyme::{Enzyme as SageEnzyme, EnzymeParameters};
use std::collections::HashSet;
use std::sync::Arc;
use timsim_chem::{parse_fasta, Enumerator, Protocol};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let max_missed: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let (min_len, max_len) = (7usize, 50usize);

    let text = std::fs::read_to_string(path).expect("read fasta");
    let proteins = parse_fasta(&text);
    println!("proteins: {}\nenzyme: trypsin (cleave KR, not before P)\nmax_missed_cleavages: {max_missed}\nlength: {min_len}-{max_len}\n", proteins.len());

    // ── SAGE ──
    let sage_params = EnzymeParameters {
        missed_cleavages: max_missed as u8,
        min_len,
        max_len,
        enzyme: SageEnzyme::new("KR", "P", true, false),
    };
    let t = std::time::Instant::now();
    let mut sage_set: HashSet<(String, String)> = HashSet::new();
    for p in &proteins {
        let acc: Arc<str> = Arc::from(p.id.as_str());
        for d in sage_params.digest(&p.sequence, acc) {
            sage_set.insert((p.id.clone(), d.sequence));
        }
    }
    let t_sage = t.elapsed();

    // ── OURS ──
    let e = Enumerator::new(
        Protocol::parse("trypsin").unwrap(),
        max_missed,
        min_len as u32,
        max_len as u32,
    )
    .unwrap();
    let input: Vec<(String, String)> = proteins.iter().map(|p| (p.id.clone(), p.sequence.clone())).collect();
    let t = std::time::Instant::now();
    let digests = e.enumerate_all(&input);
    let t_ours = t.elapsed();

    let mut our_set: HashSet<(String, String)> = HashSet::new();
    let mut n_occ = 0usize;
    for (d, p) in digests.iter().zip(&proteins) {
        for o in &d.occurrences {
            n_occ += 1;
            our_set.insert((p.id.clone(), o.sequence(&p.sequence).to_string()));
        }
    }

    // ── COMPARE ──
    let only_sage: Vec<_> = sage_set.difference(&our_set).take(8).cloned().collect();
    let only_ours: Vec<_> = our_set.difference(&sage_set).take(8).cloned().collect();
    let n_only_sage = sage_set.difference(&our_set).count();
    let n_only_ours = our_set.difference(&sage_set).count();

    println!("sage  (protein, peptide) pairs : {}   [{:.2?}]", sage_set.len(), t_sage);
    println!("ours  (protein, peptide) pairs : {}   [{:.2?}]", our_set.len(), t_ours);
    println!("ours  occurrences              : {n_occ}   (+{} vs unique pairs — repeats within a protein, which Sage discards)",
             n_occ - our_set.len());
    println!();
    println!("in SAGE but not OURS : {n_only_sage}");
    for x in &only_sage { println!("    {} / {}", x.0, x.1); }
    println!("in OURS but not SAGE : {n_only_ours}");
    for x in &only_ours { println!("    {} / {}", x.0, x.1); }
    println!();
    if n_only_sage == 0 && n_only_ours == 0 {
        println!("✓ IDENTICAL peptide sets — cleavage rules independently confirmed.");
    } else {
        println!("✗ DIVERGENCE — our cleavage rules disagree with Sage.");
        std::process::exit(1);
    }
}
