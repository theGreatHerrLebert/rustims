//! Sweeps the CLI does not yet expose: cleavage-BLOCKING modifications (the acetylome case),
//! and the modform universe vs the abundance floor. Emits JSON for the report.
use std::sync::Arc;
use timsim_chem::digest::{BlockingOccupancy, Enumerator, YieldModel};
use timsim_chem::modify::{plausible_charges, Modification, Site, Stage};
use timsim_chem::{enumerate_modforms, parse_fasta, Protocol};
use rayon::prelude::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let proteins = parse_fasta(&std::fs::read_to_string(&a[1]).unwrap());
    let input: Vec<(String, String)> = proteins.iter().take(4000)
        .map(|p| (p.id.clone(), p.sequence.clone())).collect();

    let e = Enumerator::new(Protocol::parse("trypsin/p").unwrap(), 3, 7, 50).unwrap();
    let digests = e.enumerate_all(&input);

    // ── BLOCKING: acetyl-K blocks trypsin. Raising occupancy must raise missed cleavages —
    //    which is exactly how diGly/acetylome experiments FIND the modified lysine.
    let mut blocking = Vec::new();
    for occ in [0.0, 0.05, 0.10, 0.20, 0.40] {
        let sites: Vec<(Arc<str>, u32, f64)> = digests.iter()
            .flat_map(|d| d.cleavage_sites.iter().map(move |&s| (d.protein_id.clone(), s, occ)))
            .collect();
        let bo = BlockingOccupancy::from_sites(sites).unwrap();
        let (_, st) = YieldModel::new(0.90, bo).unwrap().apply_all(&digests);
        let dist = st.missed_cleavage_distribution();
        blocking.push(format!(
            r#"{{"occupancy":{occ},"missed":[{}]}}"#,
            dist.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(",")
        ));
    }

    // ── MODFORM UNIVERSE vs abundance floor ──
    let seqs: std::collections::HashMap<&str, &str> =
        proteins.iter().map(|p| (p.id.as_str(), p.sequence.as_str())).collect();
    let mut peps: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for d in &digests {
        for o in &d.occurrences {
            let s = o.sequence(seqs[&*d.protein_id]);
            if seen.insert(s) { peps.push(s.to_string()); }
        }
    }
    let m = |n: &str, u: u32, t: &str, si: Site, o: f64, dm: f64, st: Stage, f: &str| Modification {
        name: n.into(), unimod_id: u, targets: t.into(), site: si,
        occupancy: o, mass_delta: dm, composition: f.into(), blocks_cleavage: false, stage: st };

    let mut universe = Vec::new();
    for (label, mods) in [
        ("none", vec![]),
        ("standard", vec![
            m("Carbamidomethyl",4,"C",Site::Residue,0.98,57.02146,Stage::Protein, "C2H3NO"),
            m("Oxidation",35,"M",Site::Residue,0.05,15.99491,Stage::Peptide, "O"),
            m("Acetyl(N-term)",1,"",Site::NTerm,0.01,42.01057,Stage::Protein, "C2H2O")]),
        ("+phospho 2%", vec![
            m("Carbamidomethyl",4,"C",Site::Residue,0.98,57.02146,Stage::Protein, "C2H3NO"),
            m("Oxidation",35,"M",Site::Residue,0.05,15.99491,Stage::Peptide, "O"),
            m("Acetyl(N-term)",1,"",Site::NTerm,0.01,42.01057,Stage::Protein, "C2H2O"),
            m("Phospho",21,"STY",Site::Residue,0.02,79.96633,Stage::Protein, "HO3P")]),
    ] {
        for floor in [1e-2, 1e-3, 1e-4, 1e-5] {
            let r: Vec<(u64,u64,f64)> = peps.par_iter().map(|s| {
                let (f, st) = enumerate_modforms(s, &mods, floor).unwrap();
                let z = plausible_charges(s, 4).len() as u64;
                (f.len() as u64, f.len() as u64 * z, st.mass_retained)
            }).collect();
            let mf: u64 = r.iter().map(|x| x.0).sum();
            let pc: u64 = r.iter().map(|x| x.1).sum();
            let ret: f64 = r.iter().map(|x| x.2).sum::<f64>() / r.len() as f64;
            universe.push(format!(
                r#"{{"scenario":"{label}","floor":{floor:e},"modforms":{mf},"precursors":{pc},"retained":{ret:.6}}}"#));
        }
    }

    println!(r#"{{"peptides":{},"blocking":[{}],"universe":[{}]}}"#,
             peps.len(), blocking.join(","), universe.join(","));
}
