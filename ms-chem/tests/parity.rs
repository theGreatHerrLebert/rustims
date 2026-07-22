//! ms-chem regression gate: the unified crate must reproduce the parity-proven numbers.
//!
//! Foundation increment — neutral monoisotopic peptide mass. ms-chem must agree with BOTH mscore
//! and timsim-chem (which agreed with each other to float precision in R1 Gate 1). This is the
//! guard the build phase runs against: if a table port introduced a typo, this fails immediately.

const AAS: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";

fn corpus() -> Vec<String> {
    let mut v = Vec::new();
    for &a in AAS {
        for &b in AAS {
            for &c in AAS {
                v.push(format!("{}{}{}", a as char, b as char, c as char));
            }
        }
    }
    let mut s: u64 = 0x9e3779b97f4a7c15;
    for &len in &[5usize, 10, 20, 40] {
        for _ in 0..1500 {
            let mut seq = String::with_capacity(len);
            for _ in 0..len {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                seq.push(AAS[((s >> 33) as usize) % AAS.len()] as char);
            }
            v.push(seq);
        }
    }
    v
}

#[test]
fn monoisotopic_mass_matches_mscore_and_timsim() {
    use mscore::data::peptide::PeptideSequence;

    let corpus = corpus();
    let rel_tol = 1e-7;
    let mut max_rel_ms = 0.0f64;
    let mut max_rel_ts = 0.0f64;
    let mut worst = String::new();

    for seq in &corpus {
        let x = ms_chem::monoisotopic(seq).expect("std residue");
        let m = PeptideSequence::new(seq.clone(), None).mono_isotopic_mass();
        let t = timsim_chem::mass::monoisotopic(seq).expect("std residue");

        let rms = (x - m).abs() / x.abs().max(1.0);
        let rts = (x - t).abs() / x.abs().max(1.0);
        if rms > max_rel_ms {
            max_rel_ms = rms;
            worst = seq.clone();
        }
        max_rel_ts = max_rel_ts.max(rts);

        assert!(rms < rel_tol, "ms-chem vs mscore rel {rms:.3e} on {seq:?}");
        assert!(rts < rel_tol, "ms-chem vs timsim rel {rts:.3e} on {seq:?}");
    }

    eprintln!(
        "[ms-chem/mono] corpus={}  max_rel vs mscore={max_rel_ms:.3e}  vs timsim={max_rel_ts:.3e}  worst={worst}",
        corpus.len()
    );
}

#[test]
fn selenocysteine_correct_and_surfaces_mscore_bug() {
    // ms-chem is a superset: it accepts U (selenocysteine), which timsim's residue table lacked.
    // Computed from elements as C3H5NOSe (⁸⁰Se), the residue mass is the standard 150.95364 Da.
    let u_res = ms_chem::residue_monoisotopic_mass(b'U').unwrap();
    assert!((u_res - 150.95364).abs() < 1e-4, "ms-chem U residue={u_res}");

    // FINDING (R1, surfaced by the build): mscore hard-codes U = 168.053, which is NOT C3H5NOSe —
    // it disagrees with the element-derived value by ~17 Da and is simply wrong. This is exactly
    // what "compute residue masses from elements" protects against: a hard-coded literal can be
    // incorrect; a computed one cannot drift from the element table. ms-chem adopts the correct
    // value; mscore's U is a bug to fix when it folds onto ms-chem.
    use mscore::data::peptide::PeptideSequence;
    let mscore_u = PeptideSequence::new("U".to_string(), None).mono_isotopic_mass() - ms_chem::WATER;
    assert!(
        (mscore_u - 168.053).abs() < 1e-2 && (u_res - mscore_u).abs() > 15.0,
        "expected the documented mscore U discrepancy: mscore={mscore_u}, ms-chem={u_res}"
    );
}
