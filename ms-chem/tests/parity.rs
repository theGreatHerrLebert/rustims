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
fn isotope_envelope_matches_timsim_and_mscore() {
    use mscore::data::peptide::PeptideSequence;
    use std::collections::HashMap;

    const DEPTH: usize = 6;
    let corpus = corpus();
    // ms-chem adopts CIAAW abundances + timsim's algorithm, so it should match timsim ~exactly
    // (to f32 rounding, since timsim returns f32), and match mscore within the N/S abundance-table
    // budget documented in Gate 2 (mscore uses the older values).
    let mut max_vs_timsim = 0.0f64;
    let mut max_vs_mscore = 0.0f64;

    for seq in &corpus {
        let x = ms_chem::envelope(seq, DEPTH).expect("std residue");

        let t: Vec<f64> = timsim_chem::isotope::envelope(seq, DEPTH)
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect();
        max_vs_timsim = max_vs_timsim.max(
            x.iter().zip(&t).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max),
        );

        // mscore: fine-structure distribution binned to nominal, normalized over DEPTH
        let comp: HashMap<String, i32> = PeptideSequence::new(seq.clone(), None)
            .atomic_composition()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let dist = mscore::algorithm::isotope::generate_isotope_distribution(&comp, 1e-3, 1e-9, 200);
        let mono = dist.iter().map(|(m, _)| *m).fold(f64::INFINITY, f64::min);
        let mut bins = vec![0.0f64; DEPTH];
        for (m, a) in &dist {
            let k = (m - mono).round() as usize;
            if k < DEPTH {
                bins[k] += *a;
            }
        }
        let s: f64 = bins.iter().sum();
        for b in &mut bins {
            *b /= s;
        }
        max_vs_mscore = max_vs_mscore.max(
            x.iter().zip(&bins).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max),
        );
    }

    eprintln!(
        "[ms-chem/isotope] corpus={}  max|Δ| vs timsim={max_vs_timsim:.3e}  vs mscore={max_vs_mscore:.3e}",
        corpus.len()
    );
    // vs timsim: same tables+algo → f32 rounding only.
    assert!(max_vs_timsim < 1e-6, "ms-chem vs timsim envelope {max_vs_timsim:.3e}");
    // vs mscore: bounded by the N/S abundance-table difference (Gate 2 ~1.3e-3; corpus has no
    // pathological homopolymers, so it stays small).
    assert!(max_vs_mscore < 5e-3, "ms-chem vs mscore envelope {max_vs_mscore:.3e}");
}

#[test]
fn fragment_ions_match_mscore_and_timsim() {
    use mscore::data::peptide::{FragmentType, PeptideSequence};

    // focused corpus (mscore's product-ion builder is heavy); exhaustive di/tri + a sample
    let full = corpus();
    let mut fc: Vec<&String> = full.iter().filter(|s| (2..=3).contains(&s.len())).collect();
    fc.extend(full.iter().filter(|s| s.len() >= 10).take(300));

    let mut max_vs_mscore = 0.0f64;
    let mut max_vs_timsim = 0.0f64;

    for seq in &fc {
        // ms-chem: full b+y set at charge 1
        let mut x: Vec<f64> = ms_chem::fragment_ions(seq, 1).unwrap().iter().map(|f| f.mz).collect();
        // timsim: full b+y set
        let mut t: Vec<f64> = timsim_chem::fragment::fragment_ions(seq, 1).unwrap().iter().map(|f| f.mz).collect();
        // mscore: full spectrum (returns b+y for any FragmentType; merges isobaric peaks)
        let mut m: Vec<f64> = PeptideSequence::new((*seq).clone(), None)
            .calculate_mono_isotopic_product_ion_spectrum(1, FragmentType::B)
            .mz
            .iter()
            .copied()
            .collect();
        for v in [&mut x, &mut t, &mut m] {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.dedup_by(|a, b| (*a - *b).abs() < 1e-6); // account for mscore's isobaric merge
        }
        assert_eq!(x.len(), t.len(), "ms-chem vs timsim ladder length on {seq}");
        assert_eq!(x.len(), m.len(), "ms-chem vs mscore ladder length on {seq}");
        max_vs_timsim = max_vs_timsim.max(x.iter().zip(&t).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));
        max_vs_mscore = max_vs_mscore.max(x.iter().zip(&m).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));
    }

    eprintln!(
        "[ms-chem/fragment] peptides={}  max|Δ m/z| vs mscore={max_vs_mscore:.3e}  vs timsim={max_vs_timsim:.3e}",
        fc.len()
    );
    assert!(max_vs_mscore < 1e-5, "ms-chem vs mscore fragments {max_vs_mscore:.3e}");
    assert!(max_vs_timsim < 1e-5, "ms-chem vs timsim fragments {max_vs_timsim:.3e}");
}

#[test]
fn modification_catalog_matches_mscore_and_is_coverage_consistent() {
    use mscore::chemistry::unimod::unimod_modifications_mass_numerical;

    // (1) every ms-chem modification cross-checks (mass == composition-derived mass)
    for m in ms_chem::BUILTIN_MODIFICATIONS {
        m.validate().unwrap_or_else(|e| panic!("{e}"));
    }

    // (2) where mscore's UNIMOD mass table has the same id, the masses agree (Gate 3)
    let mscore_table = unimod_modifications_mass_numerical();
    let mut checked = 0;
    for m in ms_chem::BUILTIN_MODIFICATIONS {
        if let Some(&mass) = mscore_table.get(&m.unimod_id) {
            assert!(
                (mass - m.mass_delta).abs() < 1e-3,
                "{} (id {}): ms-chem {:.5} vs mscore {:.5}",
                m.name, m.unimod_id, m.mass_delta, mass
            );
            checked += 1;
        }
    }
    assert!(checked >= 5, "expected the shared sim mods to be present in mscore's table");

    // (3) the coverage fix: GG (121) — which mscore had in its COMPOSITION table but NOT its mass
    // table — is present in ms-chem with BOTH mass and composition, cross-checked.
    let gg = ms_chem::modification_by_id(121).expect("GG present in ms-chem");
    assert_eq!(gg.name, "GG");
    gg.validate().unwrap();
    assert!(mscore_table.get(&121).is_none(), "precondition: mscore mass table lacks 121");

    eprintln!(
        "[ms-chem/mod] {} builtins cross-check; {checked} agree with mscore's mass table; \
         GG(121) coverage gap fixed",
        ms_chem::BUILTIN_MODIFICATIONS.len()
    );
}

#[test]
fn selenocysteine_correct_and_mscore_bug_fixed_by_fold() {
    // ms-chem is a superset: it accepts U (selenocysteine), which timsim's residue table lacked.
    // Computed from elements as C3H5NOSe (⁸⁰Se), the residue mass is the standard 150.95364 Da.
    let u_res = ms_chem::residue_monoisotopic_mass(b'U').unwrap();
    assert!((u_res - 150.95364).abs() < 1e-4, "ms-chem U residue={u_res}");

    // The build surfaced that mscore hard-coded U = 168.053 (not C3H5NOSe, off ~17 Da). The R1 fold
    // delegates mscore's U to ms-chem, so it is now CORRECT: mscore's residue mass for U agrees with
    // ms-chem. (This is the compute-from-elements principle closing the loop — the bug the fold
    // surfaced is the bug the fold fixes.)
    use mscore::chemistry::amino_acid::amino_acid_masses;
    let mscore_u = amino_acid_masses()["U"];
    assert!(
        (mscore_u - u_res).abs() < 1e-9,
        "post-fold mscore U should match ms-chem: mscore={mscore_u}, ms-chem={u_res}"
    );
}
