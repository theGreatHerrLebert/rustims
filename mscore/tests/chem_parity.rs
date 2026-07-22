//! R1 — ms-chem parity: differential test of `mscore/chem` against the independent
//! `timsim-chem` v2 implementation over a deterministic corpus.
//!
//! Anti-self-consistency (the xcheck rule): the two impls were written independently, so
//! agreement here is real evidence that the unified `ms-chem` can adopt either's residue /
//! element tables and mass summation without silently changing simulation output. Any
//! divergence is either a bug to fix or a semantic difference to document *before* dedup.
//!
//! This first test covers the neutral monoisotopic peptide mass (residues + water). Isotope
//! envelopes and modification masses get their own tests as R1 proceeds.

use mscore::data::peptide::PeptideSequence;

/// 20 standard amino acids. Selenocysteine (U) and edge residues are checked separately —
/// mscore accepts U, timsim-chem's residue table may not, which is itself a parity item.
const AAS: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";

/// Deterministic corpus: exhaustive 1/2/3-mers + seeded pseudo-random longer peptides.
/// No RNG crate / no Date — a fixed LCG so the corpus is identical on every run.
fn corpus() -> Vec<String> {
    let mut v = Vec::new();
    for &a in AAS {
        v.push((a as char).to_string());
    }
    for &a in AAS {
        for &b in AAS {
            v.push(format!("{}{}", a as char, b as char));
        }
    }
    for &a in AAS {
        for &b in AAS {
            for &c in AAS {
                v.push(format!("{}{}{}", a as char, b as char, c as char));
            }
        }
    }
    let mut s: u64 = 0x9e3779b97f4a7c15;
    for &len in &[5usize, 10, 20, 30, 50] {
        for _ in 0..2000 {
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
fn monoisotopic_mass_parity() {
    let corpus = corpus();
    // Relative tolerance is the right metric: the two impls represent residue masses
    // differently (mscore sums element masses; timsim-chem hardcodes residue literals),
    // so a chemically-identical residue can differ by a few ULPs that accumulate ~linearly
    // in sequence length. 1e-7 sits far above that float floor (~1e-8 on 50-mers) and far
    // below any real semantic error (a wrong element/residue mass is ~1e-4 relative or worse).
    let rel_tol = 1e-7;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut divergences = 0usize;
    let mut worst = String::new();

    for seq in &corpus {
        let m_mscore = PeptideSequence::new(seq.clone(), None).mono_isotopic_mass();
        let m_timsim = timsim_chem::mass::monoisotopic(seq).expect("valid residue");
        let d = (m_mscore - m_timsim).abs();
        let rel = d / m_mscore.abs().max(1.0);
        if rel > max_rel {
            max_rel = rel;
            max_abs = d;
            worst = seq.clone();
        }
        if rel > rel_tol {
            divergences += 1;
        }
    }

    eprintln!(
        "[parity/mono-mass] corpus={} seqs  max_rel={:.3e}  max|Δ|={:.3e} Da  \
         divergences(rel>{:.0e})={}  worst={}",
        corpus.len(),
        max_rel,
        max_abs,
        rel_tol,
        divergences,
        worst
    );

    // A true semantic divergence (wrong table entry, stray proton/water term) would show
    // up as O(1e-4) relative or a constant offset, not float rounding.
    assert_eq!(
        divergences, 0,
        "mono-mass parity: {} sequences diverge beyond rel {:.0e} (max_rel {:.3e} on {:?})",
        divergences, rel_tol, max_rel, worst
    );
}

/// Gate 2 — isotope envelope parity. mscore convolves *actual isotope masses* from its
/// `isotopic_abundance()` table (fine structure, merged at 1e-3 Da); timsim-chem convolves
/// nominal neutron-count abundances (`AB_*`) to `depth` peaks. We bin mscore's peaks to
/// nominal isotope index, normalise both envelopes to sum=1, and compare peak-by-peak.
///
/// KNOWN divergence source (documented, not a bug): the two abundance tables differ for
/// N ([0.99632,0.00368] vs [0.99636,0.00364]) and S ([0.9493,0.0076,0.0429] vs
/// [0.9499,0.0075,0.0425,_,0.0001]) — different IUPAC/CIAAW revisions. C/H/O are identical.
/// So envelopes are expected to agree to ~1e-3 (table difference), not to float precision.
/// The canonical `ms-chem` must pick ONE abundance table; this test measures the stakes.
#[test]
fn isotope_envelope_parity() {
    use mscore::data::peptide::PeptideSequence;
    use std::collections::HashMap;

    const DEPTH: usize = 6;
    let corpus = corpus();
    let mut max_peak_diff = 0.0f64;
    let mut worst = String::new();
    let mut worst_env_mscore = vec![];
    let mut worst_env_timsim = vec![];
    // histogram of the divergence to show it's a smooth table-difference, not outliers
    let mut over_1e_3 = 0usize;

    for seq in &corpus {
        // mscore: composition -> fine-structure distribution -> nominal bins
        let comp: HashMap<String, i32> = PeptideSequence::new(seq.clone(), None)
            .atomic_composition()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let dist =
            mscore::algorithm::isotope::generate_isotope_distribution(&comp, 1e-3, 1e-9, 200);
        let mono = dist.iter().map(|(m, _)| *m).fold(f64::INFINITY, f64::min);
        let mut bins = vec![0.0f64; DEPTH];
        for (m, a) in &dist {
            let k = (m - mono).round() as usize;
            if k < DEPTH {
                bins[k] += *a;
            }
        }
        let s: f64 = bins.iter().sum();
        if s > 0.0 {
            for b in &mut bins {
                *b /= s;
            }
        }

        // timsim: nominal envelope, already sum=1 over DEPTH
        let env: Vec<f64> = timsim_chem::isotope::envelope(seq, DEPTH)
            .expect("valid residue")
            .into_iter()
            .map(|x| x as f64)
            .collect();

        let d = bins
            .iter()
            .zip(&env)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if d > 1e-3 {
            over_1e_3 += 1;
        }
        if d > max_peak_diff {
            max_peak_diff = d;
            worst = seq.clone();
            worst_env_mscore = bins.clone();
            worst_env_timsim = env.clone();
        }
    }

    eprintln!(
        "[parity/isotope] corpus={} depth={}  max|Δ peak abundance|={:.3e}  \
         peptides(>1e-3)={}  worst={}",
        corpus.len(),
        DEPTH,
        max_peak_diff,
        over_1e_3,
        worst
    );
    eprintln!("  worst mscore envelope: {:?}", fmt(&worst_env_mscore));
    eprintln!("  worst timsim envelope: {:?}", fmt(&worst_env_timsim));

    // Bound reflects the KNOWN N/S abundance-table difference (~1e-3), not float precision.
    // If this ever blows past ~5e-3, something beyond the abundance tables has diverged.
    assert!(
        max_peak_diff < 5e-3,
        "isotope envelope divergence {:.3e} exceeds the abundance-table budget on {:?}",
        max_peak_diff,
        worst
    );
}

fn fmt(v: &[f64]) -> Vec<String> {
    v.iter().map(|x| format!("{:.5}", x)).collect()
}

/// Gate 4 — backbone fragment ions (b / y ladders). Both compute pure chemistry: a running residue
/// sum + terminal group + proton, so this also exercises the proton constant (mscore
/// MASS_PROTON=1.007276466621 vs timsim PROTON=1.007276466 — ~6e-10 Da apart).
///
/// API note (verified empirically): mscore's `calculate_mono_isotopic_product_ion_spectrum` returns
/// the FULL b+y spectrum regardless of the `FragmentType` argument (it builds both terminal series),
/// while timsim's `fragment_ions` returns typed b and y fragments. So we compare mscore's full
/// spectrum (one call) against timsim's full b∪y set, aligned by sorted m/z.
///
/// Uses a focused corpus (exhaustive di/tri-peptides cover every residue-pair transition; a small
/// sample of longer peptides adds ladder depth) — mscore's product-ion builder is heavy, and ladder
/// chemistry does not need 50-mers to be exercised.
#[test]
fn fragment_ion_parity() {
    use mscore::data::peptide::{FragmentType, PeptideSequence};
    use timsim_chem::fragment::fragment_ions;

    let full = corpus();
    let mut frag_corpus: Vec<&String> =
        full.iter().filter(|s| (2..=3).contains(&s.len())).collect();
    frag_corpus.extend(full.iter().filter(|s| s.len() >= 10).take(300));

    let abs_tol = 1e-5; // residue tables agree ~1e-8/residue + proton ~6e-10
    let mut max_abs = 0.0f64;
    let mut worst = String::new();
    let mut count_mismatches = 0usize;
    let mut compared = 0usize;

    for seq in &frag_corpus {
        let ps = PeptideSequence::new((*seq).clone(), None);
        // one call -> the complete b+y spectrum
        let mut ms: Vec<f64> = ps
            .calculate_mono_isotopic_product_ion_spectrum(1, FragmentType::B)
            .mz
            .iter()
            .copied()
            .collect();
        let ti = match fragment_ions(seq, 1) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut ts: Vec<f64> = ti.iter().map(|f| f.mz).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // mscore returns an MzSpectrum, which MERGES isobaric peaks (a b ion and a y ion can share
        // an m/z), while timsim lists both typed fragments. Compare unique m/z sets: dedupe near-equal
        // values in both so the merge is not counted as a divergence (the values themselves match).
        ms.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

        if ms.len() != ts.len() {
            count_mismatches += 1;
            continue;
        }
        for (a, b) in ms.iter().zip(&ts) {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
                worst = (*seq).clone();
            }
            compared += 1;
        }
    }

    eprintln!(
        "[parity/fragment] peptides={} ions_compared={}  max|Δ m/z|={:.3e}  \
         ladder-length mismatches={}  worst={}",
        frag_corpus.len(),
        compared,
        max_abs,
        count_mismatches,
        worst
    );

    assert_eq!(
        count_mismatches, 0,
        "fragment ladder-length mismatch on {} peptides (b/y enumeration differs)",
        count_mismatches
    );
    // A wrong terminal group or ion definition would show O(1 Da); proton-constant + residue
    // rounding live far below abs_tol.
    assert!(
        max_abs < abs_tol,
        "fragment m/z divergence {:.3e} exceeds {:.0e} on {:?}",
        max_abs,
        abs_tol,
        worst
    );
}

/// Gate 3 — modification masses. timsim-chem has NO UNIMOD mirror: its modifications are
/// `Modification` structs carrying a `mass_delta` + elemental `composition`, self-validated
/// (composition mono-mass must equal mass_delta). mscore owns the authoritative 2144-entry
/// UNIMOD table (`unimod_modifications_mass_numerical()`), which is canonical for ms-chem.
///
/// This checks the two independent routes agree for the modifications the simulator actually
/// uses: (1) timsim's hand-entered `mass_delta`, (2) mscore's UNIMOD table entry, and (3) the
/// composition string re-parsed through mscore's element table. All three must land on one
/// number, or a modified precursor's m/z would silently drift.
#[test]
fn modification_mass_parity() {
    use mscore::chemistry::sum_formula::SumFormula;
    use mscore::chemistry::unimod::unimod_modifications_mass_numerical;

    // (unimod_id, name, timsim mass_delta, timsim composition) — the curated set timsim-chem
    // uses, taken from its own fixtures. Compositions are additive (no losses) for this set.
    let mods: &[(u32, &str, f64, &str)] = &[
        (21, "Phospho", 79.96633, "HO3P"),
        (35, "Oxidation", 15.99491, "O"),
        (4, "Carbamidomethyl", 57.02146, "C2H3NO"),
        (1, "Acetyl", 42.01057, "C2H2O"),
        (37, "Trimethyl", 42.04695, "C3H6"),
        (121, "GG", 114.04293, "C4H6N2O2"),
    ];

    use mscore::chemistry::elements::atomic_weights_mono_isotopic;
    use mscore::chemistry::unimod::modification_atomic_composition;

    let mass_table = unimod_modifications_mass_numerical(); // 1028 entries (mass only)
    let comp_table = modification_atomic_composition(); // 17 entries (composition only)
    let weights = atomic_weights_mono_isotopic();
    let tol = 1e-3; // UNIMOD masses are tabulated to ~5 decimals
    let mut worst = 0.0f64;
    let mut mass_table_gaps = vec![];
    let mut comp_table_gaps = vec![];

    for &(id, name, timsim_delta, comp) in mods {
        // Route A: mscore's numerical mass table (may be missing — a coverage gap, not a panic).
        let route_a = mass_table.get(&id).copied();
        // Route B: mscore's composition table -> mass via its element weights (different, smaller set).
        let route_b = comp_table.get(&format!("[UNIMOD:{id}]")).map(|els| {
            els.iter()
                .fold(0.0, |acc, (el, n)| acc + weights[el] * *n as f64)
        });
        // Route C: timsim's declared composition string -> mass via mscore's parser (the oracle).
        let route_c = SumFormula::new(comp).monoisotopic_weight();

        if route_a.is_none() {
            mass_table_gaps.push((id, name));
        }
        if route_b.is_none() {
            comp_table_gaps.push((id, name));
        }

        // Every route that EXISTS must land on timsim's mass_delta.
        for (label, val) in [("mass_table", route_a), ("comp_table", route_b)] {
            if let Some(v) = val {
                let d = (v - timsim_delta).abs();
                worst = worst.max(d);
                assert!(
                    d < tol,
                    "{name} (id {id}): mscore {label} {v:.5} vs timsim {timsim_delta:.5} \
                     diverge by {d:.2e}"
                );
            }
        }
        let d_c = (route_c - timsim_delta).abs();
        worst = worst.max(d_c);
        assert!(d_c < tol, "{name}: composition {comp} -> {route_c:.5} vs {timsim_delta:.5}");

        eprintln!(
            "[parity/mod] {name:<16} id={id:<4} timsim={timsim_delta:.5}  \
             mass_table={}  comp_table={}  from_comp({comp})={route_c:.5}",
            route_a.map_or("MISSING".into(), |v| format!("{v:.5}")),
            route_b.map_or("MISSING".into(), |v| format!("{v:.5}")),
        );
    }

    eprintln!(
        "[parity/mod] {} mods checked; worst present-route Δ={:.2e}",
        mods.len(),
        worst
    );
    // FINDINGS (documented, drive the ms-chem design; not failures — the masses that DO exist agree):
    // mscore's mass table (1028) and composition table (17) are largely DISJOINT sets, so some
    // sim-used mods are in one but not the other. ms-chem must unify them into one coverage-consistent
    // table (timsim's Modification pattern: composition + mass together, cross-checked at load).
    if !mass_table_gaps.is_empty() {
        eprintln!("[parity/mod] FINDING — in mscore composition table but MISSING from its mass table: {:?}", mass_table_gaps);
    }
    if !comp_table_gaps.is_empty() {
        eprintln!("[parity/mod] FINDING — in mscore mass table but MISSING from its composition table: {:?}", comp_table_gaps);
    }
}
