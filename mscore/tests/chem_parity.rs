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
