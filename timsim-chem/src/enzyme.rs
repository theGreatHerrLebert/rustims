//! Enzyme cleavage rules.
//!
//! The cleavage-site logic is derived from Sage (MIT, Copyright (c) 2022 Michael
//! Lazear, <https://github.com/lazear/sage>), specifically `crates/sage/src/enzyme.rs`.
//!
//! What is deliberately NOT carried over: `Digest`, `decoy` / `reverse()`, and
//! `missed_cleavages` as a generative bound. Those model a *search space* — the set of
//! candidates an engine must consider. We model a *sample* — the molecules that exist,
//! and how much of each. Cleavage rules are chemistry and are shared; enumeration
//! semantics are not. See `digest.rs`.

use regex::Regex;

const VALID_AA: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";

/// Where an enzyme cuts, and where it refuses to.
#[derive(Clone, Debug)]
pub struct Enzyme {
    /// Residues after which (or before which) the enzyme cleaves.
    regex: Regex,
    /// Cleavage is blocked when the site is followed by one of these residues
    /// (indexed `b - b'A'`). Trypsin's "not before proline" rule.
    skip_suffix: [bool; 26],
    /// Cleave C-terminal to the matched residue (trypsin) or N-terminal to it (Asp-N).
    c_terminal: bool,
}

impl Enzyme {
    /// `cleave`: residues defining a cleavage site, e.g. `"KR"` for trypsin.
    /// `skip_suffix`: residues that block cleavage when they follow the site, e.g. `"P"`.
    ///
    /// Returns `None` for an empty rule (i.e. a non-specific digest).
    pub fn new(cleave: &str, skip_suffix: &str, c_terminal: bool) -> Option<Self> {
        assert!(
            cleave.bytes().all(|b| VALID_AA.contains(&b)),
            "enzyme cleavage residues contain non-amino-acid characters: {cleave}"
        );
        assert!(
            skip_suffix.bytes().all(|b| VALID_AA.contains(&b)),
            "enzyme cleavage restriction contains non-amino-acid characters: {skip_suffix}"
        );

        if cleave.is_empty() {
            return None;
        }

        Some(Enzyme {
            regex: Regex::new(&format!("[{cleave}]")).expect("valid character class"),
            skip_suffix: {
                let mut arr = [false; 26];
                for b in skip_suffix.bytes() {
                    arr[(b - b'A') as usize] = true;
                }
                arr
            },
            c_terminal,
        })
    }

    /// Internal cleavage-site positions within `sequence`, as residue offsets.
    ///
    /// A position `k` means "the bond between residue `k-1` and residue `k`". Protein
    /// termini (0 and `sequence.len()`) are **not** included — they are not cleavage
    /// events, and conflating them with cleavage sites is a modelling error that gives
    /// terminal peptides the wrong yield. See `digest::Digester`.
    pub fn cleavage_positions(&self, sequence: &str) -> Vec<usize> {
        let bytes = sequence.as_bytes();
        let mut positions = Vec::new();

        for mat in self.regex.find_iter(sequence) {
            let pos = if self.c_terminal {
                mat.end()
            } else {
                mat.start()
            };

            // A cut at 0 or at len() is not a cut — it is a terminus.
            if pos == 0 || pos >= sequence.len() {
                continue;
            }

            // Blocked by the residue following the site (e.g. trypsin's `not_before = P`).
            if bytes
                .get(pos)
                .is_some_and(|b| self.skip_suffix[(b - b'A') as usize])
            {
                continue;
            }

            positions.push(pos);
        }

        positions.dedup();
        positions
    }
}

/// A digestion protocol: one or more enzymes applied together.
///
/// Multiple enzymes take the **union** of their cleavage sites, which is what a double
/// digest (e.g. trypsin + Lys-C) actually does.
#[derive(Clone, Debug, Default)]
pub struct Protocol {
    enzymes: Vec<Enzyme>,
}

impl Protocol {
    pub fn new(enzymes: Vec<Enzyme>) -> Self {
        Protocol { enzymes }
    }

    /// Look up a protocol by name. `+` unions enzymes: `"trypsin+lysc"`.
    ///
    /// Enzyme definitions are data, not code — this table is the built-in default and is
    /// expected to be overridable from a TOML file. Chemists will ask for this.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut enzymes = Vec::new();
        for name in spec.split('+') {
            let name = name.trim().to_ascii_lowercase();
            let enzyme = match name.as_str() {
                // (cleave, skip_suffix, c_terminal)
                "trypsin" => Enzyme::new("KR", "P", true),
                "trypsin/p" | "trypsin_p" => Enzyme::new("KR", "", true),
                "lysc" => Enzyme::new("K", "P", true),
                "lysn" => Enzyme::new("K", "", false),
                "argc" => Enzyme::new("R", "P", true),
                "gluc" => Enzyme::new("DE", "P", true),
                "aspn" => Enzyme::new("D", "", false),
                "chymotrypsin" => Enzyme::new("FWYL", "P", true),
                other => return Err(format!("unknown enzyme: {other:?}")),
            };
            enzymes.push(enzyme.ok_or_else(|| format!("empty cleavage rule for {name:?}"))?);
        }
        if enzymes.is_empty() {
            return Err("empty enzyme specification".to_string());
        }
        Ok(Protocol { enzymes })
    }

    /// Union of every enzyme's cleavage positions, sorted and deduplicated.
    pub fn cleavage_positions(&self, sequence: &str) -> Vec<usize> {
        let mut positions: Vec<usize> = self
            .enzymes
            .iter()
            .flat_map(|e| e.cleavage_positions(sequence))
            .collect();
        positions.sort_unstable();
        positions.dedup();
        positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trypsin_cuts_after_k_and_r() {
        // PEPTIDEK | SEQUENCER | ENDS
        let p = Protocol::parse("trypsin").unwrap();
        assert_eq!(p.cleavage_positions("PEPTIDEKSEQUENCERENDS"), vec![8, 17]);
    }

    #[test]
    fn trypsin_does_not_cut_before_proline() {
        // The K at index 7 is followed by P, so the bond at position 8 is not cleaved.
        let p = Protocol::parse("trypsin").unwrap();
        assert_eq!(p.cleavage_positions("PEPTIDEKPEPTIDER"), Vec::<usize>::new());

        // trypsin/p ignores the proline rule.
        let pp = Protocol::parse("trypsin/p").unwrap();
        assert_eq!(pp.cleavage_positions("PEPTIDEKPEPTIDER"), vec![8]);
    }

    #[test]
    fn terminal_residues_are_not_cleavage_sites() {
        // A trailing K is the protein C-terminus, not a cut. A protein is not cleaved
        // off its own end — this is the bug that gives terminal peptides p instead of 1.
        let p = Protocol::parse("trypsin").unwrap();
        assert_eq!(p.cleavage_positions("PEPTIDEK"), Vec::<usize>::new());
        assert_eq!(p.cleavage_positions("PEPTIDEKAAAK"), vec![8]);
    }

    #[test]
    fn aspn_cuts_n_terminal_to_d() {
        // Asp-N cleaves *before* D, so the site is at the index of D itself.
        let p = Protocol::parse("aspn").unwrap();
        assert_eq!(p.cleavage_positions("AAADAAA"), vec![3]);
    }

    #[test]
    fn double_digest_unions_sites() {
        // Trypsin cuts after K and R; Glu-C after D and E. Together: all four.
        let t = Protocol::parse("trypsin").unwrap();
        let g = Protocol::parse("gluc").unwrap();
        let both = Protocol::parse("trypsin+gluc").unwrap();

        let seq = "AAKAAEAARAADAA";
        let mut expected: Vec<usize> = t
            .cleavage_positions(seq)
            .into_iter()
            .chain(g.cleavage_positions(seq))
            .collect();
        expected.sort_unstable();
        expected.dedup();

        assert_eq!(both.cleavage_positions(seq), expected);
        assert!(both.cleavage_positions(seq).len() > t.cleavage_positions(seq).len());
    }

    #[test]
    fn unknown_enzyme_is_an_error() {
        assert!(Protocol::parse("nuclease").is_err());
    }
}
