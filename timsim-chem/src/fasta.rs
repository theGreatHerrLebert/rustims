//! FASTA reading.
//!
//! Derived from Sage (MIT, Copyright (c) 2022 Michael Lazear). Decoy generation is
//! deliberately not carried over: a simulator models a sample, and a sample contains no
//! decoys. Decoys are search-engine apparatus.

/// One protein as read from a FASTA file. Carries no amount — abundance is a *quantity*,
/// and quantities live on a separate axis so that structure can be shared across samples.
#[derive(Clone, Debug, PartialEq)]
pub struct Protein {
    /// Accession, parsed from the header (`sp|P12345|NAME_HUMAN` → `P12345`).
    pub id: String,
    /// The full header line, minus the leading `>`.
    pub description: String,
    pub sequence: String,
}

/// Parse FASTA contents. Sequences are uppercased; `*` (stop) and whitespace are stripped.
///
/// Residues outside the 20 standard amino acids (`B`, `J`, `O`, `U`, `X`, `Z`) are
/// retained in the sequence — they are real entries in real databases — but they are not
/// cleavage sites and the caller is responsible for deciding what to do with peptides
/// containing them.
pub fn parse(contents: &str) -> Vec<Protein> {
    let mut proteins = Vec::new();
    let mut description: Option<String> = None;
    let mut sequence = String::new();

    let mut flush = |description: &mut Option<String>, sequence: &mut String| {
        if let Some(desc) = description.take() {
            if !sequence.is_empty() {
                proteins.push(Protein {
                    id: accession(&desc).to_string(),
                    description: desc,
                    sequence: std::mem::take(sequence),
                });
            }
        }
        sequence.clear();
    };

    for line in contents.lines() {
        let line = line.trim();
        if let Some(header) = line.strip_prefix('>') {
            flush(&mut description, &mut sequence);
            description = Some(header.to_string());
        } else if description.is_some() {
            for c in line.chars().filter(|c| c.is_ascii_alphabetic()) {
                sequence.push(c.to_ascii_uppercase());
            }
        }
    }
    flush(&mut description, &mut sequence);

    proteins
}

/// Extract an accession from a FASTA header.
///
/// `sp|P02768|ALBU_HUMAN Serum albumin` → `P02768`
/// `ENSP00000493376.2 pep chromosome:...` → `ENSP00000493376.2`
fn accession(header: &str) -> &str {
    let mut parts = header.split('|');
    let first = parts.next().unwrap_or("");
    match parts.next() {
        // UniProt style: db|accession|name
        Some(acc) if matches!(first, "sp" | "tr" | "SP" | "TR") && !acc.is_empty() => acc,
        _ => header.split_ascii_whitespace().next().unwrap_or(header),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_uniprot_headers_and_multiline_sequences() {
        let fasta = ">sp|P02768|ALBU_HUMAN Serum albumin OS=Homo sapiens\n\
                     MKWVTFISLL\n\
                     LLFSSAYSRG\n\
                     >tr|Q9Y6K9|Q9Y6K9_HUMAN NEMO\n\
                     PEPTIDEK\n";
        let proteins = parse(fasta);

        assert_eq!(proteins.len(), 2);
        assert_eq!(proteins[0].id, "P02768");
        assert_eq!(proteins[0].sequence, "MKWVTFISLLLLFSSAYSRG");
        assert_eq!(proteins[1].id, "Q9Y6K9");
        assert_eq!(proteins[1].sequence, "PEPTIDEK");
    }

    #[test]
    fn falls_back_to_the_first_token_for_non_uniprot_headers() {
        let proteins = parse(">ENSP00000493376.2 pep chromosome:GRCh38\nPEPTIDEK\n");
        assert_eq!(proteins[0].id, "ENSP00000493376.2");
    }

    #[test]
    fn strips_stop_codons_and_lowercase() {
        let proteins = parse(">x\npepTIDEk*\n");
        assert_eq!(proteins[0].sequence, "PEPTIDEK");
    }

    #[test]
    fn ignores_a_header_with_no_sequence() {
        assert!(parse(">empty\n>x\nPEPTIDEK\n").len() == 1);
    }
}
