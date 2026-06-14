//! Fast spectral-library writer: decode Prosit intensity vectors into a DiaNN/
//! Spectronaut-style transition `.tsv` in parallel.
//!
//! The per-peptide decode (`PeptideSequence::associate_with_predicted_intensities`)
//! is already a Rust kernel in `mscore`; the slow part of a Python library builder is
//! calling it millions of times across the PyO3 boundary and assembling a giant
//! DataFrame. This does the whole batch in Rust with `rayon` and streams the rows
//! straight to disk — turning a multi-hour build into seconds.
//!
//! The caller (Python) supplies the cheap string columns it already has
//! (`ModifiedPeptide` in DiaNN `(UniMod:n)` form, the stripped `PeptideSequence`) so
//! this module needs no mod-notation regex; ordinals come from the fragment ion's own
//! `amino_acid_count()`.

use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use mscore::data::peptide::{FragmentType, PeptideSequence};
use rayon::prelude::*;

/// DiaNN/Spectronaut transition columns, in output order.
pub const LIBRARY_HEADER: &str = "ProteinId\tGenes\tModifiedPeptide\tPeptideSequence\t\
PrecursorCharge\tPrecursorMz\tTr_recalibrated\tFragmentType\tFragmentCharge\t\
FragmentSeriesNumber\tProductMz\tLibraryIntensity";

/// Decode one batch of peptides (parallel) and write/append their transitions to
/// `out_path` as a DiaNN-style `.tsv`.
///
/// Per precursor: b/y ions are read off the reshaped Prosit array, intensities are
/// renormalised to max=1, non-finite / non-positive intensities and m/z are dropped,
/// and a precursor with fewer than `min_fragments` surviving fragments is skipped
/// entirely (it emits no rows). Returns `(transitions_written, precursors_written)`.
///
/// `prosit_flat` is row-major: peptide `i`'s vector is `prosit_flat[i*n_cols..(i+1)*n_cols]`.
/// All metadata slices must have the same length (the peptide count); `n_cols` is the
/// Prosit vector width (174). With `append=false` the file is truncated and the header
/// written first; with `append=true` rows are appended (no header) — for chunked builds.
#[allow(clippy::too_many_arguments)]
pub fn build_spectral_library_tsv(
    out_path: &Path,
    modified_sequences: &[String],
    modpep_diann: &[String],
    stripped: &[String],
    charges: &[i32],
    prosit_flat: &[f64],
    n_cols: usize,
    precursor_mz: &[f64],
    retention_time: &[f64],
    protein_ids: &[String],
    genes: &[String],
    min_fragments: usize,
    append: bool,
) -> io::Result<(usize, usize)> {
    let n = modified_sequences.len();
    for (name, len) in [
        ("modpep_diann", modpep_diann.len()),
        ("stripped", stripped.len()),
        ("charges", charges.len()),
        ("precursor_mz", precursor_mz.len()),
        ("retention_time", retention_time.len()),
        ("protein_ids", protein_ids.len()),
        ("genes", genes.len()),
    ] {
        if len != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("column '{name}' length {len} != peptide count {n}"),
            ));
        }
    }
    if prosit_flat.len() != n * n_cols {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "prosit_flat length {} != peptides {} * n_cols {}",
                prosit_flat.len(),
                n,
                n_cols
            ),
        ));
    }

    // Decode each peptide independently; emit a String block of tsv rows (empty if the
    // precursor is skipped). Pure per-item work -> embarrassingly parallel.
    let blocks: Vec<String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let intensities = prosit_flat[i * n_cols..(i + 1) * n_cols].to_vec();
            let pep = PeptideSequence::new(modified_sequences[i].clone(), None);
            let coll = pep.associate_with_predicted_intensities(
                charges[i],
                FragmentType::B,
                intensities,
                true,
                true,
            );
            // Gather surviving fragments and the per-precursor max for renormalisation.
            let mut frags: Vec<(String, i32, usize, f64, f64)> = Vec::new();
            let mut max_i = 0.0_f64;
            for series in &coll.peptide_ions {
                for ion in series.n_ions.iter().chain(series.c_ions.iter()) {
                    let inten = ion.ion.intensity;
                    let mz = ion.mz();
                    if !inten.is_finite() || inten <= 0.0 || !mz.is_finite() {
                        continue;
                    }
                    let ordinal = ion.ion.sequence.amino_acid_count();
                    if inten > max_i {
                        max_i = inten;
                    }
                    frags.push((ion.kind.to_string(), ion.ion.charge, ordinal, mz, inten));
                }
            }
            if frags.len() < min_fragments || max_i <= 0.0 {
                return String::new();
            }
            let mut block = String::with_capacity(frags.len() * 80);
            for (kind, fch, ord, mz, inten) in &frags {
                // ProteinId Genes ModifiedPeptide PeptideSequence PrecursorCharge
                // PrecursorMz Tr FragmentType FragmentCharge FragmentSeriesNumber
                // ProductMz LibraryIntensity
                block.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    protein_ids[i],
                    genes[i],
                    modpep_diann[i],
                    stripped[i],
                    charges[i],
                    precursor_mz[i],
                    retention_time[i],
                    kind,
                    fch,
                    ord,
                    mz,
                    inten / max_i,
                ));
            }
            block
        })
        .collect();

    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(append)
        .truncate(!append)
        .open(out_path)?;
    let mut w = BufWriter::new(file);
    if !append {
        writeln!(w, "{LIBRARY_HEADER}")?;
    }
    let (mut n_trans, mut n_prec) = (0usize, 0usize);
    for block in &blocks {
        if block.is_empty() {
            continue;
        }
        n_prec += 1;
        n_trans += block.as_bytes().iter().filter(|&&b| b == b'\n').count();
        w.write_all(block.as_bytes())?;
    }
    w.flush()?;
    Ok((n_trans, n_prec))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(s: &str) -> Vec<String> {
        vec![s.to_string()]
    }

    #[test]
    fn rejects_mismatched_column_lengths() {
        let dir = std::env::temp_dir().join("rustdf_lib_badcols.tsv");
        // charges has 2 entries but everything else has 1 -> error, no file contract.
        let r = build_spectral_library_tsv(
            &dir,
            &one("PEPTIDEK"),
            &one("PEPTIDEK"),
            &one("PEPTIDEK"),
            &[2, 3],
            &vec![0.0; 174],
            174,
            &[500.0],
            &[10.0],
            &one("P"),
            &one("G"),
            3,
            false,
        );
        assert!(r.is_err(), "mismatched column lengths must error");
    }

    #[test]
    fn rejects_wrong_prosit_flat_size() {
        let dir = std::env::temp_dir().join("rustdf_lib_badflat.tsv");
        let r = build_spectral_library_tsv(
            &dir, &one("PEPK"), &one("PEPK"), &one("PEPK"), &[2],
            &vec![0.0; 100], 174, &[500.0], &[10.0], &one("P"), &one("G"), 3, false,
        );
        assert!(r.is_err(), "prosit_flat length != n*n_cols must error");
    }

    #[test]
    fn all_zero_intensities_emit_header_only() {
        // A zero Prosit vector -> every fragment intensity 0 -> precursor skipped
        // (< min_fragments). Replace mode still writes the header and an empty body.
        let out = std::env::temp_dir().join("rustdf_lib_zero.tsv");
        let (n_trans, n_prec) = build_spectral_library_tsv(
            &out, &one("PEPTIDEK"), &one("PEPTIDEK"), &one("PEPTIDEK"), &[2],
            &vec![0.0; 174], 174, &[500.0], &[10.0], &one("P"), &one("G"), 3, false,
        )
        .expect("write");
        assert_eq!((n_trans, n_prec), (0, 0));
        let body = std::fs::read_to_string(&out).expect("read");
        assert!(body.starts_with("ProteinId\t"), "header present");
        assert_eq!(body.lines().count(), 1, "header only, no transition rows");
    }
}
