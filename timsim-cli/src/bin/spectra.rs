//! `timsim-spectra` — materialise the **instrument-independent** two-spectra-per-ion artifact.
//!
//! For each peptide ion (precursor) it writes two rows: an **MS1** spectrum (precursor isotopes) and
//! an **MS2** spectrum (fragment isotopes, with Prosit intensities), both as pure `(m/z, intensity)`
//! via mscore's peptide-ion path. No instrument geometry: a downstream *projector* (`timsim-render`
//! for timsTOF, others for Thermo/Sciex) maps these peaks onto `(frame, scan, tof)` / `(scan, m/z)`.
//!
//! This is the seam that lets one spectrum computation drive any instrument — the chemistry is done
//! once here; only the projection is instrument-specific.

use anyhow::{anyhow, Result};
use arrow::array::{Array, Float32Array, UInt8Array, UInt16Array, UInt64Array};
use clap::Parser;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use timsim_cli::sequences::{load_annotated, load_bare, load_mod_info};
use timsim_cli::spectrum::{fragment_peaks, normalize_total, precursor_peaks, FragKey, Peaks, SpectrumOpts};
use timsim_schema::tables::{
    fragment_intensities as FI, ion_spectra as SP, precursors as PRE,
};

#[derive(Parser)]
#[command(name = "timsim-spectra", about = "peptide ions -> instrument-independent MS1+MS2 spectra")]
struct Args {
    #[arg(long)]
    precursors: PathBuf,
    #[arg(long)]
    peptides: PathBuf,
    #[arg(long)]
    modforms: PathBuf,
    #[arg(long)]
    modifications: PathBuf,
    #[arg(long)]
    fragment_intensities: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Cap on precursors (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

struct PrecRow {
    precursor_id: u64,
    modform_id: u64,
    charge: i32,
}

fn main() -> Result<()> {
    let a = Args::parse();

    // Pass 1: precursor rows + the modform/peptide ids they touch.
    let mut rows: Vec<PrecRow> = Vec::new();
    let (mut need_pep, mut need_mf): (HashSet<u64>, HashSet<u64>) = (HashSet::new(), HashSet::new());
    'outer: for b in timsim_schema::read(&a.precursors, PRE::TABLE)? {
        let pcid: &UInt64Array = b.column_by_name(PRE::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let pid: &UInt64Array = b.column_by_name(PRE::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let mfid: &UInt64Array = b.column_by_name(PRE::MODFORM_ID).unwrap().as_any().downcast_ref().unwrap();
        let chg: &UInt8Array = b.column_by_name(PRE::CHARGE).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            need_pep.insert(pid.value(i));
            need_mf.insert(mfid.value(i));
            rows.push(PrecRow {
                precursor_id: pcid.value(i),
                modform_id: mfid.value(i),
                charge: chg.value(i).max(1) as i32,
            });
            if a.limit > 0 && rows.len() >= a.limit {
                break 'outer;
            }
        }
    }

    // Sequences.
    let mod_info = load_mod_info(&a.modifications)?;
    let bare = load_bare(&a.peptides, &need_pep)?;
    let annotated = load_annotated(&a.modforms, &need_mf, &bare, &mod_info)?;

    // Fragment intensities: precursor_id -> (ion_type, ordinal, frag_charge) -> intensity.
    let keep: HashSet<u64> = rows.iter().map(|r| r.precursor_id).collect();
    let mut frags: HashMap<u64, HashMap<FragKey, f64>> = HashMap::new();
    for b in timsim_schema::read(&a.fragment_intensities, FI::TABLE)? {
        let pcid: &UInt64Array = b.column_by_name(FI::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let it: &arrow::array::StringArray = b.column_by_name(FI::ION_TYPE).unwrap().as_any().downcast_ref().unwrap();
        let ord: &UInt16Array = b.column_by_name(FI::ORDINAL).unwrap().as_any().downcast_ref().unwrap();
        let fc: &UInt8Array = b.column_by_name(FI::FRAG_CHARGE).unwrap().as_any().downcast_ref().unwrap();
        let inten: &Float32Array = b.column_by_name(FI::INTENSITY).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let pc = pcid.value(i);
            if !keep.contains(&pc) {
                continue;
            }
            let ion = it.value(i).chars().next().unwrap_or('?');
            frags.entry(pc).or_default().insert((ion, ord.value(i), fc.value(i)), inten.value(i) as f64);
        }
    }

    // Generate the two spectra per ion — IN PARALLEL. Each precursor is independent (the annotated/frags
    // maps are read-only), and mscore's isotope/fragment maths are pure, so this is embarrassingly
    // parallel. `flat_map_iter` runs the per-precursor closure across the rayon pool; the order-preserving
    // `collect` keeps the output deterministic (same bytes as the serial version). This is the pole at
    // 250K+ scale — serial it is ~100 min, parallel ~7 min on 16 cores.
    use rayon::prelude::*;
    let opts = SpectrumOpts::default();
    let generated: Vec<(u64, u8, Peaks)> = rows
        .par_iter()
        .flat_map_iter(|r| {
            let mut out: Vec<(u64, u8, Peaks)> = Vec::new();
            if let Some(ann) = annotated.get(&r.modform_id) {
                // MS1 — always present. Unit-total so precursor and fragments share the current scale;
                // the render supplies the absolute level (per-ion abundance).
                let mut ms1 = precursor_peaks(ann, r.charge, opts);
                if !ms1.is_empty() {
                    normalize_total(&mut ms1);
                    out.push((r.precursor_id, 1, ms1));
                }
                // MS2 — only if the ion has predicted fragments. Same unit total as MS1 (ion-current
                // conservation): raw Prosit fragments sum to ~7, which would render ~7× too hot.
                if let Some(per_ion) = frags.get(&r.precursor_id) {
                    let mut ms2 = fragment_peaks(ann, r.charge, per_ion, opts);
                    if !ms2.is_empty() {
                        normalize_total(&mut ms2);
                        out.push((r.precursor_id, 2, ms2));
                    }
                }
            }
            out
        })
        .collect();

    let n_ms1 = generated.iter().filter(|x| x.1 == 1).count() as u64;
    let n_ms2 = generated.iter().filter(|x| x.1 == 2).count() as u64;
    let mut b_pcid: Vec<u64> = Vec::with_capacity(generated.len());
    let mut b_level: Vec<u8> = Vec::with_capacity(generated.len());
    let mut b_mz: Vec<Peaks> = Vec::with_capacity(generated.len());
    for (pc, lv, pk) in generated {
        b_pcid.push(pc);
        b_level.push(lv);
        b_mz.push(pk);
    }

    write_artifact(&a.out, &b_pcid, &b_level, &b_mz)?;
    println!("  timsim-spectra: {} MS1 + {} MS2 spectra for {} precursors -> {}",
             n_ms1, n_ms2, rows.len(), a.out.display());
    Ok(())
}

fn write_artifact(out: &std::path::Path, pcid: &[u64], level: &[u8], spectra: &[Peaks]) -> Result<()> {
    use arrow::array::{Float32Builder, Float64Builder, ListBuilder, UInt64Array as U64, UInt8Array as U8};
    use arrow::record_batch::RecordBatch;

    let mut mz_b = ListBuilder::new(Float64Builder::new());
    let mut int_b = ListBuilder::new(Float32Builder::new());
    for s in spectra {
        for &(mz, inten) in s {
            mz_b.values().append_value(mz);
            int_b.values().append_value(inten as f32);
        }
        mz_b.append(true);
        int_b.append(true);
    }
    let cols: Vec<arrow::array::ArrayRef> = vec![
        std::sync::Arc::new(U64::from(pcid.to_vec())),
        std::sync::Arc::new(U8::from(level.to_vec())),
        std::sync::Arc::new(mz_b.finish()),
        std::sync::Arc::new(int_b.finish()),
    ];
    let spec = timsim_schema::tables::by_name(SP::TABLE).ok_or_else(|| anyhow!("no ion_spectra spec"))?;
    let batch = RecordBatch::try_new(spec.schema.clone(), cols)?;
    timsim_schema::write(out, SP::TABLE, &batch, "timsim-spectra", None)?;
    Ok(())
}
