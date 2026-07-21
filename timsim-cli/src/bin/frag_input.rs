//! `timsim-frag-input` — freeze the fragment-prediction input as an explicit artifact.
//!
//! The fragment intensity model needs `(precursor_id, sequence, charge)`, where the sequence is the
//! modform's `[UNIMOD:id]`-annotated sequence — so a MODIFIED precursor fragments as modified (an
//! oxidised or phosphorylated peptide has a different fragmentation pattern than its bare form). This
//! was previously hidden inside `timsim-fragments`' `--peptides` join, which used the BARE peptide
//! sequence and therefore predicted every modform identically. Making it a distinct node freezes the
//! sequence + charge + mod encoding as a cached, inspectable table, and — crucially — reuses the SAME
//! `annotate()` the spectrum builder uses for m/z, so the intensity prediction and the fragment m/z
//! agree on what the molecule is.

use anyhow::{anyhow, Result};
use arrow::array::{Array, StringArray, UInt8Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use parquet::arrow::ArrowWriter;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use timsim_cli::sequences::{load_annotated, load_bare, load_mod_info};
use timsim_schema::tables::precursors as PRE;

#[derive(Parser)]
#[command(name = "timsim-frag-input", about = "precursors + modforms -> (precursor_id, annotated sequence, charge)")]
struct Args {
    #[arg(long)] precursors: PathBuf,
    #[arg(long)] peptides: PathBuf,
    #[arg(long)] modforms: PathBuf,
    #[arg(long)] modifications: PathBuf,
    #[arg(long)] out: PathBuf,
    /// Row-group size (fragment-prediction input rows).
    #[arg(long, default_value_t = 2_000_000)] chunk: usize,
}

fn main() -> Result<()> {
    let a = Args::parse();

    // Pass 1: which peptides and modforms does the precursor set actually touch? (Don't hold per-precursor
    // rows — at PTM scale that is the memory hog; re-read in pass 2 instead.)
    let (mut need_pep, mut need_mf): (HashSet<u64>, HashSet<u64>) = (HashSet::new(), HashSet::new());
    for b in timsim_schema::read(&a.precursors, PRE::TABLE)? {
        let pid: &UInt64Array = b.column_by_name(PRE::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let mfid: &UInt64Array = b.column_by_name(PRE::MODFORM_ID).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            need_pep.insert(pid.value(i));
            need_mf.insert(mfid.value(i));
        }
    }
    let mod_info = load_mod_info(&a.modifications)?;
    let bare = load_bare(&a.peptides, &need_pep)?;
    let annotated = load_annotated(&a.modforms, &need_mf, &bare, &mod_info)?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("precursor_id", DataType::UInt64, false),
        Field::new("sequence", DataType::Utf8, false),
        Field::new("charge", DataType::UInt8, false),
    ]));
    let file = std::fs::File::create(&a.out)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;

    // Pass 2: stream (precursor_id, annotated sequence, charge) in row-groups. Peak memory is one chunk
    // plus the annotation maps (per-modform, not per-precursor).
    let (mut pc, mut sq, mut ch): (Vec<u64>, Vec<String>, Vec<u8>) = Default::default();
    let (mut written, mut missing) = (0u64, 0u64);
    let mut missing_examples: Vec<(u64, u64)> = Vec::new(); // (precursor_id, modform_id), first few
    let mut flush = |pc: &mut Vec<u64>, sq: &mut Vec<String>, ch: &mut Vec<u8>, w: &mut ArrowWriter<std::fs::File>| -> Result<()> {
        if pc.is_empty() { return Ok(()); }
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(UInt64Array::from(std::mem::take(pc))),
            Arc::new(StringArray::from(std::mem::take(sq))),
            Arc::new(UInt8Array::from(std::mem::take(ch))),
        ])?;
        w.write(&batch)?;
        Ok(())
    };
    for b in timsim_schema::read(&a.precursors, PRE::TABLE)? {
        let pcid: &UInt64Array = b.column_by_name(PRE::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let mfid: &UInt64Array = b.column_by_name(PRE::MODFORM_ID).unwrap().as_any().downcast_ref().unwrap();
        let chg: &UInt8Array = b.column_by_name(PRE::CHARGE).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            match annotated.get(&mfid.value(i)) {
                Some(seq) => {
                    pc.push(pcid.value(i));
                    sq.push(seq.clone());
                    ch.push(chg.value(i).max(1));
                    written += 1;
                }
                None => {
                    missing += 1;
                    if missing_examples.len() < 5 { missing_examples.push((pcid.value(i), mfid.value(i))); }
                }
            }
            if pc.len() >= a.chunk { flush(&mut pc, &mut sq, &mut ch, &mut writer)?; }
        }
    }
    flush(&mut pc, &mut sq, &mut ch, &mut writer)?;
    writer.close()?;
    // In a consistent artifact set every precursor's modform is in the modforms table, so `missing`
    // must be 0. A non-zero count means the precursors/modforms/peptides are out of sync — fail loudly
    // rather than silently emit a fragment input with fewer rows than precursors (which would drop MS2
    // for those precursors downstream while still producing an apparently-valid .raw + truth).
    if missing > 0 {
        return Err(anyhow!(
            "{missing} precursors had no annotated modform — precursors/modforms/peptides are out of \
             sync (e.g. {:?}). Refusing to emit a fragment input missing rows.", missing_examples));
    }
    eprintln!("wrote {written} fragment-prediction input rows -> {}", a.out.display());
    if written == 0 {
        return Err(anyhow!("no fragment-prediction input rows produced — check precursors/modforms alignment"));
    }
    Ok(())
}
