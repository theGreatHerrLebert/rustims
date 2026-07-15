//! `timsim-localization` — the site-localization answer key. STRUCTURE.
//!
//! We place every modification, so where a mod *is* is known by construction. Localisation is the
//! **search engine's** problem — recovering that position from the fragment spectrum — and this tool
//! emits what the engine is scored against.
//!
//! A modform is *ambiguous* when its modification's target residue occurs more than once in the
//! peptide (a phospho on a peptide with two S/T/Y). For each such modform this records where the mod
//! actually is (`true_position`), every place it could be (`candidate_positions`), and the b/y
//! fragments whose m/z depends on which candidate carries it (`site_determining`). Joining the last
//! against the fragments a run rendered separates *unresolvable* (the discriminating fragments were
//! never in the data) from *the search engine missed* — the distinction v1's evaluation asserts but
//! cannot check.
//!
//! Pure chemistry: it needs the modforms, the modification spec (for each mod's target residues),
//! and the peptide sequences. No model, no render.

use anyhow::{Context, Result};
use arrow::array::{
    Array, ArrayRef, ListArray, ListBuilder, StringArray, StringBuilder, UInt32Array,
    UInt32Builder, UInt64Array,
};
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use timsim_chem::fragment::{site_determining_set, IonType};
use timsim_cli::{batch, print_schema, producer};
use timsim_schema::tables::{
    localization_sites as LOC, modforms as MF, modifications as MODS, peptides as PEP,
};

#[derive(Parser)]
#[command(
    name = "timsim-localization",
    about = "modforms -> site-localization answer key (which fragments localise each ambiguous mod)"
)]
struct Args {
    #[arg(long)]
    modforms: Option<PathBuf>,
    #[arg(long)]
    modifications: Option<PathBuf>,
    #[arg(long)]
    peptides: Option<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    schema: bool,
}

fn col_str<'a>(b: &'a arrow::record_batch::RecordBatch, n: &str) -> &'a StringArray {
    b.column_by_name(n).unwrap().as_any().downcast_ref().unwrap()
}
fn col_u64<'a>(b: &'a arrow::record_batch::RecordBatch, n: &str) -> &'a UInt64Array {
    b.column_by_name(n).unwrap().as_any().downcast_ref().unwrap()
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.schema {
        return print_schema(LOC::TABLE);
    }
    let modforms = a.modforms.ok_or_else(|| anyhow::anyhow!("--modforms is required"))?;
    let mods_p = a.modifications.ok_or_else(|| anyhow::anyhow!("--modifications is required"))?;
    let peptides = a.peptides.ok_or_else(|| anyhow::anyhow!("--peptides is required"))?;
    let out = a.out.ok_or_else(|| anyhow::anyhow!("--out is required"))?;

    // modification -> target residues (e.g. "Phospho" -> "STY"). A terminal mod has empty targets
    // and is never positionally ambiguous, so it is simply absent here.
    let mut targets: HashMap<String, String> = HashMap::new();
    for b in timsim_schema::read(&mods_p, MODS::TABLE)? {
        let name = col_str(&b, MODS::NAME);
        let tgt = col_str(&b, MODS::TARGETS);
        for i in 0..b.num_rows() {
            targets.insert(name.value(i).to_string(), tgt.value(i).to_string());
        }
    }

    // peptide_id -> sequence.
    let mut seq_of: HashMap<u64, String> = HashMap::new();
    for b in timsim_schema::read(&peptides, PEP::TABLE)? {
        let id = col_u64(&b, PEP::PEPTIDE_ID);
        let sq = col_str(&b, PEP::SEQUENCE);
        for i in 0..b.num_rows() {
            seq_of.insert(id.value(i), sq.value(i).to_string());
        }
    }

    // Output builders.
    let (mut o_mfid, mut o_pid, mut o_mod, mut o_true) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut cand_b = ListBuilder::new(UInt32Builder::new());
    let mut det_b = ListBuilder::new(StringBuilder::new());

    let mut n_modforms = 0usize;
    let mut n_ambiguous = 0usize;

    for b in timsim_schema::read(&modforms, MF::TABLE)? {
        let mfid = col_u64(&b, MF::MODFORM_ID);
        let pid = col_u64(&b, MF::PEPTIDE_ID);
        let positions: &ListArray =
            b.column_by_name(MF::MOD_POSITIONS).unwrap().as_any().downcast_ref().unwrap();
        let names: &ListArray =
            b.column_by_name(MF::MOD_NAMES).unwrap().as_any().downcast_ref().unwrap();

        for i in 0..b.num_rows() {
            n_modforms += 1;
            let pos_list = positions.value(i);
            let pos_list: &UInt32Array = pos_list.as_any().downcast_ref().unwrap();
            if pos_list.is_empty() {
                continue; // the unmodified form: nothing to localise
            }
            let name_list = names.value(i);
            let name_list: &StringArray = name_list.as_any().downcast_ref().unwrap();

            let seq = seq_of
                .get(&pid.value(i))
                .with_context(|| format!("modform references unknown peptide_id {}", pid.value(i)))?;
            let bytes = seq.as_bytes();

            // Group this modform's modifications BY TYPE: the localisation question is asked per
            // modification, over the residues that type can occupy.
            let mut by_mod: HashMap<&str, Vec<usize>> = HashMap::new();
            for k in 0..pos_list.len() {
                by_mod
                    .entry(name_list.value(k))
                    .or_default()
                    .push(pos_list.value(k) as usize);
            }

            // Deterministic order (HashMap iteration is not) so the artifact is reproducible.
            let mut mod_names: Vec<&str> = by_mod.keys().copied().collect();
            mod_names.sort_unstable();

            for mod_name in mod_names {
                let occupied = &by_mod[mod_name];
                let Some(tgt) = targets.get(mod_name) else { continue };
                if tgt.is_empty() {
                    continue; // terminal / non-residue mod: not positionally ambiguous
                }
                // Every position this modification could occupy.
                let candidates: Vec<usize> = bytes
                    .iter()
                    .enumerate()
                    .filter(|(_, &r)| tgt.contains(r as char))
                    .map(|(p, _)| p)
                    .collect();

                // Ambiguous iff SOME BUT NOT ALL candidate sites carry the mod. If every candidate
                // is modified (e.g. carbamidomethyl on all cysteines) there is no "which one"; if
                // there is only one candidate it is trivially placed.
                let k = occupied.len();
                if candidates.len() < 2 || k == 0 || k >= candidates.len() {
                    continue;
                }
                n_ambiguous += 1;

                // One row per occupied site (the search must localise each). Emitting per site keeps
                // the answer key a flat "this mod is truly here" table.
                for &true_pos in occupied {
                    o_mfid.push(mfid.value(i));
                    o_pid.push(pid.value(i));
                    o_mod.push(mod_name.to_string());
                    o_true.push(true_pos as u32);
                    for &c in &candidates {
                        cand_b.values().append_value(c as u32);
                    }
                    cand_b.append(true);
                    for (t, ord) in site_determining_set(seq.len(), &candidates) {
                        let tag = format!("{}{}", if t == IonType::B { "b" } else { "y" }, ord);
                        det_b.values().append_value(&tag);
                    }
                    det_b.append(true);
                }
            }
        }
    }

    let cols: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(o_mfid)),
        Arc::new(UInt64Array::from(o_pid)),
        Arc::new(StringArray::from(o_mod)),
        Arc::new(UInt32Array::from(o_true)),
        Arc::new(cand_b.finish()),
        Arc::new(det_b.finish()),
    ];
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p)?;
    }
    timsim_schema::write(&out, LOC::TABLE, &batch(LOC::TABLE, cols)?, &producer("timsim-localization"), None)?;

    println!("  timsim-localization");
    println!("    modforms scanned    : {n_modforms:>12}");
    println!(
        "    ambiguous sites     : {n_ambiguous:>12}   (modified residue occurs ≥2× — a real "
    );
    println!("                          localisation question; the rest are trivially placed)");
    println!("    -> {}", out.display());
    println!();
    println!("  Each row lists the b/y fragments a search must observe to localise the site.");
    println!("  Join `site_determining` against a run's rendered fragments to tell 'unresolvable'");
    println!("  (evidence absent) from 'the engine missed' (evidence present).");
    Ok(())
}
