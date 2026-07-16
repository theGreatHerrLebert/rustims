//! Shared sequence plumbing: load bare peptide sequences + modforms + the modification spec, and
//! produce the `[UNIMOD:id]`-annotated sequence that mscore parses. Used by the spectrum-
//! materialisation tool — the render itself is a projector and never touches sequences.

use anyhow::{anyhow, Result};
use arrow::array::{Array, ListArray, StringArray, UInt32Array, UInt64Array};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use timsim_schema::tables::{modforms as MF, modifications as MODS, peptides as PEP};

/// Build the `[UNIMOD:id]`-annotated sequence from a bare sequence + a modform's `(position, name)`
/// mods + the `name -> (unimod id, site)` map. Mirrors v1's `annotate_modform`: an **n_term** mod is a
/// prefix, a **c_term** mod a suffix, a **residue** mod follows its residue. Residue tags are inserted
/// highest-position-first so earlier positions don't shift. Errors on a mod name absent from the
/// modifications table (a dropped mod is a wrong composition).
pub fn annotate(
    bare: &str,
    positions: &UInt32Array,
    names: &StringArray,
    mod_info: &HashMap<String, (u32, String)>,
) -> Result<String> {
    let mut prefix = String::new();
    let mut suffix = String::new();
    let mut residue: Vec<(usize, u32)> = Vec::new();
    for k in 0..positions.len() {
        let name = names.value(k);
        let (uid, site) = mod_info
            .get(name)
            .ok_or_else(|| anyhow!("modification {name:?} not in the modifications table"))?;
        match site.as_str() {
            "n_term" => prefix.push_str(&format!("[UNIMOD:{uid}]")),
            "c_term" => suffix.push_str(&format!("[UNIMOD:{uid}]")),
            _ => residue.push((positions.value(k) as usize, *uid)),
        }
    }
    let mut out = bare.to_string();
    residue.sort_by(|x, y| y.0.cmp(&x.0)); // descending: an insert never shifts a lower position
    for (pos, uid) in residue {
        let at = (pos + 1).min(out.len());
        out.insert_str(at, &format!("[UNIMOD:{uid}]"));
    }
    Ok(format!("{prefix}{out}{suffix}"))
}

/// `modification name -> (unimod id, site)`.
pub fn load_mod_info(path: &Path) -> Result<HashMap<String, (u32, String)>> {
    let mut out = HashMap::new();
    for b in timsim_schema::read(path, MODS::TABLE)? {
        let name: &StringArray = b.column_by_name(MODS::NAME).unwrap().as_any().downcast_ref().unwrap();
        let uid: &UInt32Array = b.column_by_name(MODS::UNIMOD_ID).unwrap().as_any().downcast_ref().unwrap();
        let site: &StringArray = b.column_by_name(MODS::SITE).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            out.insert(name.value(i).to_string(), (uid.value(i), site.value(i).to_string()));
        }
    }
    Ok(out)
}

/// `peptide_id -> bare sequence`, restricted to `need`.
pub fn load_bare(path: &Path, need: &HashSet<u64>) -> Result<HashMap<u64, String>> {
    let mut out = HashMap::new();
    for b in timsim_schema::read(path, PEP::TABLE)? {
        let id: &UInt64Array = b.column_by_name(PEP::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let seq: &StringArray = b.column_by_name(PEP::SEQUENCE).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if need.contains(&id.value(i)) {
                out.insert(id.value(i), seq.value(i).to_string());
            }
        }
    }
    Ok(out)
}

/// `modform_id -> annotated sequence`, restricted to `need`.
pub fn load_annotated(
    path: &Path,
    need: &HashSet<u64>,
    bare: &HashMap<u64, String>,
    mod_info: &HashMap<String, (u32, String)>,
) -> Result<HashMap<u64, String>> {
    let mut out = HashMap::new();
    for b in timsim_schema::read(path, MF::TABLE)? {
        let mfid: &UInt64Array = b.column_by_name(MF::MODFORM_ID).unwrap().as_any().downcast_ref().unwrap();
        let pid: &UInt64Array = b.column_by_name(MF::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let positions: &ListArray = b.column_by_name(MF::MOD_POSITIONS).unwrap().as_any().downcast_ref().unwrap();
        let names: &ListArray = b.column_by_name(MF::MOD_NAMES).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let mf = mfid.value(i);
            if !need.contains(&mf) {
                continue;
            }
            let Some(seq) = bare.get(&pid.value(i)) else { continue };
            let pos = positions.value(i);
            let pos: &UInt32Array = pos.as_any().downcast_ref().unwrap();
            let nm = names.value(i);
            let nm: &StringArray = nm.as_any().downcast_ref().unwrap();
            out.insert(mf, annotate(seq, pos, nm, mod_info)?);
        }
    }
    Ok(out)
}
