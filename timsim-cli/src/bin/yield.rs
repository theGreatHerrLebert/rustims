//! `timsim-yield` — how much of each peptide, in this sample. QUANTITY.
//!
//! Consumes the SHARED structure (occurrences + cleavage sites, computed once) and one
//! sample's protein amounts, and produces peptide amounts. A 20-sample A/B design runs the
//! digest once and this N times.
//!
//! `--digestion-efficiency` (chemist) and `--cleavage-p` (informatician) are the same
//! parameter under two names. Two vocabularies, one physics.

use anyhow::{bail, Result};
use arrow::array::{ArrayRef, Float64Array, StringArray, UInt32Array, UInt64Array};
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use timsim_chem::digest::{Bounds, ProteinDigest, YieldModel};
use timsim_chem::modify::{blocking_at_sites, Modification, Site, Stage};
use timsim_chem::{BlockingOccupancy, Occurrence};
use timsim_cli::{batch, print_schema, producer};
use timsim_schema::tables::{
    cleavage_sites as CS, peptide_occurrences as PO, peptide_quantities as PQ,
    modifications as MODS, protein_quantities as PRQ, proteome as PROT,
};

#[derive(Parser)]
#[command(name = "timsim-yield", about = "structure + protein amounts -> peptide amounts (quantity)")]
struct Args {
    /// The proteome. Required for the authoritative protein LENGTH — inferring it from the
    /// occurrence table is wrong, because the length filter discards C-terminal peptides.
    #[arg(long)]
    proteome: Option<PathBuf>,
    #[arg(long)]
    occurrences: Option<PathBuf>,
    #[arg(long)]
    cleavage_sites: Option<PathBuf>,
    #[arg(long)]
    protein_quantities: Option<PathBuf>,

    /// `modifications.parquet` from `timsim-modify`. Optional; without it the proteome is
    /// unmodified and nothing blocks the protease.
    ///
    /// This is a **path to the artifact**, deliberately — not a `--blocking-occupancy` number. The
    /// occupancies here are the same numbers `timsim-modify` used to build the modforms, and a flag
    /// would let the two stages disagree about them. Every B13 bug found so far had that shape.
    #[arg(long)]
    modifications: Option<PathBuf>,

    /// Per-site cleavage probability. The chemist's name.
    #[arg(long)]
    digestion_efficiency: Option<f64>,
    /// The same parameter, under the informatician's name.
    #[arg(long, conflicts_with = "digestion_efficiency")]
    cleavage_p: Option<f64>,

    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long)]
    schema: bool,
    /// Print the derived physical parameters and exit.
    #[arg(long)]
    explain: bool,
}

fn col_str<'a>(b: &'a arrow::record_batch::RecordBatch, n: &str) -> &'a StringArray {
    b.column_by_name(n).unwrap().as_any().downcast_ref().unwrap()
}
fn col_u64<'a>(b: &'a arrow::record_batch::RecordBatch, n: &str) -> &'a UInt64Array {
    b.column_by_name(n).unwrap().as_any().downcast_ref().unwrap()
}
fn col_u32<'a>(b: &'a arrow::record_batch::RecordBatch, n: &str) -> &'a UInt32Array {
    b.column_by_name(n).unwrap().as_any().downcast_ref().unwrap()
}
fn col_f64<'a>(b: &'a arrow::record_batch::RecordBatch, n: &str) -> &'a Float64Array {
    b.column_by_name(n).unwrap().as_any().downcast_ref().unwrap()
}

/// Read `modifications.parquet` back into the chemistry type. The spec is an artifact precisely so
/// that this is a *read* and not a re-entry of the same numbers on a second command line.
fn read_modifications(path: &PathBuf) -> Result<Vec<Modification>> {
    let mut out = Vec::new();
    for b in timsim_schema::read(path, MODS::TABLE)? {
        let name = col_str(&b, MODS::NAME);
        let targets = col_str(&b, MODS::TARGETS);
        let site = col_str(&b, MODS::SITE);
        let stage = col_str(&b, MODS::STAGE);
        let occ = col_f64(&b, MODS::OCCUPANCY);
        let delta = col_f64(&b, MODS::MASS_DELTA);
        let unimod: &UInt32Array =
            b.column_by_name(MODS::UNIMOD_ID).unwrap().as_any().downcast_ref().unwrap();
        let blocks: &arrow::array::BooleanArray = b
            .column_by_name(MODS::BLOCKS_CLEAVAGE)
            .unwrap()
            .as_any()
            .downcast_ref()
            .unwrap();
        for i in 0..b.num_rows() {
            out.push(Modification {
                name: name.value(i).to_string(),
                unimod_id: unimod.value(i),
                targets: targets.value(i).to_string(),
                site: match site.value(i) {
                    "residue" => Site::Residue,
                    "n_term" => Site::NTerm,
                    "c_term" => Site::CTerm,
                    o => bail!("{}: unknown site {o:?}", path.display()),
                },
                occupancy: occ.value(i),
                mass_delta: delta.value(i),
                blocks_cleavage: blocks.value(i),
                stage: match stage.value(i) {
                    "protein" => Stage::Protein,
                    "peptide" => Stage::Peptide,
                    o => bail!("{}: unknown stage {o:?}", path.display()),
                },
            });
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.schema {
        return print_schema(PQ::TABLE);
    }
    let p = a.digestion_efficiency.or(a.cleavage_p).unwrap_or(0.90);
    if a.explain {
        println!("  digestion_efficiency = {p}");
        println!("    -> per-site cleavage probability p_eff(k) = {p} * (1 - blocking_occupancy(k))");
        println!("    -> P(n missed cleavages) falls out; it is NOT a parameter");
        return Ok(());
    }
    let req = |o: Option<PathBuf>, n: &str| o.ok_or_else(|| anyhow::anyhow!("--{n} is required"));
    let (prot_p, occ_p, cs_p, pq_p, out) = (
        req(a.proteome, "proteome")?,
        req(a.occurrences, "occurrences")?,
        req(a.cleavage_sites, "cleavage-sites")?,
        req(a.protein_quantities, "protein-quantities")?,
        req(a.out, "out")?,
    );
    if !(0.0..=1.0).contains(&p) {
        bail!("digestion efficiency must be in [0, 1], got {p}");
    }

    // The enumeration bounds are READ from the structure artifact, never re-entered. Digesting
    // at 4 and yielding at a defaulted 2 would silently give every 3- and 4-missed-cleavage
    // occurrence a zero yield — a wrong number with no error. A fact the artifact knows must
    // not be retyped by the caller.
    let md = timsim_schema::metadata(&occ_p)?;
    let get = |k: &str| -> Result<u32> {
        md.get(k)
            .ok_or_else(|| anyhow::anyhow!(
                "{}: missing {k} — this artifact predates bounds metadata; re-run timsim-digest",
                occ_p.display()
            ))?
            .parse::<u32>()
            .map_err(|e| anyhow::anyhow!("{}: {k} is not an integer: {e}", occ_p.display()))
    };
    let bounds = Bounds {
        max_missed_cleavages: get(timsim_schema::meta::MAX_MISSED_CLEAVAGES)?,
        min_len: get(timsim_schema::meta::MIN_LENGTH)?,
        max_len: get(timsim_schema::meta::MAX_LENGTH)?,
    };

    // cleavage sites, grouped by protein
    let mut sites: HashMap<String, Vec<u32>> = HashMap::new();
    for b in timsim_schema::read(&cs_p, CS::TABLE)? {
        let (pr, po) = (col_str(&b, CS::PROTEIN_ID), col_u32(&b, CS::POSITION));
        for i in 0..b.num_rows() {
            sites.entry(pr.value(i).to_string()).or_default().push(po.value(i));
        }
    }
    for v in sites.values_mut() {
        v.sort_unstable();
    }

    // Protein lengths, READ not inferred. Sequences too — a blocking modification sits on a
    // *residue*, so deciding whether a cleavage site is blocked requires knowing which residue is
    // standing at it.
    let mut length: HashMap<String, u32> = HashMap::new();
    let mut sequence: HashMap<String, String> = HashMap::new();
    for b in timsim_schema::read(&prot_p, PROT::TABLE)? {
        let id = col_str(&b, PROT::PROTEIN_ID);
        let ln = col_u32(&b, PROT::LENGTH);
        let sq = col_str(&b, PROT::SEQUENCE);
        for i in 0..b.num_rows() {
            length.insert(id.value(i).to_string(), ln.value(i));
            sequence.insert(id.value(i).to_string(), sq.value(i).to_string());
        }
    }

    // occurrences, grouped by protein
    let mut occs: HashMap<String, Vec<Occurrence>> = HashMap::new();
    for b in timsim_schema::read(&occ_p, PO::TABLE)? {
        let (pid, pr) = (col_u64(&b, PO::PEPTIDE_ID), col_str(&b, PO::PROTEIN_ID));
        let (st, en) = (col_u32(&b, PO::START), col_u32(&b, PO::END));
        let mc: &arrow::array::UInt16Array =
            b.column_by_name(PO::N_MISSED_CLEAVAGES).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let prot = pr.value(i).to_string();
            let e = en.value(i);
            occs.entry(prot.clone()).or_default().push(Occurrence {
                protein_id: Arc::from(prot.as_str()),
                start: st.value(i),
                end: e,
                n_missed_cleavages: mc.value(i),
            });
            let _ = pid.value(i);
        }
    }

    // Peptide ids, aligned with the occurrence rows we reload below.
    let mut occ_pid: HashMap<(String, u32, u32), u64> = HashMap::new();
    for b in timsim_schema::read(&occ_p, PO::TABLE)? {
        let (pid, pr) = (col_u64(&b, PO::PEPTIDE_ID), col_str(&b, PO::PROTEIN_ID));
        let (st, en) = (col_u32(&b, PO::START), col_u32(&b, PO::END));
        for i in 0..b.num_rows() {
            occ_pid.insert((pr.value(i).to_string(), st.value(i), en.value(i)), pid.value(i));
        }
    }

    // protein amounts, per sample
    let mut amounts: HashMap<String, HashMap<String, f64>> = HashMap::new(); // sample -> protein -> amol
    for b in timsim_schema::read(&pq_p, PRQ::TABLE)? {
        let (pr, sa) = (col_str(&b, PRQ::PROTEIN_ID), col_str(&b, PRQ::SAMPLE_ID));
        let am = col_f64(&b, PRQ::AMOUNT_AMOL);
        for i in 0..b.num_rows() {
            amounts
                .entry(sa.value(i).to_string())
                .or_default()
                .insert(pr.value(i).to_string(), am.value(i));
        }
    }

    // The structure, reassembled once — shared by every sample below.
    // Iterate the PROTEOME, not the occurrence table.
    //
    // When the length filter removes every peptide from a protein, that protein has no rows in
    // `peptide_occurrences` — so building digests from `occs` would omit it entirely, and the
    // reported truncation loss, filter loss and missed-cleavage distribution would exclude
    // *exactly the proteins those bounds hit hardest*. The accounting has to cover the structure
    // it claims to account for, so a protein with zero surviving peptides still contributes its
    // residues to the denominator.
    for prot in occs.keys() {
        if !length.contains_key(prot) {
            anyhow::bail!("protein {prot:?} appears in the occurrences but not in the proteome");
        }
    }
    let mut names: Vec<&String> = length.keys().collect();
    names.sort();
    let mut digests: Vec<ProteinDigest> = Vec::with_capacity(names.len());
    for prot in names {
        let d = ProteinDigest {
            protein_id: Arc::from(prot.as_str()),
            length: length[prot],
            cleavage_sites: sites.get(prot).cloned().unwrap_or_default(),
            occurrences: occs.get(prot).cloned().unwrap_or_default(),
            bounds,
        };
        // Fail at the boundary, with a message — not with an underflow deep inside.
        d.validate().map_err(anyhow::Error::msg)?;
        digests.push(d);
    }

    // ── cleavage-blocking modifications ──────────────────────────────────────
    //
    // Acetyl-K, ubiquitin-GG, trimethyl-K and TMT-K physically stop trypsin: a lysine carrying one
    // is not cut. So a modification does not merely add mass to a peptide — it changes **which
    // peptides exist**, and the missed cleavage it forces at the modified residue *is the evidence*
    // that localises the site. A diGly simulation whose protease ignores the GG is not a simulation
    // of a diGly experiment.
    //
    //     p_eff(k) = p · (1 − blocking_occupancy(k))
    //
    // The occupancies are READ from `modifications.parquet` — the same artifact, and therefore the
    // same numbers, that `timsim-modify` used to build the modforms.
    // The P1-residue rule lives in `timsim_chem::modify::blocking_at_sites`, not here — it is
    // chemistry, and it is exactly the kind of off-by-one that is invisible in a CLI and testable
    // in a library.
    let mut blocking_sites: Vec<(Arc<str>, u32, f64)> = Vec::new();
    let mut blockers: Vec<Modification> = Vec::new();
    if let Some(mods_p) = &a.modifications {
        blockers = read_modifications(mods_p)?
            .into_iter()
            .filter(|m| m.blocks_cleavage)
            .collect();

        for d in &digests {
            let seq = sequence.get(&*d.protein_id).ok_or_else(|| {
                anyhow::anyhow!("protein {:?} has no sequence in the proteome", d.protein_id)
            })?;
            for (pos, occ) in blocking_at_sites(seq, &d.cleavage_sites, &blockers) {
                blocking_sites.push((d.protein_id.clone(), pos, occ));
            }
        }
    }

    let n_blocked = blocking_sites.len();
    let blocking = if blocking_sites.is_empty() {
        BlockingOccupancy::none()
    } else {
        BlockingOccupancy::from_sites(blocking_sites).map_err(anyhow::Error::msg)?
    };
    if !blockers.is_empty() {
        println!("  cleavage blocking:");
        for m in &blockers {
            println!(
                "    {:<16} on {:<4} occupancy {:.4}  -> p_eff = {p} * (1 - {:.4})",
                m.name, m.targets, m.occupancy, m.occupancy
            );
        }
        println!("    {n_blocked} cleavage sites carry a blocking modification");
        println!("    the missed cleavage this forces IS the evidence that localises the site");
        println!();
    }

    let ym = YieldModel::new(p, blocking).map_err(anyhow::Error::msg)?;

    let mut samples: Vec<&String> = amounts.keys().collect();
    samples.sort();

    let (mut o_pid, mut o_sample, mut o_amt) = (Vec::new(), Vec::new(), Vec::new());

    // The accounting is a property of the STRUCTURE, so it is computed over every protein —
    // independently of whether that protein happens to carry an amount in this design. A protein
    // skipped by `timsim-design` (a non-standard residue, say) still contributes its residues to
    // the denominator; folding the two loops together would quietly report losses that exclude
    // exactly the proteins the bounds hit hardest.
    let mut agg_stats = timsim_chem::DigestStats::default();
    let mut yields_by_protein: HashMap<&str, Vec<f64>> = HashMap::with_capacity(digests.len());
    for d in &digests {
        let (yields, st) = ym.apply(d);
        agg_merge(&mut agg_stats, &st);
        yields_by_protein.insert(&d.protein_id, yields);
    }
    let stats_first = Some(agg_stats);

    for s in &samples {
        let per_protein = &amounts[*s];
        // peptide_id -> summed amount. Amounts sum over OCCURRENCES, not proteins: the same
        // sequence can occur more than once in one protein and in more than one protein.
        let mut acc: HashMap<u64, f64> = HashMap::new();

        for d in &digests {
            let protein_amol = match per_protein.get(&*d.protein_id) {
                Some(v) => *v,
                None => continue, // no amount in this design — but its stats are already counted
            };
            let yields = &yields_by_protein[&*d.protein_id];
            for (o, y) in d.occurrences.iter().zip(yields) {
                let key = (d.protein_id.to_string(), o.start, o.end);
                if let Some(pid) = occ_pid.get(&key) {
                    *acc.entry(*pid).or_insert(0.0) += protein_amol * y;
                }
            }
        }

        let mut rows: Vec<(u64, f64)> = acc.into_iter().collect();
        rows.sort_unstable_by_key(|r| r.0); // deterministic row order
        for (pid, amt) in rows {
            o_pid.push(pid);
            o_sample.push((*s).clone());
            o_amt.push(amt);
        }
    }

    let n = o_pid.len();
    timsim_schema::write(&out, PQ::TABLE, &batch(PQ::TABLE, vec![
        Arc::new(UInt64Array::from(o_pid)) as ArrayRef,
        Arc::new(StringArray::from(o_sample)),
        Arc::new(Float64Array::from(o_amt.clone())),
    ])?, &producer("timsim-yield"), None)?;

    let st = stats_first.unwrap_or_default();
    let dist = st.missed_cleavage_distribution();
    println!("  digestion efficiency : {p}  (per-site cleavage probability)");
    print!("  missed cleavages     :");
    for (i, f) in dist.iter().enumerate() {
        print!("  {i} → {:.1}%", f * 100.0);
    }
    println!("   (molar, observed — NOT a parameter)");
    println!("  truncated at n={:<6}: omits {:.4}% of residue mass   (measured, not a formula)",
             bounds.max_missed_cleavages, st.truncation_loss() * 100.0);
    println!("    (bounds read from the structure artifact, not re-entered)");
    println!("  length filter        : omits {:.2}% of enumerated mass", st.filter_loss() * 100.0);
    println!("  samples              : {}", samples.len());
    println!("  peptide amounts      : {n} rows");
    println!("  total on column      : {:.3e} amol (sample 1)", o_amt.iter().take(n / samples.len().max(1)).sum::<f64>());

    if let Some(rp) = a.report {
        let mut t = String::new();
        t.push_str(&format!("digestion_efficiency = {p}\n"));
        t.push_str(&format!("truncation_loss = {:.6e}\n", st.truncation_loss()));
        t.push_str(&format!("filter_loss = {:.6e}\n", st.filter_loss()));
        t.push_str("[missed_cleavages]\n");
        for (i, f) in dist.iter().enumerate() {
            t.push_str(&format!("\"{i}\" = {f:.6}\n"));
        }
        std::fs::write(&rp, t)?;
        println!("  report               -> {}", rp.display());
    }
    println!("  -> {}", out.display());
    Ok(())
}

fn agg_merge(a: &mut timsim_chem::DigestStats, b: &timsim_chem::DigestStats) {
    a.residues_total += b.residues_total;
    a.residues_enumerated += b.residues_enumerated;
    a.residues_retained += b.residues_retained;
    if a.yield_by_missed_cleavages.len() < b.yield_by_missed_cleavages.len() {
        a.yield_by_missed_cleavages.resize(b.yield_by_missed_cleavages.len(), 0.0);
    }
    for (s, v) in a.yield_by_missed_cleavages.iter_mut().zip(&b.yield_by_missed_cleavages) {
        *s += v;
    }
}
