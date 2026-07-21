//! Native SCIEX **`.wiff.scan`** authoring: render a no-IM DIA `synthetic_data.db` and
//! author the simulated peaks into a real ZenoTOF template, producing a pure-synthetic
//! `.wiff` + `.wiff.scan` that ProteoWizard/DiaNN read as vendor data. The SCIEX analogue
//! of `astral_dispatch::write_astral_raw` (Thermo `.raw`), but the physical layout is far
//! less regular, so this module carries the reverse-engineered *role model* it needs.
//!
//! ## Role model (from data-file inspection, no Clearcore2 SDK)
//! A ZenoTOF SWATH `.wiff.scan` is a stream of physical peak blocks. Per cycle: one MS1
//! survey then `N` MS2 windows (`N` = the method's SWATH window count).
//! - **MS1 vs MS2** is the block's *calibration group*: MS1 survey scans use a distinct
//!   `cal_a` (the periodic minority cluster); MS2 windows share the other.
//! - **Window index** is the block's file-order rank among its cycle's MS2 blocks (the
//!   isolation window is not stored per block; it is positional, in method order).
//! - **Seed / low-mass cutoff.** The reader seeds each scan's first peak at a per-scan TOF
//!   cutoff `cut_n` and ignores the first token's own n. `cut_n` is encoded in the block's
//!   `u32` header: `hdr = 8*cut_n + Q(cal_a,cal_b)`. To author peaks that read back at the
//!   intended m/z the payload must be `[(cut_n, 1)] + real` (a cutoff sentinel, then real
//!   peaks with strictly-increasing `n > cut_n`).
//!
//! `Q(a,b)` is recovered by a closed form fit to the ZenoTOF 7600 cal regime (see
//! [`seed_cut_n`]): exact for ~95% of scans and within ±1 TOF bin (~≤16 ppm whole-scan
//! shift) for the rest. The exact-per-template `Q`-table is the documented upgrade path.
//!
//! ## Length-preserving authoring
//! Each block's token stream is rewritten **in place**, zero-padded to its original byte
//! length, so every block keeps its exact position and the `.wiff.scan` `Idx` directory stays
//! valid untouched — no offset recomputation, no risk to the ~4% of `Idx` records that point
//! at blocks `scan_blocks` does not enumerate (empty/mini blocks). The cost is a per-slot peak
//! budget (MS2 median ~55 peaks): peaks beyond what the slot holds are dropped
//! lowest-intensity-first. Growing a block past its budget (via the `sciexwiff` `rebuild` +
//! full-`Idx` retranslation) is the documented upgrade for arbitrary peaks/scan.
//! The codec (`encode_stream`) lives in the `sciexwiff` crate, proven byte-identical on real
//! files; this module decides *which* peaks go into *which* block and writes them.
//!
//! ## Modes
//! - **Pure-synthetic** (`overlay_ppm == 0`, default): each authored block's peaks are replaced
//!   by the simulated peaks; every other block is cleared. No real template signal survives.
//! - **Spike-in / overlay** (`overlay_ppm > 0`): each authored block keeps its REAL template
//!   peaks (decoded back to `(n, intensity)`) and the simulated peaks are added on top
//!   (real⊕sim) — the SCIEX analogue of the Astral `superimpose_ppm` workflow. The union is at
//!   TOF-bin resolution and, like everything here, budget-trimmed to the slot lowest-first.

#![cfg(feature = "sciex")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mscore::data::spectrum::MzSpectrum;
use sciexwiff::wiffscan::{
    block_payload, encode_stream, mz_to_n, n_to_mz, rebuild_grow, retranslate_idx, scan_blocks,
    GrowEdit, Peak, ScanBlock, MAX_INTENSITY,
};
use sciexwiff::{patch_idx_stream, read_idx_stream, read_method};

/// Tuning for a native `.wiff.scan` author.
#[derive(Debug, Clone, Copy)]
pub struct SciexWriteOptions {
    pub num_threads: usize,
    /// Quadrupole transmission edge steepness `k` (as the Bruker/Thermo paths).
    pub quad_k: f64,
    pub max_ms1_peaks: usize,
    pub max_ms2_peaks: usize,
    /// Gaussian m/z noise (≈ppm at 3σ) on MS1 precursor peaks. 0 = off.
    pub precursor_noise_ppm: f64,
    /// Gaussian m/z noise (≈ppm at 3σ) on MS2 fragment peaks. 0 = off.
    pub fragment_noise_ppm: f64,
    /// Spike-in / overlay: `> 0` keeps each authored block's REAL template peaks and adds the
    /// simulated peaks on top (real⊕sim), the SCIEX analogue of the Astral `superimpose_ppm`
    /// mode. `0` (default) = pure-synthetic (template peaks replaced). The value documents the
    /// intended merge tolerance; the union is at the block's TOF-bin (`n`) resolution.
    pub overlay_ppm: f64,
    /// Overlay-only: how strong the simulated spike-in peaks are relative to the real template
    /// background — the synthetic peaks are scaled so their per-scan max = `spike_scale × the
    /// scan's real-peak max`. `1.0` (default) ≈ comparable to the background; `> 1` makes the
    /// spike-ins dominant, `< 1` makes them trace. Ignored in pure-synthetic mode.
    pub spike_scale: f64,
    /// Keep the template's real peaks in leading/trailing partial-cycle blocks instead of
    /// clearing them. Default `false` — preserving real signal in a "synthetic" file is
    /// leakage; only enable deliberately (the summary always reports how many were kept).
    pub preserve_template_partial: bool,
}

impl Default for SciexWriteOptions {
    fn default() -> Self {
        SciexWriteOptions {
            num_threads: 4,
            quad_k: 15.0,
            // Generous caps — the grow writer sizes each block to its tokens (no slot budget),
            // so we author the full simulated peak list to match the mzML render.
            max_ms1_peaks: 2000,
            max_ms2_peaks: 800,
            precursor_noise_ppm: 0.0,
            fragment_noise_ppm: 0.0,
            overlay_ppm: 0.0,
            spike_scale: 1.0,
            preserve_template_partial: false,
        }
    }
}

/// Outcome of a native `.wiff.scan` author.
#[derive(Debug, Clone)]
pub struct SciexWriteSummary {
    pub scans: usize,
    pub ms1: usize,
    pub ms2: usize,
    pub ms2_nonempty: usize,
    /// Full-cycle blocks written with simulated peaks.
    pub blocks_authored: usize,
    /// Partial/leftover blocks cleared to a seed-only spectrum (0 unless partial regions).
    pub blocks_cleared: usize,
    /// Partial blocks whose real template peaks were preserved (only if opted in).
    pub blocks_preserved: usize,
    /// Embedded-`ffffffff` blocks left verbatim (real template peaks retained — a small
    /// leakage in pure-synthetic mode; 0 for a clean template).
    pub blocks_verbatim: usize,
    pub cycles: usize,
    pub windows: usize,
    /// False — authored via the grow rebuild (Idx retranslated), not length-preserving.
    pub length_preserving: bool,
}

// --- seed recovery -------------------------------------------------------------------------

// hdr = 8*cut_n + Q(a,b); Q fit to the ZenoTOF 7600 TOF-cal regime as Q = k0/a + k1*b/a + k2.
// Exact for ~95% of scans, within ±1 TOF bin otherwise. Upgrade path: an exact per-template
// Q-table keyed by (cal_a,cal_b) (100% exact, still pwiz-free at runtime).
const Q_K0: f64 = -91.539_696_34;
const Q_K1: f64 = 40.048_907_23;
const Q_K2: f64 = 187_648.604_9;

/// Recover a block's low-mass-cutoff seed `cut_n` from its `u32` header + calibration.
pub fn seed_cut_n(hdr: u32, cal_a: f64, cal_b: f64) -> i64 {
    let q = Q_K0 / cal_a + Q_K1 * cal_b / cal_a + Q_K2;
    ((hdr as f64 - q) / 8.0).round() as i64
}

/// Notice when the sim covers fewer cycles than the template (the rest are cleared, not an
/// error): a shorter gradient authored onto a longer template.
fn log_short(sim_cycles: usize, template_cycles: usize) {
    eprintln!(
        "write_sciex_wiff: sim has {sim_cycles} cycles, template has {template_cycles}; \
         authoring {sim_cycles}, clearing the remaining {} template cycles",
        template_cycles - sim_cycles
    );
}

fn read_hdr(scan: &[u8], ff: usize) -> Result<u32, String> {
    scan.get(ff + 4..ff + 8)
        .map(|s| u32::from_le_bytes(s.try_into().expect("len 4")))
        .ok_or_else(|| format!("block header out of range at ff={ff}"))
}

// --- role labeling -------------------------------------------------------------------------

/// One acquisition cycle mapped onto physical blocks: `blocks[0]` = MS1 survey, `blocks[1..]`
/// = the `N` MS2 windows in file (method) order.
#[derive(Debug, Clone)]
struct Cycle {
    blocks: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
struct Layout {
    cycles: Vec<Cycle>,
    /// Leading/trailing/leftover blocks (partial cycles) — cleared unless preservation is on.
    partial: Vec<usize>,
}

/// Metadata length of a block (`ff - meta`). The MS1 survey carries extra precursor
/// sub-messages, so its metadata is strictly longer than an MS2 window's — a clean,
/// drift-proof MS1/MS2 discriminator (MS1 ≥ ~57 B, MS2 ≤ ~56 B on the K562 template), unlike
/// `cal_a`, which drifts per cycle and overlaps between the two.
fn meta_len(b: &ScanBlock) -> usize {
    b.ff.saturating_sub(b.meta)
}

/// Label the physical blocks into whole `1 + N` cycles + partial regions using the role model.
/// Rejects (structured error) anything that is not a recognizable `N`-window SWATH layout.
fn label_layout(blocks: &[ScanBlock], n_windows: usize) -> Result<Layout, String> {
    if n_windows == 0 {
        return Err("method reports 0 SWATH windows".into());
    }
    let period = 1 + n_windows;
    let n = blocks.len();
    if n < period {
        return Err(format!("template has {n} blocks, fewer than one {period}-block cycle"));
    }
    // MS1 = the "large-metadata" blocks (extra precursor sub-messages). The MS1/MS2 metadata-
    // length boundary is not a clean count fraction over the whole file (some later MS2 also
    // carry extra sub-messages), so pick the threshold that MAXIMIZES clean `1 + N` cycles.
    // The greedy walk is conservative — it accepts only a run of exactly MS1 + N MS2 — so a
    // wrong threshold merely fragments the schedule into `partial` blocks (cleared later), it
    // never mislabels a cycle's window assignment. Self-tuning over the distinct meta-lengths.
    let mut candidates: Vec<usize> = blocks.iter().map(meta_len).collect();
    candidates.sort_unstable();
    candidates.dedup();
    if candidates.len() < 2 {
        return Err("cannot distinguish MS1 survey metadata from MS2 (uniform metadata length)".into());
    }
    let count_cycles = |thr: usize| -> usize {
        let is_ms1 = |b: &ScanBlock| meta_len(b) >= thr;
        let (mut i, mut c) = (0usize, 0usize);
        while i < n {
            if is_ms1(&blocks[i])
                && i + n_windows < n
                && (i + 1..=i + n_windows).all(|j| !is_ms1(&blocks[j]))
            {
                c += 1;
                i += period;
            } else {
                i += 1;
            }
        }
        c
    };
    let thr = *candidates
        .iter()
        .filter(|&&t| t > candidates[0]) // never label everything MS1
        .max_by_key(|&&t| count_cycles(t))
        .ok_or("no viable MS1/MS2 metadata-length threshold")?;
    let is_ms1 = |b: &ScanBlock| meta_len(b) >= thr;

    // Greedy walk with the chosen threshold. A full cycle = MS1 then exactly N MS2 (we do not
    // require the block after to be MS1 — a trailing partial MS2 may follow the last cycle; an
    // over-long cycle drops its extra MS2 to `partial`, and the next MS1 starts a fresh one).
    let mut layout = Layout::default();
    let mut i = 0usize;
    while i < n {
        let full_cycle = is_ms1(&blocks[i])
            && i + n_windows < n
            && (i + 1..=i + n_windows).all(|j| !is_ms1(&blocks[j]));
        if full_cycle {
            layout.cycles.push(Cycle { blocks: (i..=i + n_windows).collect() });
            i += period;
        } else {
            layout.partial.push(i);
            i += 1;
        }
    }
    if layout.cycles.is_empty() {
        return Err(format!(
            "no full {n_windows}-window SWATH cycles found ({n} blocks, meta-len threshold {thr}) \
             — role model does not fit this template"
        ));
    }
    // Guard fidelity: too many partial (would-be-cleared) blocks means the role model does not
    // fit this template — fail loudly rather than silently clear a large fraction of valid scans.
    let full = layout.cycles.len() * (1 + n_windows);
    if layout.partial.len() * 3 > n {
        return Err(format!(
            "layout not recognized: {} of {n} blocks are partial (>33%) — role model does not \
             fit this template ({} full-cycle blocks)",
            layout.partial.len(),
            full
        ));
    }
    Ok(layout)
}

// --- authoring -----------------------------------------------------------------------------

/// Apply Gaussian m/z noise (≈`ppm` at 3σ) — the FT/TOF recording-stage mass artifact that
/// gives a search engine a mass-error distribution to calibrate against. Off when `ppm <= 0`.
fn apply_mz_noise(peaks: &[(f64, f32)], ppm: f64) -> Vec<(f64, f32)> {
    if !ppm.is_finite() || ppm <= 0.0 || peaks.is_empty() {
        return peaks.to_vec();
    }
    let mz: Vec<f64> = peaks.iter().map(|(m, _)| *m).collect();
    let inten: Vec<f64> = peaks.iter().map(|(_, i)| *i as f64).collect();
    let noised = MzSpectrum::new(mz, inten).add_mz_noise_normal(ppm);
    noised
        .mz
        .iter()
        .zip(noised.intensity.iter())
        .map(|(m, i)| (*m, *i as f32))
        .collect()
}

// Simulated fragment/precursor intensities are small relative floats (e.g. 1e-3..20). The
// `.wiff.scan` codec stores integer counts, so scale each scan's intensities to this full-scale
// target before rounding — otherwise every peak collapses toward 1 and the spectrum goes flat
// (which destroys a search engine's ability to score it). Chosen to preserve ~4 orders of
// dynamic range while keeping most intensities to a 2-byte codec field.
const INTENSITY_FULL_SCALE: f64 = 50_000.0;

/// Convert simulated `(m/z, intensity)` peaks to `(n, intensity)`: m/z→n per the block cal,
/// dropping peaks at/below the cutoff, and scaling intensities per-scan so the max maps to
/// `full_scale` (preserving the relative pattern). No dedupe/cap yet (see [`finalize_peaks`]).
fn sim_to_n(
    peaks: &[(f64, f32)],
    cal_a: f64,
    cal_b: f64,
    cut_n: i64,
    noise_ppm: f64,
    full_scale: f64,
) -> Vec<(i64, u32)> {
    let src = apply_mz_noise(peaks, noise_ppm);
    let max_it = src
        .iter()
        .map(|(_, i)| *i as f64)
        .filter(|v| v.is_finite() && *v > 0.0)
        .fold(0.0f64, f64::max);
    if max_it <= 0.0 {
        return Vec::new();
    }
    let scale = full_scale.max(1.0) / max_it;
    let mut nv: Vec<(i64, u32)> = Vec::with_capacity(src.len());
    for (mz, inten) in src {
        if !mz.is_finite() || mz <= 0.0 || !inten.is_finite() || inten <= 0.0 {
            continue;
        }
        let n = mz_to_n(mz, cal_a, cal_b);
        if n <= cut_n {
            continue; // below the scan's low-mass cutoff — not representable
        }
        let iu = ((inten as f64 * scale).round() as i64).clamp(1, (MAX_INTENSITY - 1) as i64) as u32;
        nv.push((n, iu));
    }
    nv
}

/// Dedupe/merge equal n (sum intensity), cap to `cap` by intensity, re-sort by n (strictly
/// increasing — the codec requires it). Shared by the pure-synthetic and overlay paths.
fn finalize_peaks(mut nv: Vec<(i64, u32)>, cap: usize) -> Vec<(i64, u32)> {
    nv.sort_by_key(|&(n, _)| n);
    let mut merged: Vec<(i64, u32)> = Vec::with_capacity(nv.len());
    for (n, iu) in nv {
        if let Some(last) = merged.last_mut() {
            if last.0 == n {
                last.1 = last.1.saturating_add(iu).min(MAX_INTENSITY - 1);
                continue;
            }
        }
        merged.push((n, iu));
    }
    if merged.len() > cap {
        merged.sort_by(|a, b| b.1.cmp(&a.1));
        merged.truncate(cap);
        merged.sort_by_key(|&(n, _)| n);
    }
    merged
}

/// Decode a block's REAL template peaks as absolute `(n, intensity)` for overlay/spike-in.
/// `block_payload` returns peaks relative to seed 0 (element 0 is the cutoff sentinel, which
/// is dropped); the real peaks' absolute n = `cut_n + relative`. Zero-intensity padding is
/// dropped. On a decode failure the block is treated as having no real peaks (sim-only).
fn decode_template_peaks(scan: &[u8], b: &ScanBlock, cut_n: i64) -> Vec<(i64, u32)> {
    match block_payload(scan, b) {
        Ok((peaks, _term)) => peaks
            .iter()
            .skip(1) // element 0 is the original cutoff sentinel
            .filter(|&&(_, inten)| inten > 0)
            .map(|&(rel, inten)| (cut_n + rel, inten))
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Encode a grown block's payload: the cutoff seed sentinel then the finalized peaks. The grow
/// rebuild sizes the block to the tokens, so there is no slot budget to fit. `real` empty ⇒ a
/// seed-only (cleared/empty) spectrum.
fn author_tokens(real: &[(i64, u32)], cut_n: i64) -> Result<Vec<u8>, String> {
    let mut payload: Vec<Peak> = Vec::with_capacity(real.len() + 1);
    payload.push((cut_n, 1)); // cutoff seed (n forced by the reader; tiny intensity)
    payload.extend_from_slice(real);
    let mut tokens = encode_stream(&payload).map_err(|e| format!("encode: {e:?}"))?;
    // Terminator: real vendor blocks end their peak-list with a short 0xff run (observed 2-4 bytes)
    // right before the next block's metadata. Without it the ABI reader over-reads past our tokens
    // into the following metadata and accumulates a bogus delta -> absurd m/z (~594e6) on ~1.7% of
    // peaks -> 0 IDs. block_payload strips any trailing-0xff span, so this round-trips cleanly.
    tokens.extend_from_slice(&[0xff, 0xff, 0xff, 0xff]);
    Ok(tokens)
}

/// A block is "clean" if its token region holds no embedded `ffffffff` — the ~0.14% of blocks
/// that do (large surveys with an internal sentinel) are left verbatim by the grow path rather
/// than risk splitting the token stream at the wrong place.
fn is_clean_tail(scan: &[u8], b: &ScanBlock) -> bool {
    let body = &scan[b.ff + 9..b.end];
    if body.len() < 6 {
        return true;
    }
    !body[..body.len() - 6]
        .windows(4)
        .any(|w| w == [0xff, 0xff, 0xff, 0xff])
}

// --- output bundle -------------------------------------------------------------------------

/// Copy the template `.wiff` and every same-basename sibling (`.wiff2`, …) into `out_dir`,
/// writing the authored `new_scan` in place of the `.wiff.scan`. Returns the output `.wiff`.
fn copy_bundle(template_wiff: &Path, out_dir: &Path, new_scan: &[u8]) -> Result<PathBuf, String> {
    let dir = template_wiff.parent().unwrap_or_else(|| Path::new("."));
    let wiff_name = template_wiff
        .file_name()
        .ok_or("template has no file name")?
        .to_string_lossy()
        .to_string();
    let base = wiff_name.strip_suffix(".wiff").unwrap_or(&wiff_name).to_string();
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {out_dir:?}: {e}"))?;
    let mut out_wiff: Option<PathBuf> = None;
    for entry in std::fs::read_dir(dir).map_err(|e| format!("read_dir {dir:?}: {e}"))? {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let fname = entry.file_name().to_string_lossy().to_string();
        if !fname.starts_with(&base) {
            continue;
        }
        let dest = out_dir.join(&fname);
        if fname.ends_with(".wiff.scan") {
            std::fs::write(&dest, new_scan).map_err(|e| format!("write {dest:?}: {e}"))?;
        } else {
            std::fs::copy(&path, &dest).map_err(|e| format!("copy {path:?}->{dest:?}: {e}"))?;
        }
        if fname == wiff_name {
            out_wiff = Some(dest);
        }
    }
    out_wiff.ok_or_else(|| format!("output .wiff ({wiff_name}) was not placed in {out_dir:?}"))
}

// --- per-template profile (exact roles + seeds from a pwiz characterization) ----------------

/// One authorable block from a template characterization: its `scan_blocks` index, MS level, and
/// exact low-mass-cutoff seed `cut_n` (derived from pwiz's reported spectra). Blocks absent from
/// the profile's `authored` list are cleared. A profile lets the writer author ANY pwiz-readable
/// template without the metadata role model or the closed-form seed (both fit to K562) — the fix
/// for cross-lab / cross-method templates where those K562-anchored heuristics don't hold.
#[derive(serde::Deserialize)]
pub struct ProfileBlock {
    pub block: usize,
    pub ms: u8,
    pub cut_n: i64,
}

/// A template characterization: the SWATH window count + the ordered authorable blocks.
#[derive(serde::Deserialize)]
pub struct TemplateProfile {
    pub n_windows: usize,
    pub authored: Vec<ProfileBlock>,
}

/// Enumerate the physical scan blocks' `(cal_a, cal_b)` in file order — for the offline template
/// characterizer, which needs the exact same block enumeration the writer + profiles use.
pub fn scan_block_cals(scan_path: &Path) -> Result<Vec<(f64, f64)>, String> {
    let scan = std::fs::read(scan_path).map_err(|e| format!("read {scan_path:?}: {e}"))?;
    Ok(scan_blocks(&scan).iter().map(|b| (b.cal_a, b.cal_b)).collect())
}

fn load_profile(path: &Path) -> Result<TemplateProfile, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("read profile {path:?}: {e}"))?;
    serde_json::from_str(&s).map_err(|e| format!("parse profile {path:?}: {e}"))
}

/// Group a profile's authored blocks into whole `1 + N` cycles (each MS1 starts a cycle; MS2
/// blocks append in order) + a per-block `cut_n` map. Non-full cycles and unlisted blocks are
/// cleared. Returns `(layout, cut_n map, n_windows)`.
fn layout_from_profile(
    p: &TemplateProfile,
    n_blocks: usize,
) -> Result<(Layout, HashMap<usize, i64>, usize), String> {
    if p.n_windows == 0 {
        return Err("profile has 0 SWATH windows".into());
    }
    let period = 1 + p.n_windows;
    let mut cutn = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for e in &p.authored {
        if e.block >= n_blocks {
            return Err(format!("profile block {} out of range ({n_blocks} blocks)", e.block));
        }
        cutn.insert(e.block, e.cut_n);
        if e.ms == 1 && !cur.is_empty() {
            groups.push(std::mem::take(&mut cur));
        }
        cur.push(e.block);
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    // Keep only whole cycles (1 MS1 + N MS2); clear everything else.
    let mut layout = Layout::default();
    let mut authored: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for g in groups {
        if g.len() == period {
            authored.extend(&g);
            layout.cycles.push(Cycle { blocks: g });
        }
    }
    if layout.cycles.is_empty() {
        return Err(format!("profile has no whole {period}-block cycles"));
    }
    layout.partial = (0..n_blocks).filter(|i| !authored.contains(i)).collect();
    Ok((layout, cutn, p.n_windows))
}

// --- driver --------------------------------------------------------------------------------

/// Render the no-IM DIA DB at `db_path` and author it into the ZenoTOF `template_wiff`,
/// writing the pure-synthetic `.wiff` + `.wiff.scan` (and siblings) into `out_dir`.
///
/// The sim may cover fewer cycles than the template (authored first, the rest cleared) but not
/// more. Authoring uses the grow rebuild (arbitrary peaks/scan) with full `Idx` retranslation.
/// With `profile_path`, roles + seeds come from a per-template characterization (exact, works on
/// any pwiz-readable template); without it, the K562-fit role model + closed-form seed are used.
pub fn write_sciex_wiff(
    db_path: &Path,
    template_wiff: &Path,
    out_dir: &Path,
    opts: SciexWriteOptions,
    profile_path: Option<&Path>,
) -> Result<SciexWriteSummary, String> {
    use crate::sim::dia_render::render_dia_scans_each;

    for (name, v) in [
        ("precursor_noise_ppm", opts.precursor_noise_ppm),
        ("fragment_noise_ppm", opts.fragment_noise_ppm),
        ("overlay_ppm", opts.overlay_ppm),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(format!("{name} must be finite and >= 0, got {v}"));
        }
    }
    if !opts.spike_scale.is_finite() || opts.spike_scale <= 0.0 {
        return Err(format!("spike_scale must be finite and > 0, got {}", opts.spike_scale));
    }

    // Template: method (window count) + spectra bytes (mutable — edited in place).
    let method = read_method(template_wiff).map_err(|e| format!("read .wiff method: {e}"))?;
    let scan_path = format!("{}.scan", template_wiff.to_string_lossy());
    let scan = std::fs::read(&scan_path).map_err(|e| format!("read {scan_path}: {e}"))?;
    let blocks = scan_blocks(&scan);

    // Layout + per-block cut_n: from a per-template profile (exact, any template) if given, else
    // the K562-fit metadata role model + closed-form seed.
    let (layout, cutn_override, n_windows) = match profile_path {
        Some(pp) => {
            let prof = load_profile(pp)?;
            layout_from_profile(&prof, blocks.len())?
        }
        None => (label_layout(&blocks, method.swath_windows.len())?, HashMap::new(), method.swath_windows.len()),
    };
    let n_cycles = layout.cycles.len();
    // cut_n for a block: profile override (exact) or the closed-form recovery from the header.
    let cut_n_of = |bidx: usize, b: &ScanBlock| -> Result<i64, String> {
        match cutn_override.get(&bidx) {
            Some(&c) => Ok(c),
            None => Ok(seed_cut_n(read_hdr(&scan, b.ff)?, b.cal_a, b.cal_b)),
        }
    };

    // Render the simulated DB (collect; the walk yields acquisition order: MS1 then windows).
    let mut rendered = Vec::new();
    let counts = render_dia_scans_each(db_path, opts.num_threads, opts.quad_k, |d| {
        rendered.push(d);
        Ok(())
    })?;

    // The sim renders whole cycles in acquisition order (MS1 then N windows). It may cover
    // FEWER cycles than the template (a shorter gradient) — author those, clear the rest — but
    // never more than the template can hold.
    let period = 1 + n_windows;
    if rendered.len() % period != 0 {
        return Err(format!(
            "sim produced {} scans, not a whole multiple of the {period}-scan cycle \
             (1 MS1 + {n_windows} MS2)",
            rendered.len()
        ));
    }
    let sim_cycles = rendered.len() / period;
    if sim_cycles > n_cycles {
        return Err(format!(
            "sim has {sim_cycles} cycles but the template holds only {n_cycles}; shorten the \
             gradient or use a longer template"
        ));
    }
    if sim_cycles < n_cycles {
        log_short(sim_cycles, n_cycles);
    }

    // Author the first `sim_cycles` template cycles positionally (MS1 then windows, MS-level
    // cross-checked). The grow writer sizes each block to its tokens, so the full simulated peak
    // list is authored (no slot budget). Remaining full-cycle blocks AND partial blocks are
    // cleared (seed-only) — the file stays purely synthetic. Rare embedded-`ffffffff` blocks are
    // left verbatim (not grown) to avoid a bad token split.
    let mut blocks_authored = 0usize;
    let mut blocks_cleared = 0usize;
    let mut blocks_preserved = 0usize;
    let mut blocks_verbatim = 0usize;
    let mut ri = 0usize;
    let mut edits: Vec<GrowEdit> = Vec::with_capacity(blocks.len());
    // Push an edit, skipping embedded-ffffffff blocks (grown-body replace would drop their
    // mini-block). Returns whether the edit was applied.
    let mut push_edit = |bidx: usize, tokens: Vec<u8>| -> bool {
        if is_clean_tail(&scan, &blocks[bidx]) {
            edits.push(GrowEdit { block: bidx, tokens });
            true
        } else {
            blocks_verbatim += 1;
            false
        }
    };
    // A cleared block is authored seed-only but PADDED to its original body length: pwiz uses a
    // per-scan length that we do not rewrite, so shrinking a block makes it over-read — keeping
    // the original size (like the length-preserving path) avoids that, and the 0x00 padding
    // decodes to zero-intensity peaks a search engine ignores. Authored blocks grow freely.
    let clear_tokens = |b: &ScanBlock, cut_n: i64| -> Result<Vec<u8>, String> {
        let body = b.end - (b.ff + 9);
        let mut t = author_tokens(&[], cut_n)?;
        if t.len() < body {
            // Pad with 0xff (a terminator continuation), NOT 0x00 — the reader decodes trailing 0x00
            // bytes as spurious peaks, whereas a 0xff run is the block's natural end-of-peaklist.
            t.resize(body, 0xff);
        }
        Ok(t)
    };
    for (ci, cyc) in layout.cycles.iter().enumerate() {
        for (pos, &bidx) in cyc.blocks.iter().enumerate() {
            let b = &blocks[bidx];
            let cut_n = cut_n_of(bidx, b)?;
            if ci < sim_cycles {
                let desc = &rendered[ri];
                ri += 1;
                let want_ms1 = pos == 0;
                if want_ms1 != (desc.ms_level == 1) {
                    return Err(format!(
                        "MS-level misalignment at cycle block {bidx}: template expects {} but sim is MS{}",
                        if want_ms1 { "MS1" } else { "MS2" },
                        desc.ms_level
                    ));
                }
                let (cap, ppm) = if want_ms1 {
                    (opts.max_ms1_peaks, opts.precursor_noise_ppm)
                } else {
                    (opts.max_ms2_peaks, opts.fragment_noise_ppm)
                };
                // In overlay mode, scale the synthetic peaks relative to the scan's real-peak
                // background (spike_scale) so the spike-in strength is controllable; pure-
                // synthetic uses the fixed full scale.
                let (full_scale, real) = if opts.overlay_ppm > 0.0 {
                    let real = decode_template_peaks(&scan, b, cut_n);
                    let real_max = real.iter().map(|&(_, i)| i).max().unwrap_or(0) as f64;
                    let fs = if real_max > 0.0 {
                        (opts.spike_scale * real_max).max(1.0)
                    } else {
                        INTENSITY_FULL_SCALE
                    };
                    (fs, Some(real))
                } else {
                    (INTENSITY_FULL_SCALE, None)
                };
                let mut nv = sim_to_n(&desc.peaks, b.cal_a, b.cal_b, cut_n, ppm, full_scale);
                if let Some(real) = real {
                    nv.extend(real); // spike-in: real⊕sim
                }
                let tokens = author_tokens(&finalize_peaks(nv, cap), cut_n)?;
                if push_edit(bidx, tokens) {
                    blocks_authored += 1;
                }
            } else if !opts.preserve_template_partial {
                let tokens = clear_tokens(b, cut_n)?; // beyond the sim: pad-clear
                if push_edit(bidx, tokens) {
                    blocks_cleared += 1;
                }
            } else {
                blocks_preserved += 1;
            }
        }
    }
    // Partial blocks: pad-clear (seed-only, original size) unless preservation is opted in.
    if opts.preserve_template_partial {
        blocks_preserved += layout.partial.len();
    } else {
        for &bidx in &layout.partial {
            let b = &blocks[bidx];
            let cut_n = cut_n_of(bidx, b)?;
            let tokens = clear_tokens(b, cut_n)?;
            if push_edit(bidx, tokens) {
                blocks_cleared += 1;
            }
        }
    }
    if blocks_verbatim > 0 {
        eprintln!("write_sciex_wiff: {blocks_verbatim} embedded-ffffffff blocks left verbatim");
    }

    // Grow rebuild: author arbitrary token lengths, retranslate EVERY Idx offset, write out.
    let idx = read_idx_stream(template_wiff).map_err(|e| format!("read Idx: {e}"))?;
    let (new_scan, breakpoints) =
        rebuild_grow(&scan, &blocks, &edits).map_err(|e| format!("rebuild_grow: {e:?}"))?;
    if new_scan.len() > u32::MAX as usize + 44 {
        return Err(format!(
            "authored .wiff.scan is {} bytes, exceeds the u32 Idx offset ceiling",
            new_scan.len()
        ));
    }
    let new_idx =
        retranslate_idx(&idx, &breakpoints).map_err(|e| format!("retranslate_idx: {e:?}"))?;
    let out_wiff = copy_bundle(template_wiff, out_dir, &new_scan)?;
    patch_idx_stream(&out_wiff, &new_idx).map_err(|e| format!("patch Idx: {e}"))?;

    Ok(SciexWriteSummary {
        scans: counts.scans,
        ms1: counts.ms1,
        ms2: counts.ms2,
        ms2_nonempty: counts.ms2_nonempty,
        blocks_authored,
        blocks_cleared,
        blocks_preserved,
        blocks_verbatim,
        cycles: n_cycles,
        windows: n_windows,
        length_preserving: false,
    })
}

/// Count acquisition cycles (MS1 markers) in a ZenoTOF template so the sim can be built to
/// match. Reads the `.wiff` method (for `N`) + the sibling `.wiff.scan` + the `Idx`.
pub fn sciex_template_cycles(template_wiff: &Path) -> Result<usize, String> {
    let method = read_method(template_wiff).map_err(|e| format!("read .wiff method: {e}"))?;
    let scan_path = format!("{}.scan", template_wiff.to_string_lossy());
    let scan = std::fs::read(&scan_path).map_err(|e| format!("read {scan_path}: {e}"))?;
    let blocks = scan_blocks(&scan);
    let layout = label_layout(&blocks, method.swath_windows.len())?;
    Ok(layout.cycles.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Encode MS level via metadata length (ff - meta): MS1 large (60), MS2 small (29).
    fn blk(ms1: bool) -> ScanBlock {
        let ml = if ms1 { 60 } else { 29 };
        ScanBlock { meta: 0, ff: ml, end: ml + 9, cal_a: 0.00048982, cal_b: -12.9 }
    }

    #[test]
    fn label_layout_greedy_cycles() {
        // 2 leading MS2, then two full cycles of 1 MS1 + 3 MS2, then 1 trailing MS2.
        let mut b = vec![blk(false), blk(false)]; // leading partial
        for _ in 0..2 {
            b.push(blk(true));
            b.extend(std::iter::repeat_with(|| blk(false)).take(3));
        }
        b.push(blk(false)); // trailing partial
        let layout = label_layout(&b, 3).expect("label");
        assert_eq!(layout.cycles.len(), 2);
        assert_eq!(layout.partial.len(), 3); // 2 leading + 1 trailing
        assert_eq!(layout.cycles[0].blocks, vec![2, 3, 4, 5]);
        assert_eq!(layout.cycles[1].blocks, vec![6, 7, 8, 9]);
    }

    #[test]
    fn label_layout_rejects_uniform_metadata() {
        // No MS1 survey (all same metadata length) -> not a recognizable SWATH layout.
        let b = vec![blk(false), blk(false), blk(false), blk(false)];
        assert!(label_layout(&b, 2).is_err());
    }

    #[test]
    fn author_tokens_seed_first_and_increasing() {
        // The payload seeds at the cutoff and is strictly increasing in n.
        let a = 0.000489823;
        let cal_b = -12.9765;
        let cut_n = seed_cut_n(2_440_200, a, cal_b);
        // Peaks above this block's cutoff (~893 m/z); the two at 1000.0 merge to one n.
        let peaks = vec![(1200.0_f64, 5000.0_f32), (1000.0, 9000.0), (1000.0, 1000.0)];
        let real = finalize_peaks(sim_to_n(&peaks, a, cal_b, cut_n, 0.0, INTENSITY_FULL_SCALE), 10);
        assert_eq!(real.len(), 2, "1000.0 merged, 1200.0 kept");
        let payload = author_tokens(&real, cut_n).expect("author");
        assert_eq!(&payload[payload.len() - 4..], &[0xff, 0xff, 0xff, 0xff], "ends with 0xff terminator");
        let dec = sciexwiff::wiffscan::decode_stream(&payload[..payload.len() - 4], 0, cut_n, 64, false).expect("decode");
        assert_eq!(dec[0].0, cut_n, "first peak is the cutoff seed");
        assert!(dec[1].0 > dec[0].0 && dec[2].0 > dec[1].0, "strictly increasing n");
        assert_eq!(dec.len(), 3, "seed + 2 real peaks, no padding");
    }

    #[test]
    fn author_tokens_seed_only_clear() {
        // An empty real list is a valid seed-only (cleared) spectrum: just the sentinel.
        let cut_n = 300_000;
        let cleared = author_tokens(&[], cut_n).expect("clear");
        assert_eq!(&cleared[cleared.len() - 4..], &[0xff, 0xff, 0xff, 0xff], "ends with 0xff terminator");
        let dec = sciexwiff::wiffscan::decode_stream(&cleared[..cleared.len() - 4], 0, cut_n, 8, false).expect("decode");
        assert_eq!(dec.len(), 1, "seed only");
        assert_eq!(dec[0].0, cut_n);
    }

    #[test]
    fn intensity_scaling_preserves_dynamic_range() {
        // Small float intensities must scale up (not collapse to 1) to keep the pattern.
        let a = 0.000489823;
        let cal_b = -12.9765;
        let cut_n = seed_cut_n(2_440_200, a, cal_b);
        let peaks = vec![(1000.0_f64, 0.001_f32), (1100.0, 0.5), (1200.0, 20.0)];
        let nv = sim_to_n(&peaks, a, cal_b, cut_n, 0.0, INTENSITY_FULL_SCALE);
        let max = nv.iter().map(|&(_, i)| i).max().unwrap();
        let min = nv.iter().map(|&(_, i)| i).min().unwrap();
        assert!(max >= 40_000, "top peak scaled to ~full scale, got {max}");
        assert!(max / min.max(1) > 100, "dynamic range preserved (not flattened), got {min}..{max}");
    }

    #[test]
    fn overlay_decodes_template_real_peaks() {
        // Spike-in reads the block's real peaks back as absolute (n, intensity).
        let cut_n = 300_000;
        let tokens = encode_stream(&[(cut_n, 1), (cut_n + 10, 500), (cut_n + 25, 700)]).unwrap();
        let mut scan = vec![0u8; 44];
        let ff = scan.len();
        scan.extend_from_slice(&[0xff, 0xff, 0xff, 0xff]); // sentinel
        scan.extend_from_slice(&0u32.to_le_bytes()); // hdr (ignored by pwiz)
        scan.push(0x00);
        scan.extend_from_slice(&tokens);
        scan.extend_from_slice(&[0xff, 0xff, 0xff, 0xff]); // terminator (dropped on decode)
        let end = scan.len();
        let b = ScanBlock { meta: ff, ff, end, cal_a: 0.00048982, cal_b: -12.9 };
        let tpeaks = decode_template_peaks(&scan, &b, cut_n);
        assert_eq!(tpeaks, vec![(cut_n + 10, 500), (cut_n + 25, 700)]);
        // Overlay union: sim peak at a fresh n plus the two template peaks.
        let merged = finalize_peaks([vec![(cut_n + 15, 999)], tpeaks].concat(), 10);
        assert_eq!(merged.len(), 3, "sim + 2 template peaks");
        assert!(merged.windows(2).all(|w| w[1].0 > w[0].0), "strictly increasing n");
    }

    // Gated: label roles on the real template.
    // TIMSIM_SCIEX_WIFF=<...wiff> TIMSIM_SCIEX_WIFF_SCAN=<...wiff.scan>
    #[test]
    fn label_roles_real() {
        let (wiff, scan_p) = match (
            std::env::var("TIMSIM_SCIEX_WIFF"),
            std::env::var("TIMSIM_SCIEX_WIFF_SCAN"),
        ) {
            (Ok(w), Ok(s)) => (w, s),
            _ => {
                eprintln!("SKIP label_roles_real: set TIMSIM_SCIEX_WIFF + TIMSIM_SCIEX_WIFF_SCAN");
                return;
            }
        };
        let method = read_method(&wiff).expect("method");
        let n = method.swath_windows.len();
        let scan = std::fs::read(&scan_p).expect("scan");
        let blocks = scan_blocks(&scan);
        let layout = label_layout(&blocks, n).expect("label");
        assert!(layout.cycles.len() > 10, "expected many cycles, got {}", layout.cycles.len());
        for c in &layout.cycles {
            assert_eq!(c.blocks.len(), 1 + n, "cycle must be 1 MS1 + {n} MS2");
        }
        eprintln!(
            "label_roles_real OK: {} windows, {} full cycles, {} partial blocks",
            n,
            layout.cycles.len(),
            layout.partial.len()
        );
    }

    /// Debug probe: decoded-intensity VALUE distribution of a `.wiff.scan` (not encoded byte-widths).
    /// Compares real vendor intensity magnitudes (template) vs our authored ones — used to spot that our
    /// full_scale=50000 scaling makes intensities far larger than the vendor's. Gated on
    /// `TIMSIM_SCIEX_WIFF_SCAN`.
    #[test]
    fn probe_intensity() {
        let scan_path = match std::env::var("TIMSIM_SCIEX_WIFF_SCAN") {
            Ok(p) => p,
            Err(_) => return,
        };
        let scan = std::fs::read(&scan_path).expect("read");
        let blocks = scan_blocks(&scan);
        let mut all: Vec<u32> = Vec::new();
        for b in blocks.iter().take(3000) {
            if let Ok(h) = read_hdr(&scan, b.ff) {
                let cut_n = seed_cut_n(h, b.cal_a, b.cal_b);
                for (_, i) in decode_template_peaks(&scan, b, cut_n) {
                    all.push(i);
                }
            }
        }
        all.sort_unstable();
        let n = all.len();
        if n == 0 {
            eprintln!("INTENSITY: no peaks");
            return;
        }
        let (s, m, big) = (
            all.iter().filter(|&&x| x < 124).count(),
            all.iter().filter(|&&x| (124..256).contains(&x)).count(),
            all.iter().filter(|&&x| x >= 256).count(),
        );
        eprintln!(
            "INTENSITY: {} peaks | min {} median {} p95 {} max {} | value ranges: <124={} 124-255={} >=256={}",
            n, all[0], all[n / 2], all[n * 95 / 100], all[n - 1], s, m, big
        );
    }

    /// Re-author a `.wiff` from a DB + template (to validate the terminator fix via pwiz readback).
    /// Gated on TIMSIM_SCIEX_DB + TIMSIM_SCIEX_TEMPLATE + TIMSIM_SCIEX_OUT.
    #[test]
    fn probe_reauthor() {
        let (db, tpl, out) = match (
            std::env::var("TIMSIM_SCIEX_DB"),
            std::env::var("TIMSIM_SCIEX_TEMPLATE"),
            std::env::var("TIMSIM_SCIEX_OUT"),
        ) {
            (Ok(d), Ok(t), Ok(o)) => (d, t, o),
            _ => return,
        };
        std::fs::create_dir_all(&out).unwrap();
        let opts = SciexWriteOptions::default();
        let summary = write_sciex_wiff(
            std::path::Path::new(&db),
            std::path::Path::new(&tpl),
            std::path::Path::new(&out),
            opts,
            None,
        )
        .expect("write_sciex_wiff");
        eprintln!("REAUTHOR OK: {summary:?}");
    }

    /// Debug probe: dump the trailing bytes of a few clean blocks + the byte AFTER b.end, to see the
    /// real token terminator our authored blocks must reproduce. Gated on `TIMSIM_SCIEX_WIFF_SCAN`.
    #[test]
    fn probe_terminator() {
        let scan_path = match std::env::var("TIMSIM_SCIEX_WIFF_SCAN") {
            Ok(p) => p,
            Err(_) => return,
        };
        let scan = std::fs::read(&scan_path).expect("read");
        let blocks = scan_blocks(&scan);
        let mut shown = 0;
        for b in blocks.iter() {
            if !is_clean_tail(&scan, b) || b.end - (b.ff + 9) < 20 {
                continue;
            }
            // decode to find the real peaks + terminator length
            let cut_n = match read_hdr(&scan, b.ff) {
                Ok(h) => seed_cut_n(h, b.cal_a, b.cal_b),
                Err(_) => continue,
            };
            let (peaks, term) = match block_payload(&scan, b) {
                Ok(x) => x,
                Err(_) => continue,
            };
            let tail: Vec<String> = scan[(b.end - 12)..(b.end + 6).min(scan.len())]
                .iter().map(|x| format!("{x:02x}")).collect();
            eprintln!(
                "block ff={} body_len={} decoded_peaks={} term_len={} cut_n={} | [b.end-12 .. b.end+6]=[{}]",
                b.ff, b.end - (b.ff + 9), peaks.len(), term, cut_n, tail.join(" ")
            );
            shown += 1;
            if shown >= 6 {
                break;
            }
        }
    }

    /// Debug probe: does `is_clean_tail` catch EVERY block whose body contains an embedded ffffffff
    /// (a mini block)? A block with a mini block that `is_clean_tail` reports clean is authored + its
    /// mini block overwritten -> the pwiz mis-seek. Gated on `TIMSIM_SCIEX_WIFF_SCAN`.
    #[test]
    fn probe_clean_tail_coverage() {
        let scan_path = match std::env::var("TIMSIM_SCIEX_WIFF_SCAN") {
            Ok(p) => p,
            Err(_) => return,
        };
        let scan = std::fs::read(&scan_path).expect("read .wiff.scan");
        let blocks = scan_blocks(&scan);
        let (mut with_ffff, mut clean_true, mut missed) = (0usize, 0usize, 0usize);
        let mut missed_samples: Vec<(usize, usize)> = Vec::new();
        for (bi, b) in blocks.iter().enumerate() {
            let body = &scan[b.ff + 9..b.end];
            // full-body ffffffff search (what is_clean_tail SHOULD catch)
            let has = body.windows(4).any(|w| w == [0xff, 0xff, 0xff, 0xff]);
            let clean = is_clean_tail(&scan, b);
            if has {
                with_ffff += 1;
            }
            if clean {
                clean_true += 1;
            }
            if has && clean {
                // a mini block present but is_clean_tail says clean -> it WILL be authored/overwritten
                missed += 1;
                if missed_samples.len() < 6 {
                    // where is the ffffffff relative to the body end?
                    let pos = body.windows(4).position(|w| w == [0xff, 0xff, 0xff, 0xff]).unwrap();
                    missed_samples.push((bi, body.len() - pos));
                }
            }
        }
        eprintln!(
            "CLEAN_TAIL: {} blocks | {} have an embedded ffffffff | is_clean_tail=true for {} | MISSED (has ffffffff but reported clean) = {}",
            blocks.len(), with_ffff, clean_true, missed
        );
        for (bi, from_end) in &missed_samples {
            eprintln!("  block {} : ffffffff sits {} bytes from body end", bi, from_end);
        }
    }

    /// Debug probe: decode a real `.wiff.scan` with OUR codec + each block's OWN calibration and
    /// report peaks whose recovered m/z is out of range (>2000). If WE see garbage, the block
    /// genuinely encodes a bad `n` (writer or per-block cal fault); if we see none but pwiz reads
    /// ~594M, pwiz decodes with a different calibration than the block carries (a mismatch).
    /// Gated on `TIMSIM_SCIEX_WIFF_SCAN`.
    #[test]
    fn probe_garbage_mz() {
        let scan_path = match std::env::var("TIMSIM_SCIEX_WIFF_SCAN") {
            Ok(p) => p,
            Err(_) => return,
        };
        let scan = std::fs::read(&scan_path).expect("read .wiff.scan");
        let blocks = scan_blocks(&scan);
        let mut cal_hist: std::collections::BTreeMap<i64, usize> = Default::default();
        for b in &blocks {
            *cal_hist.entry((b.cal_a * 1e8).round() as i64).or_default() += 1;
        }
        let (mut total, mut garbage) = (0usize, 0usize);
        let mut samples: Vec<(f64, i64, f64, i64)> = Vec::new();
        for b in &blocks {
            let cut_n = match read_hdr(&scan, b.ff) {
                Ok(h) => seed_cut_n(h, b.cal_a, b.cal_b),
                Err(_) => continue,
            };
            for (n, _) in decode_template_peaks(&scan, b, cut_n) {
                let mz = n_to_mz(n, b.cal_a, b.cal_b);
                total += 1;
                if mz > 2000.0 {
                    garbage += 1;
                    if samples.len() < 8 {
                        samples.push((mz, n, b.cal_a, cut_n));
                    }
                }
            }
        }
        eprintln!(
            "PROBE: {} blocks, {} peaks, {} garbage(>2000 m/z) = {:.3}%",
            blocks.len(),
            total,
            garbage,
            100.0 * garbage as f64 / total.max(1) as f64
        );
        eprintln!("cal_a clusters ((cal_a*1e8).round -> count): {:?}", cal_hist);
        for (mz, n, a, cut) in &samples {
            eprintln!("  garbage: mz={:.1} n={} cal_a={:.5e} cut_n={}", mz, n, a, cut);
        }
    }
}
