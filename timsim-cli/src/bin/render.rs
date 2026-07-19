//! `timsim-render` — the measurement stage: render the instrument-independent feature space into a
//! real Bruker timsTOF `.d` (MS1-only for this milestone).
//!
//! Pipeline: read `precursors` + `peptide_rt` → place each precursor on the acquisition grid (m/z→TOF
//! and 1/K0→scan, rt_index→frame) → stream-render isotope envelopes with the proven sweep
//! ([`timsim_cli::render::stream_render_flat`]) → write frames through the Rust-native
//! [`rustdf::data::tdf_writer::TdfWriter`].
//!
//! Two calibration modes:
//!   - **`--reference-d <PATH>`** (recommended): copy the Bruker `MzCalibration`/`TimsCalibration`/
//!     `GlobalMetadata`/`Segments` verbatim from a real `.d`, derive the acquisition geometry
//!     (num_scans, TOF/mobility ranges) from it, and PLACE ions with that same calibration
//!     ([`MzCalibrator`]/[`MobilityCalibrator`], ModelType 2). A vendor reader (openTIMS/DiaNN via the
//!     Bruker SDK) then derives correct m/z and 1/K0 because placement and coefficients agree.
//!   - **fallback** (no reference): the reference-free `SimpleIndexConverter` (sqrt TOF↔m/z, linear
//!     scan↔1/K0) from CLI ranges — a valid, self-consistent file for our own tooling.

use anyhow::{anyhow, Result};
use arrow::array::{Array, Float32Array, Float64Array, ListArray, StringArray, UInt64Array, UInt8Array};
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;

use mscore::chemistry::formulas::ccs_to_one_over_reduced_mobility;
use timsim_schema::tables::ion_spectra as SP;
use timsim_schema::tables::precursor_ccs as CCS;

use rustdf::data::calibration::{MobilityCalibrator, MzCalibrator};
use rustdf::data::handle::{
    IndexConverter, SimpleIndexConverter, TimsData, TimsIndexConverter, TimsLazyLoder,
    TimsRawDataLayout,
};
use rustdf::data::meta::{read_global_meta_sql, read_meta_data_sql, read_mz_calibration, read_tims_calibration};
use rustdf::data::tdf_writer::{RenderedFrame, TdfWriter, TdfWriterConfig};
use rustdf::data::utility::flatten_scan_values;
use timsim_cli::render::{stream_render_flat, Geometry, Ion};
use timsim_schema::tables::{peptide_rt as RT, precursors as PRE};

#[derive(Parser)]
#[command(name = "timsim-render", about = "feature space -> streaming render -> MS1 Bruker .d")]
struct Args {
    #[arg(long)]
    precursors: PathBuf,
    #[arg(long)]
    peptide_rt: PathBuf,
    /// The instrument-independent spectra (`timsim-spectra` output). The render is a pure PROJECTOR:
    /// it reads each ion's materialised MS1 spectrum and places its peaks onto the acquisition grid.
    #[arg(long)]
    ion_spectra: PathBuf,
    /// `precursor_ccs` artifact. When given, each precursor's mobility is CCS→1/K0 (Mason-Schamp,
    /// per-run gas/temperature) — physical mobility, required for a search engine. Without it, mobility
    /// falls back to a non-physical m/z trend (fine for format checks, not for DiaNN).
    #[arg(long)]
    precursor_ccs: Option<PathBuf>,
    /// `peptide_quantities` artifact — the per-sample biological abundance axis. When given, each ion's
    /// intensity is scaled by `amount_amol × ionization_propensity × modform_fraction` (v1's `events`),
    /// restoring the ~6-order dynamic range real DIA data has. WITHOUT it only `charge_fraction` varies,
    /// so every peptide renders at ~the same level (~1 order) — no intense anchors, which a search
    /// engine needs to calibrate against.
    #[arg(long)]
    peptide_quantities: Option<PathBuf>,
    /// Which sample of the design to render (default: first sorted). One render is one sample.
    #[arg(long)]
    sample: Option<String>,
    /// Output `.d` directory (overwritten if it exists).
    #[arg(long)]
    out: PathBuf,
    /// Reference `.d` to copy Bruker calibration/metadata from and place ions with. Omit for the
    /// reference-free Simple calibration.
    #[arg(long)]
    reference_d: Option<PathBuf>,
    #[arg(long, default_value_t = 3000, value_parser = clap::value_parser!(u32).range(1..))]
    n_frames: u32,
    /// Mobility scans per frame (ignored in --reference-d mode: taken from the reference).
    #[arg(long, default_value_t = 709, value_parser = clap::value_parser!(u32).range(1..))]
    n_scans: u32,
    #[arg(long, default_value_t = 30.0)]
    sigma_frames: f64,
    #[arg(long, default_value_t = 4.0)]
    sigma_scans: f64,
    #[arg(long, default_value_t = 3.0)]
    n_sigma: f64,
    /// Simple-mode m/z and 1/K0 ranges + digitizer size (ignored in --reference-d mode).
    #[arg(long, default_value_t = 100.0)]
    mz_min: f64,
    #[arg(long, default_value_t = 1700.0)]
    mz_max: f64,
    #[arg(long, default_value_t = 0.6)]
    im_min: f64,
    #[arg(long, default_value_t = 1.6)]
    im_max: f64,
    #[arg(long, default_value_t = 400_000)]
    digitizer_num_samples: u32,
    #[arg(long, default_value_t = 0.1)]
    cycle_seconds: f64,
    #[arg(long, default_value_t = 100_000.0)]
    intensity_scale: f64,
    /// Drop quantised (scan, tof) bins whose intensity is below this floor. Emulates a peak-picking
    /// cutoff: without it, the 2-D Gaussian spread emits a haze of intensity-1 bins that dominate the
    /// peak count and drown the chromatographic shape in quantisation noise. `1` keeps every non-zero
    /// bin (legacy behaviour).
    #[arg(long, default_value_t = 1)]
    min_peak_intensity: u32,
    /// Incomplete fragmentation (DIA only): the fraction of each precursor that survives the quad
    /// INTACT and bleeds into the MS2 windows, drawn per-ion (identity-keyed on precursor id) in
    /// `[min, max]`. Default `0..0` = full fragmentation. Mirrors v1's `precursor_survival_min/max`;
    /// set e.g. `--precursor-survival-max 0.3` for v1's PFRAG-MED regime.
    #[arg(long, default_value_t = 0.0)]
    precursor_survival_min: f64,
    #[arg(long, default_value_t = 0.0)]
    precursor_survival_max: f64,
    /// Debug cap: render only the first N precursors in file order (0 = all). NOTE this caps INPUT
    /// precursors, not surviving ions — a precursor later dropped for lacking spectra still consumes a
    /// slot, so under `--limit` the ion set can differ from an unlimited run. Fine for quick smoke tests;
    /// don't use it when byte-for-byte comparing against another render.
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// DIA render: number of apex-ordered frame chunks to stream the ion spectra in (0 = auto by a
    /// memory budget). Peak memory is one chunk's active ions, not the whole dataset — the render stays
    /// bounded by the elution set regardless of how many precursors there are. Force a value to test the
    /// chunk-stitching (any N ≥ 1 must produce byte-identical output to N = 1).
    #[arg(long, default_value_t = 0)]
    render_chunks: u32,
    /// Render a DIA run: interleave MS1 + MS2 frames on the reference `.d`'s cycle, gate fragments by
    /// the diagonal quadrupole transmission. Requires `--reference-d` (a DIA `.d` for the schedule).
    #[arg(long, default_value_t = false)]
    dia: bool,
    /// Render a DDA-PASEF run: MS1 surveys every `--precursors-every` frames, top-N precursor selection
    /// with dynamic exclusion, band-limited MS2. Writes a sidecar answer key (`--dda-truth`) tying each
    /// selection event to the true precursor. Requires `--reference-d`. (Oracle-isolation baseline: clean
    /// single-precursor MS2; in-window co-isolation contaminants are a follow-up.)
    #[arg(long, default_value_t = false)]
    dda: bool,
    /// DDA: MS1 survey cadence — every Nth frame is a precursor (MS1) frame; the N-1 between are MS2.
    #[arg(long, default_value_t = 10)]
    precursors_every: u32,
    /// DDA: max precursors packed into one MS2 (PASEF) frame.
    #[arg(long, default_value_t = 25)]
    max_precursors: usize,
    /// DDA: minimum MS1 intensity (abundance × elution) for a precursor to be selectable. Note this is
    /// the currently-uncalibrated abundance scale (see RENDER_CALIBRATION.md); default 0 = top-N only.
    #[arg(long, default_value_t = 0.0)]
    intensity_threshold: f64,
    /// DDA: dynamic-exclusion window in frames — an ion isn't re-selected until this many frames after
    /// its last selection.
    #[arg(long, default_value_t = 25)]
    exclusion_width: u32,
    /// DDA: path for the sidecar answer key (Parquet). Default: `<out>.dda_selected.parquet`.
    #[arg(long)]
    dda_truth: Option<PathBuf>,
    /// After writing, reopen the `.d` through the rustims reader and report what round-trips.
    #[arg(long, default_value_t = false)]
    verify: bool,
}

/// The acquisition geometry + calibration used to place ions. Closures hide whether the calibration
/// is the reference's Bruker model or the Simple fallback.
struct Placement {
    n_scans: u32,
    tof_max: u32,
    mz_min: f64,
    mz_max: f64,
    im_min: f64,
    im_max: f64,
    reference_d: Option<String>,
    to_tof: Box<dyn Fn(f64) -> u32>,
    to_scan: Box<dyn Fn(f64) -> u32>,
    to_mz: Box<dyn Fn(u32) -> f64>,
}

fn build_placement(a: &Args) -> Result<Placement> {
    match &a.reference_d {
        Some(ref_d) => {
            let ref_s = ref_d.to_str().unwrap().to_string();
            let gm = read_global_meta_sql(&ref_s).map_err(|e| anyhow!("read reference GlobalMetadata: {e}"))?;
            let frames = read_meta_data_sql(&ref_s).map_err(|e| anyhow!("read reference Frames: {e}"))?;
            let f0 = frames.first().ok_or_else(|| anyhow!("reference .d has no frames"))?;
            let n_scans = frames.iter().map(|f| f.num_scans).max().unwrap_or(0) as u32;

            // Build the pure-Rust ModelType-2 calibrators from the reference coefficients + frame temps.
            // Select the SAME calibration rows the copied Frames reference (f0's ids) — a reference with
            // several calibrations would otherwise place peaks with coefficients that disagree with what
            // the output stores, so a vendor reader would derive wrong m/z / mobility.
            let mzc_row = read_mz_calibration(&ref_s).map_err(|e| anyhow!("{e}"))?
                .into_iter().find(|c| c.id == f0.mz_calibration)
                .ok_or_else(|| anyhow!("no MzCalibration with id {} in reference", f0.mz_calibration))?;
            let mz = MzCalibrator::new(
                mzc_row.model_type, mzc_row.digitizer_timebase, mzc_row.digitizer_delay,
                mzc_row.t1, mzc_row.t2, mzc_row.dc1, mzc_row.dc2,
                mzc_row.c0, mzc_row.c1, mzc_row.c2, mzc_row.c3, mzc_row.c4, f0.t_1, f0.t_2,
            );
            let mz_for_tof = MzCalibrator::new(
                mzc_row.model_type, mzc_row.digitizer_timebase, mzc_row.digitizer_delay,
                mzc_row.t1, mzc_row.t2, mzc_row.dc1, mzc_row.dc2,
                mzc_row.c0, mzc_row.c1, mzc_row.c2, mzc_row.c3, mzc_row.c4, f0.t_1, f0.t_2,
            );
            let tc_row = read_tims_calibration(&ref_s).map_err(|e| anyhow!("{e}"))?
                .into_iter().find(|c| c.id == f0.tims_calibration)
                .ok_or_else(|| anyhow!("no TimsCalibration with id {} in reference", f0.tims_calibration))?;
            let mob = MobilityCalibrator::new(
                tc_row.c0, tc_row.c1, tc_row.c2, tc_row.c3, tc_row.c4,
                tc_row.c5, tc_row.c6, tc_row.c7, tc_row.c8, tc_row.c9,
            );

            eprintln!(
                "  reference .d: {}  (num_scans {}, tof_max {}, m/z {:.0}-{:.0}, 1/K0 {:.2}-{:.2})",
                ref_s, n_scans, gm.tof_max_index, gm.mz_acquisition_range_lower,
                gm.mz_acquisition_range_upper, gm.one_over_k0_range_lower, gm.one_over_k0_range_upper,
            );
            Ok(Placement {
                n_scans,
                tof_max: gm.tof_max_index,
                mz_min: gm.mz_acquisition_range_lower,
                mz_max: gm.mz_acquisition_range_upper,
                im_min: gm.one_over_k0_range_lower,
                im_max: gm.one_over_k0_range_upper,
                reference_d: Some(ref_s),
                to_tof: Box::new(move |m| mz_for_tof.mz_to_tof(m)),
                to_scan: Box::new(move |k0| mob.one_over_k0_to_scan(k0)),
                to_mz: Box::new(move |tof| mz.tof_to_mz(tof)),
            })
        }
        None => {
            let conv = SimpleIndexConverter::from_boundaries(
                a.mz_min, a.mz_max, a.digitizer_num_samples, a.im_min, a.im_max, a.n_scans - 1,
            );
            let (tof_intercept, tof_slope) = (conv.tof_intercept, conv.tof_slope);
            let (scan_intercept, scan_slope) = (conv.scan_intercept, conv.scan_slope);
            Ok(Placement {
                n_scans: a.n_scans,
                tof_max: a.digitizer_num_samples,
                mz_min: a.mz_min,
                mz_max: a.mz_max,
                im_min: a.im_min,
                im_max: a.im_max,
                reference_d: None,
                // tof = (sqrt(mz) - tof_intercept) / tof_slope ; scan = (1/K0 - scan_intercept) / scan_slope
                to_tof: Box::new(move |m| ((m.sqrt() - tof_intercept) / tof_slope).max(0.0) as u32),
                to_scan: Box::new(move |k0| ((k0 - scan_intercept) / scan_slope).max(0.0) as u32),
                to_mz: Box::new(move |tof| {
                    let c = SimpleIndexConverter {
                        tof_intercept,
                        tof_slope,
                        scan_intercept,
                        scan_slope,
                    };
                    c.tof_to_mz(0, &vec![tof])[0]
                }),
            })
        }
    }
}

fn main() -> Result<()> {
    let a = Args::parse();
    if !(a.intensity_scale.is_finite() && a.intensity_scale > 0.0) {
        return Err(anyhow!("--intensity-scale must be finite and > 0, got {}", a.intensity_scale));
    }
    let p = build_placement(&a)?;
    let g = Geometry {
        n_frames: a.n_frames,
        n_scans: p.n_scans,
        sigma_frames: a.sigma_frames,
        sigma_scans: a.sigma_scans,
        n_sigma: a.n_sigma,
    };

    // peptide_id -> rt_index.
    let mut rt: HashMap<u64, f64> = HashMap::new();
    for b in timsim_schema::read(&a.peptide_rt, RT::TABLE)? {
        let id: &UInt64Array = b.column_by_name(RT::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let idx: &Float64Array = b.column_by_name(RT::RT_INDEX).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if Array::is_valid(idx, i) {
                rt.insert(id.value(i), idx.value(i));
            }
        }
    }
    // The index -> frame mapping MUST use the artifact's fixed reference range (stamped over the whole
    // peptide space), NOT the min/max of whatever subset is loaded — otherwise the same peptide lands
    // at a different frame depending on `--limit` or which sample is rendered, defeating the whole
    // point of a portable RT index (rt.py stamps these precisely so no consumer re-derives them).
    let md = timsim_schema::metadata(&a.peptide_rt)?;
    let parse = |key: &str| -> Result<f64> {
        md.get(key)
            .ok_or_else(|| anyhow!("peptide_rt missing {key} — cannot map rt index to frames"))?
            .trim()
            .parse::<f64>()
            .map_err(|e| anyhow!("bad {key}: {e}"))
    };
    let lo = parse("timsim.rt.index_min")?;
    let hi = parse("timsim.rt.index_max")?;
    let span = (hi - lo).max(1e-9);

    if a.dda {
        return run_dda(&a, &p, &g, &rt, lo, span);
    }
    if a.dia {
        return run_dia(&a, &p, &g, &rt, lo, span);
    }

    // ── project: place each ion's materialised MS1 spectrum onto the grid ──────
    // The instrument-independent MS1 spectrum lives in ion_spectra; the render only PROJECTS it. Load
    // precursor_id -> MS1 peaks (ms_level 1). (MS2 rows are the projector's business once DIA lands.)
    let mut ms1: HashMap<u64, Vec<(f64, f32)>> = HashMap::new();
    for b in timsim_schema::read(&a.ion_spectra, SP::TABLE)? {
        let pcid: &UInt64Array = b.column_by_name(SP::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let level: &UInt8Array = b.column_by_name(SP::MS_LEVEL).unwrap().as_any().downcast_ref().unwrap();
        let mz: &ListArray = b.column_by_name(SP::MZ).unwrap().as_any().downcast_ref().unwrap();
        let inten: &ListArray = b.column_by_name(SP::INTENSITY).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if level.value(i) != 1 {
                continue;
            }
            let mzv = mz.value(i);
            let mzv: &Float64Array = mzv.as_any().downcast_ref().unwrap();
            let iv = inten.value(i);
            let iv: &Float32Array = iv.as_any().downcast_ref().unwrap();
            let peaks: Vec<(f64, f32)> = (0..mzv.len()).map(|k| (mzv.value(k), iv.value(k))).collect();
            ms1.insert(pcid.value(i), peaks);
        }
    }

    // Precursors give each ion its placement coordinates: CCS -> mobility scan, peptide_id -> elution
    // frame. The peaks themselves come from the materialised spectrum, projected via m/z -> TOF.
    let ccs = load_ccs(&a.precursor_ccs)?;
    let amounts = load_amounts(&a.peptide_quantities, &a.sample)?;
    let mut ions: Vec<Ion> = Vec::new();
    let mut skipped = 0usize;
    'outer: for b in timsim_schema::read(&a.precursors, PRE::TABLE)? {
        let pcid: &UInt64Array = b.column_by_name(PRE::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let pid: &UInt64Array = b.column_by_name(PRE::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let mz: &Float64Array = b.column_by_name(PRE::MZ).unwrap().as_any().downcast_ref().unwrap();
        let chg: &UInt8Array = b.column_by_name(PRE::CHARGE).unwrap().as_any().downcast_ref().unwrap();
        let frac: &Float32Array = b.column_by_name(PRE::CHARGE_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        let ionz: &Float32Array = b.column_by_name(PRE::IONIZATION_PROPENSITY).unwrap().as_any().downcast_ref().unwrap();
        let mff: &Float32Array = b.column_by_name(PRE::MODFORM_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let Some(spec) = ms1.get(&pcid.value(i)) else { continue }; // no materialised spectrum
            let Some(&rt_index) = rt.get(&pid.value(i)) else { continue };
            // Map the index range onto the LAST valid 0-based frame (n_frames - 1); scaling by n_frames
            // puts index_max one frame past the end (and disagrees with the DIA path).
            let apex_frame = (rt_index - lo) / span * (a.n_frames as f64 - 1.0);
            let scan = place_scan(pcid.value(i), mz.value(i), chg.value(i).max(1) as u32, &ccs, &p);

            let peaks: Vec<(u32, f32)> = spec
                .iter()
                .filter_map(|&(m, inten)| {
                    if m < p.mz_min || m > p.mz_max {
                        None // out of the acquisition range — the instrument wouldn't record it
                    } else {
                        // Highest valid TOF index is DigitizerNumSamples = tof_max - 1; tof_max itself
                        // is one past the declared range and some readers reject it.
                        Some(((p.to_tof)(m).min(p.tof_max.saturating_sub(1)), inten))
                    }
                })
                .collect();
            if peaks.is_empty() {
                skipped += 1;
                continue;
            }
            let amount = amounts.get(&pid.value(i)).copied().unwrap_or(1.0);
            let abundance = amount * ionz.value(i) as f64 * mff.value(i) as f64 * frac.value(i) as f64;
            ions.push(Ion { apex_frame, scan_center: scan as f64, abundance, peaks });
            if a.limit > 0 && ions.len() >= a.limit {
                break 'outer;
            }
        }
    }
    eprintln!(
        "  projected {} ions ({} skipped: no in-range peaks) — rendering to {}",
        ions.len(), skipped, a.out.display()
    );

    // ── render -> write ──────────────────────────────────────────────────────
    let _ = std::fs::remove_dir_all(&a.out);
    let cfg = TdfWriterConfig {
        num_scans: p.n_scans,
        digitizer_num_samples: p.tof_max.saturating_sub(1),
        mz_range: (p.mz_min, p.mz_max),
        one_over_k0_range: (p.im_min, p.im_max),
        compression_level: 1,
        scan_mode: 9,
        reference_d: p.reference_d.clone(),
    };
    let mut writer = TdfWriter::create(&a.out, cfg).map_err(|e| anyhow!("{e}"))?;

    let mut next_fid: u32 = 1;
    let mut total_peaks: u64 = 0;
    let mut err: Result<()> = Ok(());
    stream_render_flat(&ions, &g, |e| {
        if err.is_err() {
            return;
        }
        let target = e.frame + 1;
        while next_fid < target {
            if let Err(x) = write_frame(&mut writer, next_fid, 0, a.cycle_seconds, Vec::new(), Vec::new(), Vec::new()) {
                err = Err(x);
                return;
            }
            next_fid += 1;
        }
        let (scans, tofs, ints) = dedup_and_quantise(e.triples, a.intensity_scale, a.min_peak_intensity);
        total_peaks += scans.len() as u64;
        if let Err(x) = write_frame(&mut writer, target, 0, a.cycle_seconds, scans, tofs, ints) {
            err = Err(x);
            return;
        }
        next_fid = target + 1;
    });
    err?;
    while next_fid <= a.n_frames {
        write_frame(&mut writer, next_fid, 0, a.cycle_seconds, Vec::new(), Vec::new(), Vec::new())?;
        next_fid += 1;
    }
    writer.finalize().map_err(|e| anyhow!("{e}"))?;
    println!(
        "  wrote {} frames, {} MS1 peaks ({} calibration) -> {}",
        a.n_frames, total_peaks,
        if p.reference_d.is_some() { "reference Bruker" } else { "Simple" },
        a.out.display()
    );

    if a.verify {
        verify(&a.out, &p)?;
    }
    Ok(())
}

/// One-shot latch so a saturating `intensity_scale` warns once, not once per frame.
static SATURATION_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sum a frame's (possibly duplicated) triples in f64 per (scan, tof), then quantise. Summing before
/// quantising keeps co-eluting sub-quantum signal (the invariant the throughput bench enforces).
///
/// The quantum is `intensity_scale`. Bins below `floor` counts are dropped (a peak-picking cutoff);
/// bins whose scaled value exceeds `u32::MAX` would be silently clipped by the `as u32` saturating cast,
/// so we detect that and warn ONCE — a saturated frame means `intensity_scale` is too hot for the most
/// abundant ion and the dynamic range is being crushed at the top (calibrate the scale down).
fn dedup_and_quantise(triples: &[(u32, u32, f64)], scale: f64, floor: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    debug_assert!(scale.is_finite() && scale > 0.0, "intensity_scale must be finite and > 0");
    const CEIL: f64 = u32::MAX as f64;
    let mut summed: HashMap<(u32, u32), f64> = HashMap::with_capacity(triples.len());
    for &(scan, tof, v) in triples {
        *summed.entry((scan, tof)).or_insert(0.0) += v;
    }
    let (mut scans, mut tofs, mut ints) = (Vec::new(), Vec::new(), Vec::new());
    for ((scan, tof), v) in summed {
        let scaled = v * scale;
        if scaled >= CEIL
            && !SATURATION_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "  WARNING: intensity saturated a u32 bin (scaled {:.3e} >= {:.3e}) — --intensity-scale \
                 is too high for the most abundant ion; top of the dynamic range is being clipped",
                scaled, CEIL
            );
        }
        let q = scaled.min(CEIL) as u32;
        if q < floor.max(1) {
            continue;
        }
        scans.push(scan);
        tofs.push(tof);
        ints.push(q);
    }
    (scans, tofs, ints)
}

/// Standard TIMS gas / temperature for Mason-Schamp (N2 at ~305 K — the imspy defaults the CCS model
/// was trained against). These are the "per-run" settings the CCS→1/K0 conversion needs.
const MASS_GAS: f64 = 28.013;
const TEMP: f64 = 31.85;
const T_DIFF: f64 = 273.15;

/// `precursor_id -> CCS` (Å²), or an empty map if no artifact is given.
fn load_ccs(path: &Option<PathBuf>) -> Result<HashMap<u64, f64>> {
    let mut out = HashMap::new();
    let Some(path) = path else { return Ok(out) };
    for b in timsim_schema::read(path, CCS::TABLE)? {
        let pcid: &UInt64Array = b.column_by_name(CCS::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let ccs: &Float64Array = b.column_by_name(CCS::CCS).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            out.insert(pcid.value(i), ccs.value(i));
        }
    }
    Ok(out)
}

/// `peptide_id -> amount_amol` for one sample of the design (the first sorted `sample_id` if `sample`
/// is None). Empty map when no `peptide_quantities` path is given — the caller then falls back to a
/// unit amount, i.e. abundance driven by charge/ionisation propensities only.
fn load_amounts(path: &Option<PathBuf>, sample: &Option<String>) -> Result<HashMap<u64, f64>> {
    use timsim_schema::tables::peptide_quantities as PQ;
    let mut out = HashMap::new();
    let Some(path) = path else { return Ok(out) };

    // Resolve the sample: the caller's choice, or the first sorted id present.
    let chosen = match sample {
        Some(s) => s.clone(),
        None => {
            let mut samples: Vec<String> = Vec::new();
            for b in timsim_schema::read(path, PQ::TABLE)? {
                let s: &StringArray = b.column_by_name(PQ::SAMPLE_ID).unwrap().as_any().downcast_ref().unwrap();
                for i in 0..b.num_rows() {
                    samples.push(s.value(i).to_string());
                }
            }
            samples.sort();
            samples.dedup();
            samples.into_iter().next().ok_or_else(|| anyhow!("{} has no samples", path.display()))?
        }
    };

    for b in timsim_schema::read(path, PQ::TABLE)? {
        let pid: &UInt64Array = b.column_by_name(PQ::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let sid: &StringArray = b.column_by_name(PQ::SAMPLE_ID).unwrap().as_any().downcast_ref().unwrap();
        let amt: &Float64Array = b.column_by_name(PQ::AMOUNT_AMOL).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if sid.value(i) == chosen {
                out.insert(pid.value(i), amt.value(i));
            }
        }
    }
    Ok(out)
}

/// Deterministic `u64 -> [0, 1)` (splitmix64 finaliser). Identity-keyed randomness: the same id always
/// maps to the same value, so per-ion draws (e.g. survival) don't reshuffle when the ion set changes.
fn hash01(x: u64) -> f64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// The mobility scan for a precursor: physical CCS→1/K0 (Mason-Schamp) when its CCS is known, else a
/// non-physical m/z trend. The 1/K0 is clamped to the acquisition band, then mapped to a scan by the
/// run's mobility calibration.
fn place_scan(pcid: u64, mz: f64, charge: u32, ccs: &HashMap<u64, f64>, p: &Placement) -> f64 {
    let one_over_k0 = match ccs.get(&pcid) {
        Some(&c) => ccs_to_one_over_reduced_mobility(c, mz, charge, MASS_GAS, TEMP, T_DIFF),
        None => {
            let f = ((mz - p.mz_min) / (p.mz_max - p.mz_min)).clamp(0.0, 1.0);
            p.im_min + (p.im_max - p.im_min) * f
        }
    };
    (p.to_scan)(one_over_k0.clamp(p.im_min, p.im_max)).min(p.n_scans - 1) as f64
}

/// v1's precursor isolation-m/z metadata from the raw MS1 isotope envelope: keep isotopes above 5% of the
/// max, then mono = first, the envelope span's far end = last, `IsolationMz` = the most-intense isotope.
fn iso_metadata(ms1: &[(f64, f32)], precursor_mz: f64) -> (f64, f64, f64) {
    if ms1.is_empty() {
        return (precursor_mz, precursor_mz, precursor_mz);
    }
    let max_i = ms1.iter().map(|&(_, i)| i).fold(0.0f32, f32::max);
    let kept: Vec<(f64, f32)> = ms1.iter().copied().filter(|&(_, i)| i > 0.05 * max_i).collect();
    let kept = if kept.is_empty() { ms1.to_vec() } else { kept };
    let mono = kept.first().map(|&(m, _)| m).unwrap_or(precursor_mz);
    let last = kept.last().map(|&(m, _)| m).unwrap_or(precursor_mz);
    let largest = kept.iter().fold((mono, 0.0f32), |acc, &(m, i)| if i > acc.1 { (m, i) } else { acc }).0;
    (mono, largest, (mono + last) / 2.0)
}

/// Mobility (scan) window `[lo, hi]` an ion deposits across (n_sigma × sigma_scans around its apex scan).
fn scan_window_dda(scan: i64, g: &Geometry, n_scans: u32) -> (u32, u32) {
    let h = g.n_sigma * g.sigma_scans;
    let lo = (scan as f64 - h).max(0.0) as u32;
    let hi = ((scan as f64 + h) as u32).min(n_scans.saturating_sub(1));
    (lo, hi)
}

/// DDA-PASEF render — MS1 surveys + top-N selection (`timsim_cli::dda`) + band-limited MS2, plus a sidecar
/// answer key tying each selection event to the true precursor. Oracle-isolation baseline: clean single
/// target per band; in-window co-isolation contaminants (and DDA memory streaming) are follow-ups.
fn run_dda(a: &Args, p: &Placement, g: &Geometry, rt: &HashMap<u64, f64>, lo: f64, span: f64) -> Result<()> {
    use rustdf::data::tdf_writer::{DdaPasefWindow, DdaPrecursor, TdfWriter, TdfWriterConfig};
    use timsim_cli::dda::{schedule, Candidate, SelectionParams};
    use timsim_cli::render::gauss_frac;

    let ref_d = a.reference_d.as_ref().ok_or_else(|| anyhow!("--dda requires --reference-d"))?;
    let _ = ref_d;
    let project = |peaks: &[(f64, f32)]| -> Vec<(u32, f32)> {
        peaks.iter().filter_map(|&(m, iv)| {
            if m < p.mz_min || m > p.mz_max { None } else { Some(((p.to_tof)(m).min(p.tof_max.saturating_sub(1)), iv)) }
        }).collect()
    };
    let ccs = load_ccs(&a.precursor_ccs)?;
    let amounts = load_amounts(&a.peptide_quantities, &a.sample)?;

    // Load ALL ion_spectra (raw m/z peaks). DDA memory streaming is a follow-up (the DIA path is chunked).
    let (mut ms1_raw, mut ms2_raw): (HashMap<u64, Vec<(f64, f32)>>, HashMap<u64, Vec<(f64, f32)>>) = (HashMap::new(), HashMap::new());
    for b in timsim_schema::read_stream(&a.ion_spectra, SP::TABLE)? {
        let b = b?;
        let pcid: &UInt64Array = b.column_by_name(SP::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let level: &UInt8Array = b.column_by_name(SP::MS_LEVEL).unwrap().as_any().downcast_ref().unwrap();
        let mz: &ListArray = b.column_by_name(SP::MZ).unwrap().as_any().downcast_ref().unwrap();
        let inten: &ListArray = b.column_by_name(SP::INTENSITY).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let mzv = mz.value(i); let mzv: &Float64Array = mzv.as_any().downcast_ref().unwrap();
            let iv = inten.value(i); let iv: &Float32Array = iv.as_any().downcast_ref().unwrap();
            let peaks: Vec<(f64, f32)> = (0..mzv.len()).map(|k| (mzv.value(k), iv.value(k))).collect();
            match level.value(i) { 1 => { ms1_raw.insert(pcid.value(i), peaks); } 2 => { ms2_raw.insert(pcid.value(i), peaks); } _ => {} }
        }
    }

    struct DdaIon { peptide_id: u64, apex_frame: f64, scan: i64, abundance: f64, ms1: Vec<(u32, f32)>, ms2: Vec<(u32, f32)> }
    let mut ions: HashMap<u64, DdaIon> = HashMap::new();
    let mut cands: Vec<Candidate> = Vec::new();
    let mut order: u32 = 0;
    'outer: for b in timsim_schema::read_stream(&a.precursors, PRE::TABLE)? {
        let b = b?;
        let pcid: &UInt64Array = b.column_by_name(PRE::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let pid: &UInt64Array = b.column_by_name(PRE::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let mz: &Float64Array = b.column_by_name(PRE::MZ).unwrap().as_any().downcast_ref().unwrap();
        let chg: &UInt8Array = b.column_by_name(PRE::CHARGE).unwrap().as_any().downcast_ref().unwrap();
        let frac: &Float32Array = b.column_by_name(PRE::CHARGE_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        let ionz: &Float32Array = b.column_by_name(PRE::IONIZATION_PROPENSITY).unwrap().as_any().downcast_ref().unwrap();
        let mff: &Float32Array = b.column_by_name(PRE::MODFORM_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let Some(&rt_index) = rt.get(&pid.value(i)) else { continue };
            let Some(ms1raw) = ms1_raw.remove(&pcid.value(i)) else { continue };
            let apex_frame = 1.0 + (rt_index - lo) / span * (a.n_frames as f64 - 1.0);
            let scan = place_scan(pcid.value(i), mz.value(i), chg.value(i).max(1) as u32, &ccs, p) as i64;
            let amount = amounts.get(&pid.value(i)).copied().unwrap_or(1.0);
            let abundance = amount * ionz.value(i) as f64 * mff.value(i) as f64 * frac.value(i) as f64;
            let (mono_mz, largest_mz, average_mz) = iso_metadata(&ms1raw, mz.value(i));
            let ms1 = project(&ms1raw);
            let ms2 = ms2_raw.remove(&pcid.value(i)).map(|s| project(&s)).unwrap_or_default();
            if ms1.is_empty() && ms2.is_empty() { continue; }
            cands.push(Candidate {
                precursor_id: pcid.value(i), order, apex_frame, scan_apex: scan,
                mono_mz, largest_mz, average_mz, charge: chg.value(i).max(1) as i64, abundance,
                sigma_frames: g.sigma_frames, n_sigma: g.n_sigma,
            });
            ions.insert(pcid.value(i), DdaIon { peptide_id: pid.value(i), apex_frame, scan, abundance, ms1, ms2 });
            order += 1;
            if a.limit > 0 && ions.len() >= a.limit { break 'outer; }
        }
    }

    let params = SelectionParams {
        precursors_every: a.precursors_every.max(1), max_precursors: a.max_precursors,
        intensity_threshold: a.intensity_threshold, exclusion_frames: a.exclusion_width,
        band_half_width: 11, n_scans: p.n_scans, ce_bias: 54.1984, ce_slope: -0.0345,
    };
    let sched = schedule(&cands, &params, a.n_frames);
    eprintln!("  DDA: {} of {} precursors selected, {} MS2 events", sched.precursors.len(), cands.len(), sched.events.len());

    // Sequential TDF precursor ids (vendor requires 1..N; our u64 hash overflows i64). our_id -> tdf_id.
    let tdf_id: HashMap<u64, i64> = sched.precursors.iter().enumerate().map(|(i, c)| (c.precursor_id, i as i64 + 1)).collect();
    let mut events_by_frame: HashMap<i64, Vec<&timsim_cli::dda::SelectionEvent>> = HashMap::new();
    for e in &sched.events { events_by_frame.entry(e.ms2_frame).or_default().push(e); }

    let _ = std::fs::remove_dir_all(&a.out);
    let cfg = TdfWriterConfig {
        num_scans: p.n_scans, digitizer_num_samples: p.tof_max.saturating_sub(1),
        mz_range: (p.mz_min, p.mz_max), one_over_k0_range: (p.im_min, p.im_max),
        compression_level: 1, scan_mode: 8, reference_d: p.reference_d.clone(),
    };
    let mut writer = TdfWriter::create(&a.out, cfg).map_err(|e| anyhow!("{e}"))?;

    // Active-set sweep over ions by apex frame, for the MS1 survey deposition.
    let win: Vec<(u32, u32, u64)> = ions.iter().map(|(&id, io)| {
        let h = g.n_sigma * g.sigma_frames;
        ((io.apex_frame - h).max(1.0) as u32, ((io.apex_frame + h) as u32).min(a.n_frames), id)
    }).collect();
    let mut order_start: Vec<usize> = (0..win.len()).collect();
    order_start.sort_unstable_by_key(|&i| win[i].0);
    let mut cursor = 0usize;
    let mut active: Vec<usize> = Vec::new();
    let per = a.precursors_every.max(1);
    let (mut ms1_n, mut ms2_n) = (0u64, 0u64);

    for frame in 1..=a.n_frames {
        while cursor < win.len() && win[order_start[cursor]].0 <= frame { active.push(order_start[cursor]); cursor += 1; }
        active.retain(|&i| win[i].1 >= frame);
        let is_ms1 = (frame - 1) % per == 0;
        let f = frame as f64;
        let mut tri: Vec<(u32, u32, f64)> = Vec::new();
        if is_ms1 {
            for &i in &active {
                let io = &ions[&win[i].2];
                let ew = gauss_frac(f - 0.5, f + 0.5, io.apex_frame, g.sigma_frames);
                if ew <= 0.0 { continue; }
                let (slo, shi) = scan_window_dda(io.scan, g, p.n_scans);
                for scan in slo..=shi {
                    let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, io.scan as f64, g.sigma_scans);
                    if mw <= 0.0 { continue; }
                    let base = io.abundance * ew * mw;
                    for &(tof, iv) in &io.ms1 { tri.push((scan, tof, base * iv as f64)); }
                }
            }
        } else if let Some(evs) = events_by_frame.get(&(frame as i64)) {
            for e in evs {
                let io = &ions[&e.precursor_id];
                let ew = gauss_frac(f - 0.5, f + 0.5, io.apex_frame, g.sigma_frames);
                if ew <= 0.0 { continue; }
                let s0 = e.scan_begin.max(0) as u32;
                let s1 = (e.scan_end.min(p.n_scans as i64 - 1)).max(0) as u32;
                for scan in s0..=s1 {
                    let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, io.scan as f64, g.sigma_scans);
                    if mw <= 0.0 { continue; }
                    let base = io.abundance * ew * mw;
                    for &(tof, iv) in &io.ms2 { tri.push((scan, tof, base * iv as f64)); }
                }
            }
        }
        let (scans, tofs, ints) = dedup_and_quantise(&tri, a.intensity_scale, a.min_peak_intensity);
        if is_ms1 { ms1_n += scans.len() as u64 } else { ms2_n += scans.len() as u64 }
        write_frame(&mut writer, frame, if is_ms1 { 0 } else { 8 }, a.cycle_seconds, scans, tofs, ints)?;
    }

    let precursors: Vec<DdaPrecursor> = sched.precursors.iter().map(|c| DdaPrecursor {
        id: tdf_id[&c.precursor_id], largest_peak_mz: c.largest_mz, average_mz: c.average_mz,
        monoisotopic_mz: c.mono_mz, charge: c.charge, scan_number: c.scan_apex as f64,
        intensity: c.abundance, parent: c.parent_ms1_frame,
    }).collect();
    let pasef: Vec<DdaPasefWindow> = sched.events.iter().map(|e| DdaPasefWindow {
        frame: e.ms2_frame, scan_num_begin: e.scan_begin, scan_num_end: e.scan_end,
        isolation_mz: e.isolation_mz, isolation_width: e.isolation_width, collision_energy: e.collision_energy,
        precursor: tdf_id[&e.precursor_id],
    }).collect();
    writer.set_dda_schedule(precursors, pasef);
    writer.finalize().map_err(|e| anyhow!("{e}"))?;

    // Sidecar answer key: one row per selection EVENT, keyed on (ms2_frame, scan_begin).
    {
        use arrow::array::{Float64Array, Int64Array, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        let (mut fr, mut sb, mut se, mut td, mut pc, mut pe, mut ch, mut iso, mut mo, mut pa, mut it, mut rtc):
            (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<u64>, Vec<u64>, Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>, Vec<f64>, Vec<f64>) = Default::default();
        for e in &sched.events {
            let io = &ions[&e.precursor_id];
            fr.push(e.ms2_frame); sb.push(e.scan_begin); se.push(e.scan_end);
            td.push(tdf_id[&e.precursor_id]); pc.push(e.precursor_id); pe.push(io.peptide_id);
            ch.push(e.charge); iso.push(e.isolation_mz); mo.push(e.mono_mz);
            pa.push(e.parent_ms1_frame); it.push(e.event_intensity); rtc.push(io.apex_frame * a.cycle_seconds);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("ms2_frame", DataType::Int64, false),
            Field::new("scan_begin", DataType::Int64, false),
            Field::new("scan_end", DataType::Int64, false),
            Field::new("tdf_precursor_id", DataType::Int64, false),
            Field::new("precursor_id", DataType::UInt64, false),
            Field::new("peptide_id", DataType::UInt64, false),
            Field::new("charge", DataType::Int64, false),
            Field::new("isolation_mz", DataType::Float64, false),
            Field::new("mono_mz", DataType::Float64, false),
            Field::new("parent_ms1_frame", DataType::Int64, false),
            Field::new("event_intensity", DataType::Float64, false),
            Field::new("rt_seconds", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(Int64Array::from(fr)), Arc::new(Int64Array::from(sb)), Arc::new(Int64Array::from(se)),
            Arc::new(Int64Array::from(td)), Arc::new(UInt64Array::from(pc)), Arc::new(UInt64Array::from(pe)),
            Arc::new(Int64Array::from(ch)), Arc::new(Float64Array::from(iso)), Arc::new(Float64Array::from(mo)),
            Arc::new(Int64Array::from(pa)), Arc::new(Float64Array::from(it)), Arc::new(Float64Array::from(rtc)),
        ])?;
        let truth_path = a.dda_truth.clone().unwrap_or_else(|| a.out.with_extension("dda_selected.parquet"));
        let file = std::fs::File::create(&truth_path)?;
        let mut w = ArrowWriter::try_new(file, schema, None)?;
        w.write(&batch)?;
        w.close()?;
        println!("  wrote DDA .d ({} MS1 + {} MS2 peaks) + {} answer-key events -> {}", ms1_n, ms2_n, sched.events.len(), truth_path.display());
    }
    Ok(())
}

/// DIA render: MS1+MS2 frames on the reference's cycle, fragments gated by the diagonal transmission.
fn run_dia(a: &Args, p: &Placement, g: &Geometry, rt: &HashMap<u64, f64>, lo: f64, span: f64) -> Result<()> {
    use timsim_cli::dia::DiaSchedule;
    use timsim_cli::ms2::{active_frames, dia_render_range, DiaIon};

    let ref_d = a.reference_d.as_ref().ok_or_else(|| anyhow!("--dia requires --reference-d (a DIA .d for the window schedule)"))?;
    let sched = DiaSchedule::from_reference(ref_d.to_str().unwrap(), a.n_frames, p.n_scans)?;
    eprintln!("  DIA schedule: cycle_len {}, {} windows", sched.cycle_len, sched.windows.len());

    // ion_spectra is NOT loaded up front — it is streamed once per apex-chunk below, so peak memory is
    // one chunk's active ions rather than every precursor's spectra at once.

    // Project each precursor's spectra to tof and build DIA ions.
    let project = |peaks: &[(f64, f32)]| -> Vec<(u32, f32)> {
        peaks
            .iter()
            .filter_map(|&(m, inten)| {
                if m < p.mz_min || m > p.mz_max {
                    None
                } else {
                    Some(((p.to_tof)(m).min(p.tof_max.saturating_sub(1)), inten))
                }
            })
            .collect()
    };
    let ccs = load_ccs(&a.precursor_ccs)?;
    let amounts = load_amounts(&a.peptide_quantities, &a.sample)?;
    if amounts.is_empty() {
        eprintln!("  WARNING: no peptide_quantities — abundance is charge/ionisation only (~1 order of dynamic range, no anchor precursors)");
    }
    // Incomplete fragmentation is on only when max > 0; clamp to [0,1] and keep min <= max.
    let survival_span: Option<(f64, f64)> = if a.precursor_survival_max > 0.0 {
        if a.precursor_survival_min > a.precursor_survival_max {
            eprintln!(
                "  WARNING: --precursor-survival-min ({}) > --precursor-survival-max ({}); clamping min to max",
                a.precursor_survival_min, a.precursor_survival_max
            );
        }
        let mx = a.precursor_survival_max.clamp(0.0, 1.0);
        let mn = a.precursor_survival_min.clamp(0.0, mx);
        eprintln!("  incomplete fragmentation: precursor survival drawn per-ion in [{mn:.3}, {mx:.3}]");
        Some((mn, mx))
    } else {
        None
    };
    // Metadata pass: one lightweight record per precursor — apex frame, scan, abundance, survival, and its
    // file-order rank. NO peaks. This is the only O(n_precursors) structure that stays resident, and it is
    // tiny (~tens of bytes/ion); the heavy spectra are streamed per chunk below. `order` preserves the
    // precursor-file order so each chunk can visit its ions in the same order the single-pass render did —
    // keeping the per-frame deposit sequence, and thus the output, byte-identical regardless of chunking.
    struct IonMeta {
        apex_frame: f64,
        scan: f64,
        abundance: f64,
        precursor_mz: f64,
        survival: f64,
        order: u32,
    }
    let mut meta: HashMap<u64, IonMeta> = HashMap::new();
    let mut order: u32 = 0;
    'outer: for b in timsim_schema::read_stream(&a.precursors, PRE::TABLE)? {
        let b = b?;
        let pcid: &UInt64Array = b.column_by_name(PRE::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let pid: &UInt64Array = b.column_by_name(PRE::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let mz: &Float64Array = b.column_by_name(PRE::MZ).unwrap().as_any().downcast_ref().unwrap();
        let chg: &UInt8Array = b.column_by_name(PRE::CHARGE).unwrap().as_any().downcast_ref().unwrap();
        let frac: &Float32Array = b.column_by_name(PRE::CHARGE_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        let ionz: &Float32Array = b.column_by_name(PRE::IONIZATION_PROPENSITY).unwrap().as_any().downcast_ref().unwrap();
        let mff: &Float32Array = b.column_by_name(PRE::MODFORM_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let Some(&rt_index) = rt.get(&pid.value(i)) else { continue };
            // 1-based apex frame (the DIA schedule + Frames.Id are 1-based).
            let apex_frame = 1.0 + (rt_index - lo) / span * (a.n_frames as f64 - 1.0);
            let scan = place_scan(pcid.value(i), mz.value(i), chg.value(i).max(1) as u32, &ccs, p);
            // v1's `events`: amount_amol (per sample) × ionisation propensity × modform fraction ×
            // charge fraction. amount 1.0 when no quantities given (propensities-only fallback).
            let amount = amounts.get(&pid.value(i)).copied().unwrap_or(1.0);
            let abundance = amount * ionz.value(i) as f64 * mff.value(i) as f64 * frac.value(i) as f64;
            // Incomplete-fragmentation survival, drawn per-ion in [min, max] — IDENTITY-KEYED on the
            // precursor id, not a bulk RNG, so adding an ion doesn't reshuffle everyone else's survival.
            let survival = survival_span
                .map(|(mn, mx)| mn + (mx - mn) * hash01(pcid.value(i)))
                .unwrap_or(0.0);
            meta.insert(
                pcid.value(i),
                IonMeta { apex_frame, scan, abundance, precursor_mz: mz.value(i), survival, order },
            );
            order += 1;
            if a.limit > 0 && meta.len() >= a.limit {
                break 'outer;
            }
        }
    }

    // Chunk the run into apex-ordered frame ranges; each chunk streams ion_spectra once and holds only the
    // ions active in its range. Auto-size to a memory budget (a chunk ≈ budget worth of ions), capped so
    // FD/scan counts stay sane; `--render-chunks` overrides. Peak memory is thus one chunk, not the dataset.
    const BYTES_PER_ION_EST: usize = 3072;
    const BUDGET_BYTES: usize = 512 * 1024 * 1024;
    let n_chunks = if a.render_chunks > 0 {
        a.render_chunks
    } else {
        (meta.len() * BYTES_PER_ION_EST / BUDGET_BYTES + 1).clamp(1, 128) as u32
    }
    .clamp(1, a.n_frames.max(1));
    // Split the frame axis at equal-ION-COUNT quantiles of the elution windows (frame_start), NOT into
    // equal-width slices: elution clusters in time, so equal-width slices would put most ions in the busy
    // middle chunk. Quantile boundaries give every chunk ~n/n_chunks ions with a WELL-DISTRIBUTED elution;
    // peak memory is then one chunk's active set. (It cannot go below the active set itself: a pathological
    // elution window as wide as the whole run makes every ion active everywhere, i.e. O(total) — inherent,
    // not a chunking failure.) `starts` is a transient O(n) u32 vector, dropped once `bounds` is built.
    // bounds[0]=1 .. bounds[n_chunks]=n_frames+1, NON-decreasing (the clamp can repeat a value, which just
    // yields an empty range, handled below); chunk c renders [bounds[c], bounds[c+1]-1].
    let bounds: Vec<u32> = {
        let mut starts: Vec<u32> = meta.values().map(|m| active_frames(m.apex_frame, g).0).collect();
        starts.sort_unstable();
        let mut bounds: Vec<u32> = Vec::with_capacity(n_chunks as usize + 1);
        bounds.push(1);
        for c in 1..n_chunks {
            let mut f = if starts.is_empty() {
                1 + a.n_frames * c / n_chunks
            } else {
                starts[(starts.len() * c as usize / n_chunks as usize).min(starts.len() - 1)]
            };
            let prev = *bounds.last().unwrap();
            f = f.max(prev + 1).min(a.n_frames); // keep non-decreasing; leave room for the final chunk
            bounds.push(f);
        }
        bounds.push(a.n_frames + 1);
        bounds
    };
    eprintln!(
        "  streaming render: {} precursors in {} apex-chunk(s) (equal-ion) -> {}",
        meta.len(), n_chunks, a.out.display()
    );

    // Render + write, per-frame MsMsType.
    let _ = std::fs::remove_dir_all(&a.out);
    let cfg = TdfWriterConfig {
        num_scans: p.n_scans,
        digitizer_num_samples: p.tof_max.saturating_sub(1),
        mz_range: (p.mz_min, p.mz_max),
        one_over_k0_range: (p.im_min, p.im_max),
        compression_level: 1,
        scan_mode: 9,
        reference_d: p.reference_d.clone(),
    };
    let mut writer = TdfWriter::create(&a.out, cfg).map_err(|e| anyhow!("{e}"))?;
    // Persist OUR replayed frame -> window group (DiaFrameMsMsInfo) + the reference's windows.
    let frame_to_group: Vec<(u32, u32)> = (1..=a.n_frames)
        .filter_map(|f| sched.window_group(f).map(|g| (f, g)))
        .collect();
    writer.set_dia_schedule(frame_to_group);

    let mut next_fid: u32 = 1;
    let (mut ms1_peaks, mut ms2_peaks) = (0u64, 0u64);
    let gap_ms = |f: u32| if sched.ms_level(f) == 1 { 0u8 } else { 9u8 };

    for chunk in 0..n_chunks {
        let fc0 = bounds[chunk as usize];
        let fc1 = bounds[chunk as usize + 1] - 1;
        if fc1 < fc0 {
            continue; // empty range (more chunks than distinct apex starts)
        }

        // Build this chunk's active ions by streaming ion_spectra once and keeping only the precursors
        // whose elution window overlaps [fc0, fc1]. (had_ms1, ms1_peaks, ms2_peaks) per precursor.
        let mut builders: HashMap<u64, (bool, Vec<(u32, f32)>, Vec<(u32, f32)>)> = HashMap::new();
        for b in timsim_schema::read_stream(&a.ion_spectra, SP::TABLE)? {
            let b = b?;
            let pcid: &UInt64Array = b.column_by_name(SP::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
            let level: &UInt8Array = b.column_by_name(SP::MS_LEVEL).unwrap().as_any().downcast_ref().unwrap();
            let mz: &ListArray = b.column_by_name(SP::MZ).unwrap().as_any().downcast_ref().unwrap();
            let inten: &ListArray = b.column_by_name(SP::INTENSITY).unwrap().as_any().downcast_ref().unwrap();
            for i in 0..b.num_rows() {
                let pc = pcid.value(i);
                let Some(m) = meta.get(&pc) else { continue };
                let (fs, fe) = active_frames(m.apex_frame, g);
                if fe < fc0 || fs > fc1 {
                    continue; // not active anywhere in this chunk's frame range
                }
                let mzv = mz.value(i);
                let mzv: &Float64Array = mzv.as_any().downcast_ref().unwrap();
                let iv = inten.value(i);
                let iv: &Float32Array = iv.as_any().downcast_ref().unwrap();
                let raw: Vec<(f64, f32)> = (0..mzv.len()).map(|k| (mzv.value(k), iv.value(k))).collect();
                let proj = project(&raw);
                let e = builders.entry(pc).or_insert((false, Vec::new(), Vec::new()));
                match level.value(i) {
                    1 => { e.0 = true; e.1 = proj; }
                    2 => { e.2 = proj; }
                    _ => {}
                }
            }
        }

        // Assemble DiaIons — same keep rule as the single-pass render (an MS1 spectrum must exist and at
        // least one projected spectrum is non-empty) — sorted back into precursor-file order so the sweep
        // deposits in the same sequence as the unchunked render (byte-identical output).
        let mut ions: Vec<(u32, DiaIon)> = builders
            .into_iter()
            .filter_map(|(pc, (had_ms1, ms1p, ms2p))| {
                if !had_ms1 || (ms1p.is_empty() && ms2p.is_empty()) {
                    return None;
                }
                let m = meta.get(&pc)?;
                Some((
                    m.order,
                    DiaIon {
                        apex_frame: m.apex_frame,
                        scan_center: m.scan,
                        abundance: m.abundance,
                        precursor_mz: m.precursor_mz,
                        ms1_peaks: ms1p,
                        ms2_peaks: ms2p,
                        survival: m.survival,
                    },
                ))
            })
            .collect();
        ions.sort_unstable_by_key(|x| x.0);
        let ions: Vec<DiaIon> = ions.into_iter().map(|x| x.1).collect();

        let mut err: Result<()> = Ok(());
        dia_render_range(&ions, &sched, g, fc0, fc1, |frame, ms_type, tri| {
            if err.is_err() {
                return;
            }
            while next_fid < frame {
                if let Err(x) = write_frame(&mut writer, next_fid, gap_ms(next_fid), a.cycle_seconds, Vec::new(), Vec::new(), Vec::new()) {
                    err = Err(x);
                    return;
                }
                next_fid += 1;
            }
            let (scans, tofs, ints) = dedup_and_quantise(tri, a.intensity_scale, a.min_peak_intensity);
            if ms_type == 0 { ms1_peaks += scans.len() as u64 } else { ms2_peaks += scans.len() as u64 }
            if let Err(x) = write_frame(&mut writer, frame, ms_type, a.cycle_seconds, scans, tofs, ints) {
                err = Err(x);
                return;
            }
            next_fid = frame + 1;
        });
        err?;
    }
    while next_fid <= a.n_frames {
        write_frame(&mut writer, next_fid, gap_ms(next_fid), a.cycle_seconds, Vec::new(), Vec::new(), Vec::new())?;
        next_fid += 1;
    }
    writer.finalize().map_err(|e| anyhow!("{e}"))?;
    println!("  wrote {} frames ({} MS1 + {} MS2 peaks) -> {}", a.n_frames, ms1_peaks, ms2_peaks, a.out.display());
    if a.verify {
        verify(&a.out, p)?;
    }
    Ok(())
}

fn write_frame(
    writer: &mut TdfWriter,
    frame_id: u32,
    ms_ms_type: u8,
    cycle_seconds: f64,
    scans: Vec<u32>,
    tofs: Vec<u32>,
    intensities: Vec<u32>,
) -> Result<()> {
    writer
        .write_frame(&RenderedFrame {
            frame_id,
            retention_time: frame_id as f64 * cycle_seconds,
            ms_ms_type,
            scans,
            tofs,
            intensities,
        })
        .map_err(|e| anyhow!("{e}"))
}

/// Reopen the `.d` through the rustims reader and report what round-trips.
fn verify(dir: &std::path::Path, p: &Placement) -> Result<()> {
    let layout = TimsRawDataLayout::new(dir.to_str().unwrap());
    let n = layout.frame_meta_data.len();
    // Any converter suffices for raw reads; m/z below is computed via the placement's own calibration.
    let sic = SimpleIndexConverter::from_boundaries(p.mz_min, p.mz_max, p.tof_max, p.im_min, p.im_max, p.n_scans - 1);
    let reader = TimsLazyLoder { raw_data_layout: layout, index_converter: TimsIndexConverter::Simple(sic) };

    let mut total = 0u64;
    let mut non_empty = 0u64;
    let mut best: (u32, usize) = (0, 0);
    for fid in 1..=n as u32 {
        let raw = reader.get_raw_frame(fid);
        let peaks = raw.tof.len();
        total += peaks as u64;
        if peaks > 0 {
            non_empty += 1;
            if peaks > best.1 {
                best = (fid, peaks);
            }
        }
    }
    println!();
    println!("  ── verify (reopened through the rustims reader) ────────────");
    println!("  frames read           : {n}");
    println!("  non-empty frames      : {non_empty}");
    println!("  total MS1 peaks        : {total}");
    if best.1 > 0 {
        let raw = reader.get_raw_frame(best.0);
        let scans = flatten_scan_values(&raw.scan, true);
        let mut idx: Vec<usize> = (0..raw.intensity.len()).collect();
        idx.sort_by(|&x, &y| raw.intensity[y].partial_cmp(&raw.intensity[x]).unwrap());
        println!("  busiest frame          : id {} with {} peaks", best.0, best.1);
        for &j in idx.iter().take(3) {
            let mz = (p.to_mz)(raw.tof[j]);
            println!(
                "      scan {:>4}  tof {:>7}  m/z {:>9.4}  intensity {:.0}",
                scans[j], raw.tof[j], mz, raw.intensity[j]
            );
        }
    }
    Ok(())
}
