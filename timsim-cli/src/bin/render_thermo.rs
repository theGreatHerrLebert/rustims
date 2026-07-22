//! `timsim-render-thermo` (M2) — render the instrument-independent feature space into a real Thermo
//! Orbitrap **Astral DIA `.raw`** by authoring into a template's scan slots (no IMS dimension).
//!
//! The template IS the acquisition schedule: we walk its slots in order and, for each, deposit the
//! eluting biology at that slot's own retention time —
//!   - **MS1 (FTMS profile):** every active precursor's isotope CENTROIDS (peak-shape de-risk settled:
//!     author centroids, not shapes), scaled by `abundance · elution(rt)`.
//!   - **MS2 (ASTMS centroid, DIA):** the fragment centroids of active precursors whose m/z falls in
//!     that slot's inherited isolation window.
//! Out-of-range peaks drop-and-account in the writer; the run-level lost ion current is reported.
//!
//! Multi-device hook (David's idea): MS1 isotopes are instrument-agnostic, so they come from
//! `--ion-spectra`; the MS2 fragment intensities are instrument-DEPENDENT, so `--fragment-spectra` can
//! point at a different predictor's output (e.g. Orbitrap-HCD for Astral) while everything else is held
//! fixed — the same sample "acquired" on another device.

use anyhow::{anyhow, Result};
use arrow::array::{Array, Float32Array, Float64Array, ListArray, StringArray, UInt64Array, UInt8Array};
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;

use mscore::timstof::quadrupole::WindowTransmission;
use ms_io::sim::acquisition::{AcquisitionWriter, ScanDescriptor, ThermoRawWriter};
use timsim_schema::tables::ion_spectra as SP;
use timsim_schema::tables::{peptide_rt as RT, precursors as PRE};

#[derive(Parser)]
#[command(name = "timsim-render-thermo", about = "feature space -> Thermo Astral DIA .raw (template-based)")]
struct Args {
    #[arg(long)] precursors: PathBuf,
    #[arg(long)] peptide_rt: PathBuf,
    /// MS1 isotope spectra (instrument-agnostic). Also the MS2 source unless --fragment-spectra is set.
    #[arg(long)] ion_spectra: PathBuf,
    /// Optional MS2 fragment spectra from a device-specific predictor (e.g. Orbitrap-HCD). Overrides the
    /// level-2 rows of --ion-spectra — the multi-device hook.
    #[arg(long)] fragment_spectra: Option<PathBuf>,
    #[arg(long)] peptide_quantities: Option<PathBuf>,
    #[arg(long)] sample: Option<String>,
    /// A real Astral DIA `.raw` — supplies the scan schedule + isolation windows we author into.
    #[arg(long)] template: PathBuf,
    #[arg(long)] out: PathBuf,
    /// Chromatographic peak width (Gaussian sigma) in SECONDS of the template gradient.
    #[arg(long, default_value_t = 3.0)] sigma_seconds: f64,
    #[arg(long, default_value_t = 3.0)] n_sigma: f64,
    /// Quadrupole edge steepness `k` (sigmoid) for the isolation-window transmission — same as the
    /// timsTOF TimsTransmissionDIA default.
    #[arg(long, default_value_t = 15.0)] transmission_k: f64,
    /// Fraction of the template RT span trimmed at each end (avoid loading/wash regions).
    #[arg(long, default_value_t = 0.05)] gradient_trim: f64,
    #[arg(long, default_value_t = 1.0e5)] intensity_scale: f64,
    #[arg(long, default_value_t = 1.0)] min_peak_intensity: f64,
    /// Sidecar answer key (per-precursor DIA truth).
    #[arg(long)] thermo_truth: Option<PathBuf>,
    /// Durable run manifest (JSON): renderer identity, template digest, method, counts, truth schema.
    #[arg(long)] manifest: Option<PathBuf>,
    /// Fragment model that produced --ion-spectra (recorded in the manifest for reproducibility).
    #[arg(long, default_value = "")] frag_model: String,
    /// Acquisition method label recorded in the manifest (the windows come from the template).
    #[arg(long, default_value = "DIA")] method: String,
    /// The collision energy (NCE) the fragments were predicted at — validated against the template's
    /// actual NCE (a mismatch means the library was built for a different CE than the template was run at).
    #[arg(long)] expected_ce: Option<f64>,
}

fn load_list_spectra(path: &PathBuf, want_level: u8) -> Result<HashMap<u64, Vec<(f64, f32)>>> {
    let mut out: HashMap<u64, Vec<(f64, f32)>> = HashMap::new();
    for b in timsim_schema::read_stream(path, SP::TABLE)? {
        let b = b?;
        let pcid: &UInt64Array = b.column_by_name(SP::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let level: &UInt8Array = b.column_by_name(SP::MS_LEVEL).unwrap().as_any().downcast_ref().unwrap();
        let mz: &ListArray = b.column_by_name(SP::MZ).unwrap().as_any().downcast_ref().unwrap();
        let inten: &ListArray = b.column_by_name(SP::INTENSITY).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if level.value(i) != want_level { continue; }
            let mzv = mz.value(i); let mzv: &Float64Array = mzv.as_any().downcast_ref().unwrap();
            let iv = inten.value(i); let iv: &Float32Array = iv.as_any().downcast_ref().unwrap();
            let peaks: Vec<(f64, f32)> = (0..mzv.len()).map(|k| (mzv.value(k), iv.value(k))).collect();
            out.insert(pcid.value(i), peaks);
        }
    }
    Ok(out)
}

fn load_amounts(path: &Option<PathBuf>, sample: &Option<String>) -> Result<HashMap<u64, f64>> {
    use timsim_schema::tables::peptide_quantities as PQ;
    let mut out = HashMap::new();
    let Some(path) = path else { return Ok(out) };
    let chosen = match sample {
        Some(s) => s.clone(),
        None => {
            let mut samples: Vec<String> = Vec::new();
            for b in timsim_schema::read(path, PQ::TABLE)? {
                let s: &StringArray = b.column_by_name(PQ::SAMPLE_ID).unwrap().as_any().downcast_ref().unwrap();
                for i in 0..b.num_rows() { samples.push(s.value(i).to_string()); }
            }
            samples.sort(); samples.dedup();
            samples.into_iter().next().ok_or_else(|| anyhow!("{} has no samples", path.display()))?
        }
    };
    for b in timsim_schema::read(path, PQ::TABLE)? {
        let pid: &UInt64Array = b.column_by_name(PQ::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let sid: &StringArray = b.column_by_name(PQ::SAMPLE_ID).unwrap().as_any().downcast_ref().unwrap();
        let amt: &Float64Array = b.column_by_name(PQ::AMOUNT_AMOL).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if sid.value(i) == chosen { out.insert(pid.value(i), amt.value(i)); }
        }
    }
    Ok(out)
}

struct Prec {
    precursor_id: u64,
    peptide_id: u64,
    mz: f64,
    charge: i64,
    abundance: f64,
    apex_rt: f64,
    ms1: Vec<(f64, f32)>,
    ms2: Vec<(f64, f32)>,
}

fn main() -> Result<()> {
    let a = Args::parse();
    if !(a.sigma_seconds.is_finite() && a.sigma_seconds > 0.0) {
        return Err(anyhow!("--sigma-seconds must be finite and > 0"));
    }
    if !(a.n_sigma.is_finite() && a.n_sigma >= 0.0) {
        return Err(anyhow!("--n-sigma must be finite and >= 0"));
    }
    if !(a.gradient_trim.is_finite() && (0.0..0.5).contains(&a.gradient_trim)) {
        return Err(anyhow!("--gradient-trim must be in [0, 0.5)"));
    }
    if !(a.intensity_scale.is_finite() && a.intensity_scale > 0.0) {
        return Err(anyhow!("--intensity-scale must be finite and > 0"));
    }

    // peptide_id -> rt_index, and the artifact's fixed reference range (stamped over the whole space).
    let mut rt: HashMap<u64, f64> = HashMap::new();
    for b in timsim_schema::read(&a.peptide_rt, RT::TABLE)? {
        let id: &UInt64Array = b.column_by_name(RT::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let idx: &Float64Array = b.column_by_name(RT::RT_INDEX).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if Array::is_valid(idx, i) { rt.insert(id.value(i), idx.value(i)); }
        }
    }
    let md = timsim_schema::metadata(&a.peptide_rt)?;
    let parse = |k: &str| -> Result<f64> {
        md.get(k).ok_or_else(|| anyhow!("peptide_rt missing {k}"))?.trim().parse::<f64>().map_err(|e| anyhow!("bad {k}: {e}"))
    };
    let (lo, hi) = (parse("timsim.rt.index_min")?, parse("timsim.rt.index_max")?);
    let span = (hi - lo).max(1e-9);

    let amounts = load_amounts(&a.peptide_quantities, &a.sample)?;
    let mut ms1_raw = load_list_spectra(&a.ion_spectra, 1)?;
    let mut ms2_raw = load_list_spectra(a.fragment_spectra.as_ref().unwrap_or(&a.ion_spectra), 2)?;

    // Open the template: schedule (rt + isolation per slot) drives the whole render.
    let _ = std::fs::remove_dir_all(&a.out);
    let mut writer = ThermoRawWriter::from_template(&a.template, &a.out).map_err(|e| anyhow!("{e}"))?;
    let manifest = writer.manifest().to_vec();
    // Thermo stores scan retention time in MINUTES; convert to seconds so --sigma-seconds is literal and
    // the answer-key rt_seconds is in seconds.
    let schedule: Vec<(f64, Option<ms_io::sim::acquisition::IsolationWindow>)> =
        writer.schedule().into_iter().map(|(t, iso)| (t * 60.0, iso)).collect();
    let (ms1_cap, ms2_cap) = writer.capacity();
    // The active-set sweep requires slot RTs finite and nondecreasing in manifest (acquisition) order.
    let mut prev = f64::NEG_INFINITY;
    for (i, (t, iso)) in schedule.iter().enumerate() {
        if !t.is_finite() {
            return Err(anyhow!("template slot {i} has non-finite retention time"));
        }
        if *t + 1e-6 < prev {
            return Err(anyhow!(
                "template slot RTs not monotonic at slot {i} ({t}s < {prev}s) — the sweep needs acquisition order"
            ));
        }
        prev = *t;
        if let Some(w) = iso {
            if !(w.center_mz.is_finite() && w.width_mz.is_finite() && w.width_mz > 0.0) {
                return Err(anyhow!("template slot {i} has a degenerate isolation window"));
            }
        }
    }
    // Gradient window from the (validated monotonic) schedule ends; trim the loading/wash edges.
    let (t0, t1) = (schedule.first().unwrap().0, schedule.last().unwrap().0);
    let trim = (t1 - t0) * a.gradient_trim;
    let (g0, g1) = (t0 + trim, t1 - trim);
    let gspan = (g1 - g0).max(1e-9);
    eprintln!("template: {} slots (MS1={ms1_cap}, MS2={ms2_cap}), gradient {:.1}..{:.1}s", manifest.len(), g0, g1);

    // Build precursors: rt_index -> apex_rt on the analytical gradient (quantile-lite: linear on the
    // trimmed span). abundance = amount (peptide-level); m/z-native isotopes/fragments already predicted.
    let mut precs: Vec<Prec> = Vec::new();
    for b in timsim_schema::read_stream(&a.precursors, PRE::TABLE)? {
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
            let apex_rt = g0 + (rt_index - lo) / span * gspan;
            let amount = amounts.get(&pid.value(i)).copied().unwrap_or(1.0);
            let abundance = amount * ionz.value(i) as f64 * mff.value(i) as f64 * frac.value(i) as f64;
            // Skip non-finite apex/abundance (a NaN rt_index or quantity would poison the sweep sort and
            // the deposition); m/z must be finite for the window test.
            if !(apex_rt.is_finite() && abundance.is_finite() && mz.value(i).is_finite()) { continue; }
            let ms1 = ms1_raw.remove(&pcid.value(i)).unwrap_or_default();
            let ms2 = ms2_raw.remove(&pcid.value(i)).unwrap_or_default();
            if ms1.is_empty() && ms2.is_empty() { continue; }
            precs.push(Prec {
                precursor_id: pcid.value(i), peptide_id: pid.value(i), mz: mz.value(i),
                charge: chg.value(i).max(1) as i64, abundance, apex_rt, ms1, ms2,
            });
        }
    }
    eprintln!("precursors: {} eligible", precs.len());

    // Fail fast: validate the template is compatible with the requested acquisition BEFORE the expensive
    // authoring sweep — a mismatched template should be rejected in milliseconds, not after a multi-minute
    // render that produces a subtly-wrong .raw.
    let n_ms1_slots = manifest.iter().filter(|(_, l, _)| *l == 1).count();
    let n_ms2_slots = manifest.len() - n_ms1_slots;
    let n_window_slots = schedule.iter().filter(|(_, iso)| iso.is_some()).count();
    if precs.is_empty() {
        return Err(anyhow!(
            "no eligible precursors to render — the feature space is empty after MS1/MS2 spectrum \
             filtering; check --ion-spectra / --peptide-quantities for sample {:?}", a.sample));
    }
    if n_ms1_slots == 0 {
        return Err(anyhow!("template {} has no MS1 scans — cannot author precursor signal", a.template.display()));
    }
    if a.method.eq_ignore_ascii_case("DIA") {
        if n_ms2_slots == 0 {
            return Err(anyhow!(
                "template {} has no MS2 scans — it is not a DIA acquisition (method=DIA)", a.template.display()));
        }
        if n_window_slots == 0 {
            return Err(anyhow!(
                "template {} has {} MS2 scans but no parseable isolation windows — cannot author DIA fragments",
                a.template.display(), n_ms2_slots));
        }
    }
    // Soft m/z-coverage check: a template whose isolation windows don't span the sample's precursor m/z
    // range will leave most precursors unfragmented. Warn (don't fail) — it is usually a template mismatch.
    let (mut wlo, mut whi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (_, iso) in schedule.iter() {
        if let Some(w) = iso { wlo = wlo.min(w.center_mz - w.width_mz / 2.0); whi = whi.max(w.center_mz + w.width_mz / 2.0); }
    }
    if a.method.eq_ignore_ascii_case("DIA") && wlo.is_finite() {
        let outside = precs.iter().filter(|p| p.mz < wlo || p.mz > whi).count();
        let frac = outside as f64 / precs.len() as f64;
        if frac > 0.5 {
            eprintln!(
                "  WARNING: {:.0}% of precursors ({}/{}) fall outside the template isolation range \
                 [{:.1}, {:.1}] Th — most will not be fragmented; is this the right template for the sample?",
                frac * 100.0, outside, precs.len(), wlo, whi);
        }
    }
    eprintln!(
        "template check OK: {} MS1 + {} MS2 slots ({} with windows), method={}",
        n_ms1_slots, n_ms2_slots, n_window_slots, a.method);

    // CE validation (#8): the template's own MS2 scans carry the NCE it was acquired at. Compare it to the
    // CE the fragment library was predicted at (--expected-ce); a mismatch means the library and the
    // template disagree on collision energy, so the fragment intensities are for the wrong regime.
    let mut ces: Vec<f64> = schedule.iter()
        .filter_map(|(_, iso)| iso.map(|w| w.collision_energy))
        .filter(|c| c.is_finite() && *c > 0.0)
        .collect();
    let (mut template_nce, mut template_nce_min, mut template_nce_max) = (None, None, None);
    if ces.is_empty() {
        eprintln!("  note: template exposes no per-scan NCE — cannot validate collision energy");
    } else {
        ces.sort_by(f64::total_cmp);
        let (cmin, cmax, median) = (ces[0], ces[ces.len() - 1], ces[ces.len() / 2]);
        template_nce = Some(median);
        template_nce_min = Some(cmin);
        template_nce_max = Some(cmax);
        // A single fragment CE cannot represent a stepped/mixed-NCE acquisition — check the SPREAD across
        // windows, not just the median (which can pass while half the windows are off).
        let stepped = (cmax - cmin) / median.max(1.0) > 0.15;
        if let Some(ece) = a.expected_ce {
            if stepped {
                eprintln!(
                    "  WARNING: template uses stepped/mixed NCE [{:.1}..{:.1}] (median {:.1}) — a single \
                     fragment CE {:.1} cannot match every window", cmin, cmax, median, ece);
            } else if (median - ece).abs() / median.max(1.0) > 0.15 {
                eprintln!(
                    "  WARNING: fragment CE {:.1} differs from template NCE {:.1} by {:.0}% — the library \
                     was predicted at a collision energy the template was not acquired at",
                    ece, median, (median - ece).abs() / median.max(1.0) * 100.0);
            } else {
                eprintln!("  CE check OK: fragment CE {:.1} ≈ template NCE {:.1} [{:.1}..{:.1}]", ece, median, cmin, cmax);
            }
        }
    }

    // Active-set sweep over slots (schedule RT is monotonic). A precursor is active in [apex ± nσ·σ].
    let half = a.n_sigma * a.sigma_seconds;
    let mut order: Vec<usize> = (0..precs.len()).collect();
    order.sort_by(|&x, &y| precs[x].apex_rt.total_cmp(&precs[y].apex_rt)); // total_cmp: NaN-safe (guarded finite above)
    let two_sig2 = 2.0 * a.sigma_seconds * a.sigma_seconds;
    let floor = a.min_peak_intensity as f32;

    // The .raw peak count is a u32 on disk, but the thermorawfile author_centroids/author_profile
    // functions currently guard at u16::MAX (65_535) and also must fit the template scan's existing
    // packet budget — so authoring more than this errors. We respect that here. (FOLLOW-UP: relaxing the
    // writer's u16 guard to u32 + a repack path would let very dense scans keep all peaks; the format
    // supports it. Until then this keeps the most intense peaks — realistic centroiding — and accounts
    // for the rest.)
    const MAX_PEAKS: usize = 65_535;
    let (mut cursor, mut ms1_n, mut ms2_n) = (0usize, 0u64, 0u64);
    let (mut capped_slots, mut capped_peaks) = (0u64, 0u64);
    let mut active: Vec<usize> = Vec::new();
    for (slot, (&(_scan, ms_level, _is_profile), &(t, iso))) in manifest.iter().zip(schedule.iter()).enumerate() {
        // Advance/retract the active set to slot time t.
        while cursor < order.len() && precs[order[cursor]].apex_rt - half <= t { active.push(order[cursor]); cursor += 1; }
        active.retain(|&i| precs[i].apex_rt + half >= t);

        let mut peaks: Vec<(f64, f32)> = Vec::new();
        if ms_level == 1 {
            for &i in &active {
                let p = &precs[i];
                let w = (-((t - p.apex_rt).powi(2)) / two_sig2).exp();
                if w <= 1e-6 { continue; }
                let base = p.abundance * w * a.intensity_scale;
                for &(m, iv) in &p.ms1 {
                    let v = (base * iv as f64) as f32;
                    if v >= floor { peaks.push((m, v)); }
                }
            }
        } else if let Some(w) = iso {
            // Quadrupole isolation is a flat-top passband with sigmoid soft edges (mscore's
            // WindowTransmission — the no-IMS sibling of the TimsTransmissionDIA curve the timsTOF path
            // uses), NOT a hard rectangle: an edge precursor is only partially transmitted, so its
            // fragments contribute proportionally.
            let wt = WindowTransmission::new(w.center_mz, w.width_mz, a.transmission_k);
            for &i in &active {
                let p = &precs[i];
                let tprob = wt.probabilities(&[p.mz])[0];
                if tprob <= 1e-3 { continue; }
                let ew = (-((t - p.apex_rt).powi(2)) / two_sig2).exp();
                if ew <= 1e-6 { continue; }
                let base = p.abundance * ew * tprob * a.intensity_scale;
                for &(m, iv) in &p.ms2 {
                    let v = (base * iv as f64) as f32;
                    if v >= floor { peaks.push((m, v)); }
                }
            }
        }
        // Respect the format's per-spectrum peak ceiling: at very high co-elution density a slot can
        // exceed 65_535 peaks. Keep the most intense (what a real instrument's centroiding does) rather
        // than aborting the whole render, and account for what was dropped so the cap is never silent.
        if peaks.len() > MAX_PEAKS {
            peaks.sort_unstable_by(|x, y| y.1.total_cmp(&x.1)); // intensity desc
            capped_peaks += (peaks.len() - MAX_PEAKS) as u64;
            capped_slots += 1;
            peaks.truncate(MAX_PEAKS);
            peaks.sort_unstable_by(|x, y| x.0.total_cmp(&y.0)); // restore m/z order for the writer
        }
        if ms_level == 1 { ms1_n += peaks.len() as u64; } else { ms2_n += peaks.len() as u64; }
        // isolation:None preserves the template's inherited DIA window (we don't re-window).
        let desc = ScanDescriptor { ms_level, retention_time: t, isolation: None, peaks };
        writer.write_scan(&desc).map_err(|e| anyhow!("slot {slot}: {e}"))?;
    }
    writer.finalize().map_err(|e| anyhow!("{e}"))?;
    let ps = writer.profile_summary();
    eprintln!(
        "wrote Astral DIA .raw ({ms1_n} MS1 + {ms2_n} MS2 authored peaks) -> {}\n  MS1 drop tally: {} bins written, {} peaks dropped (ion current {:.3e})\n  per-slot peak cap: {} slots capped at {} peaks, {} peaks dropped",
        a.out.display(), ps.written_bins, ps.dropped_total(), ps.dropped_intensity,
        capped_slots, MAX_PEAKS, capped_peaks
    );

    // Answer key: per-precursor DIA truth (join in the harness by peptide_id -> sequence + charge + mz).
    if let Some(truth) = &a.thermo_truth {
        use arrow::array::{BooleanArray, Float64Array as F64, Int64Array, UInt64Array as U64};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        // Distinct MS2 isolation windows (the DIA scheme repeats), as quad-transmission profiles for the
        // in-window eligibility flag (transmitted > 0.5 by any window — consistent with the render).
        let mut wpairs: Vec<(f64, f64)> = schedule.iter()
            .filter_map(|(_, iso)| iso.map(|w| (w.center_mz, w.width_mz)))
            .collect();
        wpairs.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.total_cmp(&y.1)));
        wpairs.dedup_by(|x, y| (x.0 - y.0).abs() < 1e-6 && (x.1 - y.1).abs() < 1e-6);
        let windows: Vec<WindowTransmission> = wpairs.iter()
            .map(|&(c, w)| WindowTransmission::new(c, w, a.transmission_k)).collect();
        let (mut pc, mut pe, mut ch, mut mo, mut rtc, mut ab, mut hm, mut iw):
            (Vec<u64>, Vec<u64>, Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>, Vec<bool>) = Default::default();
        for p in &precs {
            pc.push(p.precursor_id); pe.push(p.peptide_id); ch.push(p.charge);
            mo.push(p.mz); rtc.push(p.apex_rt); ab.push(p.abundance);
            // Eligibility for DIA: a precursor can only be identified if it has fragments AND its m/z
            // falls in some inherited isolation window. The harness uses these to define the denominator.
            hm.push(!p.ms2.is_empty());
            iw.push(windows.iter().any(|wt| wt.probabilities(&[p.mz])[0] > 0.5));
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("precursor_id", DataType::UInt64, false),
            Field::new("peptide_id", DataType::UInt64, false),
            Field::new("charge", DataType::Int64, false),
            Field::new("mz", DataType::Float64, false),
            Field::new("rt_seconds", DataType::Float64, false),
            Field::new("abundance", DataType::Float64, false),
            Field::new("has_ms2", DataType::Boolean, false),
            Field::new("in_any_window", DataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(U64::from(pc)), Arc::new(U64::from(pe)), Arc::new(Int64Array::from(ch)),
            Arc::new(F64::from(mo)), Arc::new(F64::from(rtc)), Arc::new(F64::from(ab)),
            Arc::new(BooleanArray::from(hm)), Arc::new(BooleanArray::from(iw)),
        ])?;
        let file = std::fs::File::create(truth)?;
        let mut w = ArrowWriter::try_new(file, schema, None)?;
        w.write(&batch)?; w.close()?;
        eprintln!("  answer key ({} precursors) -> {}", precs.len(), truth.display());
    }

    // Durable run manifest: the auditable boundary for a render. Records renderer identity, template
    // identity (path + size + mtime — the robust file identity the flow also hashes for invalidation),
    // the fragment model / method, the content-addressed input paths (their hashes ARE the artifact ids),
    // and the render's own counts. This is what makes a `.raw` reproducible after the fact.
    if let Some(mpath) = &a.manifest {
        let tmeta = std::fs::metadata(&a.template).ok();
        let tbytes = tmeta.as_ref().map(|m| m.len());
        let tmtime = tmeta.as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        let manifest_json = serde_json::json!({
            "renderer": { "name": "timsim-render-thermo", "version": env!("CARGO_PKG_VERSION") },
            "acquisition": {
                "method": a.method,
                "windows_from_template": true,
                "template_nce": template_nce,
                "template_nce_min": template_nce_min,
                "template_nce_max": template_nce_max,
                "fragment_ce": a.expected_ce,
            },
            "template": { "path": a.template.display().to_string(), "bytes": tbytes, "mtime_unix": tmtime },
            "fragment_model": a.frag_model,
            "sample": a.sample,
            "intensity_scale": a.intensity_scale,
            "inputs": {
                "precursors": a.precursors.display().to_string(),
                "ion_spectra": a.ion_spectra.display().to_string(),
                "peptide_rt": a.peptide_rt.display().to_string(),
                "peptide_quantities": a.peptide_quantities.as_ref().map(|p| p.display().to_string()),
            },
            "counts": {
                "precursors_eligible": precs.len(),
                "template_slots": manifest.len(),
                "ms1_slots": n_ms1_slots,
                "ms2_slots": n_ms2_slots,
                "ms1_peaks_authored": ms1_n,
                "ms2_peaks_authored": ms2_n,
            },
            "peak_cap": { "max_peaks_per_slot": MAX_PEAKS, "slots_capped": capped_slots, "peaks_dropped": capped_peaks },
            "ms1_profile_drop": { "bins_written": ps.written_bins, "peaks_dropped": ps.dropped_total(), "ion_current_dropped": ps.dropped_intensity },
            "truth": {
                "path": a.thermo_truth.as_ref().map(|p| p.display().to_string()),
                "rows": precs.len(),
                "columns": ["precursor_id","peptide_id","charge","mz","rt_seconds","abundance","has_ms2","in_any_window"],
            },
        });
        std::fs::write(mpath, serde_json::to_string_pretty(&manifest_json)?)?;
        eprintln!("  run manifest -> {}", mpath.display());
    }
    Ok(())
}
