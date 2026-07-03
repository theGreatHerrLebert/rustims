//! timsTOF driver for the sparse LSH spectral-match index (Phase 1).
//!
//! Turns a [`TimsFrame`] into indexable **units**, where a unit is a
//! **mobility moving-average window** over the native scan axis: for a center
//! scan `s` it aggregates the sparse peaks of scans `[s-w, s+w]`. In diaPASEF a
//! fragment inherits its precursor's mobility, so a peptide's fragment bag
//! concentrates in a narrow scan band; the window integrates that IM peak
//! profile (SNR + completeness), it is not full deconvolution.
//!
//! Peaks are mapped to `i64` feature ids by [`MzFeatureMap`] — log-ppm bins
//! with a tolerance-width triangular splat, so sub-tolerance mass drift moves
//! weight smoothly instead of jumping across a bin edge. Feature vectors are
//! L2-normalized, ready to hand to any [`crate::algorithm::lsh::LshScheme`].
//!
//! **DIA safety:** moving-average windows must not cross DIA isolation-tile
//! scan boundaries (fragments from different quadrupole selections would mix).
//! The driver takes an optional `segment_boundaries` list (the frame's
//! `ProgramSlice` scan ranges); windows are clamped within a segment. `mscore`
//! stays DIA-agnostic — the caller (`rustdf`, which owns `DiaIndex`) supplies
//! the boundaries. An empty list means no clamp (e.g. MS1).

use std::collections::HashMap;

use crate::data::spectrum::MsType;
use crate::timstof::frame::TimsFrame;

/// Intensity transform applied to each peak before splatting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntensityTransform {
    /// Raw intensity.
    None,
    /// `sqrt(intensity)` — compresses dynamic range.
    Sqrt,
    /// `ln(1 + intensity)`.
    Log1p,
}

impl IntensityTransform {
    /// Apply the transform (negative inputs clamped to 0).
    #[inline]
    pub fn apply(&self, intensity: f64) -> f64 {
        let x = intensity.max(0.0);
        match self {
            IntensityTransform::None => x,
            IntensityTransform::Sqrt => x.sqrt(),
            IntensityTransform::Log1p => x.ln_1p(),
        }
    }
}

/// m/z → feature-index mapping: log-ppm bins + tolerance-width triangular splat.
///
/// `bin_ppm` is the (fine) bin width in ppm; `tol_ppm` is the matching
/// tolerance and the half-width of the triangular splat kernel. Keep
/// `bin_ppm <= tol_ppm` (bins finer than tolerance) so a peak spreads over
/// several bins and two peaks within `tol_ppm` retain high overlap.
#[derive(Clone, Copy, Debug)]
pub struct MzFeatureMap {
    bin_ppm: f64,
    tol_ppm: f64,
    /// `1 / ln(1 + bin_ppm·1e-6)` — converts `ln(mz)` to bin coordinate.
    scale: f64,
    /// Splat kernel half-width in bin units, `tol_ppm / bin_ppm`.
    half_width_bins: f64,
}

impl MzFeatureMap {
    /// Create a mapping. `bin_ppm` must be `> 0` and `tol_ppm >= bin_ppm`.
    pub fn new(bin_ppm: f64, tol_ppm: f64) -> Result<Self, String> {
        if !(bin_ppm > 0.0) {
            return Err(format!("bin_ppm must be > 0, got {bin_ppm}"));
        }
        if !(tol_ppm >= bin_ppm) {
            return Err(format!(
                "tol_ppm ({tol_ppm}) must be >= bin_ppm ({bin_ppm}) so the kernel spans >= 1 bin"
            ));
        }
        Ok(Self {
            bin_ppm,
            tol_ppm,
            scale: 1.0 / (1.0 + bin_ppm * 1e-6).ln(),
            half_width_bins: tol_ppm / bin_ppm,
        })
    }

    /// Bin width in ppm (recorded in the index config header).
    #[inline]
    pub fn bin_ppm(&self) -> f64 {
        self.bin_ppm
    }

    /// Matching tolerance / splat half-width in ppm (recorded in the header).
    #[inline]
    pub fn tol_ppm(&self) -> f64 {
        self.tol_ppm
    }

    /// Continuous bin coordinate for an m/z (`ln(mz) · scale`).
    #[inline]
    pub fn coord(&self, mz: f64) -> f64 {
        mz.ln() * self.scale
    }

    /// Splat one peak of `value` at `mz` into triangular-weighted bin
    /// contributions (weights sum to 1), accumulated into `acc`.
    fn splat_into(&self, mz: f64, value: f64, acc: &mut HashMap<i64, f64>) {
        if mz <= 0.0 || value == 0.0 {
            return;
        }
        let pos = self.coord(mz);
        let hw = self.half_width_bins;
        let lo = (pos - hw).ceil() as i64;
        let hi = (pos + hw).floor() as i64;

        // First pass: triangular weights and their sum.
        let mut sum = 0.0;
        let mut b = lo;
        while b <= hi {
            let w = 1.0 - (b as f64 - pos).abs() / hw;
            if w > 0.0 {
                sum += w;
            }
            b += 1;
        }
        if sum <= 0.0 {
            // Degenerate kernel: dump full value on the nearest bin.
            *acc.entry(pos.round() as i64).or_insert(0.0) += value;
            return;
        }
        // Second pass: normalized contributions.
        let mut b = lo;
        while b <= hi {
            let w = 1.0 - (b as f64 - pos).abs() / hw;
            if w > 0.0 {
                *acc.entry(b).or_insert(0.0) += value * (w / sum);
            }
            b += 1;
        }
    }

    /// Build an L2-normalized sparse feature vector (sorted by feature id) from
    /// a list of `(mz, intensity)` peaks — the entry point for query spectra
    /// (predicted bags) as well as frame windows.
    pub fn features(
        &self,
        peaks: &[(f64, f64)],
        transform: IntensityTransform,
    ) -> Vec<(i64, f32)> {
        let mut acc = HashMap::new();
        for &(mz, intensity) in peaks {
            self.splat_into(mz, transform.apply(intensity), &mut acc);
        }
        finalize(acc)
    }
}

/// L2-normalize an accumulated feature map into a sorted sparse vector.
/// Returns empty if the vector is zero-norm.
fn finalize(acc: HashMap<i64, f64>) -> Vec<(i64, f32)> {
    let norm: f64 = acc.values().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 {
        return Vec::new();
    }
    let mut features: Vec<(i64, f32)> =
        acc.into_iter().map(|(k, v)| (k, (v / norm) as f32)).collect();
    features.sort_unstable_by_key(|&(k, _)| k);
    features
}

/// Configuration for the mobility moving-average driver.
#[derive(Clone, Debug)]
pub struct WindowConfig {
    /// Half-width `w` of the moving-average window, in scans. Each unit
    /// aggregates scans `[center-w, center+w]`. `0` = raw scan.
    pub half_width: usize,
    /// Step between window centers, in occupied scans (`1` = one unit/scan).
    pub stride: usize,
    /// Intensity transform applied before splatting.
    pub transform: IntensityTransform,
    /// m/z → feature-index mapping.
    pub feature_map: MzFeatureMap,
    /// Optional top-N peak picking per window (by transformed intensity) before
    /// splatting — turns a dense noisy mobility window into a spectrum-like
    /// sparse bag, the way real spectral matching does. `None` keeps all peaks.
    pub top_n: Option<usize>,
}

/// Metadata for one indexed unit. DIA isolation-window fields (`window_group`)
/// are attached by the `rustdf` layer, which owns `DiaIndex`.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitMeta {
    pub frame_id: i32,
    pub center_scan: i32,
    pub retention_time: f64,
    pub mobility: f64,
    pub ms_type: MsType,
}

/// A produced unit: metadata plus its L2-normalized sparse feature vector
/// (sorted by feature id), ready to hand to an `LshScheme`.
#[derive(Clone, Debug)]
pub struct FrameUnit {
    pub meta: UnitMeta,
    pub features: Vec<(i64, f32)>,
}

/// Contiguous `[start, end)` peak-index range of one scan in a frame's flat
/// SoA arrays. Mirrors `rustdf`'s `build_scan_slices` (which `mscore` cannot
/// depend on) and assumes scans are grouped contiguously and ascending, as the
/// TDF reader produces.
#[derive(Clone, Copy, Debug)]
struct ScanSlice {
    scan: i32,
    start: usize,
    end: usize,
}

fn build_scan_slices(frame: &TimsFrame) -> Vec<ScanSlice> {
    let sc = &frame.scan;
    let mut out = Vec::new();
    if sc.is_empty() {
        return out;
    }
    let mut cur = sc[0];
    let mut start = 0usize;
    for i in 1..sc.len() {
        // Reader guarantee: scans are grouped and ascending. `partition_point`
        // in the caller relies on it, so assert it in debug builds.
        debug_assert!(sc[i] >= cur, "scan array must be ascending: {} after {}", sc[i], cur);
        if sc[i] != cur {
            // Skip sentinel/negative scans (parity with rustdf's build_scan_slices).
            if cur >= 0 {
                out.push(ScanSlice { scan: cur, start, end: i });
            }
            cur = sc[i];
            start = i;
        }
    }
    if cur >= 0 {
        out.push(ScanSlice { scan: cur, start, end: sc.len() });
    }
    out
}

/// Produce indexable units from a frame: one mobility moving-average window per
/// occupied center scan (stepped by `stride`), clamped to the enclosing
/// segment.
///
/// `segment_boundaries` are inclusive `(scan_lo, scan_hi)` ranges (a frame's
/// `ProgramSlice` scan ranges). Windows never cross a boundary. An empty slice
/// means the whole frame is one segment (no clamp).
pub fn frame_to_units(
    frame: &TimsFrame,
    cfg: &WindowConfig,
    segment_boundaries: &[(i32, i32)],
) -> Vec<FrameUnit> {
    let slices = build_scan_slices(frame);
    if slices.is_empty() {
        return Vec::new();
    }

    let mz = &frame.ims_frame.mz;
    let intensity = &frame.ims_frame.intensity;
    let mobility = &frame.ims_frame.mobility;
    let rt = frame.ims_frame.retention_time;
    let w = cfg.half_width as i32;
    let stride = cfg.stride.max(1);

    // Whole-frame fallback segment when no boundaries are supplied.
    let frame_seg = (slices[0].scan, slices[slices.len() - 1].scan);
    let seg_for = |s: i32| -> Option<(i32, i32)> {
        if segment_boundaries.is_empty() {
            Some(frame_seg)
        } else {
            segment_boundaries.iter().copied().find(|&(lo, hi)| s >= lo && s <= hi)
        }
    };

    let mut out = Vec::new();
    for ci in (0..slices.len()).step_by(stride) {
        let center = slices[ci];
        let seg = match seg_for(center.scan) {
            Some(s) => s,
            None => continue, // center outside every segment; skip
        };
        let win_lo = (center.scan - w).max(seg.0);
        let win_hi = (center.scan + w).min(seg.1);

        // Slices are sorted by scan → the window is a contiguous run. Gather
        // the window's transformed peaks first (so top-N can pick among them).
        let first = slices.partition_point(|sl| sl.scan < win_lo);
        let mut wpeaks: Vec<(f64, f64)> = Vec::new();
        let mut i = first;
        while i < slices.len() && slices[i].scan <= win_hi {
            let sl = &slices[i];
            for idx in sl.start..sl.end {
                let val = cfg.transform.apply(intensity[idx]);
                // Require a valid m/z here so invalid peaks can't consume a
                // top-N slot and then get discarded in the splat.
                if val > 0.0 && mz[idx] > 0.0 {
                    wpeaks.push((mz[idx], val));
                }
            }
            i += 1;
        }
        // Optional top-N peak picking by transformed intensity.
        if let Some(n) = cfg.top_n {
            if wpeaks.len() > n {
                wpeaks.select_nth_unstable_by(n, |a, b| b.1.total_cmp(&a.1));
                wpeaks.truncate(n);
            }
        }
        let mut acc: HashMap<i64, f64> = HashMap::new();
        for &(m, v) in &wpeaks {
            cfg.feature_map.splat_into(m, v, &mut acc);
        }

        let features = finalize(acc);
        if features.is_empty() {
            continue;
        }
        out.push(FrameUnit {
            meta: UnitMeta {
                frame_id: frame.frame_id,
                center_scan: center.scan,
                retention_time: rt,
                mobility: mobility[center.start],
                ms_type: frame.ms_type.clone(),
            },
            features,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::lsh::simhash::{CosineSimHash, Projection};
    use crate::algorithm::lsh::LshScheme;
    use crate::timstof::frame::ImsFrame;

    fn map() -> MzFeatureMap {
        MzFeatureMap::new(5.0, 15.0).unwrap()
    }

    /// Build a frame from parallel per-peak arrays (tof unused by the driver).
    fn make_frame(
        frame_id: i32,
        ms_type: MsType,
        rt: f64,
        scan: Vec<i32>,
        mz: Vec<f64>,
        intensity: Vec<f64>,
        mobility: Vec<f64>,
    ) -> TimsFrame {
        let n = scan.len();
        TimsFrame {
            frame_id,
            ms_type,
            scan,
            tof: vec![0; n],
            ims_frame: ImsFrame::new(rt, mobility, mz, intensity),
        }
    }

    #[test]
    fn intensity_transform_apply() {
        assert_eq!(IntensityTransform::None.apply(4.0), 4.0);
        assert_eq!(IntensityTransform::Sqrt.apply(4.0), 2.0);
        assert!((IntensityTransform::Log1p.apply(std::f64::consts::E - 1.0) - 1.0).abs() < 1e-12);
        assert_eq!(IntensityTransform::Sqrt.apply(-3.0), 0.0); // clamped
    }

    #[test]
    fn feature_map_validates_and_normalizes() {
        assert!(MzFeatureMap::new(0.0, 10.0).is_err());
        assert!(MzFeatureMap::new(20.0, 10.0).is_err()); // tol < bin
        let m = map();
        let f = m.features(&[(500.0, 100.0), (900.0, 50.0)], IntensityTransform::None);
        assert!(!f.is_empty());
        // L2 norm ~ 1.
        let norm: f64 = f.iter().map(|&(_, v)| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        // Sorted by feature id.
        assert!(f.windows(2).all(|w| w[0].0 <= w[1].0));
    }

    #[test]
    fn log_ppm_spacing_is_one_bin_per_bin_ppm() {
        let m = map(); // bin_ppm = 5
        let a = m.coord(500.0);
        let b = m.coord(500.0 * (1.0 + 5.0e-6)); // +5 ppm ≈ +1 bin
        assert!((b - a - 1.0).abs() < 1e-3);
    }

    #[test]
    fn splat_degrades_gracefully_with_mass_shift() {
        // The triangular splat makes cosine fall off *smoothly* with mass
        // drift (no bin-edge cliff), then collapse once peaks leave the kernel.
        // Note: with a half-width-3 triangular kernel (tol/bin = 3), a
        // half-tolerance shift already costs ~28% cosine — "tolerance" is soft;
        // the kernel shape/width is a Phase-1.5 knob.
        let m = map(); // bin_ppm 5, tol_ppm 15
        let h = CosineSimHash::new(1, 4, 8, Projection::Gaussian).unwrap();
        let peaks = [(200.0, 1.0), (500.0, 1.0), (900.0, 1.0)];
        let a = m.features(&peaks, IntensityTransform::None);

        let cos = |shift_ppm: f64| {
            let shifted: Vec<(f64, f64)> =
                peaks.iter().map(|&(mz, i)| (mz * (1.0 + shift_ppm * 1e-6), i)).collect();
            h.verify(&a, &m.features(&shifted, IntensityTransform::None))
        };
        let c0 = cos(0.0);
        let c_small = cos(2.5); // ~tol/6
        let c_half = cos(7.5); // tol/2
        let c_far = cos(150.0); // 10·tol → disjoint bins

        assert!(c0 > 0.999, "no-shift cosine {c0}");
        assert!(c_small > 0.9, "small-shift cosine {c_small}");
        assert!(c_half > 0.6, "half-tol cosine {c_half}");
        assert!(c_far < 0.2, "far-shift cosine {c_far}");
        // Monotone graceful degradation.
        assert!(c0 >= c_small && c_small > c_half && c_half > c_far);
    }

    #[test]
    fn frame_to_units_basic() {
        // Three occupied scans, raw-scan units (w = 0).
        let frame = make_frame(
            7,
            MsType::FragmentDia,
            123.0,
            vec![0, 0, 1, 1, 2],
            vec![100.0, 200.0, 100.0, 300.0, 150.0],
            vec![10.0, 10.0, 10.0, 10.0, 10.0],
            vec![1.00, 1.00, 1.05, 1.05, 1.10],
        );
        let cfg = WindowConfig {
            half_width: 0,
            stride: 1,
            transform: IntensityTransform::None,
            feature_map: map(),
            top_n: None,
        };
        let units = frame_to_units(&frame, &cfg, &[]);
        assert_eq!(units.len(), 3);
        assert_eq!(units.iter().map(|u| u.meta.center_scan).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(units[1].meta.mobility, 1.05);
        assert_eq!(units[0].meta.frame_id, 7);
        assert_eq!(units[0].meta.ms_type, MsType::FragmentDia);
        for u in &units {
            let norm: f64 =
                u.features.iter().map(|&(_, v)| (v as f64) * (v as f64)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn moving_average_gathers_co_occurring_fragments() {
        // Fragments A/B/C on adjacent scans; centering on scan 1 with w=1 must
        // union all three, w=0 only B.
        let frame = make_frame(
            1,
            MsType::FragmentDia,
            0.0,
            vec![0, 1, 2],
            vec![200.0, 300.0, 400.0],
            vec![10.0, 10.0, 10.0],
            vec![1.0, 1.05, 1.10],
        );
        let raw = WindowConfig {
            half_width: 0,
            stride: 1,
            transform: IntensityTransform::None,
            feature_map: map(),
            top_n: None,
        };
        let win = WindowConfig { half_width: 1, ..raw.clone() };

        let center_raw = frame_to_units(&frame, &raw, &[])
            .into_iter()
            .find(|u| u.meta.center_scan == 1)
            .unwrap();
        let center_win = frame_to_units(&frame, &win, &[])
            .into_iter()
            .find(|u| u.meta.center_scan == 1)
            .unwrap();

        assert!(center_win.features.len() > center_raw.features.len());
        // The windowed unit must contain bins near all three fragment m/z.
        let m = map();
        for mz in [200.0, 300.0, 400.0] {
            let c = m.coord(mz).round() as i64;
            assert!(
                center_win.features.iter().any(|&(id, _)| (id - c).abs() <= 4),
                "windowed unit missing fragment at m/z {mz}"
            );
        }
    }

    #[test]
    fn windows_do_not_cross_segment_boundaries() {
        // Segment A (scans 0,1) at m/z 100, segment B (scans 2,3) at m/z 1000.
        let frame = make_frame(
            1,
            MsType::FragmentDia,
            0.0,
            vec![0, 1, 2, 3],
            vec![100.0, 100.0, 1000.0, 1000.0],
            vec![10.0, 10.0, 10.0, 10.0],
            vec![1.0, 1.05, 1.10, 1.15],
        );
        let cfg = WindowConfig {
            half_width: 10, // large enough to span the whole frame if unclamped
            stride: 1,
            transform: IntensityTransform::None,
            feature_map: map(),
            top_n: None,
        };
        let m = map();
        let hi_region = m.coord(1000.0) - 1000.0; // comfortably below the 1000-m/z bins

        // Clamped: unit at scan 1 sees only segment A → no high-m/z features.
        let clamped = frame_to_units(&frame, &cfg, &[(0, 1), (2, 3)]);
        let u1 = clamped.iter().find(|u| u.meta.center_scan == 1).unwrap();
        assert!(
            u1.features.iter().all(|&(id, _)| (id as f64) < hi_region),
            "clamped unit leaked across the isolation-tile boundary"
        );

        // Unclamped: the same window pulls in the 1000-m/z peaks.
        let unclamped = frame_to_units(&frame, &cfg, &[]);
        let u1u = unclamped.iter().find(|u| u.meta.center_scan == 1).unwrap();
        assert!(
            u1u.features.iter().any(|&(id, _)| (id as f64) >= hi_region),
            "unclamped window should have reached the high-m/z segment"
        );
    }

    #[test]
    fn top_n_picks_strongest_peaks() {
        // One scan with 5 peaks of increasing intensity; top_n=2 must keep only
        // the two strongest (highest m/z here) → far fewer features than all-5.
        let frame = make_frame(
            1,
            MsType::FragmentDia,
            0.0,
            vec![0, 0, 0, 0, 0],
            vec![200.0, 400.0, 600.0, 800.0, 1000.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0; 5],
        );
        let base = WindowConfig {
            half_width: 0,
            stride: 1,
            transform: IntensityTransform::None,
            feature_map: map(),
            top_n: None,
        };
        let all = frame_to_units(&frame, &base, &[]);
        let top2 = frame_to_units(&frame, &WindowConfig { top_n: Some(2), ..base.clone() }, &[]);
        assert!(top2[0].features.len() < all[0].features.len());
        // The two strongest peaks are at m/z 800 and 1000 → their bins survive.
        let m = map();
        for mz in [800.0, 1000.0] {
            let c = m.coord(mz).round() as i64;
            assert!(top2[0].features.iter().any(|&(id, _)| (id - c).abs() <= 8));
        }
        // The weakest (m/z 200) must be gone.
        let weak = m.coord(200.0).round() as i64;
        assert!(top2[0].features.iter().all(|&(id, _)| (id - weak).abs() > 8));
    }

    #[test]
    fn top_n_ignores_invalid_mz() {
        // A high-intensity peak at m/z 0 must not steal a top-N slot and drop a
        // valid weaker peak.
        let frame = make_frame(
            1,
            MsType::FragmentDia,
            0.0,
            vec![0, 0, 0],
            vec![0.0, 500.0, 900.0], // m/z 0 is invalid but strongest
            vec![100.0, 5.0, 4.0],
            vec![1.0; 3],
        );
        let cfg = WindowConfig {
            half_width: 0,
            stride: 1,
            transform: IntensityTransform::None,
            feature_map: map(),
            top_n: Some(2),
        };
        let units = frame_to_units(&frame, &cfg, &[]);
        let m = map();
        for mz in [500.0, 900.0] {
            let c = m.coord(mz).round() as i64;
            assert!(
                units[0].features.iter().any(|&(id, _)| (id - c).abs() <= 8),
                "valid peak at m/z {mz} was displaced by the invalid one"
            );
        }
    }

    #[test]
    fn negative_scans_are_skipped() {
        // A sentinel scan of -1 must not produce a unit, and its peak must not
        // leak into any window (parity with rustdf's build_scan_slices).
        let frame = make_frame(
            1,
            MsType::FragmentDia,
            0.0,
            vec![-1, 0, 0, 1],
            vec![999.0, 100.0, 200.0, 300.0],
            vec![10.0, 10.0, 10.0, 10.0],
            vec![0.0, 1.0, 1.0, 1.05],
        );
        let cfg = WindowConfig {
            half_width: 5, // large window would otherwise reach the -1 peak
            stride: 1,
            transform: IntensityTransform::None,
            feature_map: map(),
            top_n: None,
        };
        let units = frame_to_units(&frame, &cfg, &[]);
        assert!(units.iter().all(|u| u.meta.center_scan >= 0));
        let bad = map().coord(999.0).round() as i64;
        assert!(units
            .iter()
            .all(|u| u.features.iter().all(|&(id, _)| (id - bad).abs() > 4)));
    }

    #[test]
    fn end_to_end_identical_bags_hash_identically() {
        let frame = make_frame(
            1,
            MsType::FragmentDia,
            0.0,
            vec![0, 1],
            vec![250.0, 700.0],
            vec![5.0, 8.0],
            vec![1.0, 1.05],
        );
        let cfg = WindowConfig {
            half_width: 0,
            stride: 1,
            transform: IntensityTransform::Sqrt,
            feature_map: map(),
            top_n: None,
        };
        let units = frame_to_units(&frame, &cfg, &[]);
        let h = CosineSimHash::new(0xABCD, 32, 12, Projection::Gaussian).unwrap();
        // Same feature vector hashed twice is identical.
        let sig = h.signature(&units[0].features);
        assert_eq!(sig, h.signature(&units[0].features));
    }
}
