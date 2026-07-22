//! The MS2 (DIA fragment) emission + its **independent** conservation oracle — built before the
//! production sweep wires MS2 into the projector, exactly as the MS1 oracle was.
//!
//! An ion deposits its fragment spectrum into every MS2 frame whose window group transmits its
//! *precursor* m/z at that scan (the diagonal), scaled by `abundance × elution × mobility ×
//! transmission`. Fragments are gated by the **parent**, never re-gated at their own m/z.
//!
//! The oracle discipline (Codex): the reference must share *nothing* with the production path except
//! pure physics it can't get wrong. So [`ms2_render`] gates with mscore's [`TimsTransmissionDIA`]
//! (frame-major), while [`ms2_reference`] is **ion-major**, rebuilds the frame→group map from the
//! schedule pattern, rebuilds the `(group, scan)→window` map from the raw windows, and recomputes the
//! transmission with its OWN logistic — replicating mscore's exact edge convention (soft sigmoids
//! centred at `midpoint ± (half_window + 0.25)`, `k = 15`). If the two agree at every bin, the
//! emission loop, the diagonal lookup, AND the transmission maths are all corroborated independently.

use crate::dia::DiaSchedule;
use crate::render::{gauss_frac, Geometry};
use mscore::timstof::quadrupole::IonTransmission;
use std::collections::{BTreeMap, HashMap};

/// The soft-edge steepness the DIA transmission uses by default (mscore `TimsTransmissionDIA` `k`).
const K: f64 = 15.0;
/// Transmission below this is treated as zero — the logistic never reaches 0, so a cutoff is required
/// (Codex). BOTH paths use the same `EPS`, so it never causes a disagreement.
const EPS: f64 = 1e-3;

/// One ion to render: its elution/mobility placement, abundance, the **precursor** m/z the quad gates
/// on, and both already-projected spectra as `(tof, relative intensity)` — `ms1_peaks` (precursor
/// isotopes, deposited into MS1 frames) and `ms2_peaks` (fragments, into transmission-gated MS2 frames).
#[derive(Clone, Debug)]
pub struct DiaIon {
    pub apex_frame: f64,
    pub scan_center: f64,
    pub abundance: f64,
    pub precursor_mz: f64,
    pub ms1_peaks: Vec<(u32, f32)>,
    pub ms2_peaks: Vec<(u32, f32)>,
    /// Fraction of the precursor that survives fragmentation *intact* (incomplete fragmentation). When
    /// `> 0`, that fraction of the precursor's isotope envelope is deposited into the MS2 windows too,
    /// through the same quad diagonal — modelling residual precursor bleed-through. `0.0` (default) is
    /// full fragmentation. Mirrors v1's `precursor_survival`.
    pub survival: f64,
}

/// Elution frame window `[start, end]` (1-based, to match the DIA schedule / `Frames.Id`). Public so the
/// chunked streaming render can decide which ions are active in a given frame range without duplicating
/// the window maths.
pub fn active_frames(apex: f64, g: &Geometry) -> (u32, u32) {
    let half = g.n_sigma * g.sigma_frames;
    let start = (apex - half).max(1.0) as u32;
    let end = ((apex + half) as u32).min(g.n_frames);
    (start, end)
}

fn scan_window(scan_center: f64, g: &Geometry) -> (u32, u32) {
    let lo = (scan_center - g.n_sigma * g.sigma_scans).max(0.0) as u32;
    let hi = ((scan_center + g.n_sigma * g.sigma_scans) as u32).min(g.n_scans - 1);
    (lo, hi)
}

/// PRODUCTION MS2 emission: frame-major, gated by mscore's `TimsTransmissionDIA`. Only MS2 frames emit
/// fragments; the gate returns 0 for a precursor outside the frame's window at that scan.
pub fn ms2_render(ions: &[DiaIon], sched: &DiaSchedule, g: &Geometry) -> BTreeMap<(u32, u32, u32), f64> {
    let mut out: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
    for frame in 1..=g.n_frames {
        if sched.ms_level(frame) != 2 {
            continue; // MS1 frames carry no fragments
        }
        let f = frame as f64;
        for ion in ions {
            let (fs, fe) = active_frames(ion.apex_frame, g);
            if frame < fs || frame > fe {
                continue;
            }
            let ew = gauss_frac(f - 0.5, f + 0.5, ion.apex_frame, g.sigma_frames);
            if ew <= 0.0 {
                continue;
            }
            let (s_lo, s_hi) = scan_window(ion.scan_center, g);
            for scan in s_lo..=s_hi {
                let t = sched
                    .transmission
                    .apply_transmission(frame as i32, scan as i32, &vec![ion.precursor_mz])[0];
                if t <= EPS {
                    continue;
                }
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, ion.scan_center, g.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                let base = ion.abundance * ew * mw * t;
                for &(tof, iv) in &ion.ms2_peaks {
                    let v = base * iv as f64;
                    if v <= 0.0 {
                        continue;
                    }
                    *out.entry((frame, scan, tof)).or_insert(0.0) += v;
                }
            }
        }
    }
    out
}

/// mscore's exact quad transfer for one window, reimplemented independently: two sigmoids `k` steep,
/// centred `0.25 Th` outside each window edge, differenced. Matches `apply_transmission(mid, width, k)`.
fn logistic(mid: f64, width: f64, mz: f64) -> f64 {
    let hw = width / 2.0;
    let up_mid = mid - hw - 0.25;
    let down_mid = mid + hw + 0.25;
    let up = 1.0 / (1.0 + (-K * (mz - up_mid)).exp());
    let down = 1.0 / (1.0 + (-K * (mz - down_mid)).exp());
    up - down
}

/// Independent `(window_group, scan) -> (isolation m/z, width)`, rebuilt from the raw windows with the
/// same inclusive expansion and last-write-wins order mscore uses — so overlapping per-scan windows
/// resolve identically without touching `TimsTransmissionDIA`.
fn independent_window_map(sched: &DiaSchedule) -> HashMap<(u32, u32), (f64, f64)> {
    let mut m = HashMap::new();
    for w in &sched.windows {
        for scan in w.scan_num_begin..=w.scan_num_end {
            m.insert((w.window_group, scan), (w.isolation_mz, w.isolation_width));
        }
    }
    m
}

/// INDEPENDENT reference MS2 emission: ion-major, own frame→group (from the pattern), own window map,
/// own logistic. Shares only `gauss_frac`.
pub fn ms2_reference(ions: &[DiaIon], sched: &DiaSchedule, g: &Geometry) -> BTreeMap<(u32, u32, u32), f64> {
    let wmap = independent_window_map(sched);
    let mut out: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
    for ion in ions {
        let (fs, fe) = active_frames(ion.apex_frame, g);
        for frame in fs..=fe {
            let Some(group) = sched.pattern[((frame - 1) % sched.cycle_len) as usize] else {
                continue; // MS1 frame — no fragments
            };
            let f = frame as f64;
            let ew = gauss_frac(f - 0.5, f + 0.5, ion.apex_frame, g.sigma_frames);
            if ew <= 0.0 {
                continue;
            }
            let (s_lo, s_hi) = scan_window(ion.scan_center, g);
            for scan in s_lo..=s_hi {
                let Some(&(iso_mz, iso_w)) = wmap.get(&(group, scan)) else {
                    continue; // no window at this (group, scan) — mscore blocks it too
                };
                let t = logistic(iso_mz, iso_w, ion.precursor_mz);
                if t <= EPS {
                    continue;
                }
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, ion.scan_center, g.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                let base = ion.abundance * ew * mw * t;
                for &(tof, iv) in &ion.ms2_peaks {
                    let v = base * iv as f64;
                    if v <= 0.0 {
                        continue;
                    }
                    *out.entry((frame, scan, tof)).or_insert(0.0) += v;
                }
            }
        }
    }
    out
}

/// The full DIA render: a single 1-based frame-major pass in which MS1 frames receive each ion's
/// precursor isotopes (ungated) and MS2 frames receive its fragments gated by the diagonal
/// transmission. Calls `emit(frame, ms_ms_type, &triples)` once per non-empty frame — the same
/// `MsMsType` (0 / 9) the writer stamps. The MS2 half is exactly the emission the oracle pins; the MS1
/// half is the same isotope placement the MS1 sweep uses, restricted to MS1 frames.
///
/// Active-set sweep-line (the deferred "active-set-optimised" speed-up; the frame-major version rescanned
/// every ion for every frame — O(frames × ions), which dominated render time at scale). Ions are entered
/// in `frame_start` order against a moving frame cursor and dropped once past `frame_end`, so each frame
/// touches only its active ions: ~O(frames + Σ active). The output is identical to the frame-major
/// version — the same ions are active at each frame, and each frame's active set is visited in original
/// ion-index order, so the per-frame deposit sequence (summed downstream, order-independent anyway) is
/// unchanged.
pub fn dia_render<F: FnMut(u32, u8, &[(u32, u32, f64)])>(
    ions: &[DiaIon],
    sched: &DiaSchedule,
    g: &Geometry,
    emit: F,
) {
    dia_render_range(ions, sched, g, 1, g.n_frames, emit);
}

/// The sweep restricted to frames `[frame_lo, frame_hi]` — the primitive the chunked streaming render
/// calls once per apex-bucket, so only that bucket's ions are ever resident. `ions` must already contain
/// every ion active anywhere in the range (frame_start ≤ frame_hi and frame_end ≥ frame_lo); ions whose
/// window opened before `frame_lo` are correctly entered at `frame_lo`. Emits only frames in the range,
/// so a caller stitching ranges together writes each frame exactly once.
pub fn dia_render_range<F: FnMut(u32, u8, &[(u32, u32, f64)])>(
    ions: &[DiaIon],
    sched: &DiaSchedule,
    g: &Geometry,
    frame_lo: u32,
    frame_hi: u32,
    mut emit: F,
) {
    let n = ions.len();
    let windows: Vec<(u32, u32)> = ions.iter().map(|io| active_frames(io.apex_frame, g)).collect();
    // Enter in frame_start order; sort indices, not the ions (peaks stay put).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| windows[i].0);
    let mut cursor = 0usize;
    let mut active: Vec<usize> = Vec::new();

    for frame in frame_lo..=frame_hi {
        // Enter ions whose window has opened.
        while cursor < n && windows[order[cursor]].0 <= frame {
            active.push(order[cursor]);
            cursor += 1;
        }
        // Leave ions whose window has closed.
        active.retain(|&i| windows[i].1 >= frame);
        if active.is_empty() {
            continue;
        }
        active.sort_unstable(); // visit in original-index order -> deterministic, frame-major-identical
        let ms_level = sched.ms_level(frame);
        let f = frame as f64;
        let mut buf: Vec<(u32, u32, f64)> = Vec::new();
        for &idx in &active {
            let ion = &ions[idx];
            let ew = gauss_frac(f - 0.5, f + 0.5, ion.apex_frame, g.sigma_frames);
            if ew <= 0.0 {
                continue;
            }
            let (s_lo, s_hi) = scan_window(ion.scan_center, g);
            for scan in s_lo..=s_hi {
                // MS1 frames pass everything; MS2 frames gate the precursor through the quad diagonal.
                let t = if ms_level == 1 {
                    1.0
                } else {
                    let tt = sched.transmission.apply_transmission(frame as i32, scan as i32, &vec![ion.precursor_mz])[0];
                    if tt <= EPS {
                        continue;
                    }
                    tt
                };
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, ion.scan_center, g.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                let base = ion.abundance * ew * mw * t;
                let mut deposit = |peaks: &[(u32, f32)], gain: f64| {
                    for &(tof, iv) in peaks {
                        let v = base * gain * iv as f64;
                        if v > 0.0 {
                            buf.push((scan, tof, v));
                        }
                    }
                };
                if ms_level == 1 {
                    // MS1 is a separate survey scan: it always shows the whole intact precursor,
                    // regardless of what the same ions do downstream in MS2. Not part of the partition.
                    deposit(&ion.ms1_peaks, 1.0);
                } else {
                    // Incomplete fragmentation PARTITIONS the selected precursor current: a `survival`
                    // fraction passes the quad intact (deposited at the precursor isotopes) and the
                    // remaining `1 - survival` fragments. The two must sum to the whole (conservation) —
                    // emitting full fragments + survival would over-count (survival=1 → 200%). Same gate
                    // `t` for both: the quad selects the parent before fragmentation-or-survival.
                    deposit(&ion.ms2_peaks, 1.0 - ion.survival);
                    if ion.survival > 0.0 {
                        deposit(&ion.ms1_peaks, ion.survival);
                    }
                }
            }
        }
        if !buf.is_empty() {
            emit(frame, if ms_level == 1 { 0 } else { 9 }, &buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::cube_diff;
    use ms_io::data::meta::{DiaMsMisInfo, DiaMsMsWindow};

    fn geom() -> Geometry {
        Geometry { n_frames: 9, n_scans: 3, sigma_frames: 2.0, sigma_scans: 1.0, n_sigma: 2.0 }
    }

    /// 3-frame cycle [MS1, g1(iso 500±5), g2(iso 600±5)], windows over scans 0..=2.
    fn schedule() -> DiaSchedule {
        let info = vec![
            DiaMsMisInfo { frame_id: 2, window_group: 1 },
            DiaMsMisInfo { frame_id: 3, window_group: 2 },
            DiaMsMisInfo { frame_id: 5, window_group: 1 },
            DiaMsMisInfo { frame_id: 6, window_group: 2 },
        ];
        let win = |g: u32, mz: f64| DiaMsMsWindow {
            window_group: g, scan_num_begin: 0, scan_num_end: 2, isolation_mz: mz, isolation_width: 10.0, collision_energy: 25.0,
        };
        DiaSchedule::build(&info, vec![win(1, 500.0), win(2, 600.0)], 9, 3).unwrap()
    }

    fn ion(precursor_mz: f64) -> DiaIon {
        DiaIon {
            apex_frame: 5.0,
            scan_center: 1.0,
            abundance: 100.0,
            precursor_mz,
            ms1_peaks: vec![],
            ms2_peaks: vec![(300, 1.0), (450, 0.4)],
            survival: 0.0,
        }
    }

    /// The load-bearing test: production (mscore-gated, frame-major) and the independent reference
    /// (own logistic, ion-major) agree at every bin.
    #[test]
    fn production_matches_independent_reference_every_bin() {
        let (s, g) = (schedule(), geom());
        // Include a precursor AT the g1 window edge (505 = midpoint+half_window), where transmission
        // is partial (~0.5) and the exact soft-edge convention matters — otherwise the every-bin test
        // can't distinguish edge conventions (verified: mutating the 0.25 offset then fails here).
        let ions = vec![ion(500.0), ion(600.0), ion(505.0)];
        let prod = ms2_render(&ions, &s, &g);
        let refr = ms2_reference(&ions, &s, &g);
        assert!(!prod.is_empty(), "fixture emitted nothing — test would be vacuous");
        let (worst, only_p, only_r) = cube_diff(&prod, &refr);
        assert_eq!((only_p, only_r), (0, 0), "bin set differs ({only_p}, {only_r})");
        assert!(worst < 1e-12, "worst per-bin diff {worst:.3e}");
    }

    /// A precursor outside every window at its scan emits ZERO MS2 signal.
    #[test]
    fn precursor_outside_all_windows_emits_nothing() {
        let (s, g) = (schedule(), geom());
        let prod = ms2_render(&[ion(1000.0)], &s, &g); // 1000 is far from both 500 and 600
        assert!(prod.is_empty(), "an untransmitted precursor produced {} bins", prod.len());
    }

    /// Duplicating an ion exactly doubles its fragment bins (linearity / no cross-ion interference).
    #[test]
    fn duplicating_an_ion_doubles_its_bins() {
        let (s, g) = (schedule(), geom());
        let one = ms2_render(&[ion(500.0)], &s, &g);
        let two = ms2_render(&[ion(500.0), ion(500.0)], &s, &g);
        assert_eq!(one.keys().collect::<Vec<_>>(), two.keys().collect::<Vec<_>>(), "bin set changed");
        for (k, &v1) in &one {
            assert!((two[k] - 2.0 * v1).abs() < 1e-9, "bin {k:?} not doubled");
        }
    }

    /// Fragments are gated by the PRECURSOR, never their own m/z: moving a fragment's tof changes only
    /// where its peaks land, never WHICH (frame, scan) cells emit.
    #[test]
    fn fragments_are_not_re_gated_at_their_own_mz() {
        let (s, g) = (schedule(), geom());
        let mut moved = ion(500.0);
        moved.ms2_peaks = vec![(99_999, 1.0), (1, 0.4)]; // wildly different tof (=m/z)
        let base_cells: std::collections::BTreeSet<(u32, u32)> =
            ms2_render(&[ion(500.0)], &s, &g).keys().map(|&(f, sc, _)| (f, sc)).collect();
        let moved_cells: std::collections::BTreeSet<(u32, u32)> =
            ms2_render(&[moved], &s, &g).keys().map(|&(f, sc, _)| (f, sc)).collect();
        assert_eq!(base_cells, moved_cells, "fragment m/z changed the emitting (frame, scan) set");
    }

    /// `dia_render`'s MS2 half must reproduce the oracle-pinned `ms2_render` exactly, and its MS1
    /// frames must carry the precursor isotopes (never fragments).
    #[test]
    fn dia_render_ms2_matches_the_oracle_and_ms1_carries_isotopes() {
        let (s, g) = (schedule(), geom());
        let mut ions = vec![ion(500.0), ion(600.0), ion(505.0)];
        for io in &mut ions {
            io.ms1_peaks = vec![(700, 1.0), (704, 0.5)]; // precursor isotopes at some tof
        }
        let mut ms2_cube: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
        let mut ms1_frames = 0u32;
        dia_render(&ions, &s, &g, |frame, ms_type, tri| {
            if ms_type == 9 {
                for &(scan, tof, v) in tri {
                    *ms2_cube.entry((frame, scan, tof)).or_insert(0.0) += v;
                }
            } else {
                ms1_frames += 1;
                // MS1 frames carry only the precursor isotope tofs (700, 704), never fragment tofs.
                assert!(tri.iter().all(|&(_, tof, _)| tof == 700 || tof == 704), "MS1 frame has non-isotope tof");
            }
        });
        let oracle = ms2_render(&ions, &s, &g);
        let (worst, a, b) = cube_diff(&ms2_cube, &oracle);
        assert_eq!((a, b), (0, 0), "dia_render MS2 bins differ from ms2_render");
        assert!(worst < 1e-12, "dia_render MS2 values differ (worst {worst:.3e})");
        assert!(ms1_frames >= 1, "expected at least one MS1 frame with content");
    }

    /// Incomplete fragmentation PARTITIONS the MS2 current: fragments scale by `1 - survival` and the
    /// intact precursor by `survival`, so the surviving-precursor : fragment ratio is `s/(1-s)` (not `s`)
    /// and the two sum to the whole. `survival = 0` (default) leaves the MS2 frames fragment-only.
    #[test]
    fn survival_bleeds_precursor_into_ms2() {
        let (s, g) = (schedule(), geom());
        let mut base = ion(500.0); // group-1 precursor; ms2 tof 300 (iv 1.0), 450 (0.4)
        base.ms1_peaks = vec![(700, 1.0)]; // one precursor isotope at a tof distinct from any fragment

        // survival = 0: the precursor tof must never appear in an MS2 frame.
        let mut off = base.clone();
        off.survival = 0.0;
        dia_render(&[off], &s, &g, |_f, ms_type, tri| {
            if ms_type == 9 {
                assert!(tri.iter().all(|&(_, tof, _)| tof != 700), "survival=0 leaked precursor into MS2");
            }
        });

        // survival > 0: the precursor tof appears at survival × the co-located fragment (tof 300, iv 1.0).
        let mut on = base;
        on.survival = 0.3;
        let mut seen = false;
        dia_render(&[on], &s, &g, |_f, ms_type, tri| {
            if ms_type != 9 {
                return;
            }
            let mut by: HashMap<u32, (Option<f64>, Option<f64>)> = HashMap::new();
            for &(scan, tof, v) in tri {
                let e = by.entry(scan).or_insert((None, None));
                if tof == 300 { e.0 = Some(v); }
                if tof == 700 { e.1 = Some(v); }
            }
            for (_, (frag, prec)) in by {
                if let (Some(fr), Some(pr)) = (frag, prec) {
                    seen = true;
                    // s=0.3 -> precursor/fragment = 0.3/0.7; fragments carry (1-s), precursor s.
                    let want = 0.3 / 0.7;
                    assert!((pr / fr - want).abs() < 1e-9, "precursor:fragment should be s/(1-s)={want}; got {}", pr / fr);
                }
            }
        });
        assert!(seen, "survival>0 produced no precursor bleed-through in MS2");
    }

    /// A precursor in group 1's window emits only in g1 MS2 frames (2, 5, 8), not g2 frames — the
    /// diagonal selects exactly the transmitting group.
    #[test]
    fn precursor_emits_only_in_its_transmitting_group() {
        let (s, g) = (schedule(), geom());
        let prod = ms2_render(&[ion(500.0)], &s, &g);
        let frames: std::collections::BTreeSet<u32> = prod.keys().map(|&(f, _, _)| f).collect();
        for f in &frames {
            assert_eq!(s.window_group(*f), Some(1), "emitted in frame {f} which is not group 1");
        }
        assert!(frames.contains(&5), "expected emission in g1 frame 5");
    }
}
