//! The streaming frame render — sweep-line core, plus an **independent** reference render used only
//! to prove the sweep correct.
//!
//! The production path is [`stream_render`]: a 1-D temporal sweep that, for each frame, holds an
//! active set (min-heap on `frame_end`), accumulates every active ion's contribution into a sparse
//! per-frame `(scan, tof)` buffer, hands that buffer to a callback, and drops it. Its working set is
//! bounded by the elution window, not the run length (see `TIMSIM_V2_RENDER.md` §7).
//!
//! # Why a second, independent render lives here
//!
//! The load-bearing correctness claim is "the sweep emits each ion's mass **exactly once** — no
//! double-count across the frames it is active in, no leak, no off-by-one at a window edge." A
//! conservation check that reconstructs the expected mass from the *same* frame/scan partitioning and
//! index math the sweep uses is only a *consistency* check: a fault duplicated in both paths passes.
//!
//! So [`reference_render`] is written to share **nothing** with the sweep except the pure Gaussian
//! weight (which is physics, not indexing, and is unit-tested on its own): it is **ion-major** (the
//! sweep is frame-major), it discovers each ion's frame window by a direct `fs..=fe` loop (the sweep
//! discovers it through heap enter/leave against a moving frame cursor), it uses no active set, no
//! per-frame buffer, and no input sort. If the sweep's heap lifetime logic drops or duplicates a
//! frame, the two renders disagree at that bin — and the tests below compare **every bin**, not just
//! totals. The metamorphic tests (duplicate → exactly 2×, permute-order invariance, chunk-union
//! linearity) catch the bugs a single reference render can't.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

/// The acquisition geometry a render needs: the frame/scan grid and the peak widths (as Gaussian
/// sigmas in frame/scan units) with a truncation radius. This is the render-time image of the
/// portable `[0,1]` elution/mobility shapes — the gradient/ramp mapping happens upstream.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub n_frames: u32,
    pub n_scans: u32,
    pub sigma_frames: f64,
    pub sigma_scans: f64,
    /// Truncate each peak at this many sigma (the `target_p` analog).
    pub n_sigma: f64,
}

/// One ion to render: an elution apex (in frames), a mobility apex (in scans), a total abundance,
/// and the `(tof, relative-intensity)` peaks it deposits at that locus. MS1 isotope envelopes and
/// MS2 fragment lists both reduce to this shape — the render does not care which.
#[derive(Clone, Debug)]
pub struct Ion {
    pub apex_frame: f64,
    pub scan_center: f64,
    pub abundance: f64,
    pub peaks: Vec<(u32, f32)>,
}

/// erf via Abramowitz-Stegun 7.1.26 (max error ~1.5e-7). The two renders share this, so its error
/// cancels in their bin-for-bin comparison; its absolute accuracy is pinned by [`tests`] separately.
fn erf(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    s * y
}

/// Mass of a Gaussian(mean, sigma) between `a` and `b` — an exact CDF difference over the bin.
/// Pure physics; both renders call it identically.
#[inline]
pub fn gauss_frac(a: f64, b: f64, mean: f64, sigma: f64) -> f64 {
    let z = |x: f64| 0.5 * (1.0 + erf((x - mean) / (sigma * std::f64::consts::SQRT_2)));
    z(b) - z(a)
}

/// The frame window `[frame_start, frame_end]` an ion is active in — the **production** derivation
/// (truncate then clamp). [`reference_render`] deliberately recomputes this a different way so a bug
/// here cannot hide.
#[inline]
/// The bin range whose *centres* fall within ±`n_sigma·sigma` of the apex. NOTE: bins are selected by
/// centre, not by interval overlap, so for a very NARROW peak (`sigma_frames` ≲ 1 bin) this can select a
/// single bin that misses the bin actually holding most of the mass — the emitted fraction then depends
/// on `sigma`/`n_sigma` rather than being a fixed truncation. Harmless at the widths we run
/// (`sigma_frames`≈12, `sigma_scans`≈6, many bins across the peak) and it does NOT distort the
/// precursor↔fragment ratio (MS1 and MS2 share these weights), but if sub-bin peaks are ever needed,
/// select bins whose *intervals* overlap the support and renormalise. Same caveat in [`scan_window`].
fn active_window(apex_frame: f64, g: &Geometry) -> (u32, u32) {
    let half = g.n_sigma * g.sigma_frames;
    let start = (apex_frame - half).max(0.0) as u32;
    let end = ((apex_frame + half) as u32).min(g.n_frames - 1);
    (start, end)
}

/// Per-frame emission the callback sees: the frame index, the active-set size at that frame (for the
/// memory bound), and the sparse `(scan, tof) -> intensity` buffer. The buffer is borrowed and
/// dropped right after — the callback must not retain it if the streaming memory property is to hold.
pub struct FrameEmission<'a> {
    pub frame: u32,
    pub active: usize,
    pub buffer: &'a HashMap<(u32, u32), f64>,
}

/// The streaming sweep-line render. Calls `emit` once per frame that has any active ion, with that
/// frame's sparse buffer, then clears it. Working set stays bounded by the elution window.
///
/// This is the single production render path — the benchmark and [`sweep_render`] both drive it, so
/// the code the tests exercise is the code that runs.
pub fn stream_render<F: FnMut(FrameEmission)>(ions: &[Ion], g: &Geometry, emit: F) {
    stream_render_range(ions, g, 0, g.n_frames, emit)
}

/// [`stream_render`] restricted to the frame sub-range `[frame_lo, frame_hi)`. This is the unit of
/// **parallel render-by-chunk**: contiguous frame ranges are disjoint in their *output*, so K chunks
/// render on K cores and their emissions concatenate — no summing, no double-emit. A boundary ion
/// simply appears in the active set of both chunks it straddles (a read-only input), but each chunk
/// emits only its own frames. Correctness is pinned by `frame_range_partition_equals_whole`.
///
/// Starting the sweep at `frame_lo` with a fresh cursor is what rebuilds the active set correctly: the
/// first iteration pushes every ion with `frame_start <= frame_lo` and then pops those already expired
/// (`frame_end < frame_lo`), leaving exactly the ions alive at `frame_lo`. Give each chunk only the
/// ions overlapping its range (bucketed by the caller) so that initial push is cheap.
pub fn stream_render_range<F: FnMut(FrameEmission)>(
    ions: &[Ion],
    g: &Geometry,
    frame_lo: u32,
    frame_hi: u32,
    mut emit: F,
) {
    // A run must have at least one frame and one scan; otherwise `n_frames - 1` / `n_scans - 1`
    // underflow (panic in debug, a `u32::MAX` loop in release). The CLI also rejects zero, but the
    // library API guards itself.
    if g.n_frames == 0 || g.n_scans == 0 {
        return;
    }
    let windows: Vec<(u32, u32)> = ions.iter().map(|io| active_window(io.apex_frame, g)).collect();

    // Enter in frame_start order; sort indices, not the data (§3.6).
    let mut order: Vec<usize> = (0..ions.len()).collect();
    order.sort_unstable_by_key(|&i| windows[i].0);

    let mut active: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    let mut cursor = 0usize;
    let mut buf: HashMap<(u32, u32), f64> = HashMap::new();

    for frame in frame_lo..frame_hi {
        while cursor < order.len() && windows[order[cursor]].0 <= frame {
            let idx = order[cursor];
            active.push(Reverse((windows[idx].1, idx)));
            cursor += 1;
        }
        while let Some(&Reverse((fe, _))) = active.peek() {
            if fe < frame {
                active.pop();
            } else {
                break;
            }
        }
        if active.is_empty() {
            continue;
        }

        let f = frame as f64;
        for &Reverse((_, idx)) in active.iter() {
            let io = &ions[idx];
            let ew = gauss_frac(f - 0.5, f + 0.5, io.apex_frame, g.sigma_frames);
            if ew <= 0.0 {
                continue;
            }
            let s_lo = (io.scan_center - g.n_sigma * g.sigma_scans).max(0.0) as u32;
            let s_hi = ((io.scan_center + g.n_sigma * g.sigma_scans) as u32).min(g.n_scans - 1);
            for scan in s_lo..=s_hi {
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, io.scan_center, g.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                let base = io.abundance * ew * mw;
                for &(tof, iv) in &io.peaks {
                    let val = base * iv as f64;
                    if val <= 0.0 {
                        continue;
                    }
                    *buf.entry((scan, tof)).or_insert(0.0) += val;
                }
            }
        }

        emit(FrameEmission { frame, active: active.len(), buffer: &buf });
        buf.clear();
    }
}

/// Per-frame emission for the **flat** accumulator: a `(scan, tof, value)` list that may contain
/// duplicate `(scan, tof)` keys (co-eluting ions are appended, not summed on the fly). The consumer
/// dedups — which the real TDF block encoder does anyway — so this trades accumulate-time hashing for
/// a single dedup at encode time.
pub struct FlatEmission<'a> {
    pub frame: u32,
    pub active: usize,
    pub triples: &'a [(u32, u32, f64)],
}

/// Like [`stream_render`], but accumulates into a flat `Vec<(scan, tof, value)>` (append, no hashing)
/// instead of a per-frame `HashMap`. Bin-identical after dedup (proved by [`tests`]). Whether this or
/// the HashMap path is faster end-to-end is exactly what the throughput benchmark measures.
pub fn stream_render_flat<F: FnMut(FlatEmission)>(ions: &[Ion], g: &Geometry, emit: F) {
    stream_render_flat_range(ions, g, 0, g.n_frames, emit)
}

/// [`stream_render_flat`] restricted to `[frame_lo, frame_hi)` — the flat-accumulator unit of
/// parallel render-by-chunk. See [`stream_render_range`] for the partition correctness argument.
pub fn stream_render_flat_range<F: FnMut(FlatEmission)>(
    ions: &[Ion],
    g: &Geometry,
    frame_lo: u32,
    frame_hi: u32,
    mut emit: F,
) {
    if g.n_frames == 0 || g.n_scans == 0 {
        return;
    }
    let windows: Vec<(u32, u32)> = ions.iter().map(|io| active_window(io.apex_frame, g)).collect();
    let mut order: Vec<usize> = (0..ions.len()).collect();
    order.sort_unstable_by_key(|&i| windows[i].0);

    let mut active: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    let mut cursor = 0usize;
    let mut buf: Vec<(u32, u32, f64)> = Vec::new();

    for frame in frame_lo..frame_hi {
        while cursor < order.len() && windows[order[cursor]].0 <= frame {
            active.push(Reverse((windows[order[cursor]].1, order[cursor])));
            cursor += 1;
        }
        while let Some(&Reverse((fe, _))) = active.peek() {
            if fe < frame {
                active.pop();
            } else {
                break;
            }
        }
        if active.is_empty() {
            continue;
        }

        let f = frame as f64;
        for &Reverse((_, idx)) in active.iter() {
            let io = &ions[idx];
            let ew = gauss_frac(f - 0.5, f + 0.5, io.apex_frame, g.sigma_frames);
            if ew <= 0.0 {
                continue;
            }
            let s_lo = (io.scan_center - g.n_sigma * g.sigma_scans).max(0.0) as u32;
            let s_hi = ((io.scan_center + g.n_sigma * g.sigma_scans) as u32).min(g.n_scans - 1);
            for scan in s_lo..=s_hi {
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, io.scan_center, g.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                let base = io.abundance * ew * mw;
                for &(tof, iv) in &io.peaks {
                    let val = base * iv as f64;
                    if val <= 0.0 {
                        continue;
                    }
                    buf.push((scan, tof, val));
                }
            }
        }

        emit(FlatEmission { frame, active: active.len(), triples: &buf });
        buf.clear();
    }
}

/// Drive [`stream_render`] and materialise the whole `(frame, scan, tof) -> intensity` cube. For
/// **tests only** — this defeats the streaming memory property on purpose, so the output can be
/// compared bin-for-bin against [`reference_render`].
pub fn sweep_render(ions: &[Ion], g: &Geometry) -> BTreeMap<(u32, u32, u32), f64> {
    let mut out = BTreeMap::new();
    stream_render(ions, g, |e| {
        for (&(scan, tof), &v) in e.buffer {
            // A (scan, tof) key is unique within a frame, so no cross-frame collision here.
            out.insert((e.frame, scan, tof), v);
        }
    });
    out
}

/// The **independent** reference render (see module docs). Ion-major, no heap, no per-frame buffer,
/// no input sort; the frame window is a direct `fs..=fe` loop with its bounds recomputed via a
/// different expression than [`active_window`]. Shares only [`gauss_frac`]. Used solely to prove the
/// sweep — never on a real run.
pub fn reference_render(ions: &[Ion], g: &Geometry) -> BTreeMap<(u32, u32, u32), f64> {
    let mut out: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
    let fhalf = g.n_sigma * g.sigma_frames;
    let shalf = g.n_sigma * g.sigma_scans;
    let last_frame = (g.n_frames - 1) as f64;
    let last_scan = (g.n_scans - 1) as f64;

    for io in ions {
        // Independent window derivation: clamp on the reals, then floor — a different code path from
        // active_window's truncate-then-clamp. For non-negative values the two agree, so any
        // divergence signals a real off-by-one in one of them, which is exactly what we want to see.
        let fs = (io.apex_frame - fhalf).max(0.0).floor() as u32;
        let fe = (io.apex_frame + fhalf).min(last_frame).floor() as u32;
        let ss = (io.scan_center - shalf).max(0.0).floor() as u32;
        let se = (io.scan_center + shalf).min(last_scan).floor() as u32;

        for frame in fs..=fe {
            let ew = gauss_frac(frame as f64 - 0.5, frame as f64 + 0.5, io.apex_frame, g.sigma_frames);
            if ew <= 0.0 {
                continue;
            }
            for scan in ss..=se {
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, io.scan_center, g.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                for &(tof, iv) in &io.peaks {
                    let val = io.abundance * ew * mw * iv as f64;
                    if val <= 0.0 {
                        continue;
                    }
                    *out.entry((frame, scan, tof)).or_insert(0.0) += val;
                }
            }
        }
    }
    out
}

/// Partition `[0, n_frames)` into `k` contiguous, near-equal frame ranges — the chunks of a parallel
/// render-by-chunk. The remainder is spread across the first ranges so sizes differ by at most one.
pub fn frame_chunks(n_frames: u32, k: usize) -> Vec<(u32, u32)> {
    let k = (k.max(1) as u32).min(n_frames.max(1));
    let base = n_frames / k;
    let rem = n_frames % k;
    let mut out = Vec::with_capacity(k as usize);
    let mut lo = 0u32;
    for i in 0..k {
        let hi = lo + base + if i < rem { 1 } else { 0 };
        out.push((lo, hi));
        lo = hi;
    }
    out
}

/// For each chunk, the indices of ions whose active window overlaps that chunk's frame range. An ion
/// straddling a boundary lands in every chunk it touches (correct: each chunk emits only its own
/// frames, so the ion contributes to each without being double-emitted). Bucketing keeps each chunk's
/// sweep O(its ions), not O(all ions).
pub fn bucket_ions(ions: &[Ion], g: &Geometry, chunks: &[(u32, u32)]) -> Vec<Vec<usize>> {
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); chunks.len()];
    for (i, io) in ions.iter().enumerate() {
        let (ws, we) = active_window(io.apex_frame, g);
        for (c, &(lo, hi)) in chunks.iter().enumerate() {
            if ws < hi && we >= lo {
                buckets[c].push(i);
            }
        }
    }
    buckets
}

/// Worst relative difference between two render cubes, and the count of bins present in only one.
/// `(worst_rel, only_in_a, only_in_b)`. Rel diff uses the larger magnitude as denominator so a bin
/// that exists in one render but is ~0 in the other still registers.
pub fn cube_diff(
    a: &BTreeMap<(u32, u32, u32), f64>,
    b: &BTreeMap<(u32, u32, u32), f64>,
) -> (f64, usize, usize) {
    let mut worst = 0.0f64;
    let mut only_a = 0usize;
    for (k, &va) in a {
        match b.get(k) {
            Some(&vb) => {
                let d = (va - vb).abs();
                let denom = va.abs().max(vb.abs());
                if denom > 0.0 {
                    worst = worst.max(d / denom);
                }
            }
            None => only_a += 1,
        }
    }
    let only_b = b.keys().filter(|k| !a.contains_key(*k)).count();
    (worst, only_a, only_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> Geometry {
        Geometry { n_frames: 40, n_scans: 30, sigma_frames: 2.5, sigma_scans: 1.5, n_sigma: 3.0 }
    }

    /// A fixture that deliberately exercises the bug-prone cases Codex called out:
    ///  - an ion living across MANY frames (lifetime / enter-leave off-by-one),
    ///  - two co-eluting ions colliding into IDENTICAL (scan, tof) bins (accumulation),
    ///  - an ion pinned against the scan-0 boundary (window clamping),
    ///  - an ion pinned against the last frame (upper clamp),
    ///  - multi-peak envelopes (per-peak tof stepping).
    fn fixture() -> Vec<Ion> {
        vec![
            // long-lived, mid grid, two isotope peaks
            Ion { apex_frame: 20.0, scan_center: 15.0, abundance: 100.0, peaks: vec![(500, 1.0), (504, 0.4)] },
            // co-elutes with the next one at the SAME locus + SAME tof -> bins must add
            Ion { apex_frame: 20.3, scan_center: 15.0, abundance: 50.0, peaks: vec![(500, 1.0)] },
            Ion { apex_frame: 20.1, scan_center: 15.0, abundance: 30.0, peaks: vec![(500, 1.0)] },
            // scan-0 boundary (half the mobility peak is clamped away)
            Ion { apex_frame: 8.0, scan_center: 0.4, abundance: 70.0, peaks: vec![(300, 1.0)] },
            // last-frame boundary
            Ion { apex_frame: 39.2, scan_center: 22.0, abundance: 40.0, peaks: vec![(900, 1.0), (905, 0.2)] },
        ]
    }

    /// The load-bearing test: the streaming sweep and the independent ion-major reference must agree
    /// at EVERY (frame, scan, tof) bin, with no bin present in only one. Bit-for-bit up to float
    /// summation order (~1e-12).
    #[test]
    fn sweep_matches_independent_reference_every_bin() {
        let g = geom();
        let ions = fixture();
        let sweep = sweep_render(&ions, &g);
        let reference = reference_render(&ions, &g);

        assert!(!sweep.is_empty(), "fixture rendered nothing — the test would be vacuous");
        let (worst, only_sweep, only_ref) = cube_diff(&sweep, &reference);
        assert_eq!(only_sweep, 0, "{only_sweep} bins the sweep emitted are absent from the reference");
        assert_eq!(only_ref, 0, "{only_ref} bins the reference emitted are absent from the sweep");
        assert!(worst < 1e-12, "worst per-bin relative diff {worst:.3e} exceeds 1e-12");
    }

    /// Duplicating one ion must exactly double THAT ion's bins and change nothing else. Catches any
    /// cross-ion interference or double-counting in the accumulation.
    #[test]
    fn duplicating_one_ion_doubles_exactly_its_bins() {
        let g = geom();
        let ions = fixture();
        let base = sweep_render(&ions, &g);

        let mut dup = ions.clone();
        dup.push(ions[0].clone()); // duplicate the long-lived ion
        let with_dup = sweep_render(&dup, &g);

        // (with_dup - base) must equal exactly the render of ion 0 alone, every bin.
        let solo = sweep_render(&ions[0..1], &g);
        let mut delta: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
        for (k, &v) in &with_dup {
            delta.insert(*k, v - base.get(k).copied().unwrap_or(0.0));
        }
        // drop numerical zeros that fall out of subtraction
        delta.retain(|_, v| v.abs() > 1e-9);
        let (worst, only_delta, only_solo) = cube_diff(&delta, &solo);
        assert_eq!(only_delta, 0, "duplicate delta has {only_delta} bins the solo render lacks");
        assert_eq!(only_solo, 0, "solo render has {only_solo} bins the duplicate delta lacks");
        assert!(worst < 1e-9, "duplicate delta != solo render (worst {worst:.3e})");
    }

    /// The sweep sorts internally, so input order must not change a single bin. Catches any
    /// order-dependence in enter/leave.
    #[test]
    fn permuting_input_order_is_invariant() {
        let g = geom();
        let ions = fixture();
        let canonical = sweep_render(&ions, &g);

        let mut permuted = ions.clone();
        permuted.reverse();
        permuted.swap(0, 2);
        let out = sweep_render(&permuted, &g);

        let (worst, a, b) = cube_diff(&canonical, &out);
        assert_eq!((a, b), (0, 0), "input order changed the bin set ({a}, {b})");
        assert!(worst < 1e-12, "input order changed values (worst {worst:.3e})");
    }

    /// Rendering two arbitrary subsets and summing must equal rendering the whole set — the render is
    /// linear in the ion set, so overlapping chunks compose. Catches leakage between ions sharing a
    /// buffer bin.
    #[test]
    fn chunk_union_equals_whole() {
        let g = geom();
        let ions = fixture();
        let whole = sweep_render(&ions, &g);

        let a = sweep_render(&ions[0..2], &g);
        let b = sweep_render(&ions[2..], &g);
        let mut union: BTreeMap<(u32, u32, u32), f64> = a.clone();
        for (k, &v) in &b {
            *union.entry(*k).or_insert(0.0) += v;
        }

        let (worst, only_whole, only_union) = cube_diff(&whole, &union);
        assert_eq!((only_whole, only_union), (0, 0), "chunk union bin set differs");
        assert!(worst < 1e-12, "chunk union != whole render (worst {worst:.3e})");
    }

    /// Total emitted mass of a lone, interior ion equals abundance × (frame window mass) ×
    /// (scan window mass) × (Σ peak intensities), computed here from first principles with NO render
    /// code. Independent conservation, complementary to the bin-for-bin equality.
    #[test]
    fn lone_interior_ion_conserves_analytic_mass() {
        let g = geom();
        let ion = Ion { apex_frame: 20.0, scan_center: 15.0, abundance: 100.0, peaks: vec![(500, 1.0), (504, 0.4)] };
        let cube = sweep_render(std::slice::from_ref(&ion), &g);
        let emitted: f64 = cube.values().sum();

        let fhalf = g.n_sigma * g.sigma_frames;
        let shalf = g.n_sigma * g.sigma_scans;
        let (fs, fe) = ((ion.apex_frame - fhalf).floor() as i64, (ion.apex_frame + fhalf).floor() as i64);
        let (ss, se) = ((ion.scan_center - shalf).floor() as i64, (ion.scan_center + shalf).floor() as i64);
        let frame_mass: f64 = (fs..=fe).map(|f| gauss_frac(f as f64 - 0.5, f as f64 + 0.5, ion.apex_frame, g.sigma_frames)).sum();
        let scan_mass: f64 = (ss..=se).map(|s| gauss_frac(s as f64 - 0.5, s as f64 + 0.5, ion.scan_center, g.sigma_scans)).sum();
        let peak_sum: f64 = ion.peaks.iter().map(|&(_, iv)| iv as f64).sum();
        let expected = ion.abundance * frame_mass * scan_mass * peak_sum;

        let rel = (emitted - expected).abs() / expected;
        assert!(rel < 1e-9, "emitted {emitted:.6} vs analytic {expected:.6} (rel {rel:.3e})");
    }

    /// Parallel render-by-chunk correctness: rendering contiguous frame ranges (each given only its
    /// bucketed ions) and concatenating the emissions must reproduce the whole render exactly. This is
    /// the invariant the parallel sweep relies on — distinct from the ion-partition invariant
    /// (`chunk_union_equals_whole`), because here the partition is over OUTPUT FRAMES, not ions, so the
    /// pieces concatenate rather than sum. Boundary ions (bucketed into two chunks) must not
    /// double-emit.
    #[test]
    fn frame_range_partition_equals_whole() {
        let g = geom();
        let ions = fixture();
        let whole = sweep_render(&ions, &g);

        // Deliberately uneven chunk count so boundaries fall mid-peak.
        let chunks = frame_chunks(g.n_frames, 4);
        assert!(chunks.len() >= 2, "need multiple chunks to exercise boundaries");
        let buckets = bucket_ions(&ions, &g, &chunks);

        let mut parts: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
        for (&(lo, hi), bucket) in chunks.iter().zip(buckets.iter()) {
            let sub: Vec<Ion> = bucket.iter().map(|&i| ions[i].clone()).collect();
            stream_render_range(&sub, &g, lo, hi, |e| {
                for (&(scan, tof), &v) in e.buffer {
                    // Each frame is owned by exactly one chunk, so no cross-chunk key collision.
                    assert!(parts.insert((e.frame, scan, tof), v).is_none(), "frame emitted twice");
                }
            });
        }

        let (worst, only_whole, only_parts) = cube_diff(&whole, &parts);
        assert_eq!((only_whole, only_parts), (0, 0), "frame-partition bin set differs from whole");
        assert!(worst < 1e-12, "frame-partition render diverges from whole (worst {worst:.3e})");
    }

    /// The flat accumulator, once its duplicate `(scan, tof)` keys are summed, must reproduce the
    /// HashMap sweep's cube exactly — otherwise the two accumulators disagree and the throughput
    /// comparison would be between two different renders.
    #[test]
    fn flat_accumulator_dedups_to_the_same_cube() {
        let g = geom();
        let ions = fixture();
        let hashmap_cube = sweep_render(&ions, &g);

        let mut flat_cube: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
        stream_render_flat(&ions, &g, |e| {
            for &(scan, tof, v) in e.triples {
                *flat_cube.entry((e.frame, scan, tof)).or_insert(0.0) += v;
            }
        });

        let (worst, only_hm, only_flat) = cube_diff(&hashmap_cube, &flat_cube);
        assert_eq!((only_hm, only_flat), (0, 0), "flat vs hashmap bin set differs");
        assert!(worst < 1e-12, "flat accumulator diverges from hashmap (worst {worst:.3e})");
    }

    /// Pins the shared physics: a Gaussian's ±1σ mass ≈ 0.6827 and its mass over a wide window ≈ 1.
    /// This is the one thing both renders share, so it is verified on its own.
    #[test]
    fn gauss_frac_matches_known_values() {
        // ±1σ around 0 with sigma 1
        let one_sigma = gauss_frac(-1.0, 1.0, 0.0, 1.0);
        assert!((one_sigma - 0.6827).abs() < 1e-3, "±1σ mass {one_sigma}");
        // effectively the whole distribution
        let whole = gauss_frac(-20.0, 20.0, 0.0, 1.0);
        assert!((whole - 1.0).abs() < 1e-6, "full mass {whole}");
    }
}
