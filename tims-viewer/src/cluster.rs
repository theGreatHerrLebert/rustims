//! Grid-accelerated DBSCAN over the normalized point cloud, for coloring points by cluster
//! identity (the same analysis the proteolizard/imspy-vis Voila tool did, in the viewer).
//!
//! Points live in the normalized cube `[-1, 1]^3` (x=m/z, y=1/K0, z=RT), so a single `eps`
//! in cube units is a sensible neighborhood radius. A uniform spatial hash of cell size `eps`
//! makes each neighborhood query touch only the 27 adjacent cells, so the whole pass is about
//! O(n) for the moderate point counts the viewer clusters (it is gated to a cap by the caller).

use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Label assigned to noise / unclustered points.
pub const NOISE: i32 = -1;

/// Run DBSCAN on `points` (normalized cube positions). Returns a per-point label
/// (`NOISE` for noise, otherwise a 0-based contiguous cluster id) and the cluster count.
pub fn dbscan(points: &[[f32; 3]], eps: f32, min_pts: usize) -> (Vec<i32>, usize) {
    dbscan_with_progress(points, eps, min_pts, |_| {})
}

/// Like [`dbscan`], but invokes `progress(visited_so_far)` roughly every 1% of points (and once at
/// the end with `points.len()`). The visited count climbs monotonically — even inside one big
/// cluster's flood-fill — so it's a smooth progress proxy for a long run.
pub fn dbscan_with_progress(
    points: &[[f32; 3]],
    eps: f32,
    min_pts: usize,
    mut progress: impl FnMut(usize),
) -> (Vec<i32>, usize) {
    let n = points.len();
    if n == 0 || eps <= 0.0 {
        return (vec![NOISE; n], 0);
    }
    let inv = 1.0 / eps;
    let cell = |p: &[f32; 3]| -> (i32, i32, i32) {
        (
            (p[0] * inv).floor() as i32,
            (p[1] * inv).floor() as i32,
            (p[2] * inv).floor() as i32,
        )
    };
    // Spatial hash: cell -> point indices. FxHashMap (not SipHash) — the grid is hashed n times on
    // build and 27n times in neighbour queries, so the hash is a hot path, especially in wasm.
    let mut grid: FxHashMap<(i32, i32, i32), Vec<u32>> =
        FxHashMap::with_capacity_and_hasher(n / 4 + 1, Default::default());
    for (i, p) in points.iter().enumerate() {
        grid.entry(cell(p)).or_default().push(i as u32);
    }

    let eps2 = eps * eps;
    // Neighbors of point i within eps (inclusive), scanning the 27 surrounding cells.
    let neighbors = |i: usize, out: &mut Vec<u32>| {
        out.clear();
        let p = points[i];
        let (ci, cj, ck) = cell(&p);
        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    if let Some(v) = grid.get(&(ci + di, cj + dj, ck + dk)) {
                        for &j in v {
                            let q = points[j as usize];
                            let d = (p[0] - q[0]).powi(2)
                                + (p[1] - q[1]).powi(2)
                                + (p[2] - q[2]).powi(2);
                            if d <= eps2 {
                                out.push(j);
                            }
                        }
                    }
                }
            }
        }
    };

    let mut labels = vec![NOISE; n];
    let mut visited = vec![false; n];
    // `queued` dedups the BFS frontier: each point is enqueued at most once across the whole run, so
    // the queue stays O(n) instead of O(sum of neighbourhood sizes) — which, on dense MS peaks, would
    // otherwise balloon to billions of (mostly duplicate) entries.
    let mut queued = vec![false; n];
    let mut cid: i32 = 0;
    let mut nb: Vec<u32> = Vec::new();
    let mut nbj: Vec<u32> = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();

    // Throttled progress on the visited count (~1% granularity).
    let report_step = (n / 100).max(1);
    let mut visited_count = 0usize;
    let mut since_report = 0usize;
    let bump = |count: &mut usize, since: &mut usize, progress: &mut dyn FnMut(usize)| {
        *count += 1;
        *since += 1;
        if *since >= report_step {
            *since = 0;
            progress(*count);
        }
    };

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        bump(&mut visited_count, &mut since_report, &mut progress);
        neighbors(i, &mut nb);
        if nb.len() < min_pts {
            continue; // noise (may still be claimed as a border point later)
        }
        // Seed a new cluster and flood-fill density-reachable points.
        labels[i] = cid;
        queue.clear();
        for &j in &nb {
            if !queued[j as usize] {
                queued[j as usize] = true;
                queue.push_back(j);
            }
        }
        while let Some(j) = queue.pop_front() {
            let j = j as usize;
            if labels[j] == NOISE {
                labels[j] = cid; // claim border (or previously-noise) point; first cluster wins
            }
            if !visited[j] {
                visited[j] = true;
                bump(&mut visited_count, &mut since_report, &mut progress);
                neighbors(j, &mut nbj);
                if nbj.len() >= min_pts {
                    // j is a core point -> add its not-yet-queued neighbours to the frontier.
                    for &m in &nbj {
                        if !queued[m as usize] {
                            queued[m as usize] = true;
                            queue.push_back(m);
                        }
                    }
                }
            }
        }
        cid += 1;
    }
    progress(n); // final 100%
    (labels, cid as usize)
}

/// Per-axis multipliers applied to the normalized-cube coordinates (`x=m/z`, `y=1/K0`, `z=RT`) before
/// DBSCAN, so the clustering metric is equi-distanced *and region-independent*: at the reference eps,
/// the neighbourhood reach is a fixed physical amount on each axis — `rt_cycles` MS1 cycles,
/// `im_scans` TIMS scans, `mz_peak_widths` TOF peak-widths — regardless of how the focus region is
/// cropped. Each axis solves `reach = eps_ref · range / (2·scale)` for its target reach (the cube maps
/// each real range onto width 2), anchoring the unit to a *run-level* physical constant. An axis whose
/// anchor is unavailable (`<= 0`) falls back to `1.0` (uncalibrated, region-relative).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisScales {
    pub mz: f32,
    pub im: f32,
    pub rt: f32,
}

/// Inputs for [`cluster_axis_scales`] — the focus region (real units) plus run-level anchors + knobs.
#[derive(Clone, Copy, Debug)]
pub struct ScaleInputs {
    /// Real-unit focus-region bounds `[(mz_lo,hi), (im_lo,hi), (rt_lo,hi)]`.
    pub bounds: [(f64, f64); 3],
    /// Calibration reference eps (the live eps scales the reach proportionally about this).
    pub eps_ref: f64,
    /// Run-level seconds per MS1 cycle; `0` = unknown (RT left uncalibrated).
    pub cycle_duration: f64,
    /// Run-level 1/K0 per TIMS scan; `0` = unknown (1/K0 left uncalibrated).
    pub im_per_scan: f64,
    /// TOF resolution (peak width = m/z / resolution).
    pub mz_resolution: f64,
    pub rt_cycles: f64,
    pub im_scans: f64,
    pub mz_peak_widths: f64,
}

/// Compute the per-axis scales (see [`AxisScales`]).
pub fn cluster_axis_scales(inp: &ScaleInputs) -> AxisScales {
    // scale so the target real-unit reach maps to exactly `eps_ref` after normalize+scale.
    let scale = |range: f64, target_reach: f64| -> f32 {
        if range.is_finite() && range > 0.0 && target_reach.is_finite() && target_reach > 0.0 {
            let s = (inp.eps_ref * range / (2.0 * target_reach)) as f32;
            if s.is_finite() && s > 0.0 {
                s
            } else {
                1.0 // guard a malformed eps_ref (NaN / <=0) from producing a degenerate scale
            }
        } else {
            1.0 // anchor unavailable / degenerate range -> normalized (region-relative) fallback
        }
    };
    let mz_range = (inp.bounds[0].1 - inp.bounds[0].0).abs();
    let im_range = (inp.bounds[1].1 - inp.bounds[1].0).abs();
    let rt_range = (inp.bounds[2].1 - inp.bounds[2].0).abs();
    let mz_center = ((inp.bounds[0].0 + inp.bounds[0].1) * 0.5).abs();
    let peak_width = if inp.mz_resolution > 0.0 { mz_center / inp.mz_resolution } else { 0.0 };
    AxisScales {
        mz: scale(mz_range, inp.mz_peak_widths.max(0.0) * peak_width),
        im: scale(im_range, inp.im_scans.max(0.0) * inp.im_per_scan),
        rt: scale(rt_range, inp.rt_cycles.max(0.0) * inp.cycle_duration),
    }
}

/// The real-unit neighbourhood reach of `eps` on an axis, given its `scale` and the region's real
/// `range`. Inverse of the calibration in [`cluster_axis_scales`].
pub fn axis_reach(eps: f64, range: f64, scale: f32) -> f64 {
    if scale > 0.0 {
        eps * range / (2.0 * scale as f64)
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod scale_tests {
    use super::*;

    fn base(im: (f64, f64)) -> ScaleInputs {
        ScaleInputs {
            bounds: [(500.0, 520.0), im, (30.0, 60.0)],
            eps_ref: 0.012,
            cycle_duration: 0.1,  // s / cycle
            im_per_scan: 0.0008,  // 1/K0 / scan
            mz_resolution: 50_000.0,
            rt_cycles: 2.0,
            im_scans: 10.0,
            mz_peak_widths: 1.5,
        }
    }

    /// Normalize real points into `bounds`, apply the scales, cluster.
    fn cluster_real(
        real: &[[f64; 3]],
        bounds: [(f64, f64); 3],
        s: AxisScales,
        eps: f32,
        min_pts: usize,
    ) -> Vec<i32> {
        let pts: Vec<[f32; 3]> = real
            .iter()
            .map(|p| {
                let mut q = [0f32; 3];
                for a in 0..3 {
                    let c = (bounds[a].0 + bounds[a].1) * 0.5;
                    let h = (bounds[a].1 - bounds[a].0) * 0.5;
                    q[a] = ((p[a] - c) / h) as f32;
                }
                [q[0] * s.mz, q[1] * s.im, q[2] * s.rt]
            })
            .collect();
        dbscan(&pts, eps, min_pts).0
    }

    /// Re-label to first-seen order so cluster-id numbering doesn't matter; noise stays -1.
    fn canonical(labels: &[i32]) -> Vec<i32> {
        let mut map = std::collections::HashMap::new();
        let mut next = 0i32;
        labels
            .iter()
            .map(|&l| {
                if l < 0 {
                    -1
                } else {
                    *map.entry(l).or_insert_with(|| {
                        let v = next;
                        next += 1;
                        v
                    })
                }
            })
            .collect()
    }

    /// The realized reach equals the configured physical target on every axis — and is invariant to
    /// the focus-region width (the whole point of the calibration).
    #[test]
    fn reach_equals_target_and_is_region_independent() {
        let eps = 0.012; // == eps_ref
        for im in [(0.60, 1.50), (0.95, 1.05)] {
            let inp = base(im);
            let s = cluster_axis_scales(&inp);
            let im_reach = axis_reach(eps, im.1 - im.0, s.im);
            let rt_reach = axis_reach(eps, 60.0 - 30.0, s.rt);
            let mz_reach = axis_reach(eps, 520.0 - 500.0, s.mz);
            // Relative tolerance absorbs the f32 scale rounding; still 3 orders tighter than the ~9x
            // region-dependence bug it guards against.
            let close = |got: f64, want: f64| (got / want - 1.0).abs() < 1e-4;
            assert!(close(im_reach, 10.0 * 0.0008), "im_reach={im_reach}");
            assert!(close(rt_reach, 2.0 * 0.1), "rt_reach={rt_reach}");
            let pw = 510.0 / 50_000.0;
            assert!(close(mz_reach, 1.5 * pw), "mz_reach={mz_reach}");
        }
    }

    /// Region-invariance at the clustering level: the SAME real-unit points, normalized into a wide vs
    /// a tight IM region and scaled, produce identical DBSCAN labels.
    #[test]
    fn dbscan_labels_invariant_to_im_focus() {
        // Four points spaced 0.004 1/K0 along mobility (same m/z + RT). reach = 10·0.0008 = 0.008 >
        // 0.004, so they chain into one cluster — in BOTH regions, if the calibration is correct.
        let real = [
            [510.0, 1.000, 45.0],
            [510.0, 1.004, 45.0],
            [510.0, 1.008, 45.0],
            [510.0, 1.012, 45.0],
        ];
        let wide = (0.60, 1.50);
        let tight = (0.99, 1.02);
        let lw = cluster_real(&real, [(500.0, 520.0), wide, (30.0, 60.0)], cluster_axis_scales(&base(wide)), 0.012, 2);
        let lt = cluster_real(&real, [(500.0, 520.0), tight, (30.0, 60.0)], cluster_axis_scales(&base(tight)), 0.012, 2);
        assert_eq!(canonical(&lw), canonical(&lt), "labels must be region-independent");
        assert_eq!(*lw.iter().max().unwrap() + 1, 1, "expected exactly one cluster");
    }

    /// Documents the bug the calibration fixes: with 1/K0 left uncalibrated (`im` scale = 1, the old
    /// behavior), the same points give DIFFERENT labels in a wide vs a tight IM region.
    #[test]
    fn uncalibrated_im_diverges_with_focus() {
        let real = [
            [510.0, 1.000, 45.0],
            [510.0, 1.004, 45.0],
            [510.0, 1.008, 45.0],
            [510.0, 1.012, 45.0],
        ];
        let wide = (0.60, 1.50);
        let tight = (0.99, 1.02);
        let uncal = AxisScales { mz: 1.0, im: 1.0, rt: 1.0 };
        let lw = cluster_real(&real, [(500.0, 520.0), wide, (30.0, 60.0)], uncal, 0.012, 2);
        let lt = cluster_real(&real, [(500.0, 520.0), tight, (30.0, 60.0)], uncal, 0.012, 2);
        // wide: spacing 2·0.004/0.9 ≈ 0.0089 < eps -> chains; tight: 2·0.004/0.03 ≈ 0.27 > eps -> shatters
        assert_ne!(canonical(&lw), canonical(&lt), "uncalibrated 1/K0 is focus-dependent (the bug)");
    }

    /// The live eps (not eps_ref) scales the reach proportionally: 2x eps -> 2x reach on every axis.
    #[test]
    fn live_eps_scales_reach_proportionally() {
        let s = cluster_axis_scales(&base((0.60, 1.50)));
        for (range, scale) in [(0.9, s.im), (30.0, s.rt), (20.0, s.mz)] {
            let r1 = axis_reach(0.012, range, scale);
            let r2 = axis_reach(0.024, range, scale);
            assert!((r2 / r1 - 2.0).abs() < 1e-4, "reach should scale with eps");
        }
    }

    /// A missing run-level anchor (`im_per_scan = 0`) falls back to the unscaled (normalized) axis.
    #[test]
    fn missing_im_anchor_falls_back_to_unscaled() {
        let mut inp = base((0.60, 1.50));
        inp.im_per_scan = 0.0;
        assert_eq!(cluster_axis_scales(&inp).im, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_dense_blobs_plus_noise() {
        let mut pts = Vec::new();
        // Blob A around (-0.5,-0.5,-0.5), blob B around (0.5,0.5,0.5).
        for i in 0..50 {
            let t = i as f32 * 0.001;
            pts.push([-0.5 + t, -0.5 - t, -0.5 + t]);
            pts.push([0.5 - t, 0.5 + t, 0.5 - t]);
        }
        // A lone outlier.
        pts.push([0.95, -0.95, 0.0]);
        let (labels, k) = dbscan(&pts, 0.05, 4);
        assert_eq!(k, 2, "expected two clusters");
        assert_eq!(*labels.last().unwrap(), NOISE, "outlier should be noise");
    }

    #[test]
    fn empty_and_degenerate() {
        assert_eq!(dbscan(&[], 0.1, 4), (vec![], 0));
        let (l, k) = dbscan(&[[0.0, 0.0, 0.0]], 0.0, 1);
        assert_eq!(k, 0);
        assert_eq!(l, vec![NOISE]);
    }
}
