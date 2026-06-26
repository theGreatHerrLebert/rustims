//! Grid-accelerated DBSCAN over the normalized point cloud, for coloring points by cluster
//! identity (the same analysis the proteolizard/imspy-vis Voila tool did, in the viewer).
//!
//! Points live in the normalized cube `[-1, 1]^3` (x=m/z, y=1/K0, z=RT), so a single `eps`
//! in cube units is a sensible neighborhood radius. A uniform spatial hash of cell size `eps`
//! makes each neighborhood query touch only the 27 adjacent cells, so the whole pass is about
//! O(n) for the moderate point counts the viewer clusters (it is gated to a cap by the caller).

use std::collections::HashMap;
use std::collections::VecDeque;

/// Label assigned to noise / unclustered points.
pub const NOISE: i32 = -1;

/// Run DBSCAN on `points` (normalized cube positions). Returns a per-point label
/// (`NOISE` for noise, otherwise a 0-based contiguous cluster id) and the cluster count.
pub fn dbscan(points: &[[f32; 3]], eps: f32, min_pts: usize) -> (Vec<i32>, usize) {
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
    // Spatial hash: cell -> point indices.
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::with_capacity(n / 4 + 1);
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
    let mut cid: i32 = 0;
    let mut nb: Vec<u32> = Vec::new();
    let mut nbj: Vec<u32> = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        neighbors(i, &mut nb);
        if nb.len() < min_pts {
            continue; // noise (may still be claimed as a border point later)
        }
        // Seed a new cluster and flood-fill density-reachable points.
        labels[i] = cid;
        queue.clear();
        queue.extend(nb.iter().copied());
        while let Some(j) = queue.pop_front() {
            let j = j as usize;
            if !visited[j] {
                visited[j] = true;
                neighbors(j, &mut nbj);
                if nbj.len() >= min_pts {
                    queue.extend(nbj.iter().copied()); // j is a core point -> expand
                }
            }
            if labels[j] == NOISE {
                labels[j] = cid; // claim border (or previously-noise) point
            }
        }
        cid += 1;
    }
    (labels, cid as usize)
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
