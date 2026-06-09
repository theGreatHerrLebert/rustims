//! Synthetic timsTOF-shaped point cloud for the `DEMO` path (no Bruker data needed).
//!
//! Produces a handful of chromatographic "blobs" — gaussian clusters in (m/z, 1/K0)
//! that elute across retention time — plus a sparse noise floor, with log-normal
//! intensities. This exercises the whole pipeline (sampling, both render modes, UI,
//! camera) deterministically.

/// Tiny deterministic PRNG (xorshift64*) — avoids pulling in an RNG dependency and
/// keeps the demo reproducible across runs.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    #[inline]
    pub fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box–Muller.
    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unif().max(1e-12);
        let u2 = self.unif();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy)]
struct Blob {
    mz: f64,
    im: f64,
    rt: f64,
    rt_sigma: f64,
    mz_sigma: f64,
    im_sigma: f64,
    log_amp: f64,
    is_ms2: bool,
}

/// A raw demo point in real units (pre-normalization).
#[derive(Clone, Copy)]
pub struct RawDemoPoint {
    pub mz: f64,
    pub im: f64,
    pub rt: f64,
    pub intensity: f64,
    pub is_ms2: bool,
}

/// Generator of synthetic frames. Each call to [`Self::frame`] returns the points for
/// one retention-time frame, so the loader can stream them like real frames.
pub struct DemoSource {
    blobs: Vec<Blob>,
    rt_min: f64,
    rt_max: f64,
    mz_min: f64,
    mz_max: f64,
    im_min: f64,
    im_max: f64,
    total_points: u64,
    num_frames: usize,
}

impl DemoSource {
    pub fn new(num_frames: usize, total_points: u64) -> Self {
        let (mz_min, mz_max) = (100.0, 1700.0);
        let (im_min, im_max) = (0.6, 1.6);
        let rt_max = (num_frames.max(1) - 1) as f64 * 0.5;
        let mut rng = Rng::new(0xC0FFEE);
        let n_blobs = 60;
        let mut blobs = Vec::with_capacity(n_blobs);
        for _ in 0..n_blobs {
            blobs.push(Blob {
                mz: mz_min + rng.unif() * (mz_max - mz_min),
                im: im_min + rng.unif() * (im_max - im_min),
                rt: rng.unif() * rt_max,
                rt_sigma: rt_max * (0.01 + 0.04 * rng.unif()),
                mz_sigma: 0.3 + 2.0 * rng.unif(),
                im_sigma: 0.005 + 0.02 * rng.unif(),
                log_amp: 8.0 + 5.0 * rng.unif(), // ~e^8..e^13 dynamic range
                is_ms2: rng.unif() < 0.1,
            });
        }
        DemoSource {
            blobs,
            rt_min: 0.0,
            rt_max,
            mz_min,
            mz_max,
            im_min,
            im_max,
            total_points,
            num_frames,
        }
    }

    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// Exact point count for frame `idx`: `total_points` split across frames by
    /// quotient + remainder, so the per-frame counts sum to exactly `total_points`
    /// (some frames may legitimately get zero). This keeps the generated total in sync
    /// with the metadata estimate and GPU capacity even for tiny configs.
    fn frame_count(&self, idx: usize) -> usize {
        let nf = self.num_frames.max(1) as u64;
        let base = self.total_points / nf;
        let rem = self.total_points % nf;
        (base + if (idx as u64) < rem { 1 } else { 0 }) as usize
    }

    /// Generate the points for frame `idx` (0-based).
    pub fn frame(&self, idx: usize) -> Vec<RawDemoPoint> {
        let rt = idx as f64 * 0.5;
        // Per-frame RNG seed keeps generation deterministic and frame-independent.
        let mut rng = Rng::new(0x5EED_0000 ^ (idx as u64).wrapping_mul(0x9E37_79B9));
        let points_per_frame = self.frame_count(idx);
        if points_per_frame == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(points_per_frame);

        // Weight each blob by its gaussian elution profile at this RT.
        let weights: Vec<f64> = self
            .blobs
            .iter()
            .map(|b| {
                let d = (rt - b.rt) / b.rt_sigma;
                (-0.5 * d * d).exp()
            })
            .collect();
        let wsum: f64 = weights.iter().sum::<f64>() + 1e-9;

        // ~85% of points belong to blobs, ~15% are background noise.
        let n_signal = (points_per_frame as f64 * 0.85) as usize;
        let n_noise = points_per_frame - n_signal;

        for _ in 0..n_signal {
            // Pick a blob proportional to its current elution weight.
            let mut pick = rng.unif() * wsum;
            let mut bi = 0usize;
            for (i, w) in weights.iter().enumerate() {
                pick -= *w;
                if pick <= 0.0 {
                    bi = i;
                    break;
                }
            }
            let b = self.blobs[bi];
            let mz = (b.mz + rng.normal() * b.mz_sigma).clamp(self.mz_min, self.mz_max);
            let im = (b.im + rng.normal() * b.im_sigma).clamp(self.im_min, self.im_max);
            // Log-normal intensity scaled by elution profile.
            let prof = weights[bi].max(1e-3);
            let intensity = (b.log_amp + 0.4 * rng.normal()).exp() * prof;
            out.push(RawDemoPoint {
                mz,
                im,
                rt,
                intensity: intensity.max(1.0),
                is_ms2: b.is_ms2,
            });
        }

        for _ in 0..n_noise {
            out.push(RawDemoPoint {
                mz: self.mz_min + rng.unif() * (self.mz_max - self.mz_min),
                im: self.im_min + rng.unif() * (self.im_max - self.im_min),
                rt,
                intensity: (3.0 + 1.5 * rng.normal()).exp().max(1.0),
                is_ms2: false,
            });
        }
        let _ = self.rt_min;
        let _ = self.rt_max;
        out
    }
}
