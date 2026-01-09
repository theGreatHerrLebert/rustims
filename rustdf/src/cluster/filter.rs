//! Cluster quality filtering.
//!
//! Provides configurable quality-based filtering for clusters to reduce
//! over-clustering without losing real signal.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::cluster::cluster::ClusterResult1D;

/// Quality filter configuration for clusters.
///
/// All bounds are inclusive. Set min to 0 or max to f32::MAX to disable a bound.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterQualityFilter {
    // Intensity bounds
    pub min_raw_sum: f32,
    pub max_raw_sum: f32,

    // RT span bounds (in frames)
    pub min_rt_span: usize,
    pub max_rt_span: usize,

    // IM span bounds (in scans)
    pub min_im_span: usize,
    pub max_im_span: usize,

    // RT sigma bounds (from fit)
    pub min_rt_sigma: f32,
    pub max_rt_sigma: f32,

    // IM sigma bounds (from fit)
    pub min_im_sigma: f32,
    pub max_im_sigma: f32,

    // Fit quality (R² from moment fit, 0-1)
    pub min_rt_r2: f32,
    pub min_im_r2: f32,

    // Number of frames used
    pub min_n_frames: usize,
}

impl Default for ClusterQualityFilter {
    fn default() -> Self {
        Self {
            // Intensity: keep anything with some signal
            min_raw_sum: 0.0,
            max_raw_sum: f32::MAX,

            // RT span: at least 3 frames, not more than 500
            min_rt_span: 3,
            max_rt_span: 500,

            // IM span: at least 3 scans, not more than 150
            min_im_span: 3,
            max_im_span: 150,

            // RT sigma: not a single-frame spike, not unreasonably wide
            min_rt_sigma: 0.5,
            max_rt_sigma: 100.0,

            // IM sigma: reasonable peak width in scans
            min_im_sigma: 0.5,
            max_im_sigma: 50.0,

            // Fit quality: disabled by default (set to 0)
            min_rt_r2: 0.0,
            min_im_r2: 0.0,

            // At least used 2 frames
            min_n_frames: 2,
        }
    }
}

impl ClusterQualityFilter {
    /// Create a permissive filter that only removes obvious garbage.
    pub fn permissive() -> Self {
        Self {
            min_raw_sum: 1.0,
            max_raw_sum: f32::MAX,
            min_rt_span: 2,
            max_rt_span: 1000,
            min_im_span: 2,
            max_im_span: 200,
            min_rt_sigma: 0.3,
            max_rt_sigma: 200.0,
            min_im_sigma: 0.3,
            max_im_sigma: 100.0,
            min_rt_r2: 0.0,
            min_im_r2: 0.0,
            min_n_frames: 1,
        }
    }

    /// Create a strict filter for aggressive noise removal.
    pub fn strict() -> Self {
        Self {
            min_raw_sum: 50.0,
            max_raw_sum: f32::MAX,
            min_rt_span: 5,
            max_rt_span: 300,
            min_im_span: 5,
            max_im_span: 100,
            min_rt_sigma: 1.0,
            max_rt_sigma: 50.0,
            min_im_sigma: 1.0,
            max_im_sigma: 30.0,
            min_rt_r2: 0.1,
            min_im_r2: 0.1,
            min_n_frames: 3,
        }
    }

    /// Check if a cluster passes this quality filter.
    #[inline]
    pub fn passes(&self, c: &ClusterResult1D) -> bool {
        // Intensity check
        if c.raw_sum < self.min_raw_sum || c.raw_sum > self.max_raw_sum {
            return false;
        }

        // RT span check
        let rt_span = c.rt_window.1.saturating_sub(c.rt_window.0) + 1;
        if rt_span < self.min_rt_span || rt_span > self.max_rt_span {
            return false;
        }

        // IM span check
        let im_span = c.im_window.1.saturating_sub(c.im_window.0) + 1;
        if im_span < self.min_im_span || im_span > self.max_im_span {
            return false;
        }

        // RT sigma check
        let rt_sigma = c.rt_fit.sigma;
        if rt_sigma < self.min_rt_sigma || rt_sigma > self.max_rt_sigma {
            return false;
        }

        // IM sigma check
        let im_sigma = c.im_fit.sigma;
        if im_sigma < self.min_im_sigma || im_sigma > self.max_im_sigma {
            return false;
        }

        // R² checks (if enabled)
        if self.min_rt_r2 > 0.0 && c.rt_fit.r2 < self.min_rt_r2 {
            return false;
        }
        if self.min_im_r2 > 0.0 && c.im_fit.r2 < self.min_im_r2 {
            return false;
        }

        // Frame count check
        if c.frame_ids_used.len() < self.min_n_frames {
            return false;
        }

        true
    }

    /// Filter a slice of clusters, returning only those that pass.
    pub fn filter(&self, clusters: &[ClusterResult1D]) -> Vec<ClusterResult1D> {
        clusters
            .iter()
            .filter(|c| self.passes(c))
            .cloned()
            .collect()
    }

    /// Filter clusters in parallel.
    pub fn filter_par(&self, clusters: &[ClusterResult1D]) -> Vec<ClusterResult1D> {
        clusters
            .par_iter()
            .filter(|c| self.passes(c))
            .cloned()
            .collect()
    }

    /// Filter in place, returning count of removed clusters.
    pub fn filter_in_place(&self, clusters: &mut Vec<ClusterResult1D>) -> usize {
        let before = clusters.len();
        clusters.retain(|c| self.passes(c));
        before - clusters.len()
    }

    /// Get statistics about why clusters are failing.
    pub fn diagnose(&self, clusters: &[ClusterResult1D]) -> FilterDiagnostics {
        let mut diag = FilterDiagnostics::default();
        diag.total = clusters.len();

        for c in clusters {
            let mut dominated_by = None;

            // Check each condition in order of likely frequency
            if c.raw_sum < self.min_raw_sum {
                diag.failed_min_raw_sum += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("min_raw_sum");
                }
            }
            if c.raw_sum > self.max_raw_sum {
                diag.failed_max_raw_sum += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("max_raw_sum");
                }
            }

            let rt_span = c.rt_window.1.saturating_sub(c.rt_window.0) + 1;
            if rt_span < self.min_rt_span {
                diag.failed_min_rt_span += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("min_rt_span");
                }
            }
            if rt_span > self.max_rt_span {
                diag.failed_max_rt_span += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("max_rt_span");
                }
            }

            let im_span = c.im_window.1.saturating_sub(c.im_window.0) + 1;
            if im_span < self.min_im_span {
                diag.failed_min_im_span += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("min_im_span");
                }
            }
            if im_span > self.max_im_span {
                diag.failed_max_im_span += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("max_im_span");
                }
            }

            if c.rt_fit.sigma < self.min_rt_sigma {
                diag.failed_min_rt_sigma += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("min_rt_sigma");
                }
            }
            if c.rt_fit.sigma > self.max_rt_sigma {
                diag.failed_max_rt_sigma += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("max_rt_sigma");
                }
            }

            if c.im_fit.sigma < self.min_im_sigma {
                diag.failed_min_im_sigma += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("min_im_sigma");
                }
            }
            if c.im_fit.sigma > self.max_im_sigma {
                diag.failed_max_im_sigma += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("max_im_sigma");
                }
            }

            if self.min_rt_r2 > 0.0 && c.rt_fit.r2 < self.min_rt_r2 {
                diag.failed_min_rt_r2 += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("min_rt_r2");
                }
            }
            if self.min_im_r2 > 0.0 && c.im_fit.r2 < self.min_im_r2 {
                diag.failed_min_im_r2 += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("min_im_r2");
                }
            }

            if c.frame_ids_used.len() < self.min_n_frames {
                diag.failed_min_n_frames += 1;
                if dominated_by.is_none() {
                    dominated_by = Some("min_n_frames");
                }
            }

            if dominated_by.is_none() {
                diag.passed += 1;
            }
        }

        diag
    }
}

/// Diagnostics about filter failures.
#[derive(Clone, Debug, Default)]
pub struct FilterDiagnostics {
    pub total: usize,
    pub passed: usize,

    pub failed_min_raw_sum: usize,
    pub failed_max_raw_sum: usize,
    pub failed_min_rt_span: usize,
    pub failed_max_rt_span: usize,
    pub failed_min_im_span: usize,
    pub failed_max_im_span: usize,
    pub failed_min_rt_sigma: usize,
    pub failed_max_rt_sigma: usize,
    pub failed_min_im_sigma: usize,
    pub failed_max_im_sigma: usize,
    pub failed_min_rt_r2: usize,
    pub failed_min_im_r2: usize,
    pub failed_min_n_frames: usize,
}

impl FilterDiagnostics {
    pub fn summary(&self) -> String {
        let failed = self.total - self.passed;
        let pct = if self.total > 0 {
            (self.passed as f64 / self.total as f64) * 100.0
        } else {
            100.0
        };

        format!(
            "FilterDiagnostics: {}/{} passed ({:.1}%), {} failed\n\
             Failures by criterion:\n\
             - min_raw_sum: {}\n\
             - max_raw_sum: {}\n\
             - min_rt_span: {}\n\
             - max_rt_span: {}\n\
             - min_im_span: {}\n\
             - max_im_span: {}\n\
             - min_rt_sigma: {}\n\
             - max_rt_sigma: {}\n\
             - min_im_sigma: {}\n\
             - max_im_sigma: {}\n\
             - min_rt_r2: {}\n\
             - min_im_r2: {}\n\
             - min_n_frames: {}",
            self.passed,
            self.total,
            pct,
            failed,
            self.failed_min_raw_sum,
            self.failed_max_raw_sum,
            self.failed_min_rt_span,
            self.failed_max_rt_span,
            self.failed_min_im_span,
            self.failed_max_im_span,
            self.failed_min_rt_sigma,
            self.failed_max_rt_sigma,
            self.failed_min_im_sigma,
            self.failed_max_im_sigma,
            self.failed_min_rt_r2,
            self.failed_min_im_r2,
            self.failed_min_n_frames,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_filter() {
        let filter = ClusterQualityFilter::default();
        assert!(filter.min_raw_sum == 0.0);
        assert!(filter.min_rt_span == 3);
    }

    #[test]
    fn test_permissive_filter() {
        let filter = ClusterQualityFilter::permissive();
        assert!(filter.min_raw_sum == 1.0);
        assert!(filter.min_rt_span == 2);
    }

    #[test]
    fn test_strict_filter() {
        let filter = ClusterQualityFilter::strict();
        assert!(filter.min_raw_sum == 50.0);
        assert!(filter.min_rt_span == 5);
    }
}
