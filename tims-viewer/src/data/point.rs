//! GPU point format, axis normalization, and the shared spatial bounds.

use bytemuck::{Pod, Zeroable};

/// One renderable data point, packed for the GPU as instance data.
///
/// Layout is 32 bytes, 16-byte aligned, so a `Vec<GpuPoint>` uploads directly as a
/// vertex (instance-step) buffer via `bytemuck::cast_slice`.
///
/// `weight` is the number of original points this sample represents (`1/p` of the
/// downsample). Carrying it lets additive-density rendering stay brightness-invariant
/// regardless of the downsample ratio (see the shared density/exposure model).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPoint {
    /// Position in the normalized cube `[-1, 1]^3` (x=m/z, y=1/K0, z=RT).
    pub pos: [f32; 3],
    /// Raw intensity (counts).
    pub intensity: f32,
    /// Sample weight = `1/p` (how many source points this one stands in for).
    pub weight: f32,
    /// Bit flags. bit0: 0 = MS1/precursor, 1 = MS2/fragment.
    pub flags: u32,
    /// `_pad[0]` doubles as the DBSCAN cluster id for cluster coloring (`NO_CLUSTER` =
    /// noise/unclustered); `_pad[1]` is reserved padding to keep the 32-byte layout.
    pub _pad: [u32; 2],
}

impl GpuPoint {
    pub const MS2_FLAG: u32 = 1;
    /// Cluster-id sentinel for noise / not-yet-clustered points (rendered grey).
    pub const NO_CLUSTER: u32 = u32::MAX;

    /// `wgpu` vertex-buffer layout describing the instance step-mode attributes.
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
            0 => Float32x3, // pos
            1 => Float32,   // intensity
            2 => Float32,   // weight
            3 => Uint32,    // flags
            4 => Uint32,    // cluster id (_pad[0])
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuPoint>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &ATTRS,
        }
    }
}

/// Affine mapping between a real-unit axis range and the normalized cube `[-1, 1]`.
#[derive(Clone, Copy, Debug)]
pub struct AxisTransform {
    pub min: f64,
    pub max: f64,
}

impl AxisTransform {
    pub fn new(min: f64, max: f64) -> Self {
        // Guard against a degenerate (zero-width) range.
        if (max - min).abs() < f64::EPSILON {
            AxisTransform {
                min,
                max: min + 1.0,
            }
        } else {
            AxisTransform { min, max }
        }
    }

    /// Map a real value into `[-1, 1]`.
    #[inline]
    pub fn normalize(&self, v: f64) -> f32 {
        (2.0 * (v - self.min) / (self.max - self.min) - 1.0) as f32
    }

    /// Map a normalized `[-1, 1]` value back to real units.
    /// Used by axis-tick labeling (and tested); kept ahead of the gizmo work.
    #[allow(dead_code)]
    #[inline]
    pub fn denormalize(&self, n: f32) -> f64 {
        self.min + ((n as f64) + 1.0) * 0.5 * (self.max - self.min)
    }
}

/// The three spatial axes of the run, in real units, plus their normalizers.
#[derive(Clone, Copy, Debug)]
pub struct AxisBounds {
    /// x-axis: m/z (Th).
    pub mz: AxisTransform,
    /// y-axis: inverse ion mobility 1/K0.
    pub im: AxisTransform,
    /// z-axis: retention time (seconds).
    pub rt: AxisTransform,
}

impl AxisBounds {
    /// Normalize a `(mz, im, rt)` real-unit triple into cube coordinates.
    #[inline]
    pub fn normalize(&self, mz: f64, im: f64, rt: f64) -> [f32; 3] {
        [self.mz.normalize(mz), self.im.normalize(im), self.rt.normalize(rt)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_point_is_32_bytes() {
        assert_eq!(std::mem::size_of::<GpuPoint>(), 32);
    }

    #[test]
    fn axis_transform_round_trip() {
        let t = AxisTransform::new(100.0, 1700.0);
        for v in [100.0, 400.0, 900.5, 1700.0] {
            let n = t.normalize(v);
            let back = t.denormalize(n);
            assert!((back - v).abs() < 1e-6, "round-trip failed for {v}: {back}");
        }
        assert!((t.normalize(100.0) + 1.0).abs() < 1e-6);
        assert!((t.normalize(1700.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn axis_transform_handles_degenerate_range() {
        let t = AxisTransform::new(5.0, 5.0);
        // Must not divide by zero / produce NaN.
        assert!(t.normalize(5.0).is_finite());
    }
}
