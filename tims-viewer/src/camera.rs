//! Orbit camera around the normalized data cube.

use glam::{Mat4, Vec3};

use crate::render::uniforms::CameraUniform;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    Perspective,
    /// Hidden from the UI until its bugs are fixed; the projection math is kept.
    #[allow(dead_code)]
    Orthographic,
}

#[derive(Clone, Copy, Debug)]
pub enum AxisView {
    /// Look down the m/z axis.
    Mz,
    /// Look down the mobility axis.
    Mobility,
    /// Look down the retention-time axis.
    Rt,
}

pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Azimuth around world Y, radians.
    pub yaw: f32,
    /// Elevation, radians, clamped away from the poles.
    pub pitch: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
    pub projection: Projection,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        OrbitCamera {
            target: Vec3::ZERO,
            distance: 4.0,
            yaw: 0.7,
            pitch: 0.5,
            fovy: 45f32.to_radians(),
            znear: 0.05,
            zfar: 100.0,
            projection: Projection::Perspective,
        }
    }
}

const PITCH_LIMIT: f32 = 1.5533; // ~89 degrees

impl OrbitCamera {
    pub fn reset(&mut self) {
        let proj = self.projection;
        *self = OrbitCamera::default();
        self.projection = proj;
    }

    pub fn set_axis_view(&mut self, axis: AxisView) {
        self.target = Vec3::ZERO;
        self.distance = 4.0;
        match axis {
            AxisView::Mz => {
                self.yaw = std::f32::consts::FRAC_PI_2;
                self.pitch = 0.0;
            }
            AxisView::Mobility => {
                self.yaw = 0.0;
                self.pitch = PITCH_LIMIT;
            }
            AxisView::Rt => {
                self.yaw = 0.0;
                self.pitch = 0.0;
            }
        }
    }

    fn eye(&self) -> Vec3 {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        let dir = Vec3::new(cp * sy, sp, cp * cy);
        self.target + dir * self.distance
    }

    /// World-space right and up vectors of the camera (for billboarding).
    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let eye = self.eye();
        let forward = (self.target - eye).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        (right, up, forward)
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch = (self.pitch + dy * 0.005).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let (right, up, _) = self.basis();
        let scale = self.distance * 0.0015;
        self.target += (-right * dx + up * dy) * scale;
    }

    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (-scroll * 0.1).exp()).clamp(0.2, 50.0);
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let eye = self.eye();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = match self.projection {
            Projection::Perspective => {
                Mat4::perspective_rh(self.fovy, aspect.max(1e-3), self.znear, self.zfar)
            }
            Projection::Orthographic => {
                let half_h = self.distance * (self.fovy * 0.5).tan();
                let half_w = half_h * aspect.max(1e-3);
                Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, self.znear, self.zfar)
            }
        };
        proj * view
    }

    /// Inverse view-projection (column-major arrays), for reconstructing world-space
    /// rays in the volume raycaster.
    pub fn inv_view_proj(&self, aspect: f32) -> [[f32; 4]; 4] {
        self.view_proj(aspect).inverse().to_cols_array_2d()
    }

    pub fn to_uniform(&self, aspect: f32, viewport: [f32; 2]) -> CameraUniform {
        let (right, up, _) = self.basis();
        CameraUniform {
            view_proj: self.view_proj(aspect).to_cols_array_2d(),
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            viewport: [viewport[0], viewport[1], 0.0, 0.0],
        }
    }
}
