//! Orbit camera around the normalized data cube.

use glam::{Mat3, Mat4, Vec3};

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
    /// Which data axis points up ("topple the dice"): 0 = 1/K0, 1 = m/z, 2 = RT. Applied as a
    /// cyclic permutation of the data axes baked into the view-projection.
    pub roll: u8,
}

/// Name of the data axis currently pointing up, for the UI.
pub const ROLL_UP_AXIS: [&str; 3] = ["1/K0", "m/z", "RT"];

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
            roll: 0,
        }
    }
}

const PITCH_LIMIT: f32 = 1.5533; // ~89 degrees

impl OrbitCamera {
    pub fn reset(&mut self) {
        let (proj, roll) = (self.projection, self.roll);
        *self = OrbitCamera::default();
        self.projection = proj;
        self.roll = roll;
    }

    /// Cycle which data axis points up (0 → 1/K0, 1 → m/z, 2 → RT).
    pub fn roll_axis(&mut self) {
        self.roll = (self.roll + 1) % 3;
    }

    /// Cyclic axis-permutation matrix for the current roll, mapping data (x=m/z, y=1/K0, z=RT)
    /// to world so the chosen axis lands on world-up (Y). Identity, or a proper rotation.
    fn perm(&self) -> Mat4 {
        let m3 = match self.roll % 3 {
            // 1/K0 (data y) up: identity.
            0 => Mat3::IDENTITY,
            // m/z (data x) up: data x→world y, y→z, z→x.
            1 => Mat3::from_cols(Vec3::Y, Vec3::Z, Vec3::X),
            // RT (data z) up: data x→world z, y→x, z→y.
            _ => Mat3::from_cols(Vec3::Z, Vec3::X, Vec3::Y),
        };
        Mat4::from_mat3(m3)
    }

    pub fn set_axis_view(&mut self, axis: AxisView) {
        self.target = Vec3::ZERO;
        self.distance = 4.0;
        // `perm()` maps DATA axes onto WORLD axes under the current roll (m/z→[X,Y,Z], 1/K0→[Y,Z,X],
        // RT→[Z,X,Y] for roll 0/1/2). The camera yaw/pitch are world-space, so snap to look down the
        // WORLD axis the requested data axis currently occupies — otherwise, after "Roll up", the m/z
        // button would look down whatever data axis happens to sit on world-X. roll 0 reduces to the
        // identity mapping (Mz→X, Mobility→Y, Rt→Z), i.e. the original behavior, unchanged.
        let world_axis = match (axis, self.roll % 3) {
            (AxisView::Mz, 0) | (AxisView::Mobility, 2) | (AxisView::Rt, 1) => 0u8, // world X
            (AxisView::Mz, 1) | (AxisView::Mobility, 0) | (AxisView::Rt, 2) => 1u8, // world Y (up)
            _ => 2u8,                                                               // world Z
        };
        let (yaw, pitch) = match world_axis {
            0 => (std::f32::consts::FRAC_PI_2, 0.0),
            1 => (0.0, PITCH_LIMIT),
            _ => (0.0, 0.0),
        };
        self.yaw = yaw;
        self.pitch = pitch;
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
        // Bake the axis permutation in last: data positions are permuted, then viewed. The
        // volume raycaster reconstructs data-space coords through the inverse, so it stays
        // consistent automatically.
        proj * view * self.perm()
    }

    /// Inverse view-projection (column-major arrays), for reconstructing world-space
    /// rays in the volume raycaster.
    pub fn inv_view_proj(&self, aspect: f32) -> [[f32; 4]; 4] {
        self.view_proj(aspect).inverse().to_cols_array_2d()
    }

    pub fn to_uniform(&self, aspect: f32, viewport: [f32; 2]) -> CameraUniform {
        let (right, up, _) = self.basis();
        // Billboards expand in DATA space (before the baked permutation), so counter-rotate the
        // camera screen axes by perm⁻¹ — then perm maps them back to the world screen axes.
        let inv = self.perm().inverse();
        let right = inv.transform_vector3(right);
        let up = inv.transform_vector3(up);
        CameraUniform {
            view_proj: self.view_proj(aspect).to_cols_array_2d(),
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            viewport: [viewport[0], viewport[1], 0.0, 0.0],
        }
    }
}
