// Annotation overlay: draws line-list geometry (DDA precursor crosses / DIA isolation
// boxes) in the normalized cube, on top of the scene.

struct Camera {
    view_proj : mat4x4<f32>,
    right     : vec4<f32>,
    up        : vec4<f32>,
    viewport  : vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam : Camera;

@vertex
fn vs_main(@location(0) pos : vec3<f32>) -> @builtin(position) vec4<f32> {
    return cam.view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Bright cyan, fully opaque so markers read clearly over points or volume.
    return vec4<f32>(0.1, 0.95, 0.95, 1.0);
}
