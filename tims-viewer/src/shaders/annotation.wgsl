// Annotation overlay: draws line-list geometry (DDA precursor crosses / DIA isolation
// boxes) in the normalized cube, on top of the scene.

struct Camera {
    view_proj : mat4x4<f32>,
    right     : vec4<f32>,
    up        : vec4<f32>,
    viewport  : vec4<f32>,
};

struct Filter {
    min : vec4<f32>,
    max : vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var<uniform> flt : Filter;

@vertex
fn vs_main(@location(0) pos : vec3<f32>) -> @builtin(position) vec4<f32> {
    // Clip annotations to the active window so narrowing a range filters them too.
    if (any(pos < flt.min.xyz) || any(pos > flt.max.xyz)) {
        return vec4<f32>(2.0, 2.0, 2.0, 1.0); // outside clip volume -> culled
    }
    return cam.view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Bright cyan, fully opaque so markers read clearly over points or volume.
    return vec4<f32>(0.1, 0.95, 0.95, 1.0);
}
