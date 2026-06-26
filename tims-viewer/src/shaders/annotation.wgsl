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

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos : vec3<f32>, @location(1) color : vec3<f32>) -> VsOut {
    var out : VsOut;
    out.color = color;
    // Clip annotations to the active window so narrowing a range filters them too.
    if (any(pos < flt.min.xyz) || any(pos > flt.max.xyz)) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0); // outside clip volume -> culled
        return out;
    }
    // Focus (flt.min.w): re-fit the window box to the full cube, matching the points.
    var fpos = pos;
    if (flt.min.w > 0.5) {
        let center = (flt.min.xyz + flt.max.xyz) * 0.5;
        let halfspan = max((flt.max.xyz - flt.min.xyz) * 0.5, vec3<f32>(1e-6));
        fpos = (pos - center) / halfspan;
    }
    out.clip = cam.view_proj * vec4<f32>(fpos, 1.0);
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    // Per-vertex color, fully opaque so markers read clearly over points or volume.
    return vec4<f32>(in.color, 1.0);
}
