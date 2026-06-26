// Point-cloud splat shader: instanced screen-facing billboards with a shared
// transfer-function + colormap, additive-density and structural-opaque modes.

struct Camera {
    view_proj : mat4x4<f32>,
    right     : vec4<f32>,
    up        : vec4<f32>,
    viewport  : vec4<f32>,
};

struct Params {
    filter_min : vec4<f32>,
    filter_max : vec4<f32>,
    transfer   : vec4<f32>, // mode, i_min, i_max, exposure
    point_size : f32,
    opacity    : f32,
    focus      : f32,
    color_mode : u32,   // 0 = intensity colormap, 1 = cluster id
    ms_mask     : u32,
    colormap_id : u32,
    render_mode : u32,
    n_colormaps : u32,
};

@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var lut_tex : texture_2d<f32>;
@group(0) @binding(3) var lut_smp : sampler;

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) intensity : f32,
    @location(2) weight : f32,
    @location(3) visible : f32,
    @location(4) @interpolate(flat) cluster : u32,
};

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(5.0, 3.0, 1.0) / 6.0;
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - 3.0);
    return v * mix(vec3<f32>(1.0), clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

// Cluster id -> color: golden-ratio hue spacing; the noise sentinel renders grey.
fn cluster_color(id: u32) -> vec3<f32> {
    if (id == 0xffffffffu) {
        return vec3<f32>(0.32, 0.32, 0.38);
    }
    return hsv2rgb(fract(f32(id) * 0.6180339887), 0.65, 1.0);
}

fn transfer(i: f32) -> f32 {
    let mode = params.transfer.x;
    let imin = params.transfer.y;
    let imax = params.transfer.z;
    if (mode < 0.5) {
        return clamp((i - imin) / max(imax - imin, 1e-6), 0.0, 1.0);
    } else if (mode < 1.5) {
        let a = sqrt(max(i, 0.0));
        let lo = sqrt(max(imin, 0.0));
        let hi = sqrt(max(imax, 0.0));
        return clamp((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
    } else {
        let a = log(max(i, 1.0));
        let lo = log(max(imin, 1.0));
        let hi = log(max(imax, 1.0));
        return clamp((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid : u32,
    @location(0) pos : vec3<f32>,
    @location(1) intensity : f32,
    @location(2) weight : f32,
    @location(3) flags : u32,
    @location(4) cluster : u32,
) -> VsOut {
    var out : VsOut;
    out.cluster = cluster;

    // Window + intensity-range + MS-type filtering -> collapse off-screen if rejected.
    // The intensity range rides filter_min.w / filter_max.w (real intensity units).
    let in_window =
        all(pos >= params.filter_min.xyz) && all(pos <= params.filter_max.xyz)
        && intensity >= params.filter_min.w && intensity <= params.filter_max.w;
    let is_ms2 = (flags & 1u) != 0u;
    let show = select((params.ms_mask & 1u) != 0u, (params.ms_mask & 2u) != 0u, is_ms2);
    if (!in_window || !show) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0); // outside clip volume -> culled
        out.visible = 0.0;
        out.uv = vec2<f32>(0.0, 0.0);
        out.intensity = 0.0;
        out.weight = 0.0;
        return out;
    }

    // Quad corners for a 4-vertex triangle strip.
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    let corner = corners[vid];

    // Focus: re-fit the active window box to the full cube (zoom to selection).
    var fpos = pos;
    if (params.focus > 0.5) {
        let center = (params.filter_min.xyz + params.filter_max.xyz) * 0.5;
        let halfspan = max((params.filter_max.xyz - params.filter_min.xyz) * 0.5, vec3<f32>(1e-6));
        fpos = (pos - center) / halfspan;
    }

    // World-space billboard so size attenuates with distance (perspective).
    let bb = params.point_size * 0.0015;
    let world = fpos + (corner.x * cam.right.xyz + corner.y * cam.up.xyz) * bb;
    out.clip = cam.view_proj * vec4<f32>(world, 1.0);
    out.uv = corner;
    out.intensity = intensity;
    out.weight = weight;
    out.visible = 1.0;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (in.visible < 0.5) {
        discard;
    }
    let r = length(in.uv);
    let falloff = 1.0 - smoothstep(0.0, 1.0, r);
    if (falloff <= 0.0) {
        discard;
    }
    var color : vec3<f32>;
    if (params.color_mode == 1u) {
        color = cluster_color(in.cluster);
    } else {
        let t = transfer(in.intensity);
        let row = (f32(params.colormap_id) + 0.5) / f32(params.n_colormaps);
        color = textureSample(lut_tex, lut_smp, vec2<f32>(t, row)).rgb;
    }

    // Cluster coloring always renders opaque: additive blending would mix distinct cluster
    // hues into mush and dim them by the 1/p weight, so cluster ids would read as noise.
    if (params.render_mode == 1u || params.color_mode == 1u) {
        // Structural opaque: round point via alpha cutout, depth-written.
        if (falloff < 0.5) {
            discard;
        }
        return vec4<f32>(color, 1.0);
    }

    // Additive density: weight by 1/p so brightness is invariant to the DOWNSAMPLE
    // ratio, then scale by exposure so dense regions are tunable rather than blowing
    // out. Note: with world-size billboards, integrated brightness still varies with
    // camera distance (splat pixel-area ~ 1/dist^2) — that zoom dependence is the
    // accepted cost of perspective size attenuation, addressed in Phase 1.5 if needed.
    let exposure = params.transfer.w;
    let contrib = falloff * params.opacity * exposure * in.weight;
    return vec4<f32>(color * contrib, contrib);
}
