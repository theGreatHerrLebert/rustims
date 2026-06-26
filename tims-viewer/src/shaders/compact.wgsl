// Compaction compute pass: cull each resident point against the active window, MS
// mask, and camera frustum, and append survivors into a compacted buffer while
// counting them into the indirect-draw args. The draw pass then renders only the
// visible set via draw_indirect, instead of processing every resident instance.

struct GpuPoint {
    pos : vec3<f32>,
    intensity : f32,
    weight : f32,
    flags : u32,
    pad : vec2<u32>,
};

struct Camera {
    view_proj : mat4x4<f32>,
    right     : vec4<f32>,
    up        : vec4<f32>,
    viewport  : vec4<f32>,
};

struct Params {
    filter_min : vec4<f32>,
    filter_max : vec4<f32>,
    transfer   : vec4<f32>,
    point_size : f32,
    opacity    : f32,
    focus      : f32,      // mirrors ParamsUniform; unused here
    color_mode : u32,      // mirrors ParamsUniform; unused here
    ms_mask     : u32,
    colormap_id : u32,
    render_mode : u32,
    n_colormaps : u32,
};

// Layout matches wgpu's DrawIndirectArgs: vertex_count, instance_count, first_vertex,
// first_instance. instance_count is the atomic survivor counter.
struct DrawArgs {
    vertex_count   : u32,
    instance_count : atomic<u32>,
    first_vertex   : u32,
    first_instance : u32,
};

struct Compaction {
    point_count : u32,
    row_stride : u32,   // invocations per dispatch row (groups_x * workgroup_size)
    _p1 : u32,
    _p2 : u32,
};

@group(0) @binding(0) var<storage, read>       src : array<GpuPoint>;
@group(0) @binding(1) var<storage, read_write> dst : array<GpuPoint>;
@group(0) @binding(2) var<storage, read_write> draw_args : DrawArgs;
@group(0) @binding(3) var<uniform> cam : Camera;
@group(0) @binding(4) var<uniform> params : Params;
@group(0) @binding(5) var<uniform> comp : Compaction;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // 2D dispatch grid: rebuild the linear point index from the row stride.
    let i = gid.x + gid.y * comp.row_stride;
    if (i >= comp.point_count) {
        return;
    }
    let p = src[i];

    // Window filter (normalized cube).
    if (any(p.pos < params.filter_min.xyz) || any(p.pos > params.filter_max.xyz)) {
        return;
    }

    // MS-type mask.
    let is_ms2 = (p.flags & 1u) != 0u;
    let show = select((params.ms_mask & 1u) != 0u, (params.ms_mask & 2u) != 0u, is_ms2);
    if (!show) {
        return;
    }

    // Frustum cull (with a margin so splats near the edge aren't clipped early).
    let clip = cam.view_proj * vec4<f32>(p.pos, 1.0);
    if (clip.w <= 0.0) {
        return;
    }
    let m = clip.w * 1.2;
    if (clip.x < -m || clip.x > m || clip.y < -m || clip.y > m || clip.z < 0.0 || clip.z > clip.w) {
        return;
    }

    // Append (slot is guaranteed < capacity because survivors <= resident <= capacity).
    let slot = atomicAdd(&draw_args.instance_count, 1u);
    dst[slot] = p;
}
