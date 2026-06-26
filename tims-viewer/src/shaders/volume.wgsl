// Volume raycaster: a fullscreen triangle whose fragment shader marches a ray through
// the 3D intensity texture, sharing the point cloud's transfer-function + colormap.
// Composite (front-to-back) and maximum-intensity-projection styles. The march is
// clamped to the active window box, so range filtering works in volume mode too.

struct Volume {
    inv_view_proj : mat4x4<f32>,
    box_min : vec4<f32>,
    box_max : vec4<f32>,
    transfer : vec4<f32>, // mode, i_min, i_max, exposure
    steps : u32,
    style : u32,          // 0 = composite, 1 = MIP
    colormap_id : u32,
    n_colormaps : u32,
    density_scale : f32,  // recover raw intensity from the normalized texture
    focus : f32,          // 1.0 = re-fit the window box to the full cube (zoom to selection)
    _pad1 : f32,
    _pad2 : f32,
};

@group(0) @binding(0) var<uniform> vol : Volume;
@group(0) @binding(1) var vol_tex : texture_3d<f32>;
@group(0) @binding(2) var vol_smp : sampler;
@group(0) @binding(3) var lut_tex : texture_2d<f32>;
@group(0) @binding(4) var lut_smp : sampler;

struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) ndc : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> VsOut {
    // Fullscreen triangle.
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out : VsOut;
    out.clip = vec4<f32>(p[vid], 0.0, 1.0);
    out.ndc = p[vid];
    return out;
}

fn transfer(i: f32) -> f32 {
    let mode = vol.transfer.x;
    let imin = vol.transfer.y;
    let imax = vol.transfer.z;
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

fn colormap(t: f32) -> vec3<f32> {
    let row = (f32(vol.colormap_id) + 0.5) / f32(vol.n_colormaps);
    return textureSampleLevel(lut_tex, lut_smp, vec2<f32>(t, row), 0.0).rgb;
}

// Robust ray/AABB slab test. Returns (t_enter, t_exit); a miss has t_exit < t_enter.
fn intersect_box(o: vec3<f32>, dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    // Nudge near-zero direction components so 1/dir stays finite — avoids 0*inf = NaN
    // for box-boundary-parallel rays (common in axis/orthographic views).
    let eps = 1e-7;
    let safe = vec3<f32>(
        select(dir.x, eps, abs(dir.x) < eps),
        select(dir.y, eps, abs(dir.y) < eps),
        select(dir.z, eps, abs(dir.z) < eps),
    );
    let inv = 1.0 / safe;
    let t0 = (bmin - o) * inv;
    let t1 = (bmax - o) * inv;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_enter, t_exit);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Reconstruct the world-space ray from the inverse view-projection.
    let near = vol.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let far = vol.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let o = near.xyz / near.w;
    let dir = normalize(far.xyz / far.w - o);

    // Focus: re-fit the window box to the full cube. The ray then marches the whole cube
    // and each marched position maps back into the window box for texture sampling.
    let focus = vol.focus > 0.5;
    let center = (vol.box_min.xyz + vol.box_max.xyz) * 0.5;
    let halfspan = max((vol.box_max.xyz - vol.box_min.xyz) * 0.5, vec3<f32>(1e-6));
    let march_min = select(vol.box_min.xyz, vec3<f32>(-1.0), focus);
    let march_max = select(vol.box_max.xyz, vec3<f32>(1.0), focus);

    let hit = intersect_box(o, dir, march_min, march_max);
    let t_enter = max(hit.x, 0.0);
    let t_exit = hit.y;
    if (t_exit <= t_enter) {
        discard;
    }

    let steps = max(vol.steps, 1u);
    let span = t_exit - t_enter;
    let dt = span / f32(steps);
    let exposure = vol.transfer.w;

    if (vol.style == 1u) {
        // Maximum-intensity projection.
        var maxt = 0.0;
        for (var i = 0u; i < steps; i = i + 1u) {
            let p = o + dir * (t_enter + dt * (f32(i) + 0.5));
            let sp = select(p, center + p * halfspan, focus);
            let uvw = sp * 0.5 + vec3<f32>(0.5);
            let dens = textureSampleLevel(vol_tex, vol_smp, uvw, 0.0).r * vol.density_scale;
            maxt = max(maxt, transfer(dens));
        }
        if (maxt <= 0.0) {
            discard;
        }
        return vec4<f32>(colormap(maxt), 1.0);
    }

    // Front-to-back composite.
    var acc = vec3<f32>(0.0);
    var alpha = 0.0;
    for (var i = 0u; i < steps; i = i + 1u) {
        let p = o + dir * (t_enter + dt * (f32(i) + 0.5));
        let sp = select(p, center + p * halfspan, focus);
        let uvw = sp * 0.5 + vec3<f32>(0.5);
        let dens = textureSampleLevel(vol_tex, vol_smp, uvw, 0.0).r * vol.density_scale;
        let tval = transfer(dens);
        if (tval > 0.0) {
            let col = colormap(tval);
            // Opacity integrated over the step length so it's step-count independent.
            let a = 1.0 - exp(-tval * exposure * dt * 40.0);
            acc = acc + (1.0 - alpha) * a * col;
            alpha = alpha + (1.0 - alpha) * a;
            if (alpha > 0.99) {
                break;
            }
        }
    }
    if (alpha <= 0.0) {
        discard;
    }
    // Premultiplied (acc) with its coverage alpha, so the clear color shows through
    // where the ray didn't accumulate full opacity (pipeline uses premultiplied blend).
    return vec4<f32>(acc, alpha);
}
