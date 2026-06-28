//! Colormap lookup tables baked into a single 2D texture (256 wide × N maps tall).
//!
//! The point/volume shaders sample row `colormap_id` at column `t∈[0,1]` to map a
//! normalized intensity to a perceptual color.

/// Names in row order. Keep in sync with [`anchors`].
pub const COLORMAP_NAMES: [&str; 4] = ["Viridis", "Inferno", "Turbo", "Grayscale"];

/// Number of texels per colormap row.
const LUT_WIDTH: usize = 256;

/// Control-point anchors (sRGB 0..255) for each colormap; linearly interpolated.
fn anchors(map: usize) -> &'static [[u8; 3]] {
    match map {
        0 => &[
            [68, 1, 84],
            [59, 82, 139],
            [33, 144, 140],
            [93, 201, 99],
            [253, 231, 37],
        ],
        1 => &[
            [0, 0, 4],
            [87, 16, 110],
            [188, 55, 84],
            [249, 142, 9],
            [252, 255, 164],
        ],
        2 => &[
            [48, 18, 59],
            [54, 125, 197],
            [7, 193, 151],
            [166, 228, 48],
            [249, 189, 38],
            [180, 4, 38],
        ],
        _ => &[[0, 0, 0], [255, 255, 255]],
    }
}

/// RGB at normalized position `t∈[0,1]` for colormap `map` (CPU mirror of the LUT).
/// The egui colorbar and the GPU LUT both go through this anchor-lerp so they can't drift.
pub fn sample(map: usize, t: f32) -> [u8; 3] {
    let a = anchors(map);
    let segs = a.len() - 1;
    let t = t.clamp(0.0, 1.0);
    let fseg = t * segs as f32;
    let seg = (fseg.floor() as usize).min(segs - 1);
    let local = fseg - seg as f32;
    let c0 = a[seg];
    let c1 = a[seg + 1];
    let lerp = |i: usize| (c0[i] as f32 + (c1[i] as f32 - c0[i] as f32) * local).round() as u8;
    [lerp(0), lerp(1), lerp(2)]
}

/// Build the RGBA8 LUT pixel data, row-major (`width=256`, `height=N`).
pub fn build_lut_rgba8() -> (Vec<u8>, u32, u32) {
    let n = COLORMAP_NAMES.len();
    let mut data = vec![0u8; LUT_WIDTH * n * 4];
    for map in 0..n {
        for x in 0..LUT_WIDTH {
            let t = x as f32 / (LUT_WIDTH - 1) as f32;
            let rgb = sample(map, t);
            let idx = (map * LUT_WIDTH + x) * 4;
            data[idx] = rgb[0];
            data[idx + 1] = rgb[1];
            data[idx + 2] = rgb[2];
            data[idx + 3] = 255;
        }
    }
    (data, LUT_WIDTH as u32, n as u32)
}

/// Upload the LUT to a sampled 2D texture and return (texture, view, sampler).
pub fn create_lut_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let (data, width, height) = build_lut_rgba8();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("colormap-lut"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // sRGB so sampling returns linear color for correct blending.
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("colormap-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    (texture, view, sampler)
}

/// Distinct color per window group, by evenly spacing hue around the wheel. Shared by the
/// annotation overlay (DIA isolation windows) and the egui group legend. Render-safe (no
/// native deps), so it lives here rather than in the native loader.
pub fn group_color(group: u32, n_groups: u32) -> [f32; 3] {
    let h = (group.saturating_sub(1) as f32) / (n_groups.max(1) as f32);
    hsv_to_rgb(h, 0.7, 1.0)
}

/// HSV (all in [0,1] except hue which wraps) to linear-ish RGB for line colors.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as i32).rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_endpoints_match_anchors() {
        // Viridis anchors run [68,1,84] -> ... -> [253,231,37].
        assert_eq!(sample(0, 0.0), [68, 1, 84]);
        assert_eq!(sample(0, 1.0), [253, 231, 37]);
        // Out-of-range t clamps to the endpoints.
        assert_eq!(sample(0, -1.0), [68, 1, 84]);
        assert_eq!(sample(0, 2.0), [253, 231, 37]);
    }
}
