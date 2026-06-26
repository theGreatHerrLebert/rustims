//! egui control panel.

use crate::camera::{AxisView, OrbitCamera, Projection};
use crate::render::colormap::COLORMAP_NAMES;
use crate::render::point_cloud::PointMode;
use crate::state::{AppState, TransferMode, ViewMode, VolStyle};

/// Build the left control panel; mutates state and camera in place.
pub fn build(ctx: &egui::Context, state: &mut AppState, camera: &mut OrbitCamera) {
    egui::SidePanel::left("controls")
        .resizable(true)
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.heading("tims-viewer");
            ui.separator();

            // ---- Readouts ----
            ui.label(format!("FPS: {:.0}", state.fps));
            ui.label(format!(
                "Points: {} / {} (cap)",
                fmt_count(state.resident_points as u64),
                fmt_count(state.capacity as u64)
            ));
            ui.label(format!(
                "Run estimate: {} (×{} downsample)",
                fmt_count(state.total_estimate),
                state.downsample_stride
            ));
            if state.load_failed {
                ui.colored_label(egui::Color32::LIGHT_RED, "Load failed (see log)");
            } else if state.load_progress < 1.0 {
                ui.add(egui::ProgressBar::new(state.load_progress).show_percentage());
            } else {
                ui.label("Load complete");
            }
            ui.separator();

            // ---- Render mode ----
            ui.label("Render mode");
            ui.horizontal(|ui| {
                ui.radio_value(&mut state.view_mode, ViewMode::Points, "Points");
                ui.radio_value(&mut state.view_mode, ViewMode::Volume, "Volume");
            });
            match state.view_mode {
                ViewMode::Points => {
                    ui.horizontal(|ui| {
                        ui.radio_value(
                            &mut state.point_mode,
                            PointMode::AdditiveDensity,
                            "Additive",
                        );
                        ui.radio_value(
                            &mut state.point_mode,
                            PointMode::StructuralOpaque,
                            "Opaque",
                        );
                    });
                }
                ViewMode::Volume => {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut state.vol_style, VolStyle::Composite, "Composite");
                        ui.radio_value(&mut state.vol_style, VolStyle::Mip, "MIP");
                    });
                    ui.add(
                        egui::Slider::new(&mut state.vol_steps, 32..=1024).text("ray steps"),
                    );
                }
            }
            ui.separator();

            // ---- Intensity transfer function ----
            ui.label("Intensity transfer");
            ui.horizontal(|ui| {
                ui.radio_value(&mut state.transfer, TransferMode::Linear, "Lin");
                ui.radio_value(&mut state.transfer, TransferMode::Sqrt, "Sqrt");
                ui.radio_value(&mut state.transfer, TransferMode::Log, "Log");
            });
            ui.add(
                egui::Slider::new(&mut state.i_min, 1.0..=1e6)
                    .logarithmic(true)
                    .text("i_min"),
            );
            ui.add(
                egui::Slider::new(&mut state.i_max, 10.0..=1e8)
                    .logarithmic(true)
                    .text("i_max"),
            );
            if state.i_max <= state.i_min {
                state.i_max = state.i_min * 1.0001;
            }
            ui.add(
                egui::Slider::new(&mut state.exposure, 0.001..=5.0)
                    .logarithmic(true)
                    .text("exposure"),
            );

            egui::ComboBox::from_label("colormap")
                .selected_text(COLORMAP_NAMES[state.colormap_id as usize])
                .show_ui(ui, |ui| {
                    for (i, name) in COLORMAP_NAMES.iter().enumerate() {
                        ui.selectable_value(&mut state.colormap_id, i as u32, *name);
                    }
                });
            ui.separator();

            // ---- Appearance ----
            ui.add(egui::Slider::new(&mut state.point_size, 0.5..=12.0).text("point size"));
            ui.add(egui::Slider::new(&mut state.opacity, 0.02..=1.0).text("opacity"));
            ui.separator();

            // ---- MS filter ----
            ui.horizontal(|ui| {
                ui.checkbox(&mut state.show_ms1, "MS1");
                ui.checkbox(&mut state.show_ms2, "MS2");
            });
            ui.checkbox(&mut state.show_annotations, "Selection windows (DIA/MIDIA)");
            if state.show_annotations && state.n_window_groups > 0 {
                let n = state.n_window_groups;
                ui.horizontal(|ui| {
                    ui.label("groups:");
                    if ui.small_button("all").clicked() {
                        state.group_mask = u32::MAX;
                    }
                    if ui.small_button("none").clicked() {
                        state.group_mask = 0;
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    for g in 1..=n.min(32) {
                        let bit = 1u32 << (g - 1);
                        let mut on = state.group_mask & bit != 0;
                        let [r, gg, b] = crate::data::loader::group_color(g, n);
                        let label = egui::RichText::new(format!("{g}")).color(
                            egui::Color32::from_rgb(
                                (r * 255.0) as u8,
                                (gg * 255.0) as u8,
                                (b * 255.0) as u8,
                            ),
                        );
                        if ui.checkbox(&mut on, label).changed() {
                            if on {
                                state.group_mask |= bit;
                            } else {
                                state.group_mask &= !bit;
                            }
                        }
                    }
                });
            }
            ui.checkbox(&mut state.show_axes, "Axis frame + labels");
            ui.separator();

            // ---- Axis windows ----
            ui.label("Windows (real units)");
            axis_window(ui, "RT (s)", &mut state.rt_window, state.bounds.rt.min, state.bounds.rt.max);
            axis_window(ui, "m/z (Th)", &mut state.mz_window, state.bounds.mz.min, state.bounds.mz.max);
            axis_window(ui, "1/K0", &mut state.im_window, state.bounds.im.min, state.bounds.im.max);
            ui.checkbox(&mut state.focus, "Focus to window (zoom in)");
            if ui.button("Reset windows").clicked() {
                state.reset_windows();
            }
            ui.separator();

            // ---- Camera ----
            ui.label("Camera");
            ui.horizontal(|ui| {
                if ui.button("Reset").clicked() {
                    camera.reset();
                }
                let mut ortho = camera.projection == Projection::Orthographic;
                if ui.checkbox(&mut ortho, "Ortho").changed() {
                    camera.projection = if ortho {
                        Projection::Orthographic
                    } else {
                        Projection::Perspective
                    };
                }
            });
            ui.horizontal(|ui| {
                if ui.button("m/z view").clicked() {
                    camera.set_axis_view(AxisView::Mz);
                }
                if ui.button("mob view").clicked() {
                    camera.set_axis_view(AxisView::Mobility);
                }
                if ui.button("RT view").clicked() {
                    camera.set_axis_view(AxisView::Rt);
                }
            });
        });

    // Axis labels are drawn as a screen-space overlay (outside the panel) so they sit at
    // the projected ends of the data cube and rotate with it.
    if state.show_axes {
        draw_axis_labels(ctx, state, camera);
    }
}

/// Project the data cube's axis ends to screen and label them (name + real-unit range),
/// so the orientation of m/z, 1/K0 and RT is always readable.
fn draw_axis_labels(ctx: &egui::Context, state: &AppState, camera: &OrbitCamera) {
    let rect = ctx.screen_rect();
    let (w, h) = (rect.width(), rect.height());
    if w < 1.0 || h < 1.0 {
        return;
    }
    let vp = camera.view_proj(w / h);
    let project = |p: glam::Vec3| -> Option<egui::Pos2> {
        let clip = vp * p.extend(1.0);
        if clip.w <= 1e-4 {
            return None; // behind the camera
        }
        let nx = clip.x / clip.w;
        let ny = clip.y / clip.w;
        Some(egui::pos2((nx * 0.5 + 0.5) * w, (1.0 - (ny * 0.5 + 0.5)) * h))
    };
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("axis_labels"),
    ));
    let font = egui::FontId::proportional(14.0);
    let mut label = |end: glam::Vec3, text: String, color: egui::Color32| {
        if let Some(p) = project(end) {
            // Drop shadow so labels stay legible over a bright cloud.
            painter.text(
                p + egui::vec2(1.0, 1.0),
                egui::Align2::CENTER_CENTER,
                &text,
                font.clone(),
                egui::Color32::from_black_alpha(200),
            );
            painter.text(p, egui::Align2::CENTER_CENTER, text, font.clone(), color);
        }
    };
    // When focused, the cube represents the window, so label it with the window range.
    let (mz, im, rt) = if state.focus {
        (
            (state.mz_window.min, state.mz_window.max),
            (state.im_window.min, state.im_window.max),
            (state.rt_window.min, state.rt_window.max),
        )
    } else {
        (
            (state.bounds.mz.min, state.bounds.mz.max),
            (state.bounds.im.min, state.bounds.im.max),
            (state.bounds.rt.min, state.bounds.rt.max),
        )
    };
    // Each label sits just past the far end of its axis (from the shared min-corner).
    label(
        glam::vec3(1.16, -1.0, -1.0),
        format!("m/z {:.0}–{:.0}", mz.0, mz.1),
        egui::Color32::from_rgb(255, 150, 90),
    );
    label(
        glam::vec3(-1.0, 1.16, -1.0),
        format!("1/K0 {:.2}–{:.2}", im.0, im.1),
        egui::Color32::from_rgb(120, 220, 255),
    );
    label(
        glam::vec3(-1.0, -1.0, 1.16),
        format!("RT {:.0}–{:.0}s", rt.0, rt.1),
        egui::Color32::from_rgb(150, 235, 150),
    );
    // Mark the shared origin (min of all three axes).
    if let Some(p) = project(glam::vec3(-1.0, -1.0, -1.0)) {
        painter.circle_filled(p, 3.0, egui::Color32::from_white_alpha(170));
    }
}

fn axis_window(
    ui: &mut egui::Ui,
    label: &str,
    win: &mut crate::state::Window,
    lo: f64,
    hi: f64,
) {
    ui.label(label);
    ui.add(egui::Slider::new(&mut win.min, lo..=hi).text("min"));
    ui.add(egui::Slider::new(&mut win.max, lo..=hi).text("max"));
    // Keep the range ordered AND non-degenerate: a zero-width window would collapse the
    // focus (zoom-to-window) remap onto a single line via the shader's 1e-6 halfspan clamp.
    let eps = (hi - lo) * 1e-3;
    win.min = win.min.clamp(lo, hi - eps);
    if win.max < win.min + eps {
        win.max = win.min + eps;
    }
    win.max = win.max.min(hi);
}

fn fmt_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}
