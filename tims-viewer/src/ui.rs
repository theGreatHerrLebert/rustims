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
            ui.separator();

            // ---- Axis windows ----
            ui.label("Windows (real units)");
            axis_window(ui, "RT (s)", &mut state.rt_window, state.bounds.rt.min, state.bounds.rt.max);
            axis_window(ui, "m/z (Th)", &mut state.mz_window, state.bounds.mz.min, state.bounds.mz.max);
            axis_window(ui, "1/K0", &mut state.im_window, state.bounds.im.min, state.bounds.im.max);
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
}

fn axis_window(
    ui: &mut egui::Ui,
    label: &str,
    win: &mut crate::state::Window,
    lo: f64,
    hi: f64,
) {
    ui.label(label);
    ui.horizontal(|ui| {
        ui.add(
            egui::DragValue::new(&mut win.min)
                .speed((hi - lo) / 500.0)
                .range(lo..=hi),
        );
        ui.label("–");
        ui.add(
            egui::DragValue::new(&mut win.max)
                .speed((hi - lo) / 500.0)
                .range(lo..=hi),
        );
    });
    if win.min > win.max {
        win.max = win.min;
    }
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
