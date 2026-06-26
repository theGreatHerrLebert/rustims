//! egui control panel.

use crate::camera::{AxisView, OrbitCamera};
use crate::render::colormap::{sample, COLORMAP_NAMES};
use crate::render::point_cloud::PointMode;
use crate::state::{
    AppState, ColorMode, RefineAction, TransferMode, ViewMode, VolStyle, CLUSTER_CAP,
};
use crate::ticks::RT_MINUTES_SPAN;

/// A numeric axis-tick label for the screen-space overlay: a world-space anchor (just
/// outside the cube), its formatted value, and the axis color it is drawn in. Built by
/// `app::axis_geometry` and consumed by `draw_axis_ticks`.
pub struct TickLabel {
    pub world: glam::Vec3,
    pub text: String,
    pub axis_color: egui::Color32,
}

// Axis label colors, shared by the name labels and the per-tick numbers.
const MZ_COL: egui::Color32 = egui::Color32::from_rgb(255, 150, 90);
const IM_COL: egui::Color32 = egui::Color32::from_rgb(120, 220, 255);
const RT_COL: egui::Color32 = egui::Color32::from_rgb(150, 235, 150);

/// Build the left control panel; mutates state and camera in place.
/// `tick_labels` are the numeric axis-tick labels (rebuilt by the app on range change),
/// `vol_range` lazily yields the volume density percentile pair; it is invoked ONLY when
/// the user clicks the "Auto" transfer button, so the (multi-million-voxel) percentile
/// scan never runs on the per-frame render path.
pub fn build(
    ctx: &egui::Context,
    state: &mut AppState,
    camera: &mut OrbitCamera,
    tick_labels: &[TickLabel],
    vol_range: impl FnOnce() -> (f32, f32),
) {
    egui::SidePanel::left("controls")
        .resizable(true)
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.heading("tims-viewer");
            ui.separator();

            // ---- Readouts (always visible) ----
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

            egui::ScrollArea::vertical().show(ui, |ui| {
                filters_section(ui, state);
                selection_section(ui, state);
                rendering_section(ui, state, vol_range);
                clustering_section(ui, state);
                projections_section(ui, state);
                view_section(ui, state, camera);
            });
        });

    // Screen-space overlays (outside the panel): axis names/units + numeric ticks at the
    // projected cube edges, the intensity colorbar, and the group-color legend.
    // Bail once on a degenerate-size frame (window minimize / 0-height transient): the
    // projector would map every element to None anyway, so skip the painter/font/iteration
    // work entirely rather than spinning the full overlay machinery per element.
    let screen = ctx.screen_rect();
    if screen.width() < 1.0 || screen.height() < 1.0 {
        return;
    }
    if state.show_axes {
        draw_axis_labels(ctx, state, camera);
        draw_axis_ticks(ctx, camera, tick_labels);
    }
    if state.show_colorbar {
        draw_colorbar(ctx, state);
    }
    draw_group_legend(ctx, state);
}

/// **Filters**: MS1/MS2 · RT/m-z/1-K0 range sliders · Focus to window · Reset.
fn filters_section(ui: &mut egui::Ui, state: &mut AppState) {
    egui::CollapsingHeader::new("Filters")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut state.show_ms1, "MS1");
                ui.checkbox(&mut state.show_ms2, "MS2");
            });
            ui.separator();
            ui.label("Windows (distribution + range)");
            let (rb, mb, ib) = (state.bounds.rt, state.bounds.mz, state.bounds.im);
            hist_filter(ui, "RT (s)", &mut state.rt_window, rb.min, rb.max, &state.hist_rt, false);
            hist_filter(ui, "m/z (Th)", &mut state.mz_window, mb.min, mb.max, &state.hist_mz, false);
            hist_filter(ui, "1/K0", &mut state.im_window, ib.min, ib.max, &state.hist_im, false);
            let (idlo, idhi) = (state.i_data_lo as f64, state.i_data_hi as f64);
            hist_filter(
                ui,
                "intensity (log)",
                &mut state.intensity_window,
                idlo,
                idhi,
                &state.hist_intensity,
                true,
            );
            ui.checkbox(&mut state.focus, "Focus to window (zoom in)");
            if ui.button("Reset windows").clicked() {
                state.reset_windows();
            }
            ui.separator();
            // Level-of-detail: re-stream just this window at full resolution. "Refine" is
            // always available so you can zoom in further from an already-refined region
            // (nested refinement) without first resetting back to the full run.
            if ui
                .button("⤢ Refine to window (full res)")
                .on_hover_text("Re-stream only this RT / m-z / mobility window at full resolution")
                .clicked()
            {
                state.refine_request = Some(RefineAction::Refine);
            }
            if state.refined {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::LIGHT_GREEN, "● refined");
                    if ui.button("⟲ Back to full run").clicked() {
                        state.refine_request = Some(RefineAction::FullRun);
                    }
                });
            }
        });
}

/// **Selection windows**: show toggle · per-group colored checkboxes · all/none.
fn selection_section(ui: &mut egui::Ui, state: &mut AppState) {
    egui::CollapsingHeader::new("Selection windows")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut state.show_annotations, "Show DIA/MIDIA windows");
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
        });
}

/// **Rendering**: mode · transfer · colormap · appearance.
fn rendering_section(
    ui: &mut egui::Ui,
    state: &mut AppState,
    vol_range: impl FnOnce() -> (f32, f32),
) {
    egui::CollapsingHeader::new("Rendering")
        .default_open(true)
        .show(ui, |ui| {
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
                    ui.add(egui::Slider::new(&mut state.vol_steps, 32..=1024).text("ray steps"));
                }
            }
            ui.separator();

            // ---- Intensity transfer function ----
            ui.horizontal(|ui| {
                ui.label("Intensity transfer");
                // One-click return to auto-exposure: clears the dirty flag and re-applies.
                if ui.button("Auto").clicked() {
                    // Compute the density percentiles lazily, only on this click.
                    state.reset_transfer_auto(vol_range());
                }
            });
            ui.horizontal(|ui| {
                let mut changed = false;
                changed |= ui
                    .radio_value(&mut state.transfer, TransferMode::Linear, "Lin")
                    .changed();
                changed |= ui
                    .radio_value(&mut state.transfer, TransferMode::Sqrt, "Sqrt")
                    .changed();
                changed |= ui
                    .radio_value(&mut state.transfer, TransferMode::Log, "Log")
                    .changed();
                if changed {
                    state.transfer_user_dirty = true;
                }
            });
            if ui
                .add(
                    egui::Slider::new(&mut state.i_min, 1.0..=1e6)
                        .logarithmic(true)
                        .text("i_min"),
                )
                .changed()
            {
                state.transfer_user_dirty = true;
            }
            if ui
                .add(
                    egui::Slider::new(&mut state.i_max, 10.0..=1e8)
                        .logarithmic(true)
                        .text("i_max"),
                )
                .changed()
            {
                state.transfer_user_dirty = true;
            }
            if state.i_max <= state.i_min {
                state.i_max = state.i_min * 1.0001;
            }
            if ui
                .add(
                    egui::Slider::new(&mut state.exposure, 0.001..=5.0)
                        .logarithmic(true)
                        .text("exposure"),
                )
                .changed()
            {
                state.transfer_user_dirty = true;
            }

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
            ui.checkbox(&mut state.show_colorbar, "Colorbar");
        });
}

/// **Clustering**: color points by DBSCAN cluster id (refine to a small region first).
fn clustering_section(ui: &mut egui::Ui, state: &mut AppState) {
    egui::CollapsingHeader::new("Clustering")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("color by:");
                ui.radio_value(&mut state.color_mode, ColorMode::Intensity, "Intensity");
                ui.radio_value(&mut state.color_mode, ColorMode::Cluster, "Cluster");
            });
            ui.add(
                egui::Slider::new(&mut state.cluster_eps, 0.002..=0.1)
                    .logarithmic(true)
                    .text("eps"),
            );
            ui.add(egui::Slider::new(&mut state.cluster_min_pts, 2..=50).text("min pts"));
            // Gate on the FILTERED input size: tighten the intensity/window filters to bring a
            // dense region under the cap instead of having to shrink the spatial window.
            let n = state.cluster_input_count;
            let too_many = n as u32 > CLUSTER_CAP;
            let loading = state.load_progress < 1.0;
            ui.add_enabled_ui(!too_many && n > 0 && !loading, |ui| {
                if ui.button("Cluster (DBSCAN)").clicked() {
                    state.cluster_request = true;
                }
            });
            ui.label(format!("{} points in filter", fmt_count(n as u64)));
            if loading {
                ui.label("loading…");
            } else if too_many {
                ui.colored_label(
                    egui::Color32::from_rgb(230, 180, 80),
                    format!("filter down to ≤ {} to cluster", fmt_count(CLUSTER_CAP as u64)),
                );
            } else if state.cluster_count > 0 || state.cluster_noise > 0 {
                ui.label(format!(
                    "{} clusters · {} noise",
                    state.cluster_count,
                    fmt_count(state.cluster_noise as u64)
                ));
            }
        });
}

/// **Projections**: 2D density minimaps of the full run onto the coordinate planes, with the
/// current filter window drawn as a rectangle ("you are here").
fn projections_section(ui: &mut egui::Ui, state: &mut AppState) {
    if state.proj_mz_im.is_none() {
        return; // nothing loaded yet
    }
    egui::CollapsingHeader::new("Projections (you are here)")
        .default_open(false)
        .show(ui, |ui| {
            let (mz, im, rt) = (state.bounds.mz, state.bounds.im, state.bounds.rt);
            minimap(ui, &state.proj_mz_im, "m/z × 1/K0",
                (mz.min, mz.max), (state.mz_window.min, state.mz_window.max),
                (im.min, im.max), (state.im_window.min, state.im_window.max));
            minimap(ui, &state.proj_mz_rt, "m/z × RT",
                (mz.min, mz.max), (state.mz_window.min, state.mz_window.max),
                (rt.min, rt.max), (state.rt_window.min, state.rt_window.max));
            minimap(ui, &state.proj_im_rt, "1/K0 × RT",
                (im.min, im.max), (state.im_window.min, state.im_window.max),
                (rt.min, rt.max), (state.rt_window.min, state.rt_window.max));
        });
}

/// Draw one projection minimap (`tex`) with the current window drawn as a rectangle. `*b` are
/// the axis bounds, `*w` the current window; the vertical axis low end is at the image bottom.
fn minimap(
    ui: &mut egui::Ui,
    tex: &Option<egui::TextureHandle>,
    label: &str,
    xb: (f64, f64),
    xw: (f64, f64),
    yb: (f64, f64),
    yw: (f64, f64),
) {
    let Some(t) = tex else { return };
    ui.label(label);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(150.0, 150.0), egui::Sense::hover());
    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
    ui.painter().image(t.id(), rect, uv, egui::Color32::WHITE);
    let fr = |v: f64, b: (f64, f64)| (((v - b.0) / (b.1 - b.0).max(1e-12)).clamp(0.0, 1.0)) as f32;
    let x0 = rect.left() + fr(xw.0, xb) * rect.width();
    let x1 = rect.left() + fr(xw.1, xb) * rect.width();
    // Low value at the image bottom; egui screen y grows downward.
    let y_top = rect.bottom() - fr(yw.1, yb) * rect.height();
    let y_bot = rect.bottom() - fr(yw.0, yb) * rect.height();
    let wr = egui::Rect::from_min_max(egui::pos2(x0, y_top), egui::pos2(x1, y_bot));
    ui.painter().rect_stroke(
        wr,
        egui::Rounding::ZERO,
        egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 200, 255)),
    );
}

/// **View / Camera**: axis frame · back-face grid · ortho · m/z|mob|RT snaps.
fn view_section(ui: &mut egui::Ui, state: &mut AppState, camera: &mut OrbitCamera) {
    egui::CollapsingHeader::new("View / Camera")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut state.show_axes, "Axis frame + labels");
            ui.checkbox(&mut state.show_grid_backfaces, "Back-face grid");
            ui.horizontal(|ui| {
                if ui.button("Reset").clicked() {
                    camera.reset();
                }
                // Orthographic projection is hidden until its bugs are sorted; keep perspective.
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

/// World→screen projector for the current camera/viewport (perspective divide + viewport
/// flip). Returns `None` for points behind the camera. Shared by the axis-label, tick, and
/// (future) probe overlays so they all agree on the projection.
pub(crate) fn screen_projector(
    ctx: &egui::Context,
    camera: &OrbitCamera,
) -> impl Fn(glam::Vec3) -> Option<egui::Pos2> {
    let rect = ctx.screen_rect();
    let (w, h) = (rect.width(), rect.height());
    let vp = camera.view_proj(w / h);
    move |p: glam::Vec3| -> Option<egui::Pos2> {
        if w < 1.0 || h < 1.0 {
            return None;
        }
        let clip = vp * p.extend(1.0);
        if clip.w <= 1e-4 {
            return None; // behind the camera
        }
        let nx = clip.x / clip.w;
        let ny = clip.y / clip.w;
        Some(egui::pos2((nx * 0.5 + 0.5) * w, (1.0 - (ny * 0.5 + 0.5)) * h))
    }
}

/// Project the data cube's axis ends to screen and label them (name + units only; the
/// per-tick numbers carry the values), so the orientation of m/z, 1/K0 and RT is readable.
fn draw_axis_labels(ctx: &egui::Context, state: &AppState, camera: &OrbitCamera) {
    let project = screen_projector(ctx, camera);
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("axis_labels"),
    ));
    let font = egui::FontId::proportional(14.0);
    let label = |end: glam::Vec3, text: &str, color: egui::Color32| {
        if let Some(p) = project(end) {
            // Drop shadow so labels stay legible over a bright cloud.
            painter.text(
                p + egui::vec2(1.0, 1.0),
                egui::Align2::CENTER_CENTER,
                text,
                font.clone(),
                egui::Color32::from_black_alpha(200),
            );
            painter.text(p, egui::Align2::CENTER_CENTER, text, font.clone(), color);
        }
    };
    // RT span chooses the unit suffix (seconds vs minutes) to match the tick formatting.
    let rt_span = if state.focus {
        (state.rt_window.max - state.rt_window.min).abs()
    } else {
        (state.bounds.rt.max - state.bounds.rt.min).abs()
    };
    let rt_unit = if rt_span <= RT_MINUTES_SPAN {
        "RT (s)"
    } else {
        "RT (min)"
    };
    // Each label sits just past the far end of its axis (from the shared min-corner).
    label(glam::vec3(1.16, -1.0, -1.0), "m/z (Th)", MZ_COL);
    label(glam::vec3(-1.0, 1.16, -1.0), "1/K0", IM_COL);
    label(glam::vec3(-1.0, -1.0, 1.16), rt_unit, RT_COL);
    // Mark the shared origin (min of all three axes).
    if let Some(p) = project(glam::vec3(-1.0, -1.0, -1.0)) {
        painter.circle_filled(p, 3.0, egui::Color32::from_white_alpha(170));
    }
}

/// Draw the numeric axis-tick values at their projected world anchors, colored per axis.
fn draw_axis_ticks(ctx: &egui::Context, camera: &OrbitCamera, ticks: &[TickLabel]) {
    if ticks.is_empty() {
        return;
    }
    let project = screen_projector(ctx, camera);
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("axis_ticks"),
    ));
    let font = egui::FontId::proportional(11.0);
    for t in ticks {
        if let Some(p) = project(t.world) {
            painter.text(
                p + egui::vec2(1.0, 1.0),
                egui::Align2::CENTER_CENTER,
                &t.text,
                font.clone(),
                egui::Color32::from_black_alpha(200),
            );
            painter.text(
                p,
                egui::Align2::CENTER_CENTER,
                &t.text,
                font.clone(),
                t.axis_color,
            );
        }
    }
}

/// Vertical intensity colorbar (top-right): the active colormap gradient, i_min/i_max end
/// labels, and a transfer-mode tag. Pure screen-space; reflects the active colormap/transfer.
fn draw_colorbar(ctx: &egui::Context, state: &AppState) {
    let screen = ctx.screen_rect();
    if screen.width() < 80.0 || screen.height() < 220.0 {
        return; // too small to place legibly
    }
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("colorbar"),
    ));
    const BAR_W: f32 = 18.0;
    const BAR_H: f32 = 180.0;
    const INSET: f32 = 16.0;
    let right = screen.right() - INSET;
    let top = screen.top() + INSET + 16.0; // leave room for the transfer tag above
    let bar = egui::Rect::from_min_max(
        egui::pos2(right - BAR_W, top),
        egui::pos2(right, top + BAR_H),
    );

    // Gradient: ~64 stacked slices, t=1 (i_max) at the top.
    const SLICES: usize = 64;
    let map = state.colormap_id as usize;
    for i in 0..SLICES {
        let t = i as f32 / (SLICES - 1) as f32;
        let rgb = sample(map, t);
        let y1 = bar.bottom() - (i as f32 / SLICES as f32) * BAR_H;
        let y0 = bar.bottom() - ((i + 1) as f32 / SLICES as f32) * BAR_H;
        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(bar.left(), y0), egui::pos2(bar.right(), y1)),
            0.0,
            egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
        );
    }
    painter.rect_stroke(
        bar,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(120)),
    );

    let font = egui::FontId::proportional(11.0);
    let text = |pos: egui::Pos2, anchor: egui::Align2, s: String| {
        painter.text(
            pos + egui::vec2(1.0, 1.0),
            anchor,
            &s,
            font.clone(),
            egui::Color32::from_black_alpha(200),
        );
        painter.text(pos, anchor, s, font.clone(), egui::Color32::WHITE);
    };
    // End labels just left of the bar.
    text(
        egui::pos2(bar.left() - 4.0, bar.top()),
        egui::Align2::RIGHT_CENTER,
        fmt_intensity(state.i_max),
    );
    text(
        egui::pos2(bar.left() - 4.0, bar.bottom()),
        egui::Align2::RIGHT_CENTER,
        fmt_intensity(state.i_min),
    );
    // Transfer tag above the bar (density suffix in volume mode, where the range comes
    // from the grid percentiles rather than per-point intensity).
    let mode = match state.transfer {
        TransferMode::Linear => "lin",
        TransferMode::Sqrt => "sqrt",
        TransferMode::Log => "log",
    };
    let tag = if state.view_mode == ViewMode::Volume {
        format!("{mode} (density)")
    } else {
        mode.to_string()
    };
    text(
        egui::pos2(bar.right(), bar.top() - 14.0),
        egui::Align2::RIGHT_CENTER,
        tag,
    );
}

/// Format an intensity/density value with k/M suffixes. Fractional values below 1 (volume
/// density transfer ends) keep decimals so the colorbar legend doesn't collapse them to "0".
fn fmt_intensity(v: f32) -> String {
    let v = v as f64;
    if v >= 1_000_000.0 {
        format!("{:.1}M", v / 1e6)
    } else if v >= 1_000.0 {
        format!("{:.1}k", v / 1e3)
    } else if v >= 1.0 {
        format!("{v:.0}")
    } else if v >= 0.01 {
        format!("{v:.2}")
    } else if v > 0.0 {
        // Very small density floors (e.g. i_min = 1e-6) -> compact scientific.
        format!("{v:.0e}")
    } else {
        "0".to_string()
    }
}

/// Compact legend mapping window-group number → swatch (same `group_color` as the boxes).
/// Only shown when selection windows are active.
fn draw_group_legend(ctx: &egui::Context, state: &AppState) {
    if !state.show_annotations || state.n_window_groups == 0 {
        return;
    }
    let screen = ctx.screen_rect();
    if screen.width() < 80.0 || screen.height() < 120.0 {
        return;
    }
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("group_legend"),
    ));

    // Visible groups (mirror the overlay's per-group mask: only groups whose bit is set).
    let mask = state.group_mask;
    let visible: Vec<u32> = (1..=state.n_window_groups.min(32))
        .filter(|&g| mask & (1u32 << (g - 1)) != 0)
        .collect();
    if visible.is_empty() {
        return;
    }

    const SWATCH: f32 = 11.0;
    const ROW_H: f32 = 15.0;
    const COL_W: f32 = 38.0;
    const PAD: f32 = 6.0;
    const ROWS_PER_COL: usize = 16;
    let title_h = 16.0;
    let n = visible.len();
    let cols = ((n + ROWS_PER_COL - 1) / ROWS_PER_COL).max(1);
    let rows = n.min(ROWS_PER_COL);
    let panel_w = cols as f32 * COL_W + PAD * 2.0;
    let panel_h = title_h + rows as f32 * ROW_H + PAD * 2.0;

    // Bottom-right, below where the colorbar sits.
    let right = screen.right() - 16.0;
    let bottom = screen.bottom() - 16.0;
    let panel = egui::Rect::from_min_max(
        egui::pos2(right - panel_w, bottom - panel_h),
        egui::pos2(right, bottom),
    );
    // Translucent backdrop for contrast over the cloud.
    painter.rect_filled(panel, 4.0, egui::Color32::from_black_alpha(120));

    let font = egui::FontId::proportional(11.0);
    painter.text(
        egui::pos2(panel.left() + PAD, panel.top() + PAD),
        egui::Align2::LEFT_TOP,
        "groups",
        font.clone(),
        egui::Color32::from_gray(210),
    );

    for (idx, &g) in visible.iter().enumerate() {
        let col = idx / ROWS_PER_COL;
        let row = idx % ROWS_PER_COL;
        let x = panel.left() + PAD + col as f32 * COL_W;
        let y = panel.top() + PAD + title_h + row as f32 * ROW_H;
        let [r, gg, b] = crate::data::loader::group_color(g, state.n_window_groups);
        let sw = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(SWATCH, SWATCH));
        painter.rect_filled(
            sw,
            0.0,
            egui::Color32::from_rgb((r * 255.0) as u8, (gg * 255.0) as u8, (b * 255.0) as u8),
        );
        painter.text(
            egui::pos2(x + SWATCH + 4.0, y + SWATCH * 0.5),
            egui::Align2::LEFT_CENTER,
            format!("{g}"),
            font.clone(),
            egui::Color32::WHITE,
        );
    }
}

/// A range filter backed by a distribution histogram (image-"levels" style): a strip of bars
/// with the selected range highlighted, above min/max sliders. `log` scales both the slider
/// and the value→x mapping logarithmically (for the intensity axis). `hist` may be empty
/// (before data lands) — then only the sliders show.
fn hist_filter(
    ui: &mut egui::Ui,
    label: &str,
    win: &mut crate::state::Window,
    lo: f64,
    hi: f64,
    hist: &[u32],
    log: bool,
) {
    ui.label(label);
    if !hist.is_empty() && hi > lo {
        let height = 26.0;
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, egui::Rounding::same(2.0), egui::Color32::from_gray(22));
        // Value -> x fraction in [0,1], honoring the log axis.
        let to_frac = |v: f64| -> f32 {
            let f = if log {
                (v.max(lo).ln() - lo.ln()) / (hi.ln() - lo.ln())
            } else {
                (v - lo) / (hi - lo)
            };
            f.clamp(0.0, 1.0) as f32
        };
        let (wmin, wmax) = (to_frac(win.min), to_frac(win.max));
        let maxc = hist.iter().copied().max().unwrap_or(1).max(1) as f32;
        let n = hist.len();
        let bw = rect.width() / n as f32;
        for (i, &c) in hist.iter().enumerate() {
            let frac = (i as f32 + 0.5) / n as f32;
            // sqrt keeps low-count bins visible without flattening the peaks.
            let bh = (c as f32 / maxc).sqrt() * (height - 2.0);
            let x0 = rect.left() + i as f32 * bw;
            let bar = egui::Rect::from_min_max(
                egui::pos2(x0, rect.bottom() - bh),
                egui::pos2(x0 + bw.max(1.0), rect.bottom()),
            );
            let color = if frac >= wmin && frac <= wmax {
                egui::Color32::from_rgb(95, 170, 240)
            } else {
                egui::Color32::from_gray(72)
            };
            painter.rect_filled(bar, egui::Rounding::ZERO, color);
        }
    }
    let mut smin = egui::Slider::new(&mut win.min, lo..=hi).text("min");
    let mut smax = egui::Slider::new(&mut win.max, lo..=hi).text("max");
    if log {
        smin = smin.logarithmic(true);
        smax = smax.logarithmic(true);
    }
    ui.add(smin);
    ui.add(smax);
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
