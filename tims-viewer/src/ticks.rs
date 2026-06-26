//! "Nice" axis tick selection: 1/2/5 × 10^k steps, 4–6 ticks across a range.
//!
//! Pure, testable tick math with no egui/wgpu deps. The renderer (`app.rs`) and the
//! overlay (`ui.rs`) consume these to draw 3D tick marks plus numeric labels.

/// A computed tick: real-unit value plus its normalized cube coordinate in [-1,1].
#[derive(Clone, Copy, Debug)]
pub struct Tick {
    pub value: f64,
    pub norm: f32,
}

/// Axis identity, for per-axis label formatting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Mz,
    Im,
    Rt,
}

/// RT spans wider than this (seconds) are labeled in minutes instead of seconds.
pub const RT_MINUTES_SPAN: f64 = 600.0;

/// Choose a "nice" step so that [lo,hi] is covered by ~4–6 ticks.
/// Step ∈ {1,2,5}·10^k. Returns the step size (>0). `target` is the desired tick count.
pub fn nice_step(lo: f64, hi: f64, target: u32) -> f64 {
    let span = (hi - lo).abs();
    if span <= 0.0 || !span.is_finite() {
        return 1.0; // degenerate guard
    }
    let raw = span / target.max(1) as f64; // ideal step if steps were continuous
    let mag = 10f64.powf(raw.log10().floor()); // power-of-ten below raw
    let norm = raw / mag; // in [1,10)
    let nice = if norm < 1.5 {
        1.0
    } else if norm < 3.0 {
        2.0
    } else if norm < 7.0 {
        5.0
    } else {
        10.0
    };
    nice * mag
}

/// Generate ticks across [lo,hi] (auto-orders lo/hi). First tick = ceil(lo/step)*step,
/// stepping by `step` while <= hi (+epsilon). Each tick's `norm` is computed by `to_norm`
/// (pass the axis's AxisTransform::normalize, or a window-aware normalizer).
pub fn ticks_for<F: Fn(f64) -> f32>(lo: f64, hi: f64, target: u32, to_norm: F) -> Vec<Tick> {
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    let step = nice_step(lo, hi, target);
    let first = (lo / step).ceil() * step; // first tick >= lo on the grid
    let eps = step * 1e-6; // avoid dropping the last tick to fp error
    let mut out = Vec::new();
    let mut v = first;
    while v <= hi + eps {
        // snap near-zero to exactly 0 so formatting shows "0" not "-0"/"1e-13".
        let vs = if v.abs() < eps { 0.0 } else { v };
        out.push(Tick {
            value: vs,
            norm: to_norm(vs),
        });
        v += step;
        if out.len() > 64 {
            break; // hard safety cap
        }
    }
    out
}

/// Decimal places needed to distinguish adjacent ticks at the given step size.
/// A step of 200 needs 0; 0.2 needs 1; 0.02 needs 2, etc. Clamped to [0, 6].
fn decimals_for_step(step: f64) -> usize {
    if !step.is_finite() || step <= 0.0 {
        return 0;
    }
    // Digits after the point so the step is resolvable (e.g. step 0.2 -> 1, 0.05 -> 2).
    let d = -step.log10().floor();
    (d.max(0.0) as usize).min(6)
}

/// Format a tick value for the given axis and the visible span (span chooses RT units and,
/// together with the tick count, the decimal precision so adjacent ticks render distinctly).
pub fn fmt_tick(axis: Axis, value: f64, span: f64) -> String {
    // Recover the per-axis step from the span so a narrow focus window (sub-unit steps)
    // gets enough decimals; ~5 ticks across the span matches `ticks_for`'s target.
    let step = nice_step(0.0, span, 5);
    match axis {
        // m/z: integer Th normally, but a narrow focus window can need sub-Th precision.
        Axis::Mz => format!("{:.*}", decimals_for_step(step), value),
        // 1/K0: at least 2 decimals (its native scale), more if the window is very narrow.
        Axis::Im => format!("{:.*}", decimals_for_step(step).max(2), value),
        Axis::Rt => {
            if span <= RT_MINUTES_SPAN {
                format!("{:.*}", decimals_for_step(step), value)
            } else {
                // Minutes: span/60 sets the resolvable precision (floor at 1 decimal).
                format!("{:.*}", decimals_for_step(step / 60.0).max(1), value / 60.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Is `v` of the form {1,2,5}·10^k (within fp tolerance)?
    fn is_nice(v: f64) -> bool {
        if v <= 0.0 || !v.is_finite() {
            return false;
        }
        let mag = 10f64.powf(v.log10().floor());
        let mantissa = v / mag;
        [1.0, 2.0, 5.0, 10.0]
            .iter()
            .any(|m| (mantissa - m).abs() < 1e-6)
    }

    #[test]
    fn nice_step_is_one_two_five() {
        for &(lo, hi) in &[(0.0, 1600.0), (0.0, 0.8), (0.0, 12.0), (100.0, 1700.0)] {
            let s = nice_step(lo, hi, 5);
            assert!(is_nice(s), "step {s} for [{lo},{hi}] is not 1/2/5·10^k");
        }
        // span 1600 / 5 = 320 -> raw mag 100, norm 3.2 -> 5 -> 500.
        assert_eq!(nice_step(0.0, 1600.0, 5), 500.0);
        // span 0.8 / 5 = 0.16 -> raw mag 0.1, norm 1.6 -> 2 -> 0.2.
        assert!((nice_step(0.0, 0.8, 5) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn ticks_span_range_monotonic() {
        let ticks = ticks_for(100.0, 1700.0, 5, |v| v as f32);
        // target=5 with the {1,2,5} snap yields ~4–6; a span landing on a step boundary
        // (here step=500 over [100,1700] -> 500/1000/1500) bottoms out at 3, which the
        // spec documents as acceptable and the 64-cap bounds the top end.
        assert!(
            (3..=7).contains(&ticks.len()),
            "expected 3–7 ticks, got {}",
            ticks.len()
        );
        let step = nice_step(100.0, 1700.0, 5);
        let first = (100.0_f64 / step).ceil() * step;
        assert!((ticks[0].value - first).abs() < 1e-6);
        for t in &ticks {
            assert!(t.value >= 100.0 - 1e-6 && t.value <= 1700.0 + 1e-6);
        }
        for w in ticks.windows(2) {
            assert!(w[1].value > w[0].value, "ticks not increasing");
        }
    }

    #[test]
    fn degenerate_range_does_not_loop() {
        let ticks = ticks_for(5.0, 5.0, 5, |v| v as f32);
        assert!(!ticks.is_empty());
        assert!(ticks.iter().all(|t| t.value.is_finite()));
        assert!(ticks.len() <= 65);
    }

    #[test]
    fn fmt_tick_per_axis() {
        assert_eq!(fmt_tick(Axis::Mz, 412.7, 1600.0), "413");
        assert_eq!(fmt_tick(Axis::Im, 1.234, 1.0), "1.23");
        // RT seconds branch (span <= 600).
        assert_eq!(fmt_tick(Axis::Rt, 120.0, 300.0), "120");
        // RT minutes branch (span > 600): 1800 s -> 30.0 min.
        assert_eq!(fmt_tick(Axis::Rt, 1800.0, 3600.0), "30.0");
    }

    #[test]
    fn fmt_tick_narrow_window_gains_precision() {
        // Narrow m/z focus window (~2 Th span -> step 0.5): adjacent ticks must NOT
        // collapse to the same integer label.
        let a = fmt_tick(Axis::Mz, 412.4, 2.0);
        let b = fmt_tick(Axis::Mz, 412.6, 2.0);
        assert_ne!(a, b, "narrow m/z ticks collapsed to the same label: {a} == {b}");
        assert_eq!(a, "412.4");
        assert_eq!(b, "412.6");
        // Narrow 1/K0 window (~0.04 span -> step 0.01): needs 2 decimals minimum.
        assert_ne!(fmt_tick(Axis::Im, 0.812, 0.04), fmt_tick(Axis::Im, 0.818, 0.04));
    }
}
