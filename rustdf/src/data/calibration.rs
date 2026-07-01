//! SDK-free Bruker timsTOF axis calibration.
//!
//! Pure-Rust ports of the calibration formulas Bruker publishes for the TDF
//! format, so that TOF-index -> m/z and scan -> 1/K0 can be computed without
//! loading the proprietary `libtimsdata` SDK. The algorithms mirror the
//! implementation in PAPPSO's GPL library `libpappsomspp`
//! (`mzcalibrationmodel1.cpp`, `timsframebase.cpp`); the coefficient meanings
//! are cross-checked against Bruker's own `tims_calibration.py` reference.
//!
//! Two independent models are involved, each carrying its own `ModelType`:
//!   * m/z    : `MzCalibration` table (this data set: ModelType 2)
//!   * 1/K0   : `TimsCalibration` table (this data set: ModelType 2)
//!
//! IMPORTANT (m/z): PAPPSO only implements m/z *ModelType 1*. Modern instruments
//! write *ModelType 2*, whose first coefficients (C0,C1) still describe the same
//! `t = C0 + sqrt(1e12/C1)*sqrt(m)` base curve, but which adds a degree-6
//! correction polynomial (C8..C14) that neither PAPPSO nor this module models.
//! We therefore reproduce the *base* curve exactly (few-ppm agreement with the
//! SDK) and, for genuine ModelType-1 data, the full cubic-in-sqrt(m) curve.

/// m/z axis calibration (Bruker "model type 1" base curve + optional cubic).
///
/// Flight time from a TOF index:            `t = index * timebase + delay`
/// Calibration curve (time as fn of mass):  `t = C0 + b*s + c2*s^2 + c3*s^3`
/// with `s = sqrt(m + c4)` and `b = sqrt(1e12 / C1_tempcomp)`.
///
/// Coefficient meaning (columns of the `MzCalibration` table):
/// * `timebase` = `DigitizerTimebase` — ns per digitizer sample.
/// * `delay`    = `DigitizerDelay`    — fixed time offset (samples) before t0.
/// * `C0`       — constant term of the time/mass curve (~ the t-intercept).
/// * `C1`       — governs the dominant sqrt term; `b = sqrt(1e12 / C1)`.
/// * `c2`       — quadratic term `C2*s^2` of the curve; used by BOTH models.
/// * `c3`       — cubic term `C3*s^3`; ModelType 1 only (in ModelType 2 the C3
///               column is a duplicate of C0 and is dropped).
/// * `c4`       — "reduced mass" shift m0 (`x = m - m0`); patent US7,851,746.
/// Temperature compensation (`T1/T2` = reference temps in `MzCalibration`,
/// `dC1/dC2` its sensitivities, `T1f/T2f` = per-frame `Frames.T1/Frames.T2`):
/// `tc = 1 + (dC1*(T1-T1f) + dC2*(T2-T2f)) / 1e6`, applied as `C1 *= tc`.
#[derive(Debug, Clone)]
pub struct MzCalibrator {
    pub timebase: f64,
    pub delay: f64,
    pub c0: f64,
    pub b: f64, // sqrt(1e12 / (C1 * tc))
    pub c2: f64,
    pub c3: f64,
    pub c4: f64,
}

impl MzCalibrator {
    /// Build a calibrator from raw `MzCalibration` columns + per-frame temps.
    ///
    /// `model_type` selects whether the C2/C3 curve terms are honoured (type 1)
    /// or zeroed (type 2, base curve only).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_type: i64,
        timebase: f64,
        delay: f64,
        t1_ref: f64,
        t2_ref: f64,
        dc1: f64,
        dc2: f64,
        c0: f64,
        c1: f64,
        c2: f64,
        c3: f64,
        c4: f64,
        t1_frame: f64,
        t2_frame: f64,
    ) -> Self {
        let tc = 1.0 + (dc1 * (t1_ref - t1_frame) + dc2 * (t2_ref - t2_frame)) / 1.0e6;
        let b = (1.0e12 / (c1 * tc)).sqrt();
        // Both models share the quadratic-in-sqrt(m) curve `t = C0 + b*s + C2*s^2`
        // (empirically: a0->C0, a1->sqrt(1e12/C1), a2->C2). The cubic `C3*s^3`
        // term is real only for ModelType 1; in ModelType 2 the C3 column is a
        // duplicate of C0 and must be dropped. ModelType 2 additionally carries a
        // C8..C14 fine correction (~few ppm, worst at low m/z) that is NOT an
        // additive polynomial in m/z and is left unmodelled here.
        let c2 = c2 / tc;
        let c3 = if model_type == 1 { c3 } else { 0.0 };
        Self { timebase, delay, c0, b, c2, c3, c4 }
    }

    /// Flight time (digitizer units) for a TOF index.
    #[inline]
    fn tof_index_to_time(&self, tof_index: f64) -> f64 {
        tof_index * self.timebase + self.delay
    }

    /// TOF index -> m/z. Inverts `t = C0 + b*s + c2*s^2 + c3*s^3` for `s`.
    pub fn tof_to_mz(&self, tof_index: u32) -> f64 {
        let t = self.tof_index_to_time(tof_index as f64);
        // Base (linear-in-sqrt) closed form, exact when c2 = c3 = 0.
        let mut s = (t - self.c0) / self.b;
        if self.c2 != 0.0 || self.c3 != 0.0 {
            // Newton refinement for the ModelType-1 cubic curve.
            for _ in 0..8 {
                let f = self.c0 + self.b * s + self.c2 * s * s + self.c3 * s * s * s - t;
                let df = self.b + 2.0 * self.c2 * s + 3.0 * self.c3 * s * s;
                if df == 0.0 {
                    break;
                }
                let step = f / df;
                s -= step;
                if step.abs() < 1e-12 {
                    break;
                }
            }
        }
        s * s - self.c4
    }

    /// m/z -> TOF index (forward direction, always closed form).
    pub fn mz_to_tof(&self, mz: f64) -> u32 {
        let s = (mz + self.c4).max(0.0).sqrt();
        let t = self.c0 + self.b * s + self.c2 * s * s + self.c3 * s * s * s;
        (((t - self.delay) / self.timebase).round()).max(0.0) as u32
    }
}

/// Ion-mobility axis calibration (Bruker "model type 2", the only TIMS model).
///
/// Two steps, both exact ports of PAPPSO `timsframebase.cpp`:
///   1. scan -> trapping voltage:  `V = dv_start + slope*(scan - ttrans - ndelay)`
///      with `slope = (dv_end - dv_start) / ncycles`.  V must lie in [vmin,vmax].
///   2. voltage -> inverse mobility: `1/K0 = 1 / (C0m + C1m / V)`.
///
/// Coefficient meaning (columns of the `TimsCalibration` table, ModelType 2):
/// * `C0` = `ndelay`   — scan offset (delay), subtracted before scaling.
/// * `C1` = `ncycles`  — number of TIMS cycles; sets the voltage-vs-scan slope.
/// * `C2` = `dv_start` — trapping voltage at the start of the ramp.
/// * `C3` = `dv_end`   — trapping voltage at the end of the ramp.
/// * `C4` = `ttrans`   — transit time in cycles, subtracted before scaling.
/// * `C5`              — unused by the mobility formula (polynomial grade flag).
/// * `C6` = `C0m`      — additive constant of the mobility reciprocal.
/// * `C7` = `C1m`      — voltage-scaled term of the mobility reciprocal.
/// * `C8` = `vmin`     — lower voltage validity bound.
/// * `C9` = `vmax`     — upper voltage validity bound.
#[derive(Debug, Clone)]
pub struct MobilityCalibrator {
    pub ndelay: f64,
    pub dv_start: f64,
    pub ttrans: f64,
    pub c0m: f64,
    pub c1m: f64,
    pub vmin: f64,
    pub vmax: f64,
    pub slope: f64,
}

impl MobilityCalibrator {
    /// Build from raw `TimsCalibration` C0..C9 (ModelType must be 2).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c0: f64,
        c1: f64,
        c2: f64,
        c3: f64,
        c4: f64,
        _c5: f64,
        c6: f64,
        c7: f64,
        c8: f64,
        c9: f64,
    ) -> Self {
        Self {
            ndelay: c0,
            dv_start: c2,
            ttrans: c4,
            c0m: c6,
            c1m: c7,
            vmin: c8,
            vmax: c9,
            slope: (c3 - c2) / c1,
        }
    }

    /// scan index -> trapping voltage (clamped to the valid window).
    #[inline]
    fn voltage(&self, scan: f64) -> f64 {
        let v = self.dv_start + self.slope * (scan - self.ttrans - self.ndelay);
        v.clamp(self.vmin, self.vmax)
    }

    /// scan index -> 1/K0 (inverse reduced ion mobility).
    pub fn scan_to_one_over_k0(&self, scan: u32) -> f64 {
        1.0 / (self.c0m + self.c1m / self.voltage(scan as f64))
    }

    /// 1/K0 -> nearest scan index (exact algebraic inverse, then round).
    pub fn one_over_k0_to_scan(&self, one_over_k0: f64) -> u32 {
        // invert 1/K0 = 1/(C0m + C1m/V)  ->  V,  then V -> scan
        let inv = 1.0 / one_over_k0;
        let v = self.c1m / (inv - self.c0m);
        let scan = (v - self.dv_start) / self.slope + self.ttrans + self.ndelay;
        scan.round().max(0.0) as u32
    }
}
