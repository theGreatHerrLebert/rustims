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
///               ModelType 1 only (in ModelType 2 the C4 column is a duplicate of
///               C2 and is dropped, exactly like C3).
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
        // ModelType 2 reuses the C3 *and* C4 columns as duplicates of C0/C2
        // (verified bit-for-bit on real files: C3 == C0, C4 == C2), so neither is
        // the cubic term nor the reduced-mass shift there and both must be
        // dropped. Subtracting the duplicated C4 as if it were m0 costs ~1.4 ppm
        // mean / 6 ppm max against the SDK on ModelType-2 data.
        let (c3, c4) = if model_type == 1 { (c3, c4) } else { (0.0, 0.0) };
        Self { timebase, delay, c0, b, c2, c3, c4 }
    }

    /// Build a calibrator straight from an `MzCalibration` row plus the
    /// per-frame digitizer temperatures (`Frames.T1`, `Frames.T2`).
    pub fn from_calibration(
        cal: &crate::data::meta::MzCalibration,
        t1_frame: f64,
        t2_frame: f64,
    ) -> Self {
        Self::new(
            cal.model_type,
            cal.digitizer_timebase,
            cal.digitizer_delay,
            cal.t1,
            cal.t2,
            cal.dc1,
            cal.dc2,
            cal.c0,
            cal.c1,
            cal.c2,
            cal.c3,
            cal.c4,
            t1_frame,
            t2_frame,
        )
    }

    /// Flight time (digitizer units) for a TOF index.
    #[inline]
    fn tof_index_to_time(&self, tof_index: f64) -> f64 {
        tof_index * self.timebase + self.delay
    }

    /// TOF index -> m/z. Inverts `t = C0 + b*s + c2*s^2 + c3*s^3` for `s`.
    pub fn tof_to_mz(&self, tof_index: u32) -> f64 {
        let t = self.tof_index_to_time(tof_index as f64);
        // Linear-in-sqrt estimate; exact when c2 = c3 = 0.
        let s0 = (t - self.c0) / self.b;
        let s = if self.c3 != 0.0 {
            // ModelType-1 cubic: Newton refinement from the linear estimate.
            let mut s = s0;
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
            s
        } else if self.c2 != 0.0 {
            // ModelType-2 quadratic `c2*s^2 + b*s + (c0 - t) = 0`, solved in the
            // numerically stable ("citardauq") form so the physical root does not
            // lose precision to cancellation when |c2| is tiny. b > 0 always, so
            // q < 0 and is never zero. Falls back to the linear estimate if the
            // discriminant is negative (out-of-range tof).
            let disc = self.b * self.b - 4.0 * self.c2 * (self.c0 - t);
            if disc < 0.0 {
                s0
            } else {
                let q = -0.5 * (self.b + disc.sqrt());
                (self.c0 - t) / q
            }
        } else {
            s0
        };
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

    /// Build a mobility calibrator straight from a `TimsCalibration` row.
    pub fn from_calibration(cal: &crate::data::meta::TimsCalibration) -> Self {
        Self::new(
            cal.c0, cal.c1, cal.c2, cal.c3, cal.c4, cal.c5, cal.c6, cal.c7, cal.c8, cal.c9,
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `synchro-hela.d`: `MzCalibration` ModelType 1, frame 1 temperatures.
    fn model1() -> MzCalibrator {
        MzCalibrator::new(
            1,
            0.2,
            25585.2,
            25.432518231649194,
            22.91046717419208,
            21.0,
            0.0,
            319.4836862507276,
            156650.78463959479,
            -4.7797946742077594e-05,
            0.0,
            2.6639979911791643e-05,
            25.454778878245186,
            23.065621750666164,
        )
    }

    /// `G8602.d`: `MzCalibration` ModelType 2, frame 1 temperatures. Note that
    /// C3 duplicates C0 and C4 duplicates C2 in this model — the calibrator has
    /// to ignore both.
    fn model2() -> MzCalibrator {
        MzCalibrator::new(
            2,
            0.2,
            18290.2,
            25.3488367593942,
            25.618275892137202,
            20.0,
            0.0,
            319.95445850882476,
            154831.91077331622,
            -0.0005550791433026118,
            319.95445850882476,
            -0.0005550791433026118,
            25.37849141449188,
            26.171382195221447,
        )
    }

    fn ppm(got: f64, want: f64) -> f64 {
        (got - want).abs() / want * 1e6
    }

    /// ModelType 1 is fully modelled, so it must reproduce
    /// `tims_index_to_mz` bit-for-bit (reference values read from the SDK).
    #[test]
    fn model1_is_bit_exact_against_the_sdk() {
        let cal = model1();
        for (tof, sdk) in [
            (1u32, 100.00058181811438),
            (50000, 194.8219835433269),
            (150000, 478.458543155091),
            (250000, 887.4159830461258),
            (350000, 1421.6944158188035),
        ] {
            let got = cal.tof_to_mz(tof);
            assert!(
                (got - sdk).abs() / sdk < 1e-12,
                "tof {tof}: got {got}, SDK {sdk}"
            );
        }
    }

    /// In ModelType 2 the C3/C4 columns are duplicates of C0/C2, not the cubic
    /// term and the reduced-mass shift, so the calibrator must zero both.
    #[test]
    fn model2_drops_duplicated_c3_and_c4() {
        let cal = model2();
        assert_eq!(cal.c3, 0.0);
        assert_eq!(cal.c4, 0.0);
    }

    /// The ModelType-2 fine correction is windowed to `[C5, C6]` in m/z, so
    /// *below* that window (here C5 = 225.95) the C0/C1/C2 base curve is the
    /// whole model and must match the SDK exactly. This is what regresses --
    /// by 4.6 to 11.1 ppm -- if the duplicated C4 is subtracted as if it were m0.
    #[test]
    fn model2_is_exact_below_the_correction_window() {
        let cal = model2();
        for (tof, sdk) in [(1u32, 50.00106408611359), (50000, 121.13087702783326)] {
            let got = cal.tof_to_mz(tof);
            assert!(ppm(got, sdk) < 0.01, "tof {tof}: got {got}, SDK {sdk}");
        }
    }

    /// Inside the correction window the unmodelled C8..C14 polynomial leaves a
    /// small residual; it stays within a few ppm of the SDK.
    #[test]
    fn model2_is_within_a_few_ppm_inside_the_correction_window() {
        let cal = model2();
        for (tof, sdk) in [
            (150000u32, 356.29356381674006),
            (250000, 715.3229449956262),
            (350000, 1198.2257163072832),
        ] {
            let got = cal.tof_to_mz(tof);
            assert!(ppm(got, sdk) < 3.0, "tof {tof}: got {got}, SDK {sdk}");
        }
    }

    #[test]
    fn mz_tof_round_trips() {
        for cal in [model1(), model2()] {
            for tof in [1u32, 50_000, 150_000, 250_000, 350_000] {
                let back = cal.mz_to_tof(cal.tof_to_mz(tof));
                assert!(
                    back.abs_diff(tof) <= 1,
                    "round trip {tof} -> {} -> {back}",
                    cal.tof_to_mz(tof)
                );
            }
        }
    }

    /// `synchro-hela.d` `TimsCalibration` (ModelType 2) against
    /// `tims_scannum_to_oneoverk0`: the mobility model is the SDK computation,
    /// so agreement is at the floating-point rounding limit.
    #[test]
    fn mobility_matches_the_sdk() {
        let cal = MobilityCalibrator::new(
            1.0,
            926.0,
            174.99089000590644,
            89.34134412781896,
            33.333333333333336,
            1.0,
            0.031163511286580903,
            129.15504635187241,
            12.75853947905552,
            3135.217073581985,
        );
        for (scan, sdk) in [
            (0u32, 1.3226192435624091),
            (1, 1.321960900558156),
            (100, 1.2566451818439914),
            (400, 1.0570143319893166),
            (900, 0.7184888610899344),
        ] {
            let got = cal.scan_to_one_over_k0(scan);
            assert!(
                (got - sdk).abs() / sdk < 1e-12,
                "scan {scan}: got {got}, SDK {sdk}"
            );
            assert_eq!(cal.one_over_k0_to_scan(sdk), scan);
        }
    }
}
