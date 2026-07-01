// Head-to-head accuracy of every rustdf index-converter mode vs the Bruker SDK.
//
// Modes compared (see src/data/handle.rs):
//   Simple      - 2-point boundary model (mz: sqrt-linear from mz bounds;
//                 1/K0: linear from 1/K0 bounds). No SDK.
//   Calibrated  - mz: least-squares sqrt(mz)=a+b*tof fit (needs SDK to derive);
//                 1/K0: same linear boundary model.
//   Lookup      - mz: same SDK-derived fit; 1/K0: SDK-probed per-scan table
//                 (exact at integer scans, but built with the SDK).
//   BrukerLib   - the SDK itself (ground truth, residual 0 by definition).
//   BrukerFormula (new) - exact published curves from the calibration tables.
//                 No SDK at build or runtime.
//
// Usage:
//   cargo run --release --example compare_all_converters -- <libtimsdata.so> <data.d> [frame_id]

use rustdf::data::calibration::{MobilityCalibrator, MzCalibrator};
use rustdf::data::meta::{read_global_meta_sql, read_mz_calibration, read_tims_calibration};
use rustdf::data::raw::BrukerTimsDataLibrary;
use rusqlite::Connection;
use std::env;
use std::path::Path;

fn pctl(mut v: Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((p / 100.0) * (v.len() - 1) as f64).round() as usize]
}
fn stats(v: &[f64]) -> (f64, f64, f64) {
    (pctl(v.to_vec(), 50.0), pctl(v.to_vec(), 90.0), pctl(v.to_vec(), 100.0))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: compare_all_converters <libtimsdata.so> <data.d> [frame_id]");
        std::process::exit(2);
    }
    let lib = &args[1];
    let d = &args[2];
    let frame: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let g = read_global_meta_sql(d)?;
    let con = Connection::open(Path::new(d).join("analysis.tdf"))?;
    let (t1, t2, num_scans, mz_id, tims_id): (f64, f64, i64, i64, i64) = con.query_row(
        "SELECT T1,T2,NumScans,MzCalibration,TimsCalibration FROM Frames WHERE Id=?1",
        [frame],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
    )?;
    let mzc = read_mz_calibration(d)?.into_iter().find(|c| c.id == mz_id).unwrap();
    let tc = read_tims_calibration(d)?.into_iter().find(|c| c.id == tims_id).unwrap();
    let tof_max = g.tof_max_index;

    let sdk = BrukerTimsDataLibrary::new(lib, d)?;

    // ---- ground truth ----
    let tof_grid: Vec<u32> = (1..tof_max).step_by(23).collect();
    let tof_f: Vec<f64> = tof_grid.iter().map(|&x| x as f64).collect();
    let mut sdk_mz = vec![0.0; tof_grid.len()];
    sdk.tims_index_to_mz(frame, &tof_f, &mut sdk_mz)?;

    let scan_grid: Vec<u32> = (0..num_scans as u32).collect();
    let scan_f: Vec<f64> = scan_grid.iter().map(|&x| x as f64).collect();
    let mut sdk_im = vec![0.0; scan_grid.len()];
    sdk.tims_scan_to_inv_mob(frame, &scan_f, &mut sdk_im)?;

    // =================== m/z methods ===================
    // boundary sqrt-linear (Simple / Lookup fallback)
    let b_int = g.mz_acquisition_range_lower.sqrt();
    let b_slope = (g.mz_acquisition_range_upper.sqrt() - b_int) / tof_max as f64;
    // SDK-regression sqrt-linear (Calibrated / Lookup preferred): least squares on SDK samples
    let (mut sx, mut sy, mut sxy, mut sxx, mut nn) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (i, &tof) in tof_grid.iter().enumerate() {
        if sdk_mz[i] > 0.0 {
            let (x, y) = (tof as f64, sdk_mz[i].sqrt());
            sx += x; sy += y; sxy += x * y; sxx += x * x; nn += 1.0;
        }
    }
    let r_slope = (sxy - sx * sy / nn) / (sxx - sx * sx / nn);
    let r_int = sy / nn - r_slope * sx / nn;
    // exact formula (BrukerFormula)
    let mzconv = MzCalibrator::new(
        mzc.model_type, mzc.digitizer_timebase, mzc.digitizer_delay, mzc.t1, mzc.t2, mzc.dc1,
        mzc.dc2, mzc.c0, mzc.c1, mzc.c2, mzc.c3, mzc.c4, t1, t2,
    );

    let mut e_bound = vec![]; let mut e_reg = vec![]; let mut e_form = vec![];
    for (i, &tof) in tof_grid.iter().enumerate() {
        if sdk_mz[i] <= 0.0 { continue; }
        let m = sdk_mz[i];
        let bnd = (b_int + b_slope * tof as f64).powi(2);
        let reg = (r_int + r_slope * tof as f64).powi(2);
        let frm = mzconv.tof_to_mz(tof);
        e_bound.push(((bnd - m) / m * 1e6).abs());
        e_reg.push(((reg - m) / m * 1e6).abs());
        e_form.push(((frm - m) / m * 1e6).abs());
    }

    // =================== 1/K0 methods ===================
    // linear boundary (Simple / Calibrated): scan 0 -> im_max, scan_max -> im_min.
    // Use the frame's ACTUAL mobility endpoints (fairest: the linear model then
    // matches the SDK exactly at both ends, isolating pure nonlinearity error).
    let _ = (g.one_over_k0_range_upper, g.one_over_k0_range_lower);
    let im_hi = sdk_im[0];
    let im_lo = sdk_im[sdk_im.len() - 1];
    let scan_max = (num_scans - 1) as f64;
    let imconv = MobilityCalibrator::new(tc.c0, tc.c1, tc.c2, tc.c3, tc.c4, tc.c5, tc.c6, tc.c7, tc.c8, tc.c9);

    let mut e_lin = vec![]; let mut e_imform = vec![];
    for (i, &scan) in scan_grid.iter().enumerate() {
        let lin = im_hi + (im_lo - im_hi) / scan_max * scan as f64;
        let frm = imconv.scan_to_one_over_k0(scan);
        e_lin.push(((lin - sdk_im[i]) / sdk_im[i] * 1e6).abs());
        e_imform.push(((frm - sdk_im[i]) / sdk_im[i] * 1e6).abs());
    }
    // Lookup 1/K0 = SDK-probed table => exact at integer scans (residual 0 by construction)

    println!("dataset {}  frame {}  m/z ModelType {}  1/K0 ModelType {}\n", d, frame, mzc.model_type, tc.model_type);
    println!("m/z axis (error vs SDK, ppm)          median      p90      max   SDK-free?");
    let row = |name: &str, e: &[f64], free: &str| {
        let (m, p, mx) = stats(e);
        println!("  {:<34} {:8.3} {:8.3} {:8.3}   {}", name, m, p, mx, free);
    };
    row("Simple (boundary)", &e_bound, "yes");
    row("Calibrated / Lookup (SDK regr.)", &e_reg, "no (SDK to fit)");
    row("BrukerFormula (tables)", &e_form, "yes");
    println!();
    println!("1/K0 axis (error vs SDK, ppm)         median      p90      max   SDK-free?");
    row("Simple / Calibrated (linear)", &e_lin, "yes");
    println!("  {:<34} {:8.3} {:8.3} {:8.3}   {}", "Lookup (SDK-probed table)", 0.0, 0.0, 0.0, "no (SDK to build)");
    row("BrukerFormula (tables)", &e_imform, "yes");

    Ok(())
}
