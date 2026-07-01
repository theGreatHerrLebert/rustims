// Compare SDK-free Bruker axis calibration against the proprietary SDK.
//
// Builds the pure-Rust `MzCalibrator` / `MobilityCalibrator` (see
// src/data/calibration.rs) from the analysis.tdf calibration tables, then
// evaluates TOF-index -> m/z and scan -> 1/K0 on a grid and reports the
// residuals versus the Bruker SDK (`tims_index_to_mz`, `tims_scannum_to_oneoverk0`).
//
// Usage:
//   cargo run --release --example compare_calibration -- \
//       <path-to-libtimsdata.so> <path-to-.d-folder> [frame_id]
//
// A single frame is enough: the calibration coefficients are per-frame but
// almost always identical across a run.

use rustdf::data::calibration::{MobilityCalibrator, MzCalibrator};
use rustdf::data::meta::{read_mz_calibration, read_tims_calibration};
use rustdf::data::raw::BrukerTimsDataLibrary;
use rusqlite::Connection;
use std::env;
use std::path::Path;

/// (min, mean, max) of |v| over a slice, ignoring non-finite entries.
fn abs_stats(errs: &[f64]) -> (f64, f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = 0.0f64;
    let mut sum = 0.0f64;
    let mut n = 0.0f64;
    for &e in errs {
        if !e.is_finite() {
            continue;
        }
        let a = e.abs();
        min = min.min(a);
        max = max.max(a);
        sum += a;
        n += 1.0;
    }
    if n == 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        (min, sum / n, max)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: compare_calibration <libtimsdata.so> <data.d> [frame_id]");
        std::process::exit(2);
    }
    let lib_path = &args[1];
    let d_path = &args[2];
    let frame_id: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    // --- read calibration + per-frame context from analysis.tdf ---------------
    let tdf = Path::new(d_path).join("analysis.tdf");
    let con = Connection::open(&tdf)?;

    let (frame_t1, frame_t2, num_scans, mz_cal_id, tims_cal_id): (f64, f64, i64, i64, i64) = con
        .query_row(
            "SELECT T1, T2, NumScans, MzCalibration, TimsCalibration FROM Frames WHERE Id = ?1",
            [frame_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )?;
    let tof_max: u32 = con.query_row(
        "SELECT Value FROM GlobalMetadata WHERE Key = 'DigitizerNumSamples'",
        [],
        |r| {
            let v: String = r.get(0)?;
            Ok(v.parse::<u32>().unwrap_or(400_000))
        },
    )?;

    let mz_cal = read_mz_calibration(d_path)?
        .into_iter()
        .find(|c| c.id == mz_cal_id)
        .expect("MzCalibration row not found");
    let tims_cal = read_tims_calibration(d_path)?
        .into_iter()
        .find(|c| c.id == tims_cal_id)
        .expect("TimsCalibration row not found");

    println!("dataset            : {}", d_path);
    println!("frame_id           : {}", frame_id);
    println!(
        "m/z  ModelType     : {}   (PAPPSO implements only ModelType 1)",
        mz_cal.model_type
    );
    println!("1/K0 ModelType     : {}", tims_cal.model_type);
    println!("num_scans          : {}", num_scans);
    println!("tof_max (samples)  : {}", tof_max);
    println!();

    let mz_conv = MzCalibrator::new(
        mz_cal.model_type,
        mz_cal.digitizer_timebase,
        mz_cal.digitizer_delay,
        mz_cal.t1,
        mz_cal.t2,
        mz_cal.dc1,
        mz_cal.dc2,
        mz_cal.c0,
        mz_cal.c1,
        mz_cal.c2,
        mz_cal.c3,
        mz_cal.c4,
        frame_t1,
        frame_t2,
    );
    let im_conv = MobilityCalibrator::new(
        tims_cal.c0, tims_cal.c1, tims_cal.c2, tims_cal.c3, tims_cal.c4, tims_cal.c5, tims_cal.c6,
        tims_cal.c7, tims_cal.c8, tims_cal.c9,
    );

    // --- SDK ground truth ------------------------------------------------------
    let sdk = BrukerTimsDataLibrary::new(lib_path, d_path)?;

    // ===== TOF index -> m/z =====
    let tof_grid: Vec<f64> = (1..tof_max).step_by(97).map(|x| x as f64).collect();
    let mut sdk_mz = vec![0.0f64; tof_grid.len()];
    sdk.tims_index_to_mz(frame_id, &tof_grid, &mut sdk_mz)?;

    let mut d_mz_mda = Vec::with_capacity(tof_grid.len());
    let mut d_mz_ppm = Vec::with_capacity(tof_grid.len());
    for (i, &tof) in tof_grid.iter().enumerate() {
        if sdk_mz[i] <= 0.0 {
            continue;
        }
        let mine = mz_conv.tof_to_mz(tof as u32);
        d_mz_mda.push((mine - sdk_mz[i]) * 1000.0);
        d_mz_ppm.push((mine - sdk_mz[i]) / sdk_mz[i] * 1.0e6);
    }
    let (mn, me, mx) = abs_stats(&d_mz_mda);
    let (pn, pe, px) = abs_stats(&d_mz_ppm);
    println!("TOF index -> m/z   ({} grid points, {:.1}..{:.1} m/z)", d_mz_mda.len(), sdk_mz.iter().cloned().filter(|x| *x>0.0).fold(f64::INFINITY,f64::min), sdk_mz.iter().cloned().fold(0.0,f64::max));
    println!("  |Δ| m/z  [mDa]   min {:.4}  mean {:.4}  max {:.4}", mn, me, mx);
    println!("  |Δ| m/z  [ppm]   min {:.4}  mean {:.4}  max {:.4}", pn, pe, px);

    // round trip m/z -> tof -> m/z (formula only)
    let mut rt = Vec::new();
    for &tof in tof_grid.iter() {
        let mz = mz_conv.tof_to_mz(tof as u32);
        if mz <= 0.0 { continue; }
        let back = mz_conv.mz_to_tof(mz) as f64;
        rt.push(back - tof);
    }
    let (_, rtmean, rtmax) = abs_stats(&rt);
    println!("  round-trip tof   mean {:.3}  max {:.3}  (index units)", rtmean, rtmax);
    println!();

    // ===== scan -> 1/K0 =====
    let scan_grid: Vec<f64> = (0..num_scans as u32).map(|x| x as f64).collect();
    let mut sdk_im = vec![0.0f64; scan_grid.len()];
    sdk.tims_scan_to_inv_mob(frame_id, &scan_grid, &mut sdk_im)?;

    let mut d_im = Vec::with_capacity(scan_grid.len());
    let mut d_im_rel = Vec::with_capacity(scan_grid.len());
    for (i, &scan) in scan_grid.iter().enumerate() {
        let mine = im_conv.scan_to_one_over_k0(scan as u32);
        d_im.push(mine - sdk_im[i]);
        if sdk_im[i] != 0.0 {
            d_im_rel.push((mine - sdk_im[i]) / sdk_im[i] * 1.0e6);
        }
    }
    let (in_, ie, ix) = abs_stats(&d_im);
    let (rn, re, rx) = abs_stats(&d_im_rel);
    println!(
        "scan -> 1/K0       ({} scans, {:.4}..{:.4} 1/K0)",
        scan_grid.len(),
        sdk_im.iter().cloned().fold(f64::INFINITY, f64::min),
        sdk_im.iter().cloned().fold(0.0, f64::max)
    );
    println!("  |Δ| 1/K0        min {:.3e}  mean {:.3e}  max {:.3e}", in_, ie, ix);
    println!("  |Δ| 1/K0  [ppm] min {:.4}  mean {:.4}  max {:.4}", rn, re, rx);

    // round trip scan -> 1/K0 -> scan
    let mut rt_scan = Vec::new();
    for &scan in scan_grid.iter() {
        let im = im_conv.scan_to_one_over_k0(scan as u32);
        let back = im_conv.one_over_k0_to_scan(im) as f64;
        rt_scan.push(back - scan);
    }
    let (_, sm, sx) = abs_stats(&rt_scan);
    println!("  round-trip scan  mean {:.3}  max {:.3}  (scan units)", sm, sx);

    // `sdk` closes itself on Drop; do not call tims_close() explicitly (some SDK
    // builds abort on a double close).
    Ok(())
}
