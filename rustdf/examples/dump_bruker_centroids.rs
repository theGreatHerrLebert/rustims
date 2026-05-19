// Dump Bruker SDK-centroided spectra for a DIA-PASEF .d file.
//
// For each MS2 frame, query its window-group → quad list (scan ranges +
// iso m/z), then call tims_extract_centroided_spectrum_for_frame_v2 per
// (frame, scan_lo, scan_hi). Write the resulting (frame, quad_id, mz,
// intensity) tuples to a TSV.
//
// Usage:
//   cargo run --release --example dump_bruker_centroids -- \
//       <path-to-libtimsdata.so> <path-to-.d-folder> <out.tsv> [max_frames]
//
// The .d folder must contain analysis.tdf (sqlite) — we read DIA window
// definitions from `dia_ms_ms_windows` and `dia_ms_ms_info`.

use rusqlite::Connection;
use rustdf::data::raw::BrukerTimsDataLibrary;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: dump_bruker_centroids <libtimsdata.so> <data.d> <out.tsv> [max_frames]");
        std::process::exit(2);
    }
    let lib_path = &args[1];
    let d_path = &args[2];
    let out_path = &args[3];
    let max_frames: Option<usize> = args.get(4).and_then(|s| s.parse().ok());

    eprintln!("[bruker] opening {}", d_path);
    let bd = BrukerTimsDataLibrary::new(lib_path, d_path)?;

    // Read DIA quad definitions from analysis.tdf.
    let tdf_path = Path::new(d_path).join("analysis.tdf");
    let con = Connection::open(&tdf_path)?;

    // Quads: window_group, scan_lo, scan_hi, iso_mz, iso_width.
    let mut stmt = con.prepare(
        "SELECT WindowGroup, ScanNumBegin, ScanNumEnd, IsolationMz, IsolationWidth \
         FROM DiaFrameMsMsWindows ORDER BY WindowGroup, ScanNumBegin",
    )?;
    let mut quads: Vec<(i64, u32, u32, f64, f64)> = Vec::new();
    let rows = stmt.query_map([], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, i64>(1)? as u32,
            r.get::<_, i64>(2)? as u32,
            r.get::<_, f64>(3)?,
            r.get::<_, f64>(4)?,
        ))
    })?;
    for row in rows {
        quads.push(row?);
    }
    eprintln!("[bruker] {} quads (across all window groups)", quads.len());

    // Frame → window_group mapping.
    let mut stmt = con.prepare(
        "SELECT Frame, WindowGroup FROM DiaFrameMsMsInfo ORDER BY Frame",
    )?;
    let frame_iter = stmt.query_map([], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
    })?;
    let mut frames: Vec<(i64, i64)> = Vec::new();
    for row in frame_iter {
        frames.push(row?);
    }
    eprintln!("[bruker] {} MS2 frames", frames.len());

    // Group quads by window-group for fast lookup.
    let mut wg_to_quads: std::collections::HashMap<i64, Vec<&(i64, u32, u32, f64, f64)>> =
        std::collections::HashMap::new();
    for q in &quads {
        wg_to_quads.entry(q.0).or_default().push(q);
    }

    let f = File::create(out_path)?;
    let mut out = BufWriter::new(f);
    writeln!(out, "frame_id\tquad_id\tscan_lo\tscan_hi\tiso_mz\tiso_width\tn_peaks\tmz\tintensity")?;

    let n_iter = max_frames.unwrap_or(frames.len()).min(frames.len());
    let t0 = Instant::now();
    let mut total_calls: u64 = 0;
    let mut total_peaks: u64 = 0;
    for (fi, &(frame_id, wg)) in frames[..n_iter].iter().enumerate() {
        let qs = match wg_to_quads.get(&wg) {
            Some(v) => v,
            None => continue,
        };
        for q in qs {
            let (_, scan_lo, scan_hi, iso_mz, iso_width) = **q;
            let (mzs, intens) = bd.tims_extract_centroided_spectrum_for_frame(
                frame_id, scan_lo, scan_hi,
            )?;
            total_calls += 1;
            total_peaks += mzs.len() as u64;
            // For TSV brevity, dump compact per-spectrum row with
            // semicolon-joined peak arrays. Peak arrays can hit ~1k
            // entries; that's fine for an exploratory dump.
            let mz_str: String = mzs
                .iter()
                .map(|x| format!("{:.5}", x))
                .collect::<Vec<_>>()
                .join(";");
            let in_str: String = intens
                .iter()
                .map(|x| format!("{:.1}", x))
                .collect::<Vec<_>>()
                .join(";");
            // Quad id is a synthetic index in the sorted list — write the
            // (wg, scan_lo, scan_hi) tuple as a primary key instead.
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{}",
                frame_id, wg, scan_lo, scan_hi, iso_mz, iso_width,
                mzs.len(), mz_str, in_str
            )?;
        }
        if (fi + 1) % 100 == 0 {
            let dt = t0.elapsed().as_secs_f64();
            eprintln!(
                "[bruker] {}/{} frames; {} extract calls; {} peaks; {:.1} calls/s",
                fi + 1, n_iter, total_calls, total_peaks, total_calls as f64 / dt,
            );
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "[bruker] DONE: {} frames; {} extract calls; {} peaks total; {:.1}s ({:.1} calls/s)",
        n_iter, total_calls, total_peaks, dt, total_calls as f64 / dt,
    );
    eprintln!("[bruker] wrote {}", out_path);
    Ok(())
}
