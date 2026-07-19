//! M1 DDA writer spike: write a trivial but structurally-complete DDA-PASEF `.d` to `argv[1]`, copying
//! calibration/metadata from a real DDA reference `.d` (`argv[2]`) so a vendor/imspy reader opens it.
//! Used to validate the writer's DDA tables against a real `.d` schema and `TimsDatasetDDA` round-trip.
//!
//! Run: `cargo run -p rustdf --example dda_spike -- /tmp/dda_out.d /path/to/real_dda.d`

use rustdf::data::tdf_writer::{DdaPasefWindow, DdaPrecursor, RenderedFrame, TdfWriter, TdfWriterConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let out = &args[1];
    let reference = &args[2];
    let _ = std::fs::remove_dir_all(out);

    let cfg = TdfWriterConfig {
        num_scans: 709,
        scan_mode: 8, // DDA
        reference_d: Some(reference.clone()),
        ..Default::default()
    };
    let mut w = TdfWriter::create(out, cfg)?;

    // 6 frames: MS1, MS2, MS2, MS1, MS2, MS2 — one ion re-selected across two cycles.
    let ms_types = [0u8, 8, 8, 0, 8, 8];
    for (i, &t) in ms_types.iter().enumerate() {
        w.write_frame(&RenderedFrame {
            frame_id: i as u32 + 1,
            retention_time: i as f64 * 0.1,
            ms_ms_type: t,
            scans: vec![100 + i as u32, 400],
            tofs: vec![120_000, 260_000],
            intensities: vec![50, 90],
        })?;
    }

    let precursors = vec![
        DdaPrecursor { id: 1, largest_peak_mz: 596.87, average_mz: 596.75, monoisotopic_mz: 596.63, charge: 2, scan_number: 105.0, intensity: 4600.0, parent: 1 },
        DdaPrecursor { id: 2, largest_peak_mz: 712.40, average_mz: 712.25, monoisotopic_mz: 712.10, charge: 3, scan_number: 400.0, intensity: 2100.0, parent: 1 },
        DdaPrecursor { id: 1, largest_peak_mz: 596.87, average_mz: 596.75, monoisotopic_mz: 596.63, charge: 2, scan_number: 106.0, intensity: 3900.0, parent: 4 }, // re-selection
    ];
    let pasef = vec![
        DdaPasefWindow { frame: 2, scan_num_begin: 95,  scan_num_end: 117, isolation_mz: 596.87, isolation_width: 2.0, collision_energy: 31.5, precursor: 1 },
        DdaPasefWindow { frame: 2, scan_num_begin: 390, scan_num_end: 412, isolation_mz: 712.40, isolation_width: 3.0, collision_energy: 34.0, precursor: 2 },
        DdaPasefWindow { frame: 5, scan_num_begin: 95,  scan_num_end: 117, isolation_mz: 596.87, isolation_width: 2.0, collision_energy: 31.6, precursor: 1 },
    ];
    w.set_dda_schedule(precursors, pasef);
    w.finalize()?;
    println!("wrote DDA .d -> {out}");
    Ok(())
}
