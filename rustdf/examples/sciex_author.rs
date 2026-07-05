#[cfg(feature = "sciex")]
fn main() {
    use rustdf::sim::sciex_dispatch::{write_sciex_wiff, SciexWriteOptions};
    use std::path::Path;
    let db = std::env::args().nth(1).unwrap();
    let wiff = std::env::args().nth(2).unwrap();
    let out = std::env::args().nth(3).unwrap();
    // Optional spike-in: TIMSIM_OVERLAY_PPM > 0 keeps the template's real peaks (real⊕sim);
    // TIMSIM_SPIKE_SCALE sets the spike strength vs the real background (default 1.0).
    let overlay_ppm = std::env::var("TIMSIM_OVERLAY_PPM").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let spike_scale = std::env::var("TIMSIM_SPIKE_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let opts = SciexWriteOptions { num_threads: 8, fragment_noise_ppm: 8.0, precursor_noise_ppm: 5.0, overlay_ppm, spike_scale, ..Default::default() };
    match write_sciex_wiff(Path::new(&db), Path::new(&wiff), Path::new(&out), opts) {
        Ok(s) => println!("OK {:?}", s),
        Err(e) => { eprintln!("ERR: {e}"); std::process::exit(1); }
    }
}
#[cfg(not(feature = "sciex"))]
fn main() { eprintln!("build with --features sciex"); }
