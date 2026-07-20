//! M1 writer smoke through the ACTUAL driver seam (THERMO_PLAN.md): author every template slot via
//! `rustdf::sim::acquisition::ThermoRawWriter` (the single-cursor, slot-order, ms-level-matched path the
//! parquet driver will use), then `finalize` a complete valid `.raw`. Unlike M0 (which used the low-level
//! `RawFile` directly), this exercises `write_scan(ScanDescriptor)` → `author_profile` (MS1) /
//! `author_centroids` (MS2), the zero-residual full-coverage contract, and the finalize rebuild.
//!
//! Verifies: output checksum valid; the acquisition schedule (RT/MS order/analyzer/isolation/CE/profile
//! flag) survives bit-for-bit (isolation:None preserves the template's DIA windows); MS2 sentinels land in
//! exactly the right early/mid/late scans (no cursor drift); MS1 profile slots are authored + readable.
//!
//! Run: cargo run -p rustdf --features thermo --example thermo_m1_seam -- <template.raw> <out.raw>

use rustdf::sim::acquisition::{AcquisitionWriter, ScanDescriptor, ThermoRawWriter};
use thermorawfile::RawFile;

#[derive(PartialEq, Clone)]
struct SchedRow {
    time_bits: u64,
    ms_order: u8,
    analyzer: u8,
    iso_center_bits: u64,
    iso_width_bits: u64,
    ce_bits: u64,
    is_profile: bool,
}

fn snapshot(rf: &RawFile) -> Vec<SchedRow> {
    (rf.first_scan..=rf.last_scan)
        .map(|scan| {
            let ev = rf.scan_event(scan).expect("scan event");
            let idx = &rf.index[(scan - rf.first_scan) as usize];
            SchedRow {
                time_bits: idx.time.to_bits(),
                ms_order: ev.ms_order,
                analyzer: ev.analyzer,
                iso_center_bits: ev.isolation_center.to_bits(),
                iso_width_bits: ev.isolation_width.to_bits(),
                ce_bits: ev.collision_energy.to_bits(),
                is_profile: rf.profile(scan).is_some(),
            }
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (tpl, out) = (&a[1], &a[2]);

    let rf0 = RawFile::open(tpl).expect("open template");
    let s0 = snapshot(&rf0);
    println!("template: {} scans, checksum_valid={}", rf0.scan_count(), rf0.checksum_valid());

    let mut writer = ThermoRawWriter::from_template(tpl, out).expect("from_template");
    let manifest = writer.manifest().to_vec(); // (scan, ms_level, is_profile), acquisition order
    let (ms1_cap, ms2_cap) = writer.capacity();
    println!("seam manifest: {} slots (MS1={ms1_cap}, MS2={ms2_cap})", manifest.len());

    // Sentinel scans among the MS2 slots (early / mid / late), each a distinct centroid payload.
    let ms2_scans: Vec<u32> = manifest.iter().filter(|(_, l, _)| *l == 2).map(|(s, _, _)| *s).collect();
    let sentinels: [(u32, f32); 3] = [
        (ms2_scans[1], 1.0),
        (ms2_scans[ms2_scans.len() / 2], 2.0),
        (ms2_scans[ms2_scans.len() - 2], 3.0),
    ];
    let sentinel_payload = |tag: f32| vec![(133.7001, 1.0e4 * tag), (411.4002, 3.3e4 * tag), (777.7003, 9.9e4 * tag)];
    // MS1 profile peaks must land on the scan's frequency grid, so derive them from scan 1's OWN grid
    // via `mz_of_bin` using the SAME calibration the seam uses (`scantrailer_addr + 4`) — then
    // `author_profile` maps them straight back to those bins. Picking arbitrary in-range m/z can fall
    // between/outside bins under this calibration.
    let ms1_scan0 = manifest.iter().find(|(_, l, _)| *l == 1).map(|(s, _, _)| *s).unwrap();
    let calib = rf0.calibration_at_event(rf0.scantrailer_addr as usize + 4).expect("calibration");
    let prof0 = rf0.profile(ms1_scan0).expect("ms1 profile");
    let mut ms1_payload: Vec<(f64, f32)> = [prof0.nbins / 4, prof0.nbins / 2, 3 * prof0.nbins / 4]
        .iter()
        .enumerate()
        .map(|(k, &bin)| (prof0.mz_of_bin(bin, &calib), 5.0e5 - k as f32 * 1.0e5))
        .collect();
    // Deliberately out-of-range peaks (below/above the MS1 grid) — Change A must DROP these and author
    // the in-range remainder, not fail the scan. Every MS1 slot carries the same payload.
    ms1_payload.push((120.0, 4.0e4)); // below grid
    ms1_payload.push((3000.0, 4.0e4)); // above grid
    println!("MS1 payload: {} in-range + 2 deliberately out-of-range", ms1_payload.len() - 2);
    let ms2_filler: Vec<(f64, f32)> = vec![(200.10, 2.0e4), (500.50, 4.0e4)];

    // Author EVERY slot, in the manifest's order — the exact loop the driver runs. isolation:None keeps
    // the template's DIA windows (the seam only rewrites isolation when a window is supplied).
    for (scan, ms_level, _is_profile) in &manifest {
        let peaks = if *ms_level == 1 {
            ms1_payload.clone()
        } else if let Some(&(_, tag)) = sentinels.iter().find(|(s, _)| s == scan) {
            sentinel_payload(tag)
        } else {
            ms2_filler.clone()
        };
        let desc = ScanDescriptor { ms_level: *ms_level, retention_time: 0.0, isolation: None, peaks };
        writer.write_scan(&desc).unwrap_or_else(|e| panic!("write_scan slot {scan}: {e}"));
    }
    let complete = writer.is_complete();
    writer.finalize().expect("finalize");
    let ps = writer.profile_summary();
    println!("authored all {} slots via the seam (complete={complete}) -> {out}", manifest.len());
    println!("profile drop tally (Change A): written_bins={} dropped_below={} dropped_above={} dropped_intensity={:.0}",
        ps.written_bins, ps.dropped_below_range, ps.dropped_above_range, ps.dropped_intensity);
    // Sanity: the 2 out-of-range peaks per MS1 slot must have been dropped (not errored the run).
    let ms1_slots = manifest.iter().filter(|(_, l, _)| *l == 1).count();
    assert_eq!(ps.dropped_below_range, ms1_slots, "one below-grid drop per MS1 slot");
    assert_eq!(ps.dropped_above_range, ms1_slots, "one above-grid drop per MS1 slot");

    // Verify.
    let rf1 = RawFile::open(out).expect("reopen output");
    let mut fail = 0usize;
    if !rf1.checksum_valid() { println!("FAIL: output checksum invalid"); fail += 1; }

    let s1 = snapshot(&rf1);
    let diffs = s0.iter().zip(&s1).filter(|(a, b)| a != b).count();
    if s1.len() == s0.len() && diffs == 0 {
        println!("OK: schedule bit-identical across all {} scans", s0.len());
    } else {
        println!("FAIL: schedule changed ({diffs} rows, len {} -> {})", s0.len(), s1.len());
        fail += 1;
    }

    for &(scan, tag) in &sentinels {
        let back = rf1.centroid_peaks(scan);
        let want = sentinel_payload(tag);
        let ok = want.iter().all(|&(m, i)| back.iter().any(|p| (p.mz - m).abs() < 1e-3 && (p.intensity - i).abs() <= i * 1e-3 + 1.0));
        if ok { println!("OK: sentinel tag {tag} at scan {scan} ({} peaks)", back.len()); }
        else { println!("FAIL: sentinel tag {tag} missing at scan {scan} ({} peaks)", back.len()); fail += 1; }
    }

    // MS1 profile slots authored + readable.
    let ms1_scan = manifest.iter().find(|(_, l, _)| *l == 1).map(|(s, _, _)| *s).unwrap();
    match rf1.profile(ms1_scan).map(|p| p.point_count()) {
        Some(n) if n > 0 => println!("OK: MS1 profile scan {ms1_scan} authored+readable ({n} points)"),
        other => { println!("FAIL: MS1 profile scan {ms1_scan} not readable ({other:?})"); fail += 1; }
    }

    println!("\nM1 seam smoke: {}", if fail == 0 { "PASS — full-coverage author via the seam is valid".to_string() } else { format!("{fail} FAILURE(S)") });
    std::process::exit(if fail == 0 { 0 } else { 1 });
}
