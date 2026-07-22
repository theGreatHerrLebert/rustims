//! M0 rewrite-survival probe (THERMO_PLAN.md): does authoring into a real Astral DIA template and
//! `save()` preserve the ENTIRE acquisition schedule intact (RT, MS order, analyzer, isolation
//! center/width, CE, profile/centroid flag — bit-for-bit), and do authored sentinels land in exactly
//! the right scans (no cursor drift)? This is the load-bearing de-risk before any feature rendering:
//! if the template's method metadata does not survive a rewrite, Thermo readers / DiaNN would see a
//! file that no longer matches the original DIA method.
//!
//! Run: cargo run -p ms-io --features thermo --example thermo_m0_survival -- <template.raw> <out.raw>

use thermorawfile::RawFile;

/// The schedule fields that MUST be invariant under a rewrite (floats compared by exact bits, so any
/// re-encode drift is caught).
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
    let mut rows = Vec::with_capacity(rf.scan_count());
    for scan in rf.first_scan..=rf.last_scan {
        let ev = rf.scan_event(scan).expect("scan event");
        let idx = &rf.index[(scan - rf.first_scan) as usize];
        rows.push(SchedRow {
            time_bits: idx.time.to_bits(),
            ms_order: ev.ms_order,
            analyzer: ev.analyzer,
            iso_center_bits: ev.isolation_center.to_bits(),
            iso_width_bits: ev.isolation_width.to_bits(),
            ce_bits: ev.collision_energy.to_bits(),
            is_profile: rf.profile(scan).is_some(),
        });
    }
    rows
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (tpl, out) = (&a[1], &a[2]);

    let rf0 = RawFile::open(tpl).expect("open template");
    let (first, last) = (rf0.first_scan, rf0.last_scan);
    let s0 = snapshot(&rf0);
    println!("template: {} scans (first={first}, last={last}), checksum_valid={}", rf0.scan_count(), rf0.checksum_valid());

    // Sentinel scans: early / mid / late MS2 (ASTMS centroid), each a distinct payload.
    let ms2s: Vec<u32> = (first..=last)
        .filter(|&s| rf0.scan_event(s).map(|e| e.ms_order == 2).unwrap_or(false))
        .collect();
    assert!(ms2s.len() >= 4, "need MS2 scans");
    let sentinels: [(u32, f32); 3] = [
        (ms2s[1], 1.0),
        (ms2s[ms2s.len() / 2], 2.0),
        (ms2s[ms2s.len() - 2], 3.0),
    ];
    // A distinctive, tag-scaled peak list per sentinel — off the usual immonium/backbone grid so a
    // stray template peak can't masquerade as ours.
    let payload = |tag: f32| -> Vec<(f64, f32)> {
        vec![(133.7001, 1.0e4 * tag), (411.4002, 3.3e4 * tag), (777.7003, 9.9e4 * tag)]
    };

    // Author the sentinels and save.
    let mut rf = RawFile::open(tpl).expect("reopen for authoring");
    for &(scan, tag) in &sentinels {
        rf.author_centroids(scan, &payload(tag)).unwrap_or_else(|e| panic!("author scan {scan}: {e}"));
    }
    rf.save(out).expect("save");
    println!("authored {} MS2 sentinels -> {out}", sentinels.len());

    // Reopen and verify.
    let rf1 = RawFile::open(out).expect("reopen output");
    let mut fail = 0usize;

    // (1) Integrity checksum.
    if !rf1.checksum_valid() {
        println!("FAIL: output checksum invalid");
        fail += 1;
    }

    // (2) Full schedule bit-identical across ALL scans.
    let s1 = snapshot(&rf1);
    if s1.len() != s0.len() {
        println!("FAIL: scan count changed {} -> {}", s0.len(), s1.len());
        fail += 1;
    } else {
        let diffs = s0.iter().zip(&s1).enumerate().filter(|(_, (a, b))| a != b).count();
        if diffs == 0 {
            println!("OK: schedule bit-identical across all {} scans (RT, MS order, analyzer, isolation, CE, profile flag)", s0.len());
        } else {
            println!("FAIL: {diffs} scans changed a schedule field under rewrite");
            fail += 1;
        }
    }

    // (3) Sentinels round-trip at exactly their scans (no cursor drift).
    for &(scan, tag) in &sentinels {
        let back = rf1.centroid_peaks(scan);
        let want = payload(tag);
        let ok = back.len() >= want.len()
            && want.iter().all(|&(m, i)| back.iter().any(|p| (p.mz - m).abs() < 1e-3 && (p.intensity - i).abs() <= i * 1e-3 + 1.0));
        if ok {
            println!("OK: sentinel tag {tag} present at scan {scan} ({} peaks read back)", back.len());
        } else {
            println!("FAIL: sentinel tag {tag} NOT found at scan {scan} (read {} peaks)", back.len());
            fail += 1;
        }
    }

    // (4) An untouched MS2 scan keeps its ORIGINAL peaks (no bleed), and untouched MS1 profile still reads.
    let untouched_ms2 = ms2s.iter().copied().find(|s| !sentinels.iter().any(|(x, _)| x == s)).unwrap();
    if rf0.centroid_peaks(untouched_ms2).len() == rf1.centroid_peaks(untouched_ms2).len() {
        println!("OK: untouched MS2 scan {untouched_ms2} unchanged ({} peaks)", rf1.centroid_peaks(untouched_ms2).len());
    } else {
        println!("FAIL: untouched MS2 scan {untouched_ms2} peak count changed");
        fail += 1;
    }
    let ms1_scan = (first..=last).find(|&s| rf0.scan_event(s).map(|e| e.ms_order == 1).unwrap_or(false)).unwrap();
    match (rf0.profile(ms1_scan).map(|p| p.point_count()), rf1.profile(ms1_scan).map(|p| p.point_count())) {
        (Some(a), Some(b)) if a == b => println!("OK: untouched MS1 profile scan {ms1_scan} still reads ({a} points)"),
        (x, y) => { println!("FAIL: MS1 profile scan {ms1_scan} changed {x:?} -> {y:?}"); fail += 1; }
    }

    println!("\nM0 rewrite-survival: {}", if fail == 0 { "PASS — template survives rewrite intact".to_string() } else { format!("{fail} FAILURE(S)") });
    std::process::exit(if fail == 0 { 0 } else { 1 });
}
