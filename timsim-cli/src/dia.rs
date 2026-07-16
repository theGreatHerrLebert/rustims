//! DIA schedule **replay** + the diagonal quadrupole transmission gate.
//!
//! A DIA-PASEF run interleaves MS1 (precursor) frames with MS2 (fragment) frames on a fixed cycle;
//! each MS2 frame belongs to a *window group* that defines, per mobility scan, an isolation `(m/z,
//! width)` — the diagonal. We copy that schedule from a reference DIA `.d` rather than synthesise one:
//! per Codex, we extract the reference's **complete frame-level cycle** (its exact MS1/MS2 pattern and
//! window-group sequence, including any irregular ordering) and **repeat that whole cycle** over our
//! `n_frames`, then build mscore's [`TimsTransmissionDIA`] from the copied window definitions.
//!
//! The transmission is reused verbatim from mscore — it *is* the diagonal gate
//! (`apply_transmission(frame, scan, mz) -> probability`, soft logistic edge). This module only
//! decides which of our frames are MS1 vs MS2 (and their groups) and hands mscore the windows.

use anyhow::{anyhow, Result};
use mscore::timstof::quadrupole::TimsTransmissionDIA;
use rustdf::data::meta::{read_dia_ms_ms_info, read_dia_ms_ms_windows, DiaMsMisInfo, DiaMsMsWindow};
use std::collections::{HashMap, HashSet};

/// The replayed DIA schedule for our run + the transmission gate built from the reference windows.
pub struct DiaSchedule {
    /// Frames per cycle (e.g. 21 = 1 MS1 + 20 MS2).
    pub cycle_len: u32,
    /// One entry per cycle position: `None` = MS1/precursor, `Some(group)` = MS2 of that window group.
    pub pattern: Vec<Option<u32>>,
    /// The window definitions, copied verbatim from the reference (for the writer's `DiaFrameMsMsWindows`).
    pub windows: Vec<DiaMsMsWindow>,
    /// mscore's diagonal transmission gate over OUR frames.
    pub transmission: TimsTransmissionDIA,
}

impl DiaSchedule {
    /// Read a reference DIA `.d`'s schedule and replay it over `n_frames` (with `n_scans` scans).
    pub fn from_reference(ref_d: &str, n_frames: u32, n_scans: u32) -> Result<Self> {
        let info = read_dia_ms_ms_info(ref_d).map_err(|e| anyhow!("read DiaFrameMsMsInfo: {e}"))?;
        let windows = read_dia_ms_ms_windows(ref_d).map_err(|e| anyhow!("read DiaFrameMsMsWindows: {e}"))?;
        Self::build(&info, windows, n_frames, n_scans)
    }

    /// Build the schedule from already-read reference tables (the unit-testable core).
    pub fn build(
        info: &[DiaMsMisInfo],
        windows: Vec<DiaMsMsWindow>,
        n_frames: u32,
        n_scans: u32,
    ) -> Result<Self> {
        let (cycle_len, pattern) = extract_cycle(info)?;
        validate(&pattern, &windows, n_scans)?;

        // Replay: our MS2 frames and their groups (MS1 frames are simply absent from the transmission,
        // which is how mscore's is_precursor distinguishes them).
        let mut frame: Vec<i32> = Vec::new();
        let mut frame_group: Vec<i32> = Vec::new();
        for f in 1..=n_frames {
            if let Some(g) = pattern[((f - 1) % cycle_len) as usize] {
                frame.push(f as i32);
                frame_group.push(g as i32);
            }
        }

        // Window definitions, verbatim.
        let wg: Vec<i32> = windows.iter().map(|w| w.window_group as i32).collect();
        let ss: Vec<i32> = windows.iter().map(|w| w.scan_num_begin as i32).collect();
        let se: Vec<i32> = windows.iter().map(|w| w.scan_num_end as i32).collect();
        let imz: Vec<f64> = windows.iter().map(|w| w.isolation_mz).collect();
        let iw: Vec<f64> = windows.iter().map(|w| w.isolation_width).collect();
        let transmission = TimsTransmissionDIA::new(frame, frame_group, wg, ss, se, imz, iw, None);

        Ok(DiaSchedule { cycle_len, pattern, windows, transmission })
    }

    /// MS level of a 1-based frame: 1 = MS1/precursor, 2 = MS2/fragment.
    pub fn ms_level(&self, frame: u32) -> u8 {
        match self.pattern[((frame - 1) % self.cycle_len) as usize] {
            None => 1,
            Some(_) => 2,
        }
    }

    /// The window group of a frame, or `None` for MS1.
    pub fn window_group(&self, frame: u32) -> Option<u32> {
        self.pattern[((frame - 1) % self.cycle_len) as usize]
    }
}

/// Extract the reference's frame-level cycle: `cycle_len` and the per-position pattern (`None` = MS1,
/// `Some(group)` = MS2). Anchored at the first MS1 frame, so the pattern starts a cycle. Tiling this
/// pattern reproduces the reference's exact MS1/MS2 layout and group ordering — regular or not.
fn extract_cycle(info: &[DiaMsMisInfo]) -> Result<(u32, Vec<Option<u32>>)> {
    if info.is_empty() {
        return Err(anyhow!("reference has no DiaFrameMsMsInfo rows — not a DIA .d"));
    }
    let map: HashMap<u32, u32> = info.iter().map(|d| (d.frame_id, d.window_group)).collect();
    let max_frame = info.iter().map(|d| d.frame_id).max().unwrap();

    // First MS1 frame: the smallest frame >= 1 absent from the (MS2-only) info map.
    let mut m0 = 1u32;
    while map.contains_key(&m0) {
        m0 += 1;
    }
    // Next MS1 frame after m0 marks the cycle boundary.
    let mut m1 = m0 + 1;
    while m1 <= max_frame && map.contains_key(&m1) {
        m1 += 1;
    }
    if m1 > max_frame {
        return Err(anyhow!("could not find a full cycle (no second MS1 frame) in the reference schedule"));
    }
    let cycle_len = m1 - m0;
    let pattern: Vec<Option<u32>> = (0..cycle_len).map(|k| map.get(&(m0 + k)).copied()).collect();
    Ok((cycle_len, pattern))
}

/// Reject a schedule whose groups have no window definitions, or whose scan ranges exceed the grid —
/// a silently distorted diagonal is worse than a hard error.
fn validate(pattern: &[Option<u32>], windows: &[DiaMsMsWindow], n_scans: u32) -> Result<()> {
    if n_scans == 0 {
        return Err(anyhow!("n_scans is 0 — a run must have at least one mobility scan"));
    }
    let defined: HashSet<u32> = windows.iter().map(|w| w.window_group).collect();
    for g in pattern.iter().flatten() {
        if !defined.contains(g) {
            return Err(anyhow!("frame maps to window group {g} with no window definitions"));
        }
    }
    for w in windows {
        // A window must cover at least one emittable scan (0-based indices < n_scans). We do NOT
        // reject `scan_num_end == n_scans`: that is the real Bruker convention (mscore expands the
        // range inclusively, so the last window ends at n_scans and its one over-the-edge scan is
        // simply never queried) — rejecting it would refuse genuine reference `.d` files.
        if w.scan_num_begin >= n_scans {
            return Err(anyhow!(
                "window group {} starts at scan {} which is beyond the grid (n_scans {n_scans})",
                w.window_group, w.scan_num_begin
            ));
        }
        if w.scan_num_end > n_scans {
            return Err(anyhow!(
                "window group {} scan range [{}, {}] exceeds n_scans {n_scans}",
                w.window_group, w.scan_num_begin, w.scan_num_end
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mscore::timstof::quadrupole::IonTransmission;

    fn win(group: u32, iso_mz: f64) -> DiaMsMsWindow {
        DiaMsMsWindow { window_group: group, scan_num_begin: 0, scan_num_end: 3, isolation_mz: iso_mz, isolation_width: 10.0, collision_energy: 25.0 }
    }

    /// A tiny 3-frame cycle (MS1, g1, g2) with frames 1/4/7 as MS1 (absent from info).
    fn fixture() -> (Vec<DiaMsMisInfo>, Vec<DiaMsMsWindow>) {
        let info = vec![
            DiaMsMisInfo { frame_id: 2, window_group: 1 },
            DiaMsMisInfo { frame_id: 3, window_group: 2 },
            DiaMsMisInfo { frame_id: 5, window_group: 1 },
            DiaMsMisInfo { frame_id: 6, window_group: 2 },
        ];
        let windows = vec![win(1, 500.0), win(2, 600.0)];
        (info, windows)
    }

    #[test]
    fn extracts_and_replays_the_cycle() {
        let (info, windows) = fixture();
        let s = DiaSchedule::build(&info, windows, 9, 3).unwrap();
        assert_eq!(s.cycle_len, 3);
        assert_eq!(s.pattern, vec![None, Some(1), Some(2)]);
        // Replayed over 9 frames: MS1 at 1,4,7; g1 at 2,5,8; g2 at 3,6,9.
        assert_eq!((1..=9).map(|f| s.ms_level(f)).collect::<Vec<_>>(), vec![1, 2, 2, 1, 2, 2, 1, 2, 2]);
        assert_eq!(s.window_group(2), Some(1));
        assert_eq!(s.window_group(6), Some(2));
        assert_eq!(s.window_group(7), None);
    }

    #[test]
    fn transmission_gates_by_frame_and_window() {
        let (info, windows) = fixture();
        let s = DiaSchedule::build(&info, windows, 9, 3).unwrap();
        // MS1 frames are precursor frames (all-pass); MS2 frames gate by their group's window.
        assert!(s.transmission.is_precursor(1));
        assert!(!s.transmission.is_precursor(2));
        // Frame 2 is group 1 (iso 500 ± 5): 500 transmits, 700 does not.
        assert!(s.transmission.apply_transmission(2, 1, &vec![500.0])[0] > 0.9);
        assert!(s.transmission.apply_transmission(2, 1, &vec![700.0])[0] < 0.1);
        // Frame 3 is group 2 (iso 600): 600 transmits, 500 does not (proves the group lookup).
        assert!(s.transmission.apply_transmission(3, 1, &vec![600.0])[0] > 0.9);
        assert!(s.transmission.apply_transmission(3, 1, &vec![500.0])[0] < 0.1);
    }

    #[test]
    fn rejects_a_group_without_windows() {
        let info = vec![
            DiaMsMisInfo { frame_id: 2, window_group: 1 },
            DiaMsMisInfo { frame_id: 4, window_group: 1 },
        ];
        // Cycle [None, Some(1)] but no window defs at all.
        assert!(DiaSchedule::build(&info, vec![], 4, 3).is_err());
    }

    #[test]
    fn rejects_scan_range_beyond_grid() {
        let (info, mut windows) = fixture();
        windows[0].scan_num_end = 999; // beyond n_scans
        assert!(DiaSchedule::build(&info, windows, 9, 3).is_err());
    }

    #[test]
    fn rejects_window_starting_beyond_grid() {
        let (info, mut windows) = fixture();
        windows[0].scan_num_begin = 3; // == n_scans, so no valid (0-based) scan is covered
        assert!(DiaSchedule::build(&info, windows, 9, 3).is_err());
    }

    #[test]
    fn accepts_end_equal_to_n_scans() {
        // The Bruker convention: the last window ends at exactly n_scans. Must be accepted.
        let (info, mut windows) = fixture();
        windows[0].scan_num_end = 3; // == n_scans, begin still 0
        assert!(DiaSchedule::build(&info, windows, 9, 3).is_ok());
    }

    /// Integration against a real DIA `.d`. Run with `DIA_REF=/path/to.d cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn integration_real_reference() {
        let d = std::env::var("DIA_REF").expect("set DIA_REF to a DIA .d");
        let s = DiaSchedule::from_reference(&d, 4200, 918).unwrap();
        eprintln!("cycle_len = {}, windows = {}", s.cycle_len, s.windows.len());
        // First cycle position is MS1; a real diaPASEF cycle has many MS2 groups after it.
        assert_eq!(s.ms_level(1), 1);
        assert!(s.pattern.iter().filter(|p| p.is_some()).count() >= 1);
        // Frame 2 is the first MS2 frame — its group's window at scan 0 transmits its own isolation m/z.
        let g = s.window_group(2).expect("frame 2 should be MS2");
        let (iso_mz, _w) = *s.transmission.get_setting(g as i32, 0).expect("group 1 has a scan-0 window");
        assert!(s.transmission.apply_transmission(2, 0, &vec![iso_mz])[0] > 0.9,
                "a precursor at the window centre must transmit");
        assert!(s.transmission.apply_transmission(2, 0, &vec![iso_mz + 500.0])[0] < 0.1,
                "a precursor far from the window must not transmit");
    }
}
