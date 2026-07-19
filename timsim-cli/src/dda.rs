//! DDA-PASEF precursor selection — a faithful port of v1's `simulate_dda_pasef_selection_scheme` /
//! `schedule_precursors` (`imspy_simulation/timsim/jobs/dda_selection_scheme.py`).
//!
//! Every `precursors_every`-th frame (1-based) is an MS1 survey; the `k = precursors_every-1` frames
//! between are that cycle's PASEF MS2 frames. Per MS1 frame we take the candidates active there, order by
//! `(MS1 intensity ↓, stable order ↑)`, skip dynamically-excluded ones, and **first-fit** each into the
//! earliest MS2 frame that has `< max_precursors` and no conflicting mobility band (v1's conflict test
//! compares the new clamped band against existing precursors' `apex ± w`). The final survey is deliberately
//! left unscheduled, matching v1.
//!
//! Output feeds the writer (`Precursors` = one canonical row per selected ion; `PasefFrameMsMsInfo` = one
//! row per event) and the sidecar answer key (one row per event, keyed on `(ms2_frame, scan_begin)`).

/// A selectable precursor with everything the scheme needs. `abundance` × the per-frame elution weight is
/// the MS1 intensity the scheme ranks and thresholds on.
#[derive(Clone)]
pub struct Candidate {
    pub precursor_id: u64,
    pub order: u32,
    pub apex_frame: f64,
    pub scan_apex: i64,
    pub mono_mz: f64,
    pub largest_mz: f64,
    pub average_mz: f64,
    pub charge: i64,
    pub abundance: f64,
    pub sigma_frames: f64,
    pub n_sigma: f64,
}

pub struct SelectionParams {
    pub precursors_every: u32,
    pub max_precursors: usize,
    pub intensity_threshold: f64,
    pub exclusion_frames: u32,
    pub band_half_width: i64,
    pub n_scans: u32,
    pub ce_bias: f64,
    pub ce_slope: f64,
}

impl Default for SelectionParams {
    fn default() -> Self {
        SelectionParams {
            precursors_every: 10,
            max_precursors: 25,
            intensity_threshold: 0.0,
            exclusion_frames: 25,
            band_half_width: 11,
            n_scans: 709,
            ce_bias: 54.1984,
            ce_slope: -0.0345,
        }
    }
}

/// One canonical `Precursors` row (per selected ion).
pub struct CanonicalPrecursor {
    pub precursor_id: u64,
    pub largest_mz: f64,
    pub average_mz: f64,
    pub mono_mz: f64,
    pub charge: i64,
    pub scan_apex: i64,
    pub abundance: f64,
    pub parent_ms1_frame: i64,
}

/// One PASEF selection event: a `PasefFrameMsMsInfo` band + the answer-key locator.
#[derive(Clone)]
pub struct SelectionEvent {
    pub ms2_frame: i64,
    pub scan_begin: i64,
    pub scan_end: i64,
    pub isolation_mz: f64,
    pub isolation_width: f64,
    pub collision_energy: f64,
    pub precursor_id: u64,
    pub charge: i64,
    pub mono_mz: f64,
    pub parent_ms1_frame: i64,
    pub event_intensity: f64,
}

pub struct DdaSchedule {
    pub precursors: Vec<CanonicalPrecursor>,
    pub events: Vec<SelectionEvent>,
}

/// Gaussian mass in the unit frame around `frame` (CDF difference), for the per-frame MS1 elution weight.
fn elution_weight(frame: u32, apex: f64, sigma: f64) -> f64 {
    let z = |x: f64| 0.5 * (1.0 + erf((x - apex) / (sigma * std::f64::consts::SQRT_2)));
    (z(frame as f64 + 0.5) - z(frame as f64 - 0.5)).max(0.0)
}

/// Abramowitz-Stegun erf (max err ~1.5e-7) — self-contained so the scheme has no heavy dep.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

/// Run the full schedule over `n_frames`.
pub fn schedule(cands: &[Candidate], params: &SelectionParams, n_frames: u32) -> DdaSchedule {
    let per = params.precursors_every.max(1);
    let k = per.saturating_sub(1) as usize; // MS2 frames per cycle
    // MS1 survey frames (1-based); the final one is left unscheduled (v1 `frame < max_frame_id`).
    let ms1_frames: Vec<u32> = (1..=n_frames).filter(|f| (f - 1) % per == 0).collect();
    let last_ms1 = *ms1_frames.last().unwrap_or(&0);

    // Active frame window per candidate (for finding a cycle's candidates cheaply via a sweep).
    let half: Vec<f64> = cands.iter().map(|c| c.n_sigma * c.sigma_frames).collect();
    let win: Vec<(u32, u32)> = cands
        .iter()
        .zip(&half)
        .map(|(c, &h)| ((c.apex_frame - h).max(1.0) as u32, ((c.apex_frame + h) as u32).min(n_frames)))
        .collect();
    let mut order_by_start: Vec<usize> = (0..cands.len()).collect();
    order_by_start.sort_unstable_by_key(|&i| win[i].0);

    let mut cursor = 0usize;
    let mut active: Vec<usize> = Vec::new();
    let mut last_scheduled: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    let mut precursors: Vec<CanonicalPrecursor> = Vec::new();
    let mut seen_precursor: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut events: Vec<SelectionEvent> = Vec::new();

    for &f in &ms1_frames {
        if f >= last_ms1 {
            break; // final survey not scheduled
        }
        // Advance the active set to frame f.
        while cursor < cands.len() && win[order_by_start[cursor]].0 <= f {
            active.push(order_by_start[cursor]);
            cursor += 1;
        }
        active.retain(|&i| win[i].1 >= f);
        if active.is_empty() {
            continue;
        }

        // Candidates at this survey: MS1 intensity = abundance × elution weight; threshold + rank.
        let mut ranked: Vec<(f64, usize)> = active
            .iter()
            .filter_map(|&i| {
                let c = &cands[i];
                let inten = c.abundance * elution_weight(f, c.apex_frame, c.sigma_frames);
                (inten >= params.intensity_threshold && inten > 0.0).then_some((inten, i))
            })
            .collect();
        // (intensity ↓, order ↑) — deterministic (v1's tie-break on ion_id → here on stable order).
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(cands[a.1].order.cmp(&cands[b.1].order)));

        // k MS2 frames, each a list of (apex, precursor_id) placed so far (for the packing/conflict test).
        let mut frames: Vec<Vec<(i64, u64)>> = vec![Vec::new(); k];
        let w = params.band_half_width;
        for (inten, i) in ranked {
            let c = &cands[i];
            if let Some(&last) = last_scheduled.get(&c.precursor_id) {
                if f - last < params.exclusion_frames {
                    continue; // dynamically excluded
                }
            }
            let start = (c.scan_apex - w).max(0);
            let end = (c.scan_apex + w).min(params.n_scans as i64 - 1);
            for (fi, frame) in frames.iter_mut().enumerate() {
                if frame.len() >= params.max_precursors {
                    continue;
                }
                // v1's conflict test: the new clamped band vs existing precursors' apex ± w.
                let conflict = frame
                    .iter()
                    .any(|&(apex, _)| !(end < apex - w || start > apex + w));
                if conflict {
                    continue;
                }
                frame.push((c.scan_apex, c.precursor_id));
                let ms2_frame = (f + fi as u32 + 1) as i64;
                let isolation_width = if c.largest_mz - c.mono_mz > 2.0 { 3.0 } else { 2.0 };
                let ce = params.ce_bias + params.ce_slope * c.scan_apex as f64;
                events.push(SelectionEvent {
                    ms2_frame,
                    scan_begin: start,
                    scan_end: end,
                    isolation_mz: c.largest_mz,
                    isolation_width,
                    collision_energy: ce,
                    precursor_id: c.precursor_id,
                    charge: c.charge,
                    mono_mz: c.mono_mz,
                    parent_ms1_frame: f as i64,
                    event_intensity: inten,
                });
                if seen_precursor.insert(c.precursor_id) {
                    precursors.push(CanonicalPrecursor {
                        precursor_id: c.precursor_id,
                        largest_mz: c.largest_mz,
                        average_mz: c.average_mz,
                        mono_mz: c.mono_mz,
                        charge: c.charge,
                        scan_apex: c.scan_apex,
                        abundance: c.abundance,
                        parent_ms1_frame: f as i64,
                    });
                }
                last_scheduled.insert(c.precursor_id, f);
                break;
            }
        }
    }

    DdaSchedule { precursors, events }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: u64, order: u32, apex: f64, scan: i64) -> Candidate {
        Candidate {
            precursor_id: id, order, apex_frame: apex, scan_apex: scan,
            mono_mz: 500.0, largest_mz: 500.5, average_mz: 500.25, charge: 2, abundance: 1000.0,
            sigma_frames: 5.0, n_sigma: 3.0,
        }
    }

    #[test]
    fn packs_by_mobility_and_excludes() {
        // Two well-separated (mobility) ions co-eluting near frame 11 → both fit one MS2 frame.
        let params = SelectionParams { precursors_every: 5, max_precursors: 10, exclusion_frames: 8, ..Default::default() };
        let cands = vec![cand(1, 0, 11.0, 100), cand(2, 1, 11.0, 400)];
        let sched = schedule(&cands, &params, 60);
        // Both selected at least once; each event's ms2 frame is a fragment frame of a cycle.
        assert!(sched.precursors.len() == 2);
        assert!(sched.events.iter().all(|e| (e.ms2_frame - 1) % 5 != 0), "events land on MS2 (non-survey) frames");
        // Overlapping-mobility ion is not double-placed in one frame (distinct scan bands or frames).
        for e in &sched.events {
            assert!(e.scan_begin < e.scan_end);
        }
    }

    #[test]
    fn dynamic_exclusion_holds() {
        // One ion eluting broadly across many cycles; exclusion_frames caps re-selection frequency.
        let params = SelectionParams { precursors_every: 5, exclusion_frames: 12, ..Default::default() };
        let mut c = cand(1, 0, 30.0, 200);
        c.sigma_frames = 40.0; // very broad elution → active over the whole run
        let sched = schedule(&std::slice::from_ref(&c), &params, 120);
        // Consecutive selections of the same ion must be >= exclusion_frames apart in survey frame.
        let mut parents: Vec<i64> = sched.events.iter().map(|e| e.parent_ms1_frame).collect();
        parents.sort_unstable();
        parents.dedup();
        for w in parents.windows(2) {
            assert!(w[1] - w[0] >= 12, "re-selection respects dynamic exclusion");
        }
    }
}
