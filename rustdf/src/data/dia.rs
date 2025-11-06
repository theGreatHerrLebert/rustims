use std::sync::Arc;
use crate::data::acquisition::AcquisitionMode;
use rustc_hash::FxHashMap;
use crate::data::handle::{IndexConverter, TimsData, TimsDataLoader};
use crate::data::meta::{
    read_dia_ms_ms_info, read_dia_ms_ms_windows, read_global_meta_sql, read_meta_data_sql,
    DiaMsMisInfo, DiaMsMsWindow, FrameMeta, GlobalMetaData,
};
use mscore::timstof::frame::{RawTimsFrame, TimsFrame};
use mscore::timstof::slice::TimsSlice;
use rand::prelude::IteratorRandom;
use rayon::iter::IntoParallelRefIterator;
use crate::cluster::peak::{build_frame_bin_view, expand_many_im_peaks_along_rt, FrameBinView, ImPeak1D, RtExpandParams, RtFrames, RtPeak1D};
use crate::cluster::utility::{bin_range_for_win, scan_mz_range, MzScale};
use crate::data::utility::merge_ranges;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::cluster::cluster::{attach_raw_points_for_spec_1d_threads, evaluate_spec_1d, make_specs_from_im_and_rt_groups_threads, BuildSpecOpts, ClusterResult1D, ClusterSpec1D, Eval1DOpts};
use std::collections::{HashMap};
use crate::cluster::candidates::{CandidateOpts, PrecursorSearchIndex};


#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ProgramSlice {
    pub mz_lo: f64,
    pub mz_hi: f64,
    pub scan_lo: u32,
    pub scan_hi: u32,
}

#[derive(Debug, Clone)]
pub struct Ms2GroupProgram {
    /// Raw per-row isolation windows (normalized, ascending, finite)
    pub mz_windows: Vec<(f32, f32)>,
    /// Raw per-row active scan ranges (normalized, inclusive)
    pub scan_ranges: Vec<(u32, u32)>,
    /// Union (convex hull) over all mz_windows for the group; None if no valid rows
    pub mz_union: Option<(f32, f32)>,
    /// Merged disjoint scan ranges (inclusive) over the group; empty = permissive fallback
    pub scan_unions: Vec<(u32, u32)>,
}

#[derive(Debug, Clone)]
pub struct DiaIndex {
    /// MS2 frame_id -> window_group
    pub frame_to_group: HashMap<u32, u32>,
    /// window_group -> MS2 frame_ids (sorted by **time**)
    pub group_to_frames: HashMap<u32, Vec<u32>>,
    /// window_group -> list of (mz_lo, mz_hi) isolation ranges (normalized)
    pub group_to_isolation: HashMap<u32, Vec<(f64, f64)>>,
    /// window_group -> list of (scan_lo, scan_hi) active scan ranges (normalized, inclusive)
    pub group_to_scan_ranges: HashMap<u32, Vec<(u32, u32)>>,
    /// window_group -> convex m/z union (lo, hi)
    pub group_to_mz_union: HashMap<u32, (f64, f64)>,
    /// window_group -> merged disjoint scan unions (inclusive)
    pub group_to_scan_unions: HashMap<u32, Vec<(u32, u32)>>,
    /// frame_id -> time (seconds)
    pub frame_time: HashMap<u32, f64>,
}

impl DiaIndex {
    pub fn new(meta: &[FrameMeta], info: &[DiaMsMisInfo], wins: &[DiaMsMsWindow]) -> Self {
        // helpers
        #[inline]
        fn norm_f64_pair(a: f64, b: f64) -> Option<(f64, f64)> {
            if !a.is_finite() || !b.is_finite() { return None; }
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            if hi > lo { Some((lo, hi)) } else { None }
        }
        #[inline]
        fn norm_u32_pair(a: u32, b: u32) -> Option<(u32, u32)> {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            Some((lo, hi))
        }
        fn merge_scan_ranges(mut ranges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
            if ranges.is_empty() { return ranges; }
            ranges.sort_unstable_by_key(|&(l, r)| (l, r));
            let mut out: Vec<(u32,u32)> = Vec::with_capacity(ranges.len());
            let mut cur = ranges[0];
            for &(l, r) in &ranges[1..] {
                if l <= cur.1.saturating_add(1) {
                    if r > cur.1 { cur.1 = r; }
                } else {
                    out.push(cur);
                    cur = (l, r);
                }
            }
            out.push(cur);
            out
        }

        // frame_id -> time
        let mut frame_time: HashMap<u32, f64> = HashMap::new();
        for m in meta {
            frame_time.insert(m.id as u32, m.time);
        }

        // frame_id -> group; group -> frames
        let mut frame_to_group: HashMap<u32, u32> = HashMap::new();
        let mut group_to_frames: HashMap<u32, Vec<u32>> = HashMap::new();
        for r in info {
            let fid = r.frame_id as u32;
            frame_to_group.insert(fid, r.window_group);
            group_to_frames.entry(r.window_group).or_default().push(fid);
        }

        // raw program rows
        let mut group_to_isolation: HashMap<u32, Vec<(f64, f64)>> = HashMap::new();
        let mut group_to_scan_ranges: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        for w in wins {
            let half = 0.5 * w.isolation_width;
            if half.is_finite() && half > 0.0 && w.isolation_mz.is_finite() {
                let lo = (w.isolation_mz - half) as f64;
                let hi = (w.isolation_mz + half) as f64;
                if let Some(p) = norm_f64_pair(lo, hi) {
                    group_to_isolation.entry(w.window_group).or_default().push(p);
                }
            }
            if let Some(p) = norm_u32_pair(w.scan_num_begin as u32, w.scan_num_end as u32) {
                group_to_scan_ranges.entry(w.window_group).or_default().push(p);
            }
        }

        // unions + sort frames by time
        let mut group_to_mz_union: HashMap<u32, (f64, f64)> = HashMap::new();
        let mut group_to_scan_unions: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();

        for (g, frames) in group_to_frames.iter_mut() {
            frames.sort_unstable_by(|&a, &b| {
                let ta = frame_time.get(&a).copied().unwrap_or(f64::NAN);
                let tb = frame_time.get(&b).copied().unwrap_or(f64::NAN);
                ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(v) = group_to_isolation.get(g) {
                if !v.is_empty() {
                    let lo = v.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
                    let hi = v.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
                    if lo.is_finite() && hi.is_finite() && hi > lo {
                        group_to_mz_union.insert(*g, (lo, hi));
                    }
                }
            }
            let merged = merge_scan_ranges(group_to_scan_ranges.get(g).cloned().unwrap_or_default());
            group_to_scan_unions.insert(*g, merged);
        }

        DiaIndex {
            frame_to_group,
            group_to_frames,
            group_to_isolation,
            group_to_scan_ranges,
            group_to_mz_union,
            group_to_scan_unions,
            frame_time,
        }
    }

    /// Convenience: materialize a program description for a group.
    pub fn program_for_group(&self, g: u32) -> Ms2GroupProgram {
        let mz_windows = self.group_to_isolation
            .get(&g)
            .map(|v| v.iter().map(|&(a,b)| (a as f32, b as f32)).collect())
            .unwrap_or_else(Vec::new);

        let scan_ranges = self.group_to_scan_ranges
            .get(&g)
            .cloned()
            .unwrap_or_else(Vec::new);

        let mz_union = self.group_to_mz_union
            .get(&g)
            .map(|&(a,b)| (a as f32, b as f32));

        let scan_unions = self.group_to_scan_unions
            .get(&g)
            .cloned()
            .unwrap_or_else(Vec::new);

        Ms2GroupProgram { mz_windows, scan_ranges, mz_union, scan_unions }
    }

    #[inline]
    pub fn mz_bounds_for_window_group_core(&self, g: u32) -> Option<(f32, f32)> {
        self.group_to_mz_union.get(&g).map(|&(a,b)| (a as f32, b as f32))
    }
    #[inline]
    pub fn scan_unions_for_window_group_core(&self, g: u32) -> Option<Vec<(usize, usize)>> {
        self.group_to_scan_unions.get(&g).map(|v| v.iter().map(|&(l,r)| (l as usize, r as usize)).collect())
    }
}

pub struct TimsDatasetDIA {
    pub loader: TimsDataLoader,
    pub global_meta_data: GlobalMetaData,
    pub meta_data: Vec<FrameMeta>,
    pub dia_ms_ms_info: Vec<DiaMsMisInfo>,
    pub dia_ms_ms_windows: Vec<DiaMsMsWindow>,
    pub dia_index: DiaIndex,
}

impl TimsDatasetDIA {
    pub fn new(
        bruker_lib_path: &str,
        data_path: &str,
        in_memory: bool,
        use_bruker_sdk: bool,
    ) -> Self {
        // TODO: error handling
        let global_meta_data = read_global_meta_sql(data_path).unwrap();
        let meta_data = read_meta_data_sql(data_path).unwrap();
        let dia_ms_mis_info = read_dia_ms_ms_info(data_path).unwrap();
        let dia_ms_ms_windows = read_dia_ms_ms_windows(data_path).unwrap();

        let scan_max_index = meta_data.iter().map(|x| x.num_scans).max().unwrap() as u32;
        let im_lower = global_meta_data.one_over_k0_range_lower;
        let im_upper = global_meta_data.one_over_k0_range_upper;

        let tof_max_index = global_meta_data.tof_max_index;
        let mz_lower = global_meta_data.mz_acquisition_range_lower;
        let mz_upper = global_meta_data.mz_acquisition_range_upper;

        let loader = match in_memory {
            true => TimsDataLoader::new_in_memory(
                bruker_lib_path,
                data_path,
                use_bruker_sdk,
                scan_max_index,
                im_lower,
                im_upper,
                tof_max_index,
                mz_lower,
                mz_upper,
            ),
            false => TimsDataLoader::new_lazy(
                bruker_lib_path,
                data_path,
                use_bruker_sdk,
                scan_max_index,
                im_lower,
                im_upper,
                tof_max_index,
                mz_lower,
                mz_upper,
            ),
        };

        let dia_index = DiaIndex::new(&meta_data, &dia_ms_mis_info, &dia_ms_ms_windows);

        TimsDatasetDIA {
            loader,
            global_meta_data,
            meta_data,
            dia_ms_ms_info: dia_ms_mis_info,
            dia_ms_ms_windows,
            dia_index,
        }
    }

    /// Build a reusable MS1 search index for candidate enumeration.
    /// Use when you plan to enumerate pairs repeatedly (e.g., different MS2 subsets or knobs).
    pub fn build_precursor_search_index(
        &self,
        ms1: &[ClusterResult1D],
        opts: &CandidateOpts,
    ) -> PrecursorSearchIndex {
        PrecursorSearchIndex::build(self, ms1, opts)
    }

    /// One-shot helper: build a transient index and enumerate all (ms2_idx, ms1_idx) pairs.
    /// If you’ll call this repeatedly, prefer `build_precursor_search_index` + `enumerate_pairs`.
    pub fn enumerate_ms2_ms1_pairs(
        &self,
        ms1: &[ClusterResult1D],
        ms2: &[ClusterResult1D],
        opts: &CandidateOpts,
    ) -> Vec<(usize, usize)> {
        let idx = PrecursorSearchIndex::build(self, ms1, opts);
        idx.enumerate_pairs(ms1, ms2, opts)
    }

    pub fn program_for_group(&self, g: u32) -> Ms2GroupProgram {
        self.dia_index.program_for_group(g)
    }

    /// Collect all program slices for a window group from DiaFrameMsMsWindows.
    /// Isolation window: [isolation_mz - width/2, isolation_mz + width/2]
    pub fn program_slices_for_group(&self, group: u32) -> Vec<ProgramSlice> {
        self.dia_ms_ms_windows
            .iter()
            .filter(|w| w.window_group == group)
            .map(|w| {
                let half = 0.5 * w.isolation_width;
                ProgramSlice {
                    mz_lo: (w.isolation_mz - half) as f64,
                    mz_hi: (w.isolation_mz + half) as f64,
                    scan_lo: w.scan_num_begin as u32,
                    scan_hi: w.scan_num_end as u32,
                }
            })
            .collect()
    }

    pub fn sample_precursor_signal(
        &self,
        num_frames: usize,
        max_intensity: f64,
        take_probability: f64,
    ) -> TimsFrame {
        // get all precursor frames
        let precursor_frames = self.meta_data.iter().filter(|x| x.ms_ms_type == 0);

        // randomly sample num_frames
        let mut rng = rand::thread_rng();
        let mut sampled_frames: Vec<TimsFrame> = Vec::new();

        // go through each frame and sample the data
        for frame in precursor_frames.choose_multiple(&mut rng, num_frames) {
            let frame_id = frame.id;
            let frame_data = self
                .loader
                .get_frame(frame_id as u32)
                .filter_ranged(0.0, 2000.0, 0, 1000, 0.0, 5.0, 1.0, max_intensity)
                .generate_random_sample(take_probability);
            sampled_frames.push(frame_data);
        }

        // get the first frame
        let mut sampled_frame = sampled_frames.remove(0);

        // sum all the other frames to the first frame
        for frame in sampled_frames {
            sampled_frame = sampled_frame + frame;
        }

        sampled_frame
    }

    pub fn sample_fragment_signal(
        &self,
        num_frames: usize,
        window_group: u32,
        max_intensity: f64,
        take_probability: f64,
    ) -> TimsFrame {
        // get all fragment frames, filter by window_group
        let fragment_frames: Vec<u32> = self
            .dia_ms_ms_info
            .iter()
            .filter(|x| x.window_group == window_group)
            .map(|x| x.frame_id)
            .collect();

        // randomly sample num_frames
        let mut rng = rand::thread_rng();
        let mut sampled_frames: Vec<TimsFrame> = Vec::new();

        // go through each frame and sample the data
        for frame_id in fragment_frames
            .into_iter()
            .choose_multiple(&mut rng, num_frames)
        {
            let frame_data = self
                .loader
                .get_frame(frame_id)
                .filter_ranged(0.0, 2000.0, 0, 1000, 0.0, 5.0, 1.0, max_intensity)
                .generate_random_sample(take_probability);
            sampled_frames.push(frame_data);
        }

        // get the first frame
        let mut sampled_frame = sampled_frames.remove(0);

        // sum all the other frames to the first frame
        for frame in sampled_frames {
            sampled_frame = sampled_frame + frame;
        }

        sampled_frame
    }

    /// All DIA window_group IDs present in the file (sorted unique).
    pub fn dia_window_groups(&self) -> Vec<u32> {
        let mut gs: Vec<u32> = self
            .dia_ms_ms_info
            .iter()
            .map(|x| x.window_group as u32)
            .collect();
        gs.sort_unstable();
        gs.dedup();
        gs
    }

    /// Map frame_id -> (time, ms_ms_type) for quick lookups.
    fn frame_time_map(&self) -> FxHashMap<u32, (f32, i64)> {
        let mut m = FxHashMap::default();
        for fm in &self.meta_data {
            m.insert(fm.id as u32, (fm.time as f32, fm.ms_ms_type));
        }
        m
    }

    /// RT-sorted **fragment** frames for a given DIA group.
    pub fn fragment_frame_ids_and_times_for_group_core(&self, window_group: u32) -> (Vec<u32>, Vec<f32>) {
        let time_map = self.frame_time_map();
        let mut rows: Vec<(u32, f32)> = self
            .dia_ms_ms_info
            .iter()
            .filter(|x| x.window_group == window_group)
            .filter_map(|x| {
                time_map.get(&(x.frame_id)).map(|(t, _ms2)| (x.frame_id, *t))
            })
            .collect();

        // Just in case: keep only MS2 frames (ms_ms_type != 0)
        rows.retain(|(fid, _)| time_map.get(fid).map(|(_, ty)| *ty != 0).unwrap_or(false));

        rows.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        let (ids, times): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
        (ids, times)
    }

    /// Merged, sorted **global scan** unions for this DIA group, from DiaFrameMsMsWindows.
    /// Returns None if there are no window rows for this group.
    pub fn scan_unions_for_window_group_core(&self, window_group: u32) -> Option<Vec<(usize, usize)>> {
        let ranges: Vec<(usize, usize)> = self
            .dia_ms_ms_windows
            .iter()
            .filter(|w| w.window_group as u32 == window_group)
            .map(|w| {
                let l = w.scan_num_begin as usize;
                let r = w.scan_num_end as usize;
                if l <= r { (l, r) } else { (r, l) }
            })
            .collect();
        if ranges.is_empty() {
            return None;
        }
        Some(merge_ranges(ranges))
    }

    /// m/z unions (min..max) for the group from DiaFrameMsMsWindows (wide clamp).
    pub fn mz_bounds_for_window_group_core(&self, window_group: u32) -> Option<(f32, f32)> {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        let mut hit = false;
        for w in &self.dia_ms_ms_windows {
            if w.window_group as u32 == window_group {
                let c = w.isolation_mz as f32;
                let half = 0.5f32 * (w.isolation_width as f32);
                lo = lo.min(c - half);
                hi = hi.max(c + half);
                hit = true;
            }
        }
        if hit && hi > lo && lo.is_finite() && hi.is_finite() {
            Some((lo, hi))
        } else {
            None
        }
    }
    /// Suggest an MzScale for a DIA group. Prefer table bounds; fallback to global acquisition range.
    pub fn mzscale_for_group(&self, window_group: u32, ppm_per_bin: f32) -> MzScale {
        let (lo, hi) = self
            .mz_bounds_for_window_group_core(window_group)
            .unwrap_or((
                self.global_meta_data.mz_acquisition_range_lower as f32,
                self.global_meta_data.mz_acquisition_range_upper as f32,
            ));
        MzScale::build(lo.max(1.0), hi, ppm_per_bin)
    }

    /// Conservative: derive an MzScale by scanning actual frame data (rarely needed; slower).
    pub fn mzscale_from_frames_scan(&self, frame_ids: &[u32], ppm_per_bin: f32) -> Option<MzScale> {
        let frames: Vec<_> = frame_ids.iter().map(|&fid| self.get_frame(fid)).collect();
        scan_mz_range(&frames).map(|(lo, hi)| MzScale::build(lo.max(1.0), hi, ppm_per_bin))
    }

    /// RT-sorted FRAGMENT frames + times for a DIA group, then converted into FrameBinView rows.
    /// `ppm_per_bin` sets the m/z granularity of CSR binning.
    pub fn make_rt_frames_for_group(
        &self,
        window_group: u32,
        ppm_per_bin: f32,
    ) -> RtFrames {
        let (ids, times) = self.fragment_frame_ids_and_times_for_group_core(window_group);
        assert!(!ids.is_empty(), "No MS2 frames for window_group {}", window_group);

        // global number of scans: max in this provenance
        let global_num_scans = self.meta_data
            .iter()
            .filter(|m| ids.binary_search(&(m.id as u32)).is_ok())
            .map(|m| m.num_scans as usize)
            .max()
            .unwrap_or(0);

        // scale for this group
        let scale = self.mzscale_for_group(window_group, ppm_per_bin);

        // Build CSR rows (parallel)
        let frames: Vec<FrameBinView> = ids.par_iter()
            .map(|&fid| build_frame_bin_view(self.get_frame(fid), &scale, global_num_scans))
            .collect();

        RtFrames { frames, frame_ids: ids, rt_times: times, scale: Arc::new(scale) }
    }

    /// RT-sorted PRECURSOR frames (ms_ms_type==0) into FrameBinView rows.
    pub fn make_rt_frames_for_precursor(
        &self,
        ppm_per_bin: f32,
    ) -> RtFrames {
        // Collect (frame_id, time) for precursor frames and sort by time
        let mut rows: Vec<(u32, f32, usize)> = self.meta_data
            .iter()
            .filter(|m| m.ms_ms_type == 0)
            .map(|m| (m.id as u32, m.time as f32, m.num_scans as usize))
            .collect();
        assert!(!rows.is_empty(), "No precursor (MS1) frames found");
        rows.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());

        let frame_ids: Vec<u32> = rows.iter().map(|r| r.0).collect();
        let rt_times:  Vec<f32> = rows.iter().map(|r| r.1).collect();
        let global_num_scans = rows.iter().map(|r| r.2).max().unwrap_or(1);

        // A broad scale for MS1 (use global acquisition range from metadata)
        let lo = self.global_meta_data.mz_acquisition_range_lower as f32;
        let hi = self.global_meta_data.mz_acquisition_range_upper as f32;
        let scale = MzScale::build(lo.max(1.0), hi, ppm_per_bin);

        let frames: Vec<FrameBinView> = frame_ids.par_iter()
            .map(|&fid| build_frame_bin_view(self.get_frame(fid), &scale, global_num_scans))
            .collect();

        RtFrames { frames, frame_ids, rt_times , scale: Arc::new(scale) }
    }
    /// Expand a batch of IM peaks along RT within a DIA `window_group`.
    /// Returns Vec<Vec<RtPeak1D>> aligned to `im_peaks` order.
    pub fn expand_rt_for_im_peaks_in_group(
        &self,
        window_group: u32,
        im_peaks: &[ImPeak1D],
        p: RtExpandParams,
        ppm_per_bin: f32,   // CSR resolution; keep consistent with how im_peaks were found
    ) -> Vec<Vec<RtPeak1D>> {
        if im_peaks.is_empty() {
            return Vec::new();
        }
        // Debug guard: single-provenance assumption
        debug_assert!(im_peaks.iter().all(|x| x.window_group == Some(window_group)));

        let rt = self.make_rt_frames_for_group(window_group, ppm_per_bin);
        debug_assert!(rt.is_consistent());
        expand_many_im_peaks_along_rt(
            im_peaks,
            &rt.frames,
            rt.ctx(),
            rt.scale.as_ref(),              // <-- pass the CSR scale
            p,
        )
    }

    /// Expand a batch of IM peaks along RT in PRECURSOR space (ms_ms_type==0).
    pub fn expand_rt_for_im_peaks_in_precursor(
        &self,
        im_peaks: &[ImPeak1D],
        p: RtExpandParams,
        ppm_per_bin: f32,
    ) -> Vec<Vec<RtPeak1D>> {
        if im_peaks.is_empty() {
            return Vec::new();
        }
        // Debug guard: precursor provenance
        debug_assert!(im_peaks.iter().all(|x| x.window_group.is_none()));

        let rt = self.make_rt_frames_for_precursor(ppm_per_bin);
        debug_assert!(rt.is_consistent());
        expand_many_im_peaks_along_rt(
            im_peaks,
            &rt.frames,
            rt.ctx(),
            rt.scale.as_ref(),             // <-- pass the CSR scale
            p,
        )
    }

    /// Evaluate a batch of 1D cluster specs on the given RtFrames.
    /// - Runs spec evaluation (marginals + moment fits) in parallel.
    /// - Optionally attaches raw points using the finalized m/z window and the spec’s IM/RT windows.
    /// - `num_threads == 0` => use rayon’s global pool.
    pub fn evaluate_specs_1d_threads(
        &self,
        rt_frames: &RtFrames,
        specs: &[ClusterSpec1D],
        opts: &Eval1DOpts,
        num_threads: usize,
    ) -> Vec<ClusterResult1D> {
        if specs.is_empty() { return Vec::new(); }

        let scale = &*rt_frames.scale;

        let run = || {
            specs.par_iter()
                .map(|spec| {
                    // 1) compute fits/marginals from CSR-binned `rt_frames`
                    let mut res = evaluate_spec_1d(rt_frames, spec, opts);

                    // 2) optional raw attachment (pull real frames only when requested)
                    if opts.attach.attach_points
                        && res.raw_sum > 0.0
                        && res.mz_fit.area > 0.0
                    {
                        // Recreate the final bin bounds from the result’s m/z window
                        // (evaluate_spec_1d may have refined μ±kσ already).
                        let (bin_lo, bin_hi) = bin_range_for_win(scale, res.mz_window);

                        // Use the spec’s IM and RT windows (evaluate_spec_1d currently does not refine IM)
                        let (im_lo, im_hi) = (spec.im_lo, spec.im_hi);
                        let (rt_lo, rt_hi) = res.rt_window;

                        let raw = attach_raw_points_for_spec_1d_threads(
                            self,
                            rt_frames,
                            bin_lo, bin_hi,
                            im_lo, im_hi,
                            rt_lo, rt_hi,
                            opts.attach.max_points,
                            num_threads.max(1),
                        );
                        res.raw_points = Some(raw);
                    } else {
                        res.raw_points = None;
                    }

                    res
                })
                .collect::<Vec<_>>()
        };

        if num_threads == 0 {
            run()
        } else {
            ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap().install(run)
        }
    }

    pub fn clusters_from_im_and_rt_groups(
        &self,
        rt_frames: &RtFrames,
        im_peaks: &[ImPeak1D],
        rt_groups: &[Vec<RtPeak1D>],
        build_opts: &BuildSpecOpts,
        eval_opts: &Eval1DOpts,
        require_rt_overlap: bool,
        num_threads: usize,
    ) -> Vec<ClusterResult1D> {
        let specs = make_specs_from_im_and_rt_groups_threads(
            im_peaks, rt_groups, rt_frames, build_opts, require_rt_overlap, num_threads
        );
        self.evaluate_specs_1d_threads(rt_frames, &specs, eval_opts, num_threads)
    }

    pub fn clusters_for_group(
        &self,
        window_group: u32,
        ppm_per_bin: f32,
        im_peaks: &[ImPeak1D],
        rt_params: RtExpandParams,
        build_opts: &BuildSpecOpts,
        eval_opts: &Eval1DOpts,
        require_rt_overlap: bool,
        num_threads: usize,
    ) -> Vec<ClusterResult1D> {
        // Guard: all IM peaks must belong to this DIA group
        debug_assert!(
            im_peaks.iter().all(|p| p.window_group == Some(window_group)),
            "clusters_for_group: some IM peaks have wrong or missing window_group"
        );

        let rt = self.make_rt_frames_for_group(window_group, ppm_per_bin);

        let rt_groups = expand_many_im_peaks_along_rt(
            im_peaks, &rt.frames, rt.ctx(), rt.scale.as_ref(), rt_params
        );

        // Force MS level 2 for DIA fragments
        let build_opts_ms2 = build_opts.with_ms_level(2);

        let specs = make_specs_from_im_and_rt_groups_threads(
            im_peaks, &rt_groups, &rt, &build_opts_ms2, require_rt_overlap, num_threads
        );

        let mut out = self.evaluate_specs_1d_threads(&rt, &specs, eval_opts, num_threads);

        // --- HARDENING: force ms_level and window_group on results ---
        for r in &mut out {
            r.ms_level = 2;
            if r.window_group.is_none() {
                r.window_group = Some(window_group);
            }
        }

        #[cfg(debug_assertions)]
        {
            let wrong = out.iter().filter(|c| c.ms_level != 2).count();
            let missg = out.iter().filter(|c| c.window_group != Some(window_group)).count();
            eprintln!("[clusters_for_group] g={} total={} wrong_ms_level={} missing_group={}",
                      window_group, out.len(), wrong, missg);
            debug_assert!(wrong == 0);
            debug_assert!(missg == 0);
        }

        out
    }
    pub fn clusters_for_precursor(
        &self,
        ppm_per_bin: f32,
        im_peaks: &[ImPeak1D],
        rt_params: RtExpandParams,
        build_opts: &BuildSpecOpts,
        eval_opts: &Eval1DOpts,
        require_rt_overlap: bool,
        num_threads: usize,
    ) -> Vec<ClusterResult1D> {
        // Guard: all IM peaks must be MS1 (no window_group)
        debug_assert!(
            im_peaks.iter().all(|p| p.window_group.is_none()),
            "clusters_for_precursor: IM peaks unexpectedly carry a window_group"
        );

        let rt = self.make_rt_frames_for_precursor(ppm_per_bin);

        let rt_groups = expand_many_im_peaks_along_rt(
            im_peaks, &rt.frames, rt.ctx(), rt.scale.as_ref(), rt_params
        );

        // Force MS level 1 for precursor
        let build_opts_ms1 = build_opts.with_ms_level(1);

        let specs = make_specs_from_im_and_rt_groups_threads(
            im_peaks, &rt_groups, &rt, &build_opts_ms1, require_rt_overlap, num_threads
        );

        self.evaluate_specs_1d_threads(&rt, &specs, eval_opts, num_threads)
    }
}

impl TimsData for TimsDatasetDIA {
    fn get_frame(&self, frame_id: u32) -> TimsFrame {
        self.loader.get_frame(frame_id)
    }

    fn get_raw_frame(&self, frame_id: u32) -> RawTimsFrame {
        self.loader.get_raw_frame(frame_id)
    }

    fn get_slice(&self, frame_ids: Vec<u32>, num_threads: usize) -> TimsSlice {
        self.loader.get_slice(frame_ids, num_threads)
    }
    fn get_acquisition_mode(&self) -> AcquisitionMode {
        self.loader.get_acquisition_mode().clone()
    }

    fn get_frame_count(&self) -> i32 {
        self.loader.get_frame_count()
    }

    fn get_data_path(&self) -> &str {
        &self.loader.get_data_path()
    }
}

impl IndexConverter for TimsDatasetDIA {
    fn tof_to_mz(&self, frame_id: u32, tof_values: &Vec<u32>) -> Vec<f64> {
        self.loader
            .get_index_converter()
            .tof_to_mz(frame_id, tof_values)
    }

    fn mz_to_tof(&self, frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32> {
        self.loader
            .get_index_converter()
            .mz_to_tof(frame_id, mz_values)
    }

    fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: &Vec<u32>) -> Vec<f64> {
        self.loader
            .get_index_converter()
            .scan_to_inverse_mobility(frame_id, scan_values)
    }

    fn inverse_mobility_to_scan(
        &self,
        frame_id: u32,
        inverse_mobility_values: &Vec<f64>,
    ) -> Vec<u32> {
        self.loader
            .get_index_converter()
            .inverse_mobility_to_scan(frame_id, inverse_mobility_values)
    }
}
