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
use crate::cluster::peak::{build_frame_bin_view, expand_many_im_peaks_along_rt, FrameBinView, ImPeak1D, RtExpandParams, RtFrames};
use crate::cluster::utility::{TofScale};
use crate::data::utility::merge_ranges;
use rayon::prelude::*;
use std::collections::{HashMap};
use rayon::ThreadPoolBuilder;
use crate::cluster::candidates::{build_pseudo_spectra_end_to_end, PseudoBuildResult, ScoreOpts};
use crate::cluster::cluster::{attach_raw_points_for_spec_1d_in_ctx, bin_range_for_win, build_scan_slices, decorate_with_mz_for_cluster, evaluate_spec_1d, make_specs_from_im_and_rt_groups_threads, BuildSpecOpts, ClusterResult1D, ClusterSpec1D, Eval1DOpts, RawAttachContext, RawPoints, ScanSlice};
use crate::cluster::feature::SimpleFeature;
use crate::cluster::pseudo::{PseudoSpecOpts};
use crate::cluster::scoring::CandidateOpts;

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
            let fid = r.frame_id;
            frame_to_group.insert(fid, r.window_group);
            group_to_frames.entry(r.window_group).or_default().push(fid);
        }

        // raw program rows
        let mut group_to_isolation: HashMap<u32, Vec<(f64, f64)>> = HashMap::new();
        let mut group_to_scan_ranges: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        for w in wins {
            let half = 0.5 * w.isolation_width;
            if half.is_finite() && half > 0.0 && w.isolation_mz.is_finite() {
                let lo = w.isolation_mz - half;
                let hi = w.isolation_mz + half;
                if let Some(p) = norm_f64_pair(lo, hi) {
                    group_to_isolation.entry(w.window_group).or_default().push(p);
                }
            }
            if let Some(p) = norm_u32_pair(w.scan_num_begin, w.scan_num_end) {
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
                    mz_lo: w.isolation_mz - half,
                    mz_hi: w.isolation_mz + half,
                    scan_lo: w.scan_num_begin,
                    scan_hi: w.scan_num_end,
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
                .filter_ranged(0.0, 2000.0, 0, 1000, 0.0, 5.0, 1.0, max_intensity, 0, i32::MAX)
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
                .filter_ranged(0.0, 2000.0, 0, 1000, 0.0, 5.0, 1.0, max_intensity, 0, i32::MAX)
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
            .map(|x| x.window_group)
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
            .filter(|w| w.window_group == window_group)
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
            if w.window_group == window_group {
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

    /// RT-sorted FRAGMENT frames + times for a DIA group, converted into FrameBinView rows.
    /// `tof_step` controls TOF granularity of CSR binning.
    pub fn make_rt_frames_for_group(&self, window_group: u32, tof_step: i32) -> RtFrames {
        assert!(tof_step > 0);

        let (ids, times) = self.fragment_frame_ids_and_times_for_group_core(window_group);
        assert!(
            !ids.is_empty(),
            "No MS2 frames for window_group {}",
            window_group
        );

        let global_num_scans = self.meta_data
            .iter()
            .filter(|m| ids.binary_search(&(m.id as u32)).is_ok())
            .map(|m| m.num_scans as usize)
            .max()
            .unwrap_or(0);

        // TOF scale for this group
        let scale = self.tof_scale_for_group(window_group, tof_step);

        let frames: Vec<FrameBinView> = ids.par_iter()
            .map(|&fid| build_frame_bin_view(self.get_frame(fid), &scale, global_num_scans))
            .collect();

        RtFrames {
            frames,
            frame_ids: ids,
            rt_times: times,
            scale: Arc::new(scale),
        }
    }

    /// RT-sorted PRECURSOR frames (ms_ms_type == 0) into FrameBinView rows.
    /// `tof_step` controls TOF granularity of CSR binning.
    pub fn make_rt_frames_for_precursor(&self, tof_step: i32) -> RtFrames {
        assert!(tof_step > 0);

        let mut rows: Vec<(u32, f32, usize)> = self.meta_data
            .iter()
            .filter(|m| m.ms_ms_type == 0)
            .map(|m| (m.id as u32, m.time as f32, m.num_scans as usize))
            .collect();

        assert!(!rows.is_empty(), "No precursor (MS1) frames found");
        rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let frame_ids: Vec<u32> = rows.iter().map(|r| r.0).collect();
        let rt_times:  Vec<f32> = rows.iter().map(|r| r.1).collect();
        let global_num_scans = rows.iter().map(|r| r.2).max().unwrap_or(1);

        // Build TOF scale from precursor frames
        let frames_for_scale: Vec<_> = frame_ids.iter().map(|&fid| self.get_frame(fid)).collect();
        let scale = TofScale::build_from_tof(&frames_for_scale, tof_step)
            .expect("make_rt_frames_for_precursor: failed to build TOF scale");

        let frames: Vec<FrameBinView> = frame_ids.par_iter()
            .map(|&fid| build_frame_bin_view(self.get_frame(fid), &scale, global_num_scans))
            .collect();

        RtFrames {
            frames,
            frame_ids,
            rt_times,
            scale: Arc::new(scale),
        }
    }

    /// Build a TOF-based scale for a DIA group.
    /// `tof_step = 1` → max TOF resolution; larger steps downsample.
    pub fn tof_scale_for_group(&self, window_group: u32, tof_step: i32) -> TofScale {
        assert!(tof_step > 0);

        let (ids, _times) = self.fragment_frame_ids_and_times_for_group_core(window_group);
        assert!(
            !ids.is_empty(),
            "tof_scale_for_group: no MS2 frames for window_group {}",
            window_group
        );

        let frames: Vec<_> = ids.iter().map(|&fid| self.get_frame(fid)).collect();

        TofScale::build_from_tof(&frames, tof_step)
            .expect("tof_scale_for_group: failed to build TOF scale (empty or degenerate)")
    }

    /// Conservative: derive a TOF scale by scanning actual frame data.
    pub fn tof_scale_from_frames_scan(
        &self,
        frame_ids: &[u32],
        tof_step: i32,
    ) -> Option<TofScale> {
        assert!(tof_step > 0);
        let frames: Vec<_> = frame_ids.iter().map(|&fid| self.get_frame(fid)).collect();
        TofScale::build_from_tof(&frames, tof_step)
    }

    /// Internal helper: go from a set of IM peaks to evaluated 1D clusters
    /// on a given RtFrames grid (precursor or DIA group).
    fn clusters_for_im_peaks_on_rt_frames(
        &self,
        rt: RtFrames,
        im_peaks: &[ImPeak1D],
        rt_params: RtExpandParams,
        build_opts: &BuildSpecOpts,
        eval_opts: &Eval1DOpts,
        require_rt_overlap: bool,
        num_threads: usize,
    ) -> Vec<ClusterResult1D> {
        // 1) Expand IM peaks along RT on this grid
        let rt_groups = expand_many_im_peaks_along_rt(
            im_peaks,
            &rt.frames,
            rt.ctx(),
            rt.scale.as_ref(),
            rt_params,
        );

        // 2) Build 1D specs (RT × IM × TOF-window)
        let specs: Vec<ClusterSpec1D> = make_specs_from_im_and_rt_groups_threads(
            im_peaks,
            &rt_groups,
            &rt,
            build_opts,
            require_rt_overlap,
            num_threads,
        );

        // 3) Evaluate all specs on this RtFrames grid
        self.evaluate_specs_1d_threads(&rt, &specs, eval_opts, num_threads)
    }

    /// Evaluate a batch of 1D cluster specs on the given RtFrames.
    /// - Runs spec evaluation (marginals + moment fits) in parallel.
    /// - Optionally attaches raw points using the finalized **TOF-bin window**
    ///   and the spec’s IM/RT windows.
    /// - `num_threads == 0` => use rayon’s global pool.
    pub fn evaluate_specs_1d_threads(
        &self,
        rt_frames: &RtFrames,
        specs: &[ClusterSpec1D],
        opts: &Eval1DOpts,
        num_threads: usize,
    ) -> Vec<ClusterResult1D> {
        if specs.is_empty() {
            return Vec::new();
        }

        let scale = &*rt_frames.scale;

        let run = || {
            // Pre-build raw attach context once if needed
            let attach_ctx = if opts.attach.attach_points {
                let frame_ids_local = rt_frames.frame_ids.clone();
                let slice = self.get_slice(frame_ids_local.clone(), num_threads.max(1));
                let scan_slices = slice
                    .frames
                    .iter()
                    .map(|fr| build_scan_slices(fr))
                    .collect::<Vec<_>>();
                let rt_axis_sec = rt_frames.rt_times.clone();

                Some(RawAttachContext {
                    slice,
                    scan_slices,
                    frame_ids_local,
                    rt_axis_sec,
                })
            } else {
                None
            };

            specs
                .par_iter()
                .map(|spec| {
                    // 1) core 1D evaluation (RT/IM/TOF fits etc.)
                    let mut res = evaluate_spec_1d(rt_frames, spec, opts);

                    // 2) decorate with m/z axis + mz_fit/mz_window
                    //
                    //    - uses the already-computed TOF window in `spec.tof_win`
                    //    - does *not* depend on raw_points being attached
                    //    - keeps the logic local and cheap
                    decorate_with_mz_for_cluster(self, rt_frames, &mut res);(self, rt_frames, spec, scale, &mut res);

                    // 3) optional raw point attachment (using shared context)
                    if let Some(ref ctx) = attach_ctx {
                        if opts.attach.attach_points && res.raw_sum > 0.0 && res.tof_fit.area > 0.0 {
                            let (bin_lo, bin_hi) = bin_range_for_win(scale, spec.tof_win);
                            let (im_lo, im_hi) = (spec.im_lo, spec.im_hi);
                            let (rt_lo, rt_hi) = (spec.rt_lo, spec.rt_hi);

                            let raw = attach_raw_points_for_spec_1d_in_ctx(
                                ctx,
                                scale,
                                bin_lo,
                                bin_hi,
                                im_lo,
                                im_hi,
                                rt_lo,
                                rt_hi,
                                opts.attach.max_points,
                            );
                            res.raw_points = Some(raw);
                        }
                    }

                    res
                })
                .collect::<Vec<_>>()
        };

        if num_threads == 0 {
            run()
        } else {
            ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap()
                .install(run)
        }
    }

    pub fn clusters_for_group(
        &self,
        window_group: u32,
        tof_step: i32,
        im_peaks: &[ImPeak1D],
        rt_params: RtExpandParams,
        build_opts: &BuildSpecOpts,
        eval_opts: &Eval1DOpts,
        require_rt_overlap: bool,
        num_threads: usize,
    ) -> Vec<ClusterResult1D> {
        // Guard: all IM peaks must belong to this DIA group
        debug_assert!(
            im_peaks
                .iter()
                .all(|p| p.window_group == Some(window_group)),
            "clusters_for_group: some IM peaks have wrong or missing window_group"
        );

        // Build RT+TOF grid for this DIA isolation window group.
        let rt = self.make_rt_frames_for_group(window_group, tof_step);

        // Force MS level 2 for DIA fragments
        let build_opts_ms2 = build_opts.with_ms_level(2);

        self.clusters_for_im_peaks_on_rt_frames(
            rt,
            im_peaks,
            rt_params,
            &build_opts_ms2,
            eval_opts,
            require_rt_overlap,
            num_threads,
        )
    }

    pub fn clusters_for_precursor(
        &self,
        tof_step: i32,
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

        // RT+TOF grid for precursor (MS1) frames
        let rt = self.make_rt_frames_for_precursor(tof_step);

        // Force MS level 1 for precursor
        let build_opts_ms1 = build_opts.with_ms_level(1);

        self.clusters_for_im_peaks_on_rt_frames(
            rt,
            im_peaks,
            rt_params,
            &build_opts_ms1,
            eval_opts,
            require_rt_overlap,
            num_threads,
        )
    }

    /// High-level DIA → pseudo-DDA builder tied to the dataset.
    ///
    /// Uses CandidateOpts::default() and ScoreOpts::default().
    pub fn build_pseudo_spectra_from_clusters(
        &self,
        ms1: &[ClusterResult1D],
        ms2: &[ClusterResult1D],
        features: Option<&[SimpleFeature]>,
        pseudo_opts: &PseudoSpecOpts,
    ) -> PseudoBuildResult {
        let cand_opts = CandidateOpts::default();
        let score_opts = ScoreOpts::default();

        build_pseudo_spectra_end_to_end(
            self,
            ms1,
            ms2,
            features,
            &cand_opts,
            &score_opts,
            pseudo_opts,
        )
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

pub fn attach_raw_points_for_spec_1d_threads(
    ds: &TimsDatasetDIA,
    rt_frames: &RtFrames,
    final_bin_lo: usize,
    final_bin_hi: usize,
    final_im_lo: usize,
    final_im_hi: usize,
    final_rt_lo: usize,
    final_rt_hi: usize,
    max_points: Option<usize>,
    num_threads: usize,
) -> RawPoints {
    let scale = &*rt_frames.scale;

    // Defensive clamp of bin indices to the valid bin range.
    let n_bins = scale.num_bins();
    if n_bins == 0 {
        return RawPoints {
            mz: Vec::new(),
            rt: Vec::new(),
            im: Vec::new(),
            scan: Vec::new(),
            intensity: Vec::new(),
            tof: Vec::new(),
            frame: Vec::new(),
        };
    }

    let mut bin_lo = final_bin_lo.min(n_bins.saturating_sub(1));
    let mut bin_hi = final_bin_hi.min(n_bins.saturating_sub(1));
    if bin_lo > bin_hi {
        std::mem::swap(&mut bin_lo, &mut bin_hi);
    }

    // Axis window from bin edges:
    //   [edges[bin_lo], cushion_hi_edge(edges[bin_hi+1])]
    let mut axis_lo = scale.edges[bin_lo];
    let hi_edge_idx = (bin_hi + 1).min(scale.edges.len().saturating_sub(1));
    let mut axis_hi = cushion_hi_edge(scale, scale.edges[hi_edge_idx]);

    // RT frames and slice
    let frame_ids_local = rt_frames.frame_ids[final_rt_lo..=final_rt_hi].to_vec();
    let slice = ds.get_slice(frame_ids_local.clone(), num_threads.max(1));
    let scan_slices: Vec<Vec<ScanSlice>> =
        slice.frames.iter().map(|fr| build_scan_slices(fr)).collect();

    // 1) Count
    let mut total = 0usize;
    for (fi, fr) in slice.frames.iter().enumerate() {
        let mz = &fr.ims_frame.mz;
        for sl in &scan_slices[fi] {
            if sl.scan < final_im_lo || sl.scan > final_im_hi {
                continue;
            }
            let l = lower_bound_in(mz, sl.start, sl.end, axis_lo);
            let r = upper_bound_in(mz, sl.start, sl.end, axis_hi);
            total += r.saturating_sub(l);
        }
    }

    // 2) Last-chance widen: if empty, expand by ±1 bin
    if total == 0 {
        let lo_idx = bin_lo.saturating_sub(1);
        let hi_edge_idx_wide =
            (bin_hi + 2).min(n_bins).min(scale.edges.len().saturating_sub(1));

        let try_lo = scale.edges[lo_idx];
        let try_hi = cushion_hi_edge(scale, scale.edges[hi_edge_idx_wide]);

        let mut total2 = 0usize;
        for (fi, fr) in slice.frames.iter().enumerate() {
            let mz = &fr.ims_frame.mz;
            for sl in &scan_slices[fi] {
                if sl.scan < final_im_lo || sl.scan > final_im_hi {
                    continue;
                }
                let l = lower_bound_in(mz, sl.start, sl.end, try_lo);
                let r = upper_bound_in(mz, sl.start, sl.end, try_hi);
                total2 += r.saturating_sub(l);
            }
        }

        if total2 > 0 {
            total = total2;
            axis_lo = try_lo;
            axis_hi = try_hi;
        }
    }

    // Still empty → bail with empty container
    if total == 0 {
        return RawPoints {
            mz: Vec::new(),
            rt: Vec::new(),
            im: Vec::new(),
            scan: Vec::new(),
            intensity: Vec::new(),
            tof: Vec::new(),
            frame: Vec::new(),
        };
    }

    let stride = max_points.map(|cap| thin_stride(total, cap)).unwrap_or(1);
    let reserve = total / stride + 8;

    let mut pts = RawPoints {
        mz: Vec::with_capacity(reserve),
        rt: Vec::with_capacity(reserve),
        im: Vec::with_capacity(reserve),
        scan: Vec::with_capacity(reserve),
        intensity: Vec::with_capacity(reserve),
        tof: Vec::with_capacity(reserve),
        frame: Vec::with_capacity(reserve),
    };

    let rt_axis_sec = rt_frames.rt_times[final_rt_lo..=final_rt_hi].to_vec();

    // 3) Extraction with thinning
    let mut seen = 0usize;
    for (fi, fr) in slice.frames.iter().enumerate() {
        let mz   = &fr.ims_frame.mz;
        let it   = &fr.ims_frame.intensity;
        let ims  = &fr.ims_frame.mobility;
        let tofs = &fr.tof;

        let len_all = mz.len().min(it.len()).min(ims.len()).min(tofs.len());
        let rt_val = rt_axis_sec[fi];
        let frame_id = frame_ids_local[fi];

        for sl in &scan_slices[fi] {
            let s_abs = sl.scan;
            if s_abs < final_im_lo || s_abs > final_im_hi {
                continue;
            }

            let mut l = lower_bound_in(mz, sl.start, sl.end, axis_lo);
            let mut r = upper_bound_in(mz, sl.start, sl.end, axis_hi);
            if r > len_all {
                r = len_all;
            }
            if l >= r {
                continue;
            }

            while l < r {
                if stride == 1 || (seen % stride == 0) {
                    pts.mz.push(mz[l] as f32);
                    pts.rt.push(rt_val);
                    pts.im.push(ims[l] as f32);
                    pts.scan.push(s_abs as u32);
                    pts.intensity.push(it[l] as f32);
                    pts.frame.push(frame_id);
                    pts.tof.push(tofs[l]);
                }
                seen += 1;
                l += 1;
            }
        }
    }

    pts
}

// ----------------------------------------------------------------------
// Helpers for axis-based extraction (no ppm, no mz_min/mz_max needed)
// ----------------------------------------------------------------------

/// Slightly "cushion" the high edge used for upper_bound search.
///
/// Semantics:
/// - `hi_edge` is typically `scale.edges[bin_hi + 1]` (the upper edge of the last bin).
/// - We nudge it outward by a tiny fraction of a *typical* bin width so values
///   that sit exactly on the edge aren't accidentally excluded.
/// - This is axis-generic: it works whether the axis is TOF or m/z-like.
#[inline]
fn cushion_hi_edge(scale: &TofScale, hi_edge: f32) -> f32 {
    let edges = &scale.edges;
    if edges.len() >= 2 {
        // Use a small fraction of the first bin width as epsilon.
        let bw = (edges[1] - edges[0]).abs().max(1e-6);
        hi_edge + 0.01 * bw
    } else {
        hi_edge
    }
}

#[inline]
fn lower_bound_in(mz: &[f64], start: usize, end: usize, x: f32) -> usize {
    let mut lo = start;
    let mut hi = end;
    let xf = x as f64;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if mz[mid] < xf {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
fn upper_bound_in(mz: &[f64], start: usize, end: usize, x: f32) -> usize {
    let mut lo = start;
    let mut hi = end;
    let xf = x as f64;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if mz[mid] <= xf {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
fn thin_stride(total: usize, cap: usize) -> usize {
    if cap == 0 || total <= cap {
        1
    } else {
        (total + cap - 1) / cap
    }
}