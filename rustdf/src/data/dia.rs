use std::sync::Arc;
use crate::data::acquisition::AcquisitionMode;
use crate::data::handle::{IndexConverter, TimsData, TimsDataLoader};
use crate::data::meta::{
    read_dia_ms_ms_info, read_dia_ms_ms_windows, read_global_meta_sql, read_meta_data_sql,
    DiaMsMisInfo, DiaMsMsWindow, FrameMeta, GlobalMetaData,
};
use mscore::timstof::frame::{RawTimsFrame, TimsFrame};
use mscore::timstof::slice::TimsSlice;
use rand::prelude::IteratorRandom;
use rustc_hash::FxHashMap;
use crate::cluster::cluster_eval::{evaluate_clusters_3d, ClusterResult, ClusterSpec, EvalOptions};
use crate::cluster::feature::{build_features_from_envelopes, AveragineLut, Envelope, Feature,
                              FeatureBuildParams, GroupingParams};
use crate::cluster::utility::{build_dense_rt_by_mz_ppm, pick_peaks_all_rows, RtPeak1D, RtIndex, build_dense_im_by_rtpeaks_ppm, ImIndex, build_dense_rt_by_mz_ppm_for_group, build_dense_im_by_rtpeaks_ppm_for_group};

#[derive(Clone, Copy, Debug)]
pub struct DiaGroupBounds {
    pub mz_lo: f32,
    pub mz_hi: f32,
    pub scan_lo: usize,
    pub scan_hi: usize,
}

pub struct TimsDatasetDIA {
    pub loader: TimsDataLoader,
    pub global_meta_data: GlobalMetaData,
    pub meta_data: Vec<FrameMeta>,
    pub dia_ms_mis_info: Vec<DiaMsMisInfo>,
    pub dia_ms_ms_windows: Vec<DiaMsMsWindow>,
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

        TimsDatasetDIA {
            loader,
            global_meta_data,
            meta_data,
            dia_ms_mis_info,
            dia_ms_ms_windows,
        }
    }

    pub fn group_bounds(&self, window_group: u32) -> Option<DiaGroupBounds> {
        // DiaFrameMsMsWindows has one row per group (Bruker schema); if multiple, fuse conservatively.
        let mut mz_lo = f32::INFINITY;
        let mut mz_hi = f32::NEG_INFINITY;
        let mut scan_lo = usize::MAX;
        let mut scan_hi = 0usize;

        let mut found = false;
        for w in &self.dia_ms_ms_windows {
            if w.window_group == window_group {
                found = true;
                let lo = (w.isolation_mz - 0.5 * w.isolation_width) as f32;
                let hi = (w.isolation_mz + 0.5 * w.isolation_width) as f32;
                mz_lo = mz_lo.min(lo);
                mz_hi = mz_hi.max(hi);
                scan_lo = scan_lo.min(w.scan_num_begin as usize);
                scan_hi = scan_hi.max(w.scan_num_end as usize);
            }
        }
        if !found || !mz_lo.is_finite() || !mz_hi.is_finite() || mz_hi <= mz_lo {
            return None;
        }
        Some(DiaGroupBounds { mz_lo, mz_hi, scan_lo, scan_hi })
    }

    pub fn frame_ids_for_group(&self, window_group: u32) -> Vec<u32> {
        let mut fids: Vec<u32> = self.dia_ms_mis_info
            .iter()
            .filter(|x| x.window_group == window_group)
            .map(|x| x.frame_id)
            .collect();
        // sort by RT time to be safe
        fids.sort_unstable_by_key(|fid| {
            self.meta_data.iter().find(|m| m.id as u32 == *fid)
                .map(|m| m.time as i64).unwrap_or(0)
        });
        fids
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
            .dia_ms_mis_info
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

    pub fn get_dense_rt_by_mz_ppm(
        &self,
        maybe_sigma_frames: Option<f32>,
        truncate: f32,
        ppm_per_bin: f32,       // <— NEW: constant-ppm bin width
        mz_pad_ppm: f32,        // (optional) pad the min/max by some ppm to avoid edge cutoffs
        num_threads: usize,
    ) -> RtIndex {
        build_dense_rt_by_mz_ppm(self, maybe_sigma_frames, truncate, ppm_per_bin, mz_pad_ppm, num_threads)
    }

    /// Build dense matrix, optionally smooth, then pick peaks.
    /// Returns the RtIndex (with data + frame_times) and the list of peaks.
    pub fn pick_peaks_dense(
        &self,
        maybe_sigma_frames: Option<f32>,
        truncate: f32,
        ppm_per_bin: f32,       // <— NEW: constant-ppm bin width
        mz_pad_ppm: f32,        // (optional) pad the min/max by
        num_threads: usize,
        min_prom: f32,
        min_distance: usize,
        min_width: usize,
        pad_left: usize,          // NEW
        pad_right: usize,         // NEW

    ) -> (RtIndex, Vec<RtPeak1D>) {
        let rt = self.get_dense_rt_by_mz_ppm(maybe_sigma_frames, truncate, ppm_per_bin, mz_pad_ppm, num_threads);
        let data_raw_slice = rt.data_raw.as_deref().unwrap_or(rt.data.as_slice());

        let peaks = pick_peaks_all_rows(
            rt.data.as_slice(),
            data_raw_slice,
            rt.rows,
            rt.cols,
            Some(&rt.frame_times),
            min_prom,
            min_distance,
            min_width,
            pad_left,
            pad_right,
            Some(&rt.scale.centers),
        );
        (rt, peaks)
    }

    /// Build m/z × RT for a specific DIA window group (fragments only).
    /// If `clamp_to_group` is true, use the instrument window m/z for the scale.
    pub fn get_dense_rt_by_mz_ppm_for_group(
        &self,
        window_group: u32,
        maybe_sigma_frames: Option<f32>,
        truncate: f32,
        ppm_per_bin: f32,
        mz_pad_ppm: f32,
        clamp_to_group: bool,
        num_threads: usize,
    ) -> RtIndex {
        let frame_ids = self.frame_ids_for_group(window_group);

        let clamp = if clamp_to_group {
            self.group_bounds(window_group).map(|b| (b.mz_lo, b.mz_hi))
        } else {
            None
        };

        build_dense_rt_by_mz_ppm_for_group(
            self,
            frame_ids,
            ppm_per_bin,
            mz_pad_ppm,
            maybe_sigma_frames,
            truncate,
            clamp,
            num_threads,
        )
    }

    /// Build + pick RT peaks for a specific DIA window group.
    pub fn pick_peaks_dense_for_group(
        &self,
        window_group: u32,
        maybe_sigma_frames: Option<f32>,
        truncate: f32,
        ppm_per_bin: f32,
        mz_pad_ppm: f32,
        clamp_to_group: bool,
        num_threads: usize,
        min_prom: f32,
        min_distance: usize,
        min_width: usize,
        pad_left: usize,
        pad_right: usize,
    ) -> (RtIndex, Vec<RtPeak1D>) {
        let rt = self.get_dense_rt_by_mz_ppm_for_group(
            window_group,
            maybe_sigma_frames,
            truncate,
            ppm_per_bin,
            mz_pad_ppm,
            clamp_to_group,
            num_threads,
        );

        let data_raw_slice = rt.data_raw.as_deref().unwrap_or(rt.data.as_slice());

        let peaks = pick_peaks_all_rows(
            rt.data.as_slice(),
            data_raw_slice,
            rt.rows,
            rt.cols,
            Some(&rt.frame_times),
            min_prom,
            min_distance,
            min_width,
            pad_left,
            pad_right,
            Some(&rt.scale.centers),
        );
        (rt, peaks)
    }

    /// Merge each group's isolation m/z bands into disjoint intervals.
    pub fn group_mz_unions(&self) -> FxHashMap<u32, Vec<(f32,f32)>> {
        let mut mp: FxHashMap<u32, Vec<(f32,f32)>> = FxHashMap::default();
        for w in &self.dia_ms_ms_windows {
            let lo = (w.isolation_mz - 0.5 * w.isolation_width) as f32;
            let hi = (w.isolation_mz + 0.5 * w.isolation_width) as f32;
            mp.entry(w.window_group).or_default().push((lo.min(hi), lo.max(hi)));
        }
        for v in mp.values_mut() {
            v.sort_by(|a,b| a.0.total_cmp(&b.0));
            let mut merged: Vec<(f32,f32)> = Vec::new();
            for (lo,hi) in v.drain(..) {
                if let Some(last) = merged.last_mut() {
                    if lo <= last.1 { last.1 = last.1.max(hi); } else { merged.push((lo,hi)); }
                } else { merged.push((lo,hi)); }
            }
            *v = merged;
        }
        mp
    }

    #[inline]
    pub fn groups_covering_mz(
        &self,
        mz: f32,
        unions: &FxHashMap<u32, Vec<(f32,f32)>>
    ) -> Vec<u32> {
        let mut out = Vec::new();
        for (g, bands) in unions {
            if bands.iter().any(|&(lo,hi)| mz >= lo && mz <= hi) {
                out.push(*g);
            }
        }
        out
    }

    /// Peak picking on an already built RtIndex (e.g., after custom transforms).
    pub fn pick_peaks_on_rtindex(
        &self,
        rt: &RtIndex,
        min_prom: f32,
        min_distance: usize,
        min_width: usize,
        pad_left: usize,          // NEW
        pad_right: usize,         // NEW
    ) -> Vec<RtPeak1D> {
        let data_raw_slice: &[f32] = rt.data_raw.as_deref().unwrap_or(rt.data.as_slice());
        pick_peaks_all_rows(
            rt.data.as_slice(),
            data_raw_slice,
            rt.rows,
            rt.cols,
            Some(&rt.frame_times),
            min_prom,
            min_distance,
            min_width,
            pad_left,
            pad_right,
            Some(&rt.scale.centers),
        )
    }

    pub fn get_dense_im_by_rtpeaks_ppm(
        &self,
        peaks: Vec<RtPeak1D>,      // rows
        rt_index: &RtIndex,        // has .scale and .frames
        num_threads: usize,
        mz_ppm_window: f32,        // ±ppm around the peak center bin
        rt_extra_pad: usize,
        maybe_sigma_scans: Option<f32>,
        truncate: f32,
    ) -> ImIndex {
        build_dense_im_by_rtpeaks_ppm(
            self,
            peaks,
            rt_index,
            num_threads,
            mz_ppm_window,
            rt_extra_pad,
            maybe_sigma_scans,
            truncate,
        )
    }

    /// Build IM matrix for a DIA group using that group’s RT peaks and frames.
    /// `rt_index_group` must be the RT index you built for the SAME group.
    pub fn get_dense_im_by_rtpeaks_ppm_for_group(
        &self,
        rt_index_group: &RtIndex,   // from `get_dense_rt_by_mz_ppm_for_group`
        peaks: Vec<RtPeak1D>,       // those peaks (rows)
        window_group: u32,
        num_threads: usize,
        mz_ppm_window: f32,
        rt_extra_pad: usize,
        maybe_sigma_scans: Option<f32>,
        truncate: f32,
        clamp_scans_to_group: bool,
    ) -> ImIndex {
        // Scan clamp from instrument windows (optional)
        let scan_clamp = if clamp_scans_to_group {
            self.group_bounds(window_group).map(|b| (b.scan_lo, b.scan_hi))
        } else { None };

        let frame_ids = self.frame_ids_for_group(window_group);

        build_dense_im_by_rtpeaks_ppm_for_group(
            self,
            peaks,
            frame_ids,
            &rt_index_group.scale,   // use the SAME m/z scale to stay consistent
            num_threads,
            mz_ppm_window,
            rt_extra_pad,
            maybe_sigma_scans,
            truncate,
            scan_clamp,
        )
    }

    pub fn group_scan_unions(&self) -> FxHashMap<u32, Vec<(usize,usize)>> {
        let mut mp: FxHashMap<u32, Vec<(usize,usize)>> = FxHashMap::default();
        for w in &self.dia_ms_ms_windows {
            let lo = w.scan_num_begin as usize;
            let hi = w.scan_num_end   as usize;
            let (lo, hi) = (lo.min(hi), lo.max(hi));
            mp.entry(w.window_group).or_default().push((lo, hi));
        }
        // merge overlaps
        for v in mp.values_mut() {
            v.sort_by(|a,b| a.0.cmp(&b.0));
            let mut merged: Vec<(usize,usize)> = Vec::new();
            for (lo,hi) in v.drain(..) {
                if let Some(last) = merged.last_mut() {
                    if lo <= last.1 { last.1 = last.1.max(hi); }
                    else { merged.push((lo,hi)); }
                } else { merged.push((lo,hi)); }
            }
            *v = merged;
        }
        mp
    }

    /// Evaluate 3D clusters (RT × IM with m/z marginal) for the given specs.
    /// This forwards to cluster_eval::evaluate_clusters_3d and returns the results.
    pub fn evaluate_clusters_3d(
        &self,
        rt_index: &RtIndex,
        specs: &[ClusterSpec],
        opts: EvalOptions,
        num_threads: usize,
    ) -> Vec<ClusterResult> {
        let mut cluster_result = evaluate_clusters_3d(self, rt_index, specs, opts, num_threads);
        annotate_precursor_groups(self, &mut cluster_result);
        cluster_result
    }

    /// Build features by:
    ///  1) determining global RT span from envelopes (in precursor RT indices),
    ///  2) loading those precursor frames into RAM,
    ///  3) shifting envelope RT bounds to the local window, and
    ///  4) calling the existing `build_features_from_envelopes`.
    pub fn build_features_from_envelopes(
        &self,
        envelopes: &[Envelope],
        clusters: &[ClusterResult],
        lut: &AveragineLut,
        gp: &GroupingParams,
        fp: &FeatureBuildParams,
    ) -> Vec<Feature> {
        if envelopes.is_empty() {
            return Vec::new();
        }

        // --- 1) Map precursor RT index -> frame_id ---------------------------
        // We assume the grouping/envelopes use *precursor-only* RT indexing.
        // Build a compact lookup: precursor_rt_idx -> frame_id.
        let mut ms1_frame_ids: Vec<u32> = Vec::new();
        ms1_frame_ids.reserve(self.meta_data.len());
        for fm in &self.meta_data {
            if fm.ms_ms_type == 0 {
                ms1_frame_ids.push(fm.id as u32);
            }
        }
        if ms1_frame_ids.is_empty() {
            return Vec::new();
        }

        // --- 2) Global RT span across envelopes (still in precursor indices) -
        let mut rt_min = usize::MAX;
        let mut rt_max = 0usize;
        for e in envelopes {
            rt_min = rt_min.min(e.rt_bounds.0);
            rt_max = rt_max.max(e.rt_bounds.1);
        }
        if rt_min == usize::MAX || rt_min > rt_max {
            return Vec::new();
        }

        // Clamp to available precursor frames
        let last_rt = ms1_frame_ids.len().saturating_sub(1);
        rt_min = rt_min.min(last_rt);
        rt_max = rt_max.min(last_rt);
        if rt_min > rt_max {
            return Vec::new();
        }

        // --- 3) Preload frames into RAM --------------------------------------
        let span = rt_max - rt_min + 1;
        let mut frames: Vec<Arc<TimsFrame>> = Vec::with_capacity(span);
        for rt_idx in rt_min..=rt_max {
            let fid = ms1_frame_ids[rt_idx];
            let fr = self.loader.get_frame(fid);
            // If you want Arc<TimsFrame> without copy, ensure get_frame returns an owned TimsFrame
            // and wrap it. If it already returns Arc<TimsFrame>, drop the Arc::new.
            frames.push(Arc::new(fr));
        }

        // --- 4) Rewrite envelopes into local RT coordinates -------------------
        let mut loc_envs: Vec<Envelope> = Vec::with_capacity(envelopes.len());
        for e in envelopes {
            let (gl, gr) = e.rt_bounds;
            // shift to local window [0..span)
            let l = gl.saturating_sub(rt_min).min(span - 1);
            let r = gr.saturating_sub(rt_min).min(span - 1);
            let mut e2 = e.clone();
            e2.rt_bounds = if l <= r { (l, r) } else { (l, l) }; // guard
            loc_envs.push(e2);
        }

        // --- 5) Delegate to the existing builder -----------------------------
        build_features_from_envelopes(
            &frames,       // contiguous local RT window
            &loc_envs,     // envelopes in local RT indices
            clusters,
            lut,
            gp,
            fp,
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

pub fn annotate_precursor_groups(ds: &TimsDatasetDIA, clusters: &mut [ClusterResult]) {
    let mz_unions   = ds.group_mz_unions();
    let scan_unions = ds.group_scan_unions();

    let ppm_tol: f32 = 15.0;   // tune
    let scan_pad: usize = 3;   // tune (accounts for slight index jitter)

    for c in clusters.iter_mut() {
        // treat “no window_group” as precursor-like
        if c.window_group.is_none() {
            let mz_prec = if c.mz_fit.mu.is_finite() && c.mz_fit.mu > 0.0 {
                c.mz_fit.mu
            } else {
                c.mz_center_hint
            };

            let (sl, sr) = c.im_window; // FINAL IM window (absolute scans)
            let groups = groups_covering_precursor_2d(
                mz_prec, sl, sr, &mz_unions, &scan_unions, ppm_tol, scan_pad,
            );
            c.window_groups_covering_mz = Some(groups);
        }
    }
}
fn ppm_da(mz: f32, ppm: f32) -> f32 { mz * ppm * 1e-6 }
pub fn groups_covering_precursor_2d(
    mz: f32,
    scan_l: usize, scan_r: usize,
    mz_unions: &FxHashMap<u32, Vec<(f32,f32)>>,
    scan_unions: &FxHashMap<u32, Vec<(usize,usize)>>,
    ppm_tol: f32,
    scan_pad: usize,
) -> Vec<u32> {
    let da = ppm_da(mz, ppm_tol);
    let (loq, hiq) = (mz - da, mz + da);
    let (sl, sr) = (scan_l.saturating_sub(scan_pad), scan_r.saturating_add(scan_pad));

    let mut out = Vec::new();
    for (g, bands_mz) in mz_unions {
        // m/z overlap?
        let mz_ok = bands_mz.iter().any(|&(lo,hi)| hiq >= lo && loq <= hi);

        // scan overlap? (if scan unions exist for this group)
        let scan_ok = if let Some(bands_sc) = scan_unions.get(g) {
            bands_sc.iter().any(|&(slo,shi)| sr >= slo && sl <= shi)
        } else {
            // if no scan info, be permissive or set false — your call; permissive is typical
            true
        };

        if mz_ok && scan_ok { out.push(*g); }
    }
    out
}