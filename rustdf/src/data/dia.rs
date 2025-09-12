use crate::data::acquisition::AcquisitionMode;
use crate::data::handle::{IndexConverter, TimsData, TimsDataLoader};
use crate::data::meta::{
    read_dia_ms_ms_info, read_dia_ms_ms_windows, read_global_meta_sql, read_meta_data_sql,
    DiaMsMisInfo, DiaMsMsWindow, FrameMeta, GlobalMetaData,
};
use std::collections::HashMap;
use rayon::prelude::*;
use mscore::timstof::frame::{RawTimsFrame, TimsFrame};
use mscore::timstof::slice::TimsSlice;
use rand::prelude::IteratorRandom;
use crate::cluster::cluster_eval::ClusterSpec;
use crate::cluster::utility::{build_dense_rt_by_mz, pick_peaks_all_rows, RtPeak1D, RtIndex, build_dense_im_by_rtpeaks, ImIndex, ClusterCloud};

#[derive(Clone)]
struct SpecPre {
    scan_left: usize,
    scan_right: usize,
    mz_lo: f64,
    mz_hi: f64,
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

    pub fn get_dense_rt_by_mz(
        &self,
        maybe_sigma_frames: Option<f32>,
        truncate: f32,
        resolution: usize,
        num_threads: usize
    ) -> RtIndex {
        build_dense_rt_by_mz(self, maybe_sigma_frames, truncate, resolution, num_threads)
    }

    /// Build dense matrix, optionally smooth, then pick peaks.
    /// Returns the RtIndex (with data + frame_times) and the list of peaks.
    pub fn pick_peaks_dense(
        &self,
        maybe_sigma_frames: Option<f32>,
        truncate: f32,
        resolution: usize,
        num_threads: usize,
        min_prom: f32,
        min_distance: usize,
        min_width: usize,
        pad_left: usize,          // NEW
        pad_right: usize,         // NEW
    ) -> (RtIndex, Vec<RtPeak1D>) {
        let rt = self.get_dense_rt_by_mz(maybe_sigma_frames, truncate, resolution, num_threads);
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
        );
        (rt, peaks)
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
        )
    }

    pub fn get_dense_im_by_rtpeaks(
        &self,
        peaks: Vec<RtPeak1D>,
        rt_bins: &[u32],
        rt_frames: &[u32],
        resolution: usize,
        num_threads: usize,
        mz_ppm: f32,
        rt_extra_pad: usize,
        maybe_sigma_scans: Option<f32>,
        truncate: f32,
    ) -> ImIndex {
        build_dense_im_by_rtpeaks(
            self,
            peaks,
            rt_bins,
            rt_frames,
            resolution,
            num_threads,
            mz_ppm,
            rt_extra_pad,
            maybe_sigma_scans,
            truncate,
        )
    }
    /// Parallel extractor: walk frames once (in parallel), distribute points to matching specs.
    pub fn extract_cluster_clouds_batched(
        &self,
        specs: &[ClusterSpec],
        bins: &[u32],
        frames_rt_sorted: &[u32],
        resolution: usize,
        num_threads: usize,
    ) -> Vec<ClusterCloud> {

        // 0) Materialize frames once in RT order (RAM)
        let slice = self.get_slice(frames_rt_sorted.to_vec(), num_threads);
        let frames_in_rt: Vec<&TimsFrame> = slice.frames.iter().collect();
        let ncols = frames_in_rt.len();

        // 1) Precompute per-spec constants and col→specs membership
        let factor = 10f64.powi(resolution as i32);

        let mut pres: Vec<SpecPre> = Vec::with_capacity(specs.len());
        let mut col_to_specs: Vec<Vec<usize>> = vec![Vec::new(); ncols];

        for (idx, sp) in specs.iter().enumerate() {
            // center m/z from the rt_row bin
            let mz_center = (bins[sp.rt_row] as f64) / factor;
            let mz_tol = (sp.mz_ppm as f64) * 1e-6 * mz_center;

            // normalize & clamp RT window (inclusive)
            let rt_left = sp.rt_left.min(sp.rt_right);
            let mut rt_right = sp.rt_left.max(sp.rt_right);
            if rt_right >= ncols { rt_right = ncols.saturating_sub(1); }
            if rt_left > rt_right { continue; } // degenerate

            pres.push(SpecPre {
                scan_left: sp.scan_left,
                scan_right: sp.scan_right,
                mz_lo: mz_center - mz_tol,
                mz_hi: mz_center + mz_tol,
            });

            for c in rt_left..=rt_right {
                col_to_specs[c].push(idx);
            }
        }

        // 2) Process columns in parallel.
        // Each worker returns a sparse map: spec_idx -> Vec of tuples (fid, scan, tof, inten)
        type Point = (u32, u32, i32, f32); // (frame_id, scan, tof, intensity)
        type Chunk = HashMap<usize, Vec<Point>>;

        let partials: Vec<Chunk> = (0..ncols)
            .into_par_iter()
            .map(|col| {
                let mut out: Chunk = HashMap::new();

                let targets = &col_to_specs[col];
                if targets.is_empty() {
                    return out; // fast skip
                }

                let fr = frames_in_rt[col];
                let fid = frames_rt_sorted[col];

                let mz = &fr.ims_frame.mz;
                let inten = &fr.ims_frame.intensity;
                // these are all &Vec<_> in your frame, so borrow as slices
                let scans_slice: Option<&[i32]> = Some(fr.scan.as_ref());
                let tofs_slice: &[i32] = fr.tof.as_ref();

                // iterate points in this frame once
                // NB: we assume all arrays are same length; if not, min length
                let len = mz.len().min(inten.len()).min(tofs_slice.len());
                let sv = scans_slice;

                // Mild micro-opt: copy targets for fewer indirections in inner loop
                for i in 0..len {
                    let m = mz[i];
                    // quick coarse check for at least one target’s scan range?
                    // We’ll just do per-target checks; scan is cheap to load.
                    let s_u32: u32 = if let Some(s) = sv { s[i] as u32 } else { 0 };

                    // Test against all specs active for this column
                    for &sp_idx in targets {
                        let sp = unsafe { pres.get_unchecked(sp_idx) };

                        // scan filter (if available)
                        if let Some(s) = sv {
                            let su = s[i] as usize;
                            if su < sp.scan_left || su > sp.scan_right {
                                continue;
                            }
                        }
                        // mz filter
                        if m < sp.mz_lo || m > sp.mz_hi {
                            continue;
                        }

                        // keep the point
                        let entry = out.entry(sp_idx).or_insert_with(|| {
                            // heuristic reserve a bit; this grows if needed
                            Vec::with_capacity(256)
                        });
                        entry.push((fid, s_u32, tofs_slice[i], inten[i] as f32));
                    }
                }

                out
            })
            .collect();

        // 3) Reduce the partial maps into final per-spec clouds (no locks used above)
        let mut clouds: Vec<ClusterCloud> = specs
            .iter()
            .map(|sp| ClusterCloud {
                rt_left: sp.rt_left,
                rt_right: sp.rt_right,
                scan_left: sp.scan_left,
                scan_right: sp.scan_right,
                frame_ids: Vec::new(),
                scans: Vec::new(),
                tofs: Vec::new(),
                intensities: Vec::new(),
            })
            .collect();

        for chunk in partials {
            for (sp_idx, mut pts) in chunk {
                let cloud = &mut clouds[sp_idx];
                cloud.frame_ids.reserve(pts.len());
                cloud.scans.reserve(pts.len());
                cloud.tofs.reserve(pts.len());
                cloud.intensities.reserve(pts.len());

                for (fid, sc, tof, it) in pts.drain(..) {
                    cloud.frame_ids.push(fid);
                    cloud.scans.push(sc);
                    cloud.tofs.push(tof);
                    cloud.intensities.push(it);
                }
            }
        }

        clouds
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
