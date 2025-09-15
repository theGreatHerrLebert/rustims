use crate::data::acquisition::AcquisitionMode;
use crate::data::handle::{IndexConverter, TimsData, TimsDataLoader};
use crate::data::meta::{
    read_dia_ms_ms_info, read_dia_ms_ms_windows, read_global_meta_sql, read_meta_data_sql,
    DiaMsMisInfo, DiaMsMsWindow, FrameMeta, GlobalMetaData,
};

use mscore::timstof::frame::{RawTimsFrame, TimsFrame};
use mscore::timstof::slice::TimsSlice;
use rand::prelude::IteratorRandom;
use crate::cluster::utility::{build_dense_rt_by_mz_ppm, pick_peaks_all_rows, RtPeak1D, RtIndex, build_dense_im_by_rtpeaks_ppm, ImIndex};

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
