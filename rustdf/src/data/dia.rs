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
use crate::data::utility::merge_ranges;

pub struct TimsDatasetDIA {
    pub loader: TimsDataLoader,
    pub global_meta_data: GlobalMetaData,
    pub meta_data: Vec<FrameMeta>,
    pub dia_ms_ms_info: Vec<DiaMsMisInfo>,
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
            dia_ms_ms_info: dia_ms_mis_info,
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
