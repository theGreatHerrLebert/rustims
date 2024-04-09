use std::fs::File;
use std::io::{Read, Cursor};
use std::path::PathBuf;
use byteorder::{LittleEndian, ReadBytesExt};
use mscore::data::spectrum::MsType;
use mscore::timstof::frame::{ImsFrame, RawTimsFrame, TimsFrame};
use mscore::timstof::slice::TimsSlice;
use crate::data::meta::{FrameMeta, GlobalMetaData, read_global_meta_sql, read_meta_data_sql};
use crate::data::raw::BrukerTimsDataLibrary;
use crate::data::utility::{flatten_scan_values, parse_decompressed_bruker_binary_data, zstd_decompress};

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::data::acquisition::AcquisitionMode;

pub trait TimsData {
    fn get_frame(&self, frame_id: u32) -> TimsFrame;
    fn get_raw_frame(&self, frame_id: u32) -> RawTimsFrame;
    fn get_slice(&self, frame_ids: Vec<u32>, num_threads: usize) -> TimsSlice;
    fn get_acquisition_mode(&self) -> AcquisitionMode;
    fn get_frame_count(&self) -> i32;
    fn get_data_path(&self) -> &str;
}

pub trait IndexConverter {
    fn tof_to_mz(&self, frame_id: u32, tof_values: &Vec<u32>) -> Vec<f64>;
    fn mz_to_tof(&self, frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32>;
    fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: &Vec<u32>) -> Vec<f64>;
    fn inverse_mobility_to_scan(&self, frame_id: u32, inverse_mobility_values: &Vec<f64>) -> Vec<u32>;
}

pub struct SimpleIndexConverter;

impl IndexConverter for SimpleIndexConverter {
    fn tof_to_mz(&self, _frame_id: u32, _tof_values: &Vec<u32>) -> Vec<f64> {
        todo!()
    }

    fn mz_to_tof(&self, _frame_id: u32, _mz_values: &Vec<f64>) -> Vec<u32> {
        todo!()
    }

    fn scan_to_inverse_mobility(&self, _frame_id: u32, _scan_values: &Vec<u32>) -> Vec<f64> {
        todo!()
    }

    fn inverse_mobility_to_scan(&self, _frame_id: u32, _inverse_mobility_values: &Vec<f64>) -> Vec<u32> {
        todo!()
    }
}

pub struct BrukerLibTimsDataConverter {
    pub bruker_lib: BrukerTimsDataLibrary,
}

impl BrukerLibTimsDataConverter {
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Self {
        let bruker_lib = BrukerTimsDataLibrary::new(bruker_lib_path, data_path).unwrap();
        BrukerLibTimsDataConverter {
            bruker_lib,
        }
    }
}
impl IndexConverter for BrukerLibTimsDataConverter {
    /// translate tof to mz values calling the bruker library
    ///
    /// # Arguments
    ///
    /// * `frame_id` - A u32 that holds the frame id
    /// * `tof` - A vector of u32 that holds the tof values
    ///
    /// # Returns
    ///
    /// * `mz_values` - A vector of f64 that holds the mz values
    ///
    fn tof_to_mz(&self, frame_id: u32, tof: &Vec<u32>) -> Vec<f64> {
        let mut dbl_tofs: Vec<f64> = Vec::new();
        dbl_tofs.resize(tof.len(), 0.0);

        for (i, &val) in tof.iter().enumerate() {
            dbl_tofs[i] = val as f64;
        }

        let mut mz_values: Vec<f64> = Vec::new();
        mz_values.resize(tof.len(),  0.0);

        self.bruker_lib.tims_index_to_mz(frame_id, &dbl_tofs, &mut mz_values).expect("Bruker binary call failed at: tims_index_to_mz;");

        mz_values
    }

    fn mz_to_tof(&self, frame_id: u32, mz: &Vec<f64>) -> Vec<u32> {
        let mut dbl_mz: Vec<f64> = Vec::new();
        dbl_mz.resize(mz.len(), 0.0);

        for (i, &val) in mz.iter().enumerate() {
            dbl_mz[i] = val;
        }

        let mut tof_values: Vec<f64> = Vec::new();
        tof_values.resize(mz.len(),  0.0);

        self.bruker_lib.tims_mz_to_index(frame_id, &dbl_mz, &mut tof_values).expect("Bruker binary call failed at: tims_mz_to_index;");

        tof_values.iter().map(|&x| x.round() as u32).collect()
    }

    /// translate scan to inverse mobility values calling the bruker library
    ///
    /// # Arguments
    ///
    /// * `frame_id` - A u32 that holds the frame id
    /// * `scan` - A vector of i32 that holds the scan values
    ///
    /// # Returns
    ///
    /// * `inv_mob` - A vector of f64 that holds the inverse mobility values
    ///
    fn scan_to_inverse_mobility(&self, frame_id: u32, scan: &Vec<u32>) -> Vec<f64> {
        let mut dbl_scans: Vec<f64> = Vec::new();
        dbl_scans.resize(scan.len(), 0.0);

        for (i, &val) in scan.iter().enumerate() {
            dbl_scans[i] = val as f64;
        }

        let mut inv_mob: Vec<f64> = Vec::new();
        inv_mob.resize(scan.len(), 0.0);

        self.bruker_lib.tims_scan_to_inv_mob(frame_id, &dbl_scans, &mut inv_mob).expect("Bruker binary call failed at: tims_scannum_to_oneoverk0;");

        inv_mob
    }

    /// translate inverse mobility to scan values calling the bruker library
    ///
    /// # Arguments
    ///
    /// * `frame_id` - A u32 that holds the frame id
    /// * `inv_mob` - A vector of f64 that holds the inverse mobility values
    ///
    /// # Returns
    ///
    /// * `scan_values` - A vector of i32 that holds the scan values
    ///
    fn inverse_mobility_to_scan(&self, frame_id: u32, inv_mob: &Vec<f64>) -> Vec<u32> {
        let mut dbl_inv_mob: Vec<f64> = Vec::new();
        dbl_inv_mob.resize(inv_mob.len(), 0.0);

        for (i, &val) in inv_mob.iter().enumerate() {
            dbl_inv_mob[i] = val;
        }

        let mut scan_values: Vec<f64> = Vec::new();
        scan_values.resize(inv_mob.len(),  0.0);

        self.bruker_lib.inv_mob_to_tims_scan(frame_id, &dbl_inv_mob, &mut scan_values).expect("Bruker binary call failed at: tims_oneoverk0_to_scannum;");

        scan_values.iter().map(|&x| x.round() as u32).collect()
    }
}

enum TimsIndexConverter {
    Simple(SimpleIndexConverter),
    BrukerLib(BrukerLibTimsDataConverter)
}

impl IndexConverter for TimsIndexConverter {
    fn tof_to_mz(&self, frame_id: u32, tof_values: &Vec<u32>) -> Vec<f64> {
        match self {
            TimsIndexConverter::Simple(converter) => converter.tof_to_mz(frame_id, tof_values),
            TimsIndexConverter::BrukerLib(converter) => converter.tof_to_mz(frame_id, tof_values)
        }
    }

    fn mz_to_tof(&self, frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32> {
        match self {
            TimsIndexConverter::Simple(converter) => converter.mz_to_tof(frame_id, mz_values),
            TimsIndexConverter::BrukerLib(converter) => converter.mz_to_tof(frame_id, mz_values)
        }
    }

    fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: &Vec<u32>) -> Vec<f64> {
        match self {
            TimsIndexConverter::Simple(converter) => converter.scan_to_inverse_mobility(frame_id, scan_values),
            TimsIndexConverter::BrukerLib(converter) => converter.scan_to_inverse_mobility(frame_id, scan_values)
        }
    }

    fn inverse_mobility_to_scan(&self, frame_id: u32, inverse_mobility_values: &Vec<f64>) -> Vec<u32> {
        match self {
            TimsIndexConverter::Simple(converter) => converter.inverse_mobility_to_scan(frame_id, inverse_mobility_values),
            TimsIndexConverter::BrukerLib(converter) => converter.inverse_mobility_to_scan(frame_id, inverse_mobility_values)
        }
    }
}


pub struct TimsLazyLoder {
    pub raw_data_layout: TimsRawDataLayout,
    index_converter: TimsIndexConverter,
}

impl TimsData for TimsLazyLoder {
    fn get_frame(&self, _frame_id: u32) -> TimsFrame {
        todo!()
    }

    fn get_raw_frame(&self, _frame_id: u32) -> RawTimsFrame {
        todo!()
    }

    fn get_slice(&self, _frame_ids: Vec<u32>, _num_threads: usize) -> TimsSlice {
        todo!()
    }

    fn get_acquisition_mode(&self) -> AcquisitionMode {
        todo!()
    }

    fn get_frame_count(&self) -> i32 {
        todo!()
    }

    fn get_data_path(&self) -> &str {
        todo!()
    }
}


pub struct TimsRawDataLayout {
    pub global_meta_data: GlobalMetaData,
    pub frame_meta_data: Vec<FrameMeta>,
    pub max_scan_count: i64,
    pub frame_id_ptr: Vec<i64>,
    pub tims_offset_values: Vec<i64>,
    pub acquisition_mode: AcquisitionMode,
}

impl TimsRawDataLayout {
    pub fn new(data_path: &str) -> Self {
        // get the global and frame meta data
        let global_meta_data = read_global_meta_sql(data_path).unwrap();
        let frame_meta_data = read_meta_data_sql(data_path).unwrap();

        // get the max scan count
        let max_scan_count = frame_meta_data.iter().map(|x| x.num_scans).max().unwrap();

        let mut frame_id_ptr: Vec<i64> = Vec::new();
        frame_id_ptr.resize(frame_meta_data.len() + 1, 0);

        // get the frame id_ptr values
        for (i, row) in frame_meta_data.iter().enumerate() {
            frame_id_ptr[i + 1] = row.num_peaks + frame_id_ptr[i];
        }

        // get the tims offset values
        let tims_offset_values = frame_meta_data.iter().map(|x| x.tims_id).collect::<Vec<i64>>();

        // get the acquisition mode
        let acquisition_mode = match frame_meta_data[0].scan_mode {
            8 => AcquisitionMode::DDA,
            9 => AcquisitionMode::DIA,
            _ => AcquisitionMode::Unknown,
        };

        TimsRawDataLayout {
            global_meta_data,
            frame_meta_data,
            max_scan_count,
            frame_id_ptr,
            tims_offset_values,
            acquisition_mode
        }
    }
}

pub struct TimsInMemoryLoader {
    raw_data_layout: TimsRawDataLayout,
    index_converter: TimsIndexConverter,
    compressed_data: Vec<u8>
}

impl TimsData for TimsInMemoryLoader {
    fn get_frame(&self, frame_id: u32) -> TimsFrame {

        let raw_frame = self.get_raw_frame(frame_id);
        let tof_i32 = raw_frame.tof.iter().map(|&x| x as i32).collect();
        let scan = flatten_scan_values(&raw_frame.scan, true);

        let mz = self.index_converter.tof_to_mz(frame_id, &raw_frame.tof);
        let inverse_mobility = self.index_converter.scan_to_inverse_mobility(frame_id, &scan);

        let ims_frame = ImsFrame {
            retention_time: raw_frame.retention_time,
            mz,
            intensity: raw_frame.intensity,
            mobility: inverse_mobility,
        };

        TimsFrame {
            frame_id: frame_id as i32,
            ms_type: raw_frame.ms_type,
            scan: scan.iter().map(|&x| x as i32).collect(),
            tof: tof_i32,
            ims_frame,
        }
    }

    fn get_raw_frame(&self, frame_id: u32) -> RawTimsFrame {
        let frame_index = (frame_id - 1) as usize;
        let offset = self.raw_data_layout.tims_offset_values[frame_index] as usize;

        let bin_size_offset = offset + 4; // Assuming the size is stored immediately before the frame data
        let bin_size = Cursor::new(&self.compressed_data[offset..bin_size_offset]).read_i32::<LittleEndian>().unwrap();

        let data_offset = bin_size_offset + 4; // Adjust based on actual structure
        let frame_data = &self.compressed_data[data_offset..data_offset + bin_size as usize - 8];

        let decompressed_bytes = zstd_decompress(&frame_data).unwrap();

        let (scan, tof, intensity) = parse_decompressed_bruker_binary_data(&decompressed_bytes).unwrap();

        let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;

        let ms_type = match ms_type_raw {
            0 => MsType::Precursor,
            8 => MsType::FragmentDda,
            9 => MsType::FragmentDia,
            _ => MsType::Unknown,
        };

        let raw_frame = RawTimsFrame {
            frame_id: frame_id as i32,
            retention_time: self.raw_data_layout.frame_meta_data[(frame_id - 1) as usize].time,
            ms_type,
            scan: scan,
            tof: tof,
            intensity: intensity.iter().map(|&x| x as f64).collect(),
        };

        raw_frame
    }

    fn get_slice(&self, frame_ids: Vec<u32>, num_threads: usize) -> TimsSlice {
        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
        let frames = pool.install(|| {
            frame_ids.par_iter().map(|&frame_id| {
                self.get_frame(frame_id)
            }).collect()
        });

        TimsSlice {
            frames
        }
    }

    fn get_acquisition_mode(&self) -> AcquisitionMode {
        todo!()
    }

    fn get_frame_count(&self) -> i32 {
        todo!()
    }

    fn get_data_path(&self) -> &str {
        todo!()
    }
}

pub enum TimsDataLoader {
    InMemory(TimsInMemoryLoader),
    Lazy(TimsLazyLoder)
}

impl TimsDataLoader {
    pub fn new_lazy(bruker_lib_path: &str, data_path: &str) -> Self {
        let raw_data_layout = TimsRawDataLayout::new(data_path);
        let index_converter = TimsIndexConverter::BrukerLib(BrukerLibTimsDataConverter::new(bruker_lib_path, data_path));
        TimsDataLoader::Lazy(TimsLazyLoder {
            raw_data_layout,
            index_converter
        })
    }

    pub fn new_in_memory(bruker_lib_path: &str, data_path: &str) -> Self {
        let raw_data_layout = TimsRawDataLayout::new(data_path);
        let index_converter = TimsIndexConverter::BrukerLib(BrukerLibTimsDataConverter::new(bruker_lib_path, data_path));

        let mut file_path = PathBuf::from(data_path);
        file_path.push("analysis.tdf_bin");
        let mut infile = File::open(file_path).unwrap();
        let mut data = Vec::new();
        infile.read_to_end(&mut data).unwrap();

        TimsDataLoader::InMemory(TimsInMemoryLoader {
            raw_data_layout,
            index_converter,
            compressed_data: data
        })
    }
}

impl TimsData for TimsDataLoader {
    fn get_frame(&self, frame_id: u32) -> TimsFrame {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_frame(frame_id),
            TimsDataLoader::Lazy(loader) => loader.get_frame(frame_id)
        }
    }
    fn get_raw_frame(&self, frame_id: u32) -> RawTimsFrame {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_raw_frame(frame_id),
            TimsDataLoader::Lazy(loader) => loader.get_raw_frame(frame_id)
        }
    }

    fn get_slice(&self, frame_ids: Vec<u32>, num_threads: usize) -> TimsSlice {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_slice(frame_ids, num_threads),
            TimsDataLoader::Lazy(loader) => loader.get_slice(frame_ids, num_threads)
        }
    }

    fn get_acquisition_mode(&self) -> AcquisitionMode {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_acquisition_mode(),
            TimsDataLoader::Lazy(loader) => loader.get_acquisition_mode()
        }
    }

    fn get_frame_count(&self) -> i32 {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_frame_count(),
            TimsDataLoader::Lazy(loader) => loader.get_frame_count()
        }
    }

    fn get_data_path(&self) -> &str {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_data_path(),
            TimsDataLoader::Lazy(loader) => loader.get_data_path()
        }
    }
}