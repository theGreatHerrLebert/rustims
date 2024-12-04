use std::fs::File;
use std::io::{Read, Cursor, SeekFrom, Seek};
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

pub struct TimsRawDataLayout {
    pub raw_data_path: String,
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
            raw_data_path: data_path.to_string(),
            global_meta_data,
            frame_meta_data,
            max_scan_count,
            frame_id_ptr,
            tims_offset_values,
            acquisition_mode
        }
    }
}

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

pub enum TimsIndexConverter {
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
    pub index_converter: TimsIndexConverter,
}

impl TimsData for TimsLazyLoder {
    fn get_frame(&self, frame_id: u32) -> TimsFrame {
        let frame_index = (frame_id - 1) as usize;

        // turns out, there can be empty frames in the data, check for that, if so, return an empty frame
        let num_peaks = self.raw_data_layout.frame_meta_data[frame_index].num_peaks;

        if num_peaks == 0 {
            return TimsFrame {
                frame_id: frame_id as i32,
                ms_type: MsType::Unknown,
                scan: Vec::new(),
                tof: Vec::new(),
                ims_frame: ImsFrame { retention_time: self.raw_data_layout.frame_meta_data[(frame_id - 1) as usize].time, mobility: Vec::new(), mz: Vec::new(), intensity: Vec::new() }
            };
        }

        let offset = self.raw_data_layout.tims_offset_values[frame_index] as u64;

        let mut file_path = PathBuf::from(&self.raw_data_layout.raw_data_path);
        file_path.push("analysis.tdf_bin");
        let mut infile = File::open(&file_path).unwrap();

        infile.seek(SeekFrom::Start(offset)).unwrap();

        let mut bin_buffer = [0u8; 4];
        infile.read_exact(&mut bin_buffer).unwrap();
        let bin_size = Cursor::new(bin_buffer).read_i32::<LittleEndian>().unwrap();

        infile.read_exact(&mut bin_buffer).unwrap();

        match self.raw_data_layout.global_meta_data.tims_compression_type {
            _ if self.raw_data_layout.global_meta_data.tims_compression_type == 1 => {
                panic!("Decompression Type1 not implemented.");
            },

            // Extract from ZSTD compressed binary
            _ if self.raw_data_layout.global_meta_data.tims_compression_type == 2 => {

                let mut compressed_data = vec![0u8; bin_size as usize - 8];
                infile.read_exact(&mut compressed_data).unwrap();

                let decompressed_bytes = zstd_decompress(&compressed_data).unwrap();

                let (scan, tof, intensity) = parse_decompressed_bruker_binary_data(&decompressed_bytes).unwrap();
                let intensity_dbl = intensity.iter().map(|&x| x as f64).collect();
                let tof_i32 = tof.iter().map(|&x| x as i32).collect();
                let scan = flatten_scan_values(&scan, true);

                let mz = self.index_converter.tof_to_mz(frame_id, &tof);
                let inv_mobility = self.index_converter.scan_to_inverse_mobility(frame_id, &scan);

                let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;

                let ms_type = match ms_type_raw {
                    0 => MsType::Precursor,
                    8 => MsType::FragmentDda,
                    9 => MsType::FragmentDia,
                    _ => MsType::Unknown,
                };

                let frame = TimsFrame {
                    frame_id: frame_id as i32,
                    ms_type,
                    scan: scan.iter().map(|&x| x as i32).collect(),
                    tof: tof_i32,
                    ims_frame: ImsFrame { retention_time: self.raw_data_layout.frame_meta_data[(frame_id - 1) as usize].time, mobility: inv_mobility, mz, intensity: intensity_dbl }
                };

                return frame;
            },

            // Error on unknown compression algorithm
            _ => {
                panic!("TimsCompressionType is not 1 or 2.")
            }
        }
    }

    fn get_raw_frame(&self, frame_id: u32) -> RawTimsFrame {

        let frame_index = (frame_id - 1) as usize;
        let offset = self.raw_data_layout.tims_offset_values[frame_index] as u64;

        // turns out, there can be empty frames in the data, check for that, if so, return an empty frame
        let num_peaks = self.raw_data_layout.frame_meta_data[frame_index].num_peaks;

        if num_peaks == 0 {
            return RawTimsFrame {
                frame_id: frame_id as i32,
                retention_time: self.raw_data_layout.frame_meta_data[(frame_id - 1) as usize].time,
                ms_type: MsType::Unknown,
                scan: Vec::new(),
                tof: Vec::new(),
                intensity: Vec::new(),
            };
        }


        let mut file_path = PathBuf::from(&self.raw_data_layout.raw_data_path);
        file_path.push("analysis.tdf_bin");
        let mut infile = File::open(&file_path).unwrap();

        infile.seek(SeekFrom::Start(offset)).unwrap();

        let mut bin_buffer = [0u8; 4];
        infile.read_exact(&mut bin_buffer).unwrap();
        let bin_size = Cursor::new(bin_buffer).read_i32::<LittleEndian>().unwrap();

        infile.read_exact(&mut bin_buffer).unwrap();

        match self.raw_data_layout.global_meta_data.tims_compression_type {
            _ if self.raw_data_layout.global_meta_data.tims_compression_type == 1 => {
                panic!("Decompression Type1 not implemented.");
            },

            // Extract from ZSTD compressed binary
            _ if self.raw_data_layout.global_meta_data.tims_compression_type == 2 => {

                let mut compressed_data = vec![0u8; bin_size as usize - 8];
                infile.read_exact(&mut compressed_data).unwrap();

                let decompressed_bytes = zstd_decompress(&compressed_data).unwrap();

                let (scan, tof, intensity) = parse_decompressed_bruker_binary_data(&decompressed_bytes).unwrap();

                let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;

                let ms_type = match ms_type_raw {
                    0 => MsType::Precursor,
                    8 => MsType::FragmentDda,
                    9 => MsType::FragmentDia,
                    _ => MsType::Unknown,
                };

                let frame = RawTimsFrame {
                    frame_id: frame_id as i32,
                    retention_time: self.raw_data_layout.frame_meta_data[(frame_id - 1) as usize].time,
                    ms_type,
                    scan,
                    tof,
                    intensity: intensity.iter().map(|&x| x as f64).collect(),
                };

                return frame;
            },

            // Error on unknown compression algorithm
            _ => {
                panic!("TimsCompressionType is not 1 or 2.")
            }
        }
    }

    fn get_slice(&self, frame_ids: Vec<u32>, _num_threads: usize) -> TimsSlice {
        let result: Vec<TimsFrame> = frame_ids
            .into_iter()
            .map(|f| { self.get_frame(f) }).collect();

        TimsSlice { frames: result }
    }

    fn get_acquisition_mode(&self) -> AcquisitionMode {
        self.raw_data_layout.acquisition_mode.clone()
    }

    fn get_frame_count(&self) -> i32 {
        self.raw_data_layout.frame_meta_data.len() as i32
    }

    fn get_data_path(&self) -> &str {
        &self.raw_data_layout.raw_data_path
    }
}

pub struct TimsInMemoryLoader {
    pub raw_data_layout: TimsRawDataLayout,
    pub index_converter: TimsIndexConverter,
    compressed_data: Vec<u8>
}

impl TimsData for TimsInMemoryLoader {
    fn get_frame(&self, frame_id: u32) -> TimsFrame {

        let raw_frame = self.get_raw_frame(frame_id);

        let raw_frame = match raw_frame.ms_type {
            MsType::FragmentDda => raw_frame.smooth(1).centroid(1),
            _ => raw_frame,
        };

        // if raw frame is empty, return an empty frame
        if raw_frame.scan.is_empty() {
            return TimsFrame::default();
        }

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
            scan,
            tof,
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
        self.raw_data_layout.acquisition_mode.clone()
    }

    fn get_frame_count(&self) -> i32 {
        self.raw_data_layout.frame_meta_data.len() as i32
    }

    fn get_data_path(&self) -> &str {
        &self.raw_data_layout.raw_data_path
    }
}

pub enum TimsDataLoader {
    InMemory(TimsInMemoryLoader),
    Lazy(TimsLazyLoder)
}

impl TimsDataLoader {
    pub fn new_lazy(bruker_lib_path: &str, data_path: &str, use_bruker_sdk: bool, scan_max_index: u32, im_lower: f64, im_upper: f64, tof_max_index: u32, mz_lower: f64, mz_upper: f64) -> Self {
        let raw_data_layout = TimsRawDataLayout::new(data_path);
        
        let index_converter = match use_bruker_sdk {
            true => TimsIndexConverter::BrukerLib(BrukerLibTimsDataConverter::new(bruker_lib_path, data_path)),
            false => TimsIndexConverter::Simple(SimpleIndexConverter::from_boundaries(mz_lower, mz_upper, tof_max_index, im_lower, im_upper, scan_max_index))
        };
        
        TimsDataLoader::Lazy(TimsLazyLoder {
            raw_data_layout,
            index_converter
        })
    }

    pub fn new_in_memory(bruker_lib_path: &str, data_path: &str, use_bruker_sdk: bool, scan_max_index: u32, im_lower: f64, im_upper: f64, tof_max_index: u32, mz_lower: f64, mz_upper: f64) -> Self {
        let raw_data_layout = TimsRawDataLayout::new(data_path);
        
        let index_converter = match use_bruker_sdk {
            true => TimsIndexConverter::BrukerLib(BrukerLibTimsDataConverter::new(bruker_lib_path, data_path)),
            false => TimsIndexConverter::Simple(SimpleIndexConverter::from_boundaries(mz_lower, mz_upper, tof_max_index, im_lower, im_upper, scan_max_index))
        };

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
    pub fn get_index_converter(&self) -> &dyn IndexConverter {
        match self {
            TimsDataLoader::InMemory(loader) => &loader.index_converter,
            TimsDataLoader::Lazy(loader) => &loader.index_converter
        }
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

pub struct SimpleIndexConverter {
    pub tof_intercept: f64,
    pub tof_slope: f64,
    pub scan_intercept: f64,
    pub scan_slope: f64,  
}

impl SimpleIndexConverter {
    pub fn from_boundaries(
        mz_min: f64,
        mz_max: f64,
        tof_max_index: u32,
        im_min: f64,
        im_max: f64,
        scan_max_index: u32,
    ) -> Self {
        let tof_intercept: f64 = mz_min.sqrt();
        let tof_slope: f64 =
            (mz_max.sqrt() - tof_intercept) / tof_max_index as f64;
        
        let scan_intercept: f64 = im_max;
        let scan_slope: f64 = (im_min - scan_intercept) / scan_max_index as f64;
        Self {
            tof_intercept,
            tof_slope,
            scan_intercept,
            scan_slope,
        }
    }
}

impl IndexConverter for SimpleIndexConverter {
    fn tof_to_mz(&self, _frame_id: u32, _tof_values: &Vec<u32>) -> Vec<f64> {
        let mut mz_values: Vec<f64> = Vec::new();
        mz_values.resize(_tof_values.len(), 0.0);
        
        for (i, &val) in _tof_values.iter().enumerate() {
            mz_values[i] = (self.tof_intercept + self.tof_slope * val as f64).powi(2);
        }
        
        mz_values
    }

    fn mz_to_tof(&self, _frame_id: u32, _mz_values: &Vec<f64>) -> Vec<u32> {
        let mut tof_values: Vec<u32> = Vec::new();
        tof_values.resize(_mz_values.len(), 0);
        
        for (i, &val) in _mz_values.iter().enumerate() {
            tof_values[i] = ((val.sqrt() - self.tof_intercept) / self.tof_slope) as u32;
        }
        
        tof_values
    }

    fn scan_to_inverse_mobility(&self, _frame_id: u32, _scan_values: &Vec<u32>) -> Vec<f64> {
        let mut inv_mobility_values: Vec<f64> = Vec::new();
        inv_mobility_values.resize(_scan_values.len(), 0.0);
        
        for (i, &val) in _scan_values.iter().enumerate() {
            inv_mobility_values[i] = self.scan_intercept + self.scan_slope * val as f64;
        }
        
        inv_mobility_values
    }

    fn inverse_mobility_to_scan(&self, _frame_id: u32, _inverse_mobility_values: &Vec<f64>) -> Vec<u32> {
        let mut scan_values: Vec<u32> = Vec::new();
        scan_values.resize(_inverse_mobility_values.len(), 0);
        
        for (i, &val) in _inverse_mobility_values.iter().enumerate() {
            scan_values[i] = ((val - self.scan_intercept) / self.scan_slope) as u32;
        }
        
        scan_values
    }
}
