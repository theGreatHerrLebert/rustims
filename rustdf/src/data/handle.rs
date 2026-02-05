use crate::data::meta::{read_global_meta_sql, read_meta_data_sql, FrameMeta, GlobalMetaData};
use crate::data::raw::BrukerTimsDataLibrary;
use crate::data::utility::{
    flatten_scan_values, parse_decompressed_bruker_binary_data, zstd_decompress,
};
use byteorder::{LittleEndian, ReadBytesExt};
use mscore::data::spectrum::MsType;
use mscore::timstof::frame::{ImsFrame, RawTimsFrame, TimsFrame};
use mscore::timstof::slice::TimsSlice;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::PathBuf;

use crate::data::acquisition::AcquisitionMode;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use std::error::Error;

/// Derive m/z calibration coefficients by using the Bruker SDK.
///
/// This function temporarily loads the SDK to convert a range of TOF indices to m/z,
/// then fits a linear regression to derive accurate calibration coefficients.
///
/// Returns `Some((intercept, slope))` for the formula: `sqrt(mz) = intercept + slope * tof`
/// Returns `None` if SDK is not available (e.g., on macOS).
fn derive_mz_calibration(
    bruker_lib_path: &str,
    data_path: &str,
    tof_max_index: u32,
) -> Option<(f64, f64)> {
    // Try to create a BrukerLib converter
    let sdk_converter = match std::panic::catch_unwind(|| {
        BrukerLibTimsDataConverter::new(bruker_lib_path, data_path)
    }) {
        Ok(converter) => converter,
        Err(_) => return None,
    };

    // Generate a range of TOF indices across the spectrum
    let n_points = 1000;
    let step = tof_max_index / n_points;
    let tof_indices: Vec<u32> = (0..n_points).map(|i| i * step + step / 2).collect();

    // Convert to m/z using SDK
    let mz_values = sdk_converter.tof_to_mz(1, &tof_indices);

    // Filter out any invalid values (zero or negative)
    let valid_pairs: Vec<(f64, f64)> = tof_indices
        .iter()
        .zip(mz_values.iter())
        .filter(|(_, mz)| **mz > 0.0)
        .map(|(tof, mz)| (*tof as f64, mz.sqrt()))
        .collect();

    if valid_pairs.len() < 10 {
        return None;
    }

    // Linear regression: sqrt(mz) = intercept + slope * tof
    // Using simple least squares: slope = Cov(x,y) / Var(x), intercept = mean_y - slope * mean_x
    let n = valid_pairs.len() as f64;
    let sum_x: f64 = valid_pairs.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = valid_pairs.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = valid_pairs.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f64 = valid_pairs.iter().map(|(x, _)| x * x).sum();

    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let slope = (sum_xy - n * mean_x * mean_y) / (sum_xx - n * mean_x * mean_x);
    let intercept = mean_y - slope * mean_x;

    Some((intercept, slope))
}

fn lzf_decompress(data: &[u8], max_output_size: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let decompressed_data = lzf::decompress(data, max_output_size)
        .map_err(|e| format!("LZF decompression failed: {}", e))?;
    Ok(decompressed_data)
}

fn parse_decompressed_bruker_binary_type1(
    decompressed_bytes: &[u8],
    scan_indices: &mut [i64],
    tof_indices: &mut [u32],
    intensities: &mut [u16],
    scan_start: usize,
    scan_index: usize,
) -> usize {
    // Interpret decompressed_bytes as a slice of i32
    let int_count = decompressed_bytes.len() / 4;
    let buffer =
        unsafe { std::slice::from_raw_parts(decompressed_bytes.as_ptr() as *const i32, int_count) };

    let mut tof_index = 0i32;
    let mut previous_was_intensity = true;
    let mut current_index = scan_start;

    for &value in buffer {
        if value >= 0 {
            // positive value => intensity
            if previous_was_intensity {
                tof_index += 1;
            }
            tof_indices[current_index] = tof_index as u32;
            intensities[current_index] = value as u16;
            previous_was_intensity = true;
            current_index += 1;
        } else {
            // negative value => indicates a jump in tof_index
            tof_index -= value; // value is negative, so this adds |value| to tof_index
            previous_was_intensity = false;
        }
    }

    let scan_size = current_index - scan_start;
    scan_indices[scan_index] = scan_size as i64;
    scan_size
}

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
        let tims_offset_values = frame_meta_data
            .iter()
            .map(|x| x.tims_id)
            .collect::<Vec<i64>>();

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
            acquisition_mode,
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
    fn inverse_mobility_to_scan(
        &self,
        frame_id: u32,
        inverse_mobility_values: &Vec<f64>,
    ) -> Vec<u32>;
}

pub struct BrukerLibTimsDataConverter {
    pub bruker_lib: BrukerTimsDataLibrary,
}

impl BrukerLibTimsDataConverter {
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Self {
        let bruker_lib = BrukerTimsDataLibrary::new(bruker_lib_path, data_path).unwrap();
        BrukerLibTimsDataConverter { bruker_lib }
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
        mz_values.resize(tof.len(), 0.0);

        self.bruker_lib
            .tims_index_to_mz(frame_id, &dbl_tofs, &mut mz_values)
            .expect("Bruker binary call failed at: tims_index_to_mz;");

        mz_values
    }

    fn mz_to_tof(&self, frame_id: u32, mz: &Vec<f64>) -> Vec<u32> {
        let mut dbl_mz: Vec<f64> = Vec::new();
        dbl_mz.resize(mz.len(), 0.0);

        for (i, &val) in mz.iter().enumerate() {
            dbl_mz[i] = val;
        }

        let mut tof_values: Vec<f64> = Vec::new();
        tof_values.resize(mz.len(), 0.0);

        self.bruker_lib
            .tims_mz_to_index(frame_id, &dbl_mz, &mut tof_values)
            .expect("Bruker binary call failed at: tims_mz_to_index;");

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

        self.bruker_lib
            .tims_scan_to_inv_mob(frame_id, &dbl_scans, &mut inv_mob)
            .expect("Bruker binary call failed at: tims_scannum_to_oneoverk0;");

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
        scan_values.resize(inv_mob.len(), 0.0);

        self.bruker_lib
            .inv_mob_to_tims_scan(frame_id, &dbl_inv_mob, &mut scan_values)
            .expect("Bruker binary call failed at: tims_oneoverk0_to_scannum;");

        scan_values.iter().map(|&x| x.round() as u32).collect()
    }
}

pub enum TimsIndexConverter {
    Simple(SimpleIndexConverter),
    Calibrated(CalibratedIndexConverter),
    BrukerLib(BrukerLibTimsDataConverter),
    Lookup(LookupIndexConverter),
}

impl IndexConverter for TimsIndexConverter {
    fn tof_to_mz(&self, frame_id: u32, tof_values: &Vec<u32>) -> Vec<f64> {
        match self {
            TimsIndexConverter::Simple(converter) => converter.tof_to_mz(frame_id, tof_values),
            TimsIndexConverter::Calibrated(converter) => converter.tof_to_mz(frame_id, tof_values),
            TimsIndexConverter::BrukerLib(converter) => converter.tof_to_mz(frame_id, tof_values),
            TimsIndexConverter::Lookup(converter) => converter.tof_to_mz(frame_id, tof_values),
        }
    }

    fn mz_to_tof(&self, frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32> {
        match self {
            TimsIndexConverter::Simple(converter) => converter.mz_to_tof(frame_id, mz_values),
            TimsIndexConverter::Calibrated(converter) => converter.mz_to_tof(frame_id, mz_values),
            TimsIndexConverter::BrukerLib(converter) => converter.mz_to_tof(frame_id, mz_values),
            TimsIndexConverter::Lookup(converter) => converter.mz_to_tof(frame_id, mz_values),
        }
    }

    fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: &Vec<u32>) -> Vec<f64> {
        match self {
            TimsIndexConverter::Simple(converter) => {
                converter.scan_to_inverse_mobility(frame_id, scan_values)
            }
            TimsIndexConverter::Calibrated(converter) => {
                converter.scan_to_inverse_mobility(frame_id, scan_values)
            }
            TimsIndexConverter::BrukerLib(converter) => {
                converter.scan_to_inverse_mobility(frame_id, scan_values)
            }
            TimsIndexConverter::Lookup(converter) => {
                converter.scan_to_inverse_mobility(frame_id, scan_values)
            }
        }
    }

    fn inverse_mobility_to_scan(
        &self,
        frame_id: u32,
        inverse_mobility_values: &Vec<f64>,
    ) -> Vec<u32> {
        match self {
            TimsIndexConverter::Simple(converter) => {
                converter.inverse_mobility_to_scan(frame_id, inverse_mobility_values)
            }
            TimsIndexConverter::Calibrated(converter) => {
                converter.inverse_mobility_to_scan(frame_id, inverse_mobility_values)
            }
            TimsIndexConverter::BrukerLib(converter) => {
                converter.inverse_mobility_to_scan(frame_id, inverse_mobility_values)
            }
            TimsIndexConverter::Lookup(converter) => {
                converter.inverse_mobility_to_scan(frame_id, inverse_mobility_values)
            }
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

            let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;

            let ms_type = match ms_type_raw {
                0 => MsType::Precursor,
                8 => MsType::FragmentDda,
                9 => MsType::FragmentDia,
                _ => MsType::Unknown,
            };

            return TimsFrame {
                frame_id: frame_id as i32,
                ms_type,
                scan: Vec::new(),
                tof: Vec::new(),
                ims_frame: ImsFrame::new(
                    self.raw_data_layout.frame_meta_data[(frame_id - 1) as usize].time,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
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
            1 => {
                let scan_count =
                    self.raw_data_layout.frame_meta_data[frame_index].num_scans as usize;
                let num_peaks = num_peaks as usize;
                let compression_offset = 8 + (scan_count + 1) * 4;

                let mut scan_offsets_buffer = vec![0u8; (scan_count + 1) * 4];
                infile.read_exact(&mut scan_offsets_buffer).unwrap();

                let mut scan_offsets = Vec::with_capacity(scan_count + 1);
                {
                    let mut rdr = Cursor::new(&scan_offsets_buffer);
                    for _ in 0..(scan_count + 1) {
                        scan_offsets.push(rdr.read_i32::<LittleEndian>().unwrap());
                    }
                }

                for offs in &mut scan_offsets {
                    *offs -= compression_offset as i32;
                }

                let remaining_size = (bin_size as usize - compression_offset) as usize;
                let mut compressed_data = vec![0u8; remaining_size];
                infile.read_exact(&mut compressed_data).unwrap();

                let mut scan_indices_ = vec![0i64; scan_count];
                let mut tof_indices_ = vec![0u32; num_peaks];
                let mut intensities_ = vec![0u16; num_peaks];

                let mut scan_start = 0usize;

                for scan_index in 0..scan_count {
                    let start = scan_offsets[scan_index] as usize;
                    let end = scan_offsets[scan_index + 1] as usize;

                    if start == end {
                        continue;
                    }

                    let max_output_size = num_peaks * 8;
                    let decompressed_bytes =
                        lzf_decompress(&compressed_data[start..end], max_output_size)
                            .expect("LZF decompression failed.");

                    scan_start += parse_decompressed_bruker_binary_type1(
                        &decompressed_bytes,
                        &mut scan_indices_,
                        &mut tof_indices_,
                        &mut intensities_,
                        scan_start,
                        scan_index,
                    );
                }

                // Create a flat scan vector to match what flatten_scan_values expects
                let mut scan = Vec::with_capacity(num_peaks);
                {
                    let mut current_scan_index = 0u32;
                    for &size in &scan_indices_ {
                        let sz = size as usize;
                        for _ in 0..sz {
                            scan.push(current_scan_index);
                        }
                        current_scan_index += 1;
                    }
                }

                let intensity_dbl = intensities_.iter().map(|&x| x as f64).collect::<Vec<f64>>();
                let tof_i32 = tof_indices_.iter().map(|&x| x as i32).collect::<Vec<i32>>();

                let mz = self.index_converter.tof_to_mz(frame_id, &tof_indices_);
                let inv_mobility = self
                    .index_converter
                    .scan_to_inverse_mobility(frame_id, &scan);

                let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;
                let ms_type = match ms_type_raw {
                    0 => MsType::Precursor,
                    8 => MsType::FragmentDda,
                    9 => MsType::FragmentDia,
                    _ => MsType::Unknown,
                };

                TimsFrame {
                    frame_id: frame_id as i32,
                    ms_type,
                    scan: scan.iter().map(|&x| x as i32).collect(),
                    tof: tof_i32,
                    ims_frame: ImsFrame::new(
                        self.raw_data_layout.frame_meta_data[frame_index].time,
                        inv_mobility,
                        mz,
                        intensity_dbl,
                    ),
                }
            }

            // Existing handling of Type 2
            2 => {
                let mut compressed_data = vec![0u8; bin_size as usize - 8];
                infile.read_exact(&mut compressed_data).unwrap();

                let decompressed_bytes = zstd_decompress(&compressed_data).unwrap();

                let (scan, tof, intensity) =
                    parse_decompressed_bruker_binary_data(&decompressed_bytes).unwrap();
                let intensity_dbl = intensity.iter().map(|&x| x as f64).collect();
                let tof_i32 = tof.iter().map(|&x| x as i32).collect();
                let scan = flatten_scan_values(&scan, true);

                let mz = self.index_converter.tof_to_mz(frame_id, &tof);
                let inv_mobility = self
                    .index_converter
                    .scan_to_inverse_mobility(frame_id, &scan);

                let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;

                let ms_type = match ms_type_raw {
                    0 => MsType::Precursor,
                    8 => MsType::FragmentDda,
                    9 => MsType::FragmentDia,
                    _ => MsType::Unknown,
                };

                TimsFrame {
                    frame_id: frame_id as i32,
                    ms_type,
                    scan: scan.iter().map(|&x| x as i32).collect(),
                    tof: tof_i32,
                    ims_frame: ImsFrame::new(
                        self.raw_data_layout.frame_meta_data[frame_index].time,
                        inv_mobility,
                        mz,
                        intensity_dbl,
                    ),
                }
            }

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
            }

            // Extract from ZSTD compressed binary
            _ if self.raw_data_layout.global_meta_data.tims_compression_type == 2 => {
                let mut compressed_data = vec![0u8; bin_size as usize - 8];
                infile.read_exact(&mut compressed_data).unwrap();

                let decompressed_bytes = zstd_decompress(&compressed_data).unwrap();

                let (scan, tof, intensity) =
                    parse_decompressed_bruker_binary_data(&decompressed_bytes).unwrap();

                let ms_type_raw = self.raw_data_layout.frame_meta_data[frame_index].ms_ms_type;

                let ms_type = match ms_type_raw {
                    0 => MsType::Precursor,
                    8 => MsType::FragmentDda,
                    9 => MsType::FragmentDia,
                    _ => MsType::Unknown,
                };

                let frame = RawTimsFrame {
                    frame_id: frame_id as i32,
                    retention_time: self.raw_data_layout.frame_meta_data[(frame_id - 1) as usize]
                        .time,
                    ms_type,
                    scan,
                    tof,
                    intensity: intensity.iter().map(|&x| x as f64).collect(),
                };

                return frame;
            }

            // Error on unknown compression algorithm
            _ => {
                panic!("TimsCompressionType is not 1 or 2.")
            }
        }
    }

    fn get_slice(&self, frame_ids: Vec<u32>, _num_threads: usize) -> TimsSlice {
        let result: Vec<TimsFrame> = frame_ids.into_iter().map(|f| self.get_frame(f)).collect();

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
    compressed_data: Vec<u8>,
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
        let inverse_mobility = self
            .index_converter
            .scan_to_inverse_mobility(frame_id, &scan);

        let ims_frame = ImsFrame::new(
            raw_frame.retention_time,
            inverse_mobility,
            mz,
            raw_frame.intensity,
        );

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
        let bin_size = Cursor::new(&self.compressed_data[offset..bin_size_offset])
            .read_i32::<LittleEndian>()
            .unwrap();

        let data_offset = bin_size_offset + 4; // Adjust based on actual structure
        let frame_data = &self.compressed_data[data_offset..data_offset + bin_size as usize - 8];

        let decompressed_bytes = zstd_decompress(&frame_data).unwrap();

        let (scan, tof, intensity) =
            parse_decompressed_bruker_binary_data(&decompressed_bytes).unwrap();

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
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        let frames = pool.install(|| {
            frame_ids
                .par_iter()
                .map(|&frame_id| self.get_frame(frame_id))
                .collect()
        });

        TimsSlice { frames }
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
    Lazy(TimsLazyLoder),
}

impl TimsDataLoader {
    pub fn new_lazy(
        bruker_lib_path: &str,
        data_path: &str,
        use_bruker_sdk: bool,
        scan_max_index: u32,
        im_lower: f64,
        im_upper: f64,
        tof_max_index: u32,
        mz_lower: f64,
        mz_upper: f64,
    ) -> Self {
        let raw_data_layout = TimsRawDataLayout::new(data_path);

        let index_converter = match use_bruker_sdk {
            true => TimsIndexConverter::BrukerLib(BrukerLibTimsDataConverter::new(
                bruker_lib_path,
                data_path,
            )),
            false => {
                // Try to derive accurate calibration using SDK, fall back to simple model
                match derive_mz_calibration(bruker_lib_path, data_path, tof_max_index) {
                    Some((intercept, slope)) => {
                        TimsIndexConverter::Calibrated(CalibratedIndexConverter::new(
                            intercept,
                            slope,
                            im_lower,
                            im_upper,
                            scan_max_index,
                        ))
                    }
                    None => {
                        eprintln!(
                            "Warning: Could not derive m/z calibration from SDK. \
                            Using simple boundary model which may have ~5 Da error on some datasets. \
                            This typically happens on macOS where Bruker SDK is not available."
                        );
                        TimsIndexConverter::Simple(SimpleIndexConverter::from_boundaries(
                            mz_lower,
                            mz_upper,
                            tof_max_index,
                            im_lower,
                            im_upper,
                            scan_max_index,
                        ))
                    }
                }
            }
        };

        TimsDataLoader::Lazy(TimsLazyLoder {
            raw_data_layout,
            index_converter,
        })
    }

    pub fn new_in_memory(
        bruker_lib_path: &str,
        data_path: &str,
        use_bruker_sdk: bool,
        scan_max_index: u32,
        im_lower: f64,
        im_upper: f64,
        tof_max_index: u32,
        mz_lower: f64,
        mz_upper: f64,
    ) -> Self {
        let raw_data_layout = TimsRawDataLayout::new(data_path);

        let index_converter = match use_bruker_sdk {
            true => TimsIndexConverter::BrukerLib(BrukerLibTimsDataConverter::new(
                bruker_lib_path,
                data_path,
            )),
            false => {
                // Try to derive accurate calibration using SDK, fall back to simple model
                match derive_mz_calibration(bruker_lib_path, data_path, tof_max_index) {
                    Some((intercept, slope)) => {
                        TimsIndexConverter::Calibrated(CalibratedIndexConverter::new(
                            intercept,
                            slope,
                            im_lower,
                            im_upper,
                            scan_max_index,
                        ))
                    }
                    None => {
                        eprintln!(
                            "Warning: Could not derive m/z calibration from SDK. \
                            Using simple boundary model which may have ~5 Da error on some datasets. \
                            This typically happens on macOS where Bruker SDK is not available."
                        );
                        TimsIndexConverter::Simple(SimpleIndexConverter::from_boundaries(
                            mz_lower,
                            mz_upper,
                            tof_max_index,
                            im_lower,
                            im_upper,
                            scan_max_index,
                        ))
                    }
                }
            }
        };

        let mut file_path = PathBuf::from(data_path);
        file_path.push("analysis.tdf_bin");
        let mut infile = File::open(file_path).unwrap();
        let mut data = Vec::new();
        infile.read_to_end(&mut data).unwrap();

        TimsDataLoader::InMemory(TimsInMemoryLoader {
            raw_data_layout,
            index_converter,
            compressed_data: data,
        })
    }

    /// Create a lazy loader with pre-computed ion mobility calibration lookup table.
    ///
    /// This method enables accurate ion mobility calibration with fast parallel extraction.
    /// The im_lookup table should be pre-computed using the Bruker SDK.
    ///
    /// # Arguments
    /// * `data_path` - Path to the .d folder
    /// * `tof_max_index` - Maximum TOF index (from GlobalMetaData)
    /// * `mz_lower` - Minimum m/z value (from GlobalMetaData)
    /// * `mz_upper` - Maximum m/z value (from GlobalMetaData)
    /// * `im_lookup` - Pre-computed scan→1/K0 lookup table
    ///
    /// # Returns
    /// A new TimsDataLoader with LookupIndexConverter
    pub fn new_lazy_with_calibration(
        data_path: &str,
        tof_max_index: u32,
        mz_lower: f64,
        mz_upper: f64,
        im_lookup: Vec<f64>,
    ) -> Self {
        let raw_data_layout = TimsRawDataLayout::new(data_path);

        let index_converter = TimsIndexConverter::Lookup(LookupIndexConverter::new(
            mz_lower,
            mz_upper,
            tof_max_index,
            im_lookup,
        ));

        TimsDataLoader::Lazy(TimsLazyLoder {
            raw_data_layout,
            index_converter,
        })
    }

    /// Create an in-memory loader with pre-computed ion mobility calibration lookup table.
    ///
    /// This method enables accurate ion mobility calibration with fast parallel extraction.
    /// The im_lookup table should be pre-computed using the Bruker SDK.
    ///
    /// # Arguments
    /// * `data_path` - Path to the .d folder
    /// * `tof_max_index` - Maximum TOF index (from GlobalMetaData)
    /// * `mz_lower` - Minimum m/z value (from GlobalMetaData)
    /// * `mz_upper` - Maximum m/z value (from GlobalMetaData)
    /// * `im_lookup` - Pre-computed scan→1/K0 lookup table
    ///
    /// # Returns
    /// A new TimsDataLoader with LookupIndexConverter
    pub fn new_in_memory_with_calibration(
        data_path: &str,
        tof_max_index: u32,
        mz_lower: f64,
        mz_upper: f64,
        im_lookup: Vec<f64>,
    ) -> Self {
        let raw_data_layout = TimsRawDataLayout::new(data_path);

        let index_converter = TimsIndexConverter::Lookup(LookupIndexConverter::new(
            mz_lower,
            mz_upper,
            tof_max_index,
            im_lookup,
        ));

        let mut file_path = PathBuf::from(data_path);
        file_path.push("analysis.tdf_bin");
        let mut infile = File::open(file_path).unwrap();
        let mut data = Vec::new();
        infile.read_to_end(&mut data).unwrap();

        TimsDataLoader::InMemory(TimsInMemoryLoader {
            raw_data_layout,
            index_converter,
            compressed_data: data,
        })
    }

    /// Create a lazy loader with full calibration (both m/z and IM).
    ///
    /// This method uses regression-derived m/z calibration coefficients instead of
    /// the simple boundary model, providing more accurate m/z conversion.
    ///
    /// # Arguments
    /// * `data_path` - Path to the .d folder
    /// * `tof_intercept` - Intercept for sqrt(mz) = intercept + slope * tof
    /// * `tof_slope` - Slope for sqrt(mz) = intercept + slope * tof
    /// * `im_min` - Minimum 1/K0 value
    /// * `im_max` - Maximum 1/K0 value
    /// * `scan_max_index` - Maximum scan index
    pub fn new_lazy_with_mz_calibration(
        data_path: &str,
        tof_intercept: f64,
        tof_slope: f64,
        im_min: f64,
        im_max: f64,
        scan_max_index: u32,
    ) -> Self {
        let raw_data_layout = TimsRawDataLayout::new(data_path);

        let index_converter = TimsIndexConverter::Calibrated(CalibratedIndexConverter::new(
            tof_intercept,
            tof_slope,
            im_min,
            im_max,
            scan_max_index,
        ));

        TimsDataLoader::Lazy(TimsLazyLoder {
            raw_data_layout,
            index_converter,
        })
    }

    /// Create an in-memory loader with full calibration (both m/z and IM).
    ///
    /// This method uses regression-derived m/z calibration coefficients instead of
    /// the simple boundary model, providing more accurate m/z conversion.
    pub fn new_in_memory_with_mz_calibration(
        data_path: &str,
        tof_intercept: f64,
        tof_slope: f64,
        im_min: f64,
        im_max: f64,
        scan_max_index: u32,
    ) -> Self {
        let raw_data_layout = TimsRawDataLayout::new(data_path);

        let index_converter = TimsIndexConverter::Calibrated(CalibratedIndexConverter::new(
            tof_intercept,
            tof_slope,
            im_min,
            im_max,
            scan_max_index,
        ));

        let mut file_path = PathBuf::from(data_path);
        file_path.push("analysis.tdf_bin");
        let mut infile = File::open(file_path).unwrap();
        let mut data = Vec::new();
        infile.read_to_end(&mut data).unwrap();

        TimsDataLoader::InMemory(TimsInMemoryLoader {
            raw_data_layout,
            index_converter,
            compressed_data: data,
        })
    }

    pub fn get_index_converter(&self) -> &dyn IndexConverter {
        match self {
            TimsDataLoader::InMemory(loader) => &loader.index_converter,
            TimsDataLoader::Lazy(loader) => &loader.index_converter,
        }
    }

    /// Check if the Bruker SDK is being used for index conversion.
    /// The Bruker SDK is NOT thread-safe, so parallel operations that call
    /// the index converter must be disabled when using the SDK.
    pub fn uses_bruker_sdk(&self) -> bool {
        match self {
            TimsDataLoader::InMemory(loader) => matches!(&loader.index_converter, TimsIndexConverter::BrukerLib(_)),
            TimsDataLoader::Lazy(loader) => matches!(&loader.index_converter, TimsIndexConverter::BrukerLib(_)),
        }
    }
}

impl TimsData for TimsDataLoader {
    fn get_frame(&self, frame_id: u32) -> TimsFrame {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_frame(frame_id),
            TimsDataLoader::Lazy(loader) => loader.get_frame(frame_id),
        }
    }
    fn get_raw_frame(&self, frame_id: u32) -> RawTimsFrame {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_raw_frame(frame_id),
            TimsDataLoader::Lazy(loader) => loader.get_raw_frame(frame_id),
        }
    }

    fn get_slice(&self, frame_ids: Vec<u32>, num_threads: usize) -> TimsSlice {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_slice(frame_ids, num_threads),
            TimsDataLoader::Lazy(loader) => loader.get_slice(frame_ids, num_threads),
        }
    }

    fn get_acquisition_mode(&self) -> AcquisitionMode {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_acquisition_mode(),
            TimsDataLoader::Lazy(loader) => loader.get_acquisition_mode(),
        }
    }

    fn get_frame_count(&self) -> i32 {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_frame_count(),
            TimsDataLoader::Lazy(loader) => loader.get_frame_count(),
        }
    }

    fn get_data_path(&self) -> &str {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_data_path(),
            TimsDataLoader::Lazy(loader) => loader.get_data_path(),
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
        let tof_slope: f64 = (mz_max.sqrt() - tof_intercept) / tof_max_index as f64;

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

    fn inverse_mobility_to_scan(
        &self,
        _frame_id: u32,
        _inverse_mobility_values: &Vec<f64>,
    ) -> Vec<u32> {
        let mut scan_values: Vec<u32> = Vec::new();
        scan_values.resize(_inverse_mobility_values.len(), 0);

        for (i, &val) in _inverse_mobility_values.iter().enumerate() {
            scan_values[i] = ((val - self.scan_intercept) / self.scan_slope) as u32;
        }

        scan_values
    }
}

/// M/z calibrated index converter using regression-derived coefficients.
///
/// This provides accurate TOF to m/z conversion without requiring the Bruker SDK
/// by using linear regression coefficients derived from known precursor m/z values.
///
/// The calibration formula is:
///   sqrt(mz) = tof_intercept + tof_slope * tof_index
///
/// This is similar to SimpleIndexConverter but uses externally-provided coefficients
/// (e.g., from regression on precursor data) rather than boundary-derived values.
pub struct CalibratedIndexConverter {
    pub tof_intercept: f64,
    pub tof_slope: f64,
    pub scan_intercept: f64,
    pub scan_slope: f64,
}

impl CalibratedIndexConverter {
    /// Create a new calibrated converter with regression-derived coefficients.
    ///
    /// # Arguments
    /// * `tof_intercept` - Intercept for sqrt(mz) = intercept + slope * tof
    /// * `tof_slope` - Slope for sqrt(mz) = intercept + slope * tof
    /// * `im_min` - Minimum 1/K0 value
    /// * `im_max` - Maximum 1/K0 value
    /// * `scan_max_index` - Maximum scan index
    pub fn new(
        tof_intercept: f64,
        tof_slope: f64,
        im_min: f64,
        im_max: f64,
        scan_max_index: u32,
    ) -> Self {
        let scan_intercept = im_max;
        let scan_slope = (im_min - scan_intercept) / scan_max_index as f64;
        Self {
            tof_intercept,
            tof_slope,
            scan_intercept,
            scan_slope,
        }
    }
}

impl IndexConverter for CalibratedIndexConverter {
    fn tof_to_mz(&self, _frame_id: u32, tof_values: &Vec<u32>) -> Vec<f64> {
        let mut mz_values: Vec<f64> = Vec::with_capacity(tof_values.len());

        for &tof_index in tof_values.iter() {
            // sqrt(mz) = tof_intercept + tof_slope * tof_index
            let sqrt_mz = self.tof_intercept + self.tof_slope * tof_index as f64;
            mz_values.push(sqrt_mz * sqrt_mz);
        }

        mz_values
    }

    fn mz_to_tof(&self, _frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32> {
        let mut tof_values: Vec<u32> = Vec::with_capacity(mz_values.len());

        for &mz in mz_values.iter() {
            let sqrt_mz = mz.sqrt();
            // tof_index = (sqrt(mz) - tof_intercept) / tof_slope
            tof_values.push(((sqrt_mz - self.tof_intercept) / self.tof_slope) as u32);
        }

        tof_values
    }

    fn scan_to_inverse_mobility(&self, _frame_id: u32, scan_values: &Vec<u32>) -> Vec<f64> {
        let mut inv_mobility_values: Vec<f64> = Vec::with_capacity(scan_values.len());

        for &val in scan_values.iter() {
            inv_mobility_values.push(self.scan_intercept + self.scan_slope * val as f64);
        }

        inv_mobility_values
    }

    fn inverse_mobility_to_scan(
        &self,
        _frame_id: u32,
        inverse_mobility_values: &Vec<f64>,
    ) -> Vec<u32> {
        let mut scan_values: Vec<u32> = Vec::with_capacity(inverse_mobility_values.len());

        for &val in inverse_mobility_values.iter() {
            scan_values.push(((val - self.scan_intercept) / self.scan_slope) as u32);
        }

        scan_values
    }
}

/// Ion mobility index converter using pre-computed lookup table.
///
/// This converter uses a pre-computed scan→1/K0 lookup table extracted from the Bruker SDK.
/// It enables accurate ion mobility calibration with fast parallel extraction.
///
/// Background:
/// - The Bruker calibration formula is patented and proprietary
/// - Using the Bruker SDK gives accurate values but is slow (not thread-safe)
/// - Linear interpolation is fast but inaccurate
/// - This converter uses SDK-probed lookup for accuracy with O(1) thread-safe lookups
///
/// The lookup table is typically small (~8KB for 1000 scans) and constant across all frames.
pub struct LookupIndexConverter {
    // m/z conversion uses simple linear model (accurate enough for most purposes)
    pub tof_intercept: f64,
    pub tof_slope: f64,

    // Ion mobility: pre-computed lookup table from Bruker SDK
    // scan_index → 1/K0 value
    pub im_lookup: Vec<f64>,

    // Fallback for inverse conversion (1/K0 → scan)
    // We store the min/max for binary search bounds
    pub im_min: f64,
    pub im_max: f64,
}

impl LookupIndexConverter {
    /// Create a new LookupIndexConverter with pre-computed ion mobility lookup.
    ///
    /// # Arguments
    /// * `mz_min` - Minimum m/z value for TOF conversion
    /// * `mz_max` - Maximum m/z value for TOF conversion
    /// * `tof_max_index` - Maximum TOF index
    /// * `im_lookup` - Pre-computed scan→1/K0 lookup table from Bruker SDK
    ///
    /// # Returns
    /// A new LookupIndexConverter instance
    pub fn new(
        mz_min: f64,
        mz_max: f64,
        tof_max_index: u32,
        im_lookup: Vec<f64>,
    ) -> Self {
        let tof_intercept: f64 = mz_min.sqrt();
        let tof_slope: f64 = (mz_max.sqrt() - tof_intercept) / tof_max_index as f64;

        // Get IM bounds for inverse conversion
        let im_min = im_lookup.iter().cloned().fold(f64::INFINITY, f64::min);
        let im_max = im_lookup.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            tof_intercept,
            tof_slope,
            im_lookup,
            im_min,
            im_max,
        }
    }
}

impl IndexConverter for LookupIndexConverter {
    fn tof_to_mz(&self, _frame_id: u32, tof_values: &Vec<u32>) -> Vec<f64> {
        let mut mz_values: Vec<f64> = Vec::new();
        mz_values.resize(tof_values.len(), 0.0);

        for (i, &val) in tof_values.iter().enumerate() {
            mz_values[i] = (self.tof_intercept + self.tof_slope * val as f64).powi(2);
        }

        mz_values
    }

    fn mz_to_tof(&self, _frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32> {
        let mut tof_values: Vec<u32> = Vec::new();
        tof_values.resize(mz_values.len(), 0);

        for (i, &val) in mz_values.iter().enumerate() {
            tof_values[i] = ((val.sqrt() - self.tof_intercept) / self.tof_slope) as u32;
        }

        tof_values
    }

    fn scan_to_inverse_mobility(&self, _frame_id: u32, scan_values: &Vec<u32>) -> Vec<f64> {
        // Use the pre-computed lookup table for O(1) conversion
        scan_values
            .iter()
            .map(|&s| {
                self.im_lookup
                    .get(s as usize)
                    .copied()
                    .unwrap_or(f64::NAN)
            })
            .collect()
    }

    fn inverse_mobility_to_scan(
        &self,
        _frame_id: u32,
        inverse_mobility_values: &Vec<f64>,
    ) -> Vec<u32> {
        // Use binary search to find the closest scan index for each 1/K0 value
        // The lookup table is monotonically decreasing (higher scan = lower 1/K0)
        inverse_mobility_values
            .iter()
            .map(|&im| {
                if im.is_nan() || self.im_lookup.is_empty() {
                    return 0;
                }

                // Binary search for the closest value
                // Note: im_lookup is typically monotonically decreasing
                let mut best_scan = 0usize;
                let mut best_diff = f64::INFINITY;

                for (scan, &lookup_im) in self.im_lookup.iter().enumerate() {
                    let diff = (lookup_im - im).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_scan = scan;
                    }
                }

                best_scan as u32
            })
            .collect()
    }
}
