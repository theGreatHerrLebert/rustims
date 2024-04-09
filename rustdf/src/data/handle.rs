use super::raw::BrukerTimsDataLibrary;
use super::meta::{read_global_meta_sql, read_meta_data_sql, FrameMeta, GlobalMetaData};

use std::io::{Read};
use std::path::PathBuf;
use std::fs::File;
use std::io::{Seek, SeekFrom, Cursor};
use byteorder::{LittleEndian, ReadBytesExt};

use mscore::data::spectrum::MsType;
use mscore::timstof::frame::{TimsFrame, ImsFrame, RawTimsFrame};
use mscore::timstof::slice::TimsSlice;
use crate::data::acquisition::AcquisitionMode;

use crate::data::utility::{parse_decompressed_bruker_binary_data, zstd_decompress};

/// Interface to be shared by all acquisitions of timsTOF
pub trait TimsData {
    fn get_frame(&self, frame_id: u32) -> TimsFrame;
    fn get_slice(&self, frame_ids: Vec<u32>) -> TimsSlice;
    fn get_acquisition_mode(&self) -> AcquisitionMode;
    fn get_frame_count(&self) -> i32;
    fn get_data_path(&self) -> &str;
    fn get_bruker_lib_path(&self) -> &str;

    fn tof_to_mz(&self, frame_id: u32, tof_values: &Vec<u32>) -> Vec<f64>;
    fn mz_to_tof(&self, frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32>;

    fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: &Vec<i32>) -> Vec<f64>;
    fn inverse_mobility_to_scan(&self, frame_id: u32, inverse_mobility_values: &Vec<f64>) -> Vec<i32>;
}

pub struct TimsDataHandle {
    pub data_path: String,
    pub bruker_lib_path: String,
    pub bruker_lib: BrukerTimsDataLibrary,
    pub global_meta_data: GlobalMetaData,
    pub frame_meta_data: Vec<FrameMeta>,
    pub acquisition_mode: AcquisitionMode,
    pub max_scan_count: i64,
    pub frame_idptr: Vec<i64>,
    pub tims_offset_values: Vec<i64>,
}

impl TimsDataHandle {
    /// Creates a new TimsDataHandle
    ///
    /// # Arguments
    ///
    /// * `bruker_lib_path` - A string slice that holds the path to the bruker library
    /// * `data_path` - A string slice that holds the path to the data
    ///
    /// # Returns
    ///
    /// * `tims_dataset` - A TimsDataHandle struct
    ///
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Result<TimsDataHandle, Box<dyn std::error::Error>> {

        // Load the library
        let bruker_lib = BrukerTimsDataLibrary::new(bruker_lib_path, data_path)?;
        // get the global and frame meta data
        let global_meta_data = read_global_meta_sql(data_path)?;
        let frame_meta_data = read_meta_data_sql(data_path)?;
        // get the max scan count
        let max_scan_count = frame_meta_data.iter().map(|x| x.num_scans).max().unwrap();

        let mut frame_idptr: Vec<i64> = Vec::new();
        frame_idptr.resize(frame_meta_data.len() + 1, 0);

        // get the frame idptr values
        for (i, row) in frame_meta_data.iter().enumerate() {
            frame_idptr[i + 1] = row.num_peaks + frame_idptr[i];
        }

        // get the tims offset values
        let tims_offset_values = frame_meta_data.iter().map(|x| x.tims_id).collect::<Vec<i64>>();

        // get the acquisition mode
        let acquisition_mode = match frame_meta_data[0].scan_mode {
            8 => AcquisitionMode::DDA,
            9 => AcquisitionMode::DIA,
            _ => AcquisitionMode::Unknown,
        };

        Ok(TimsDataHandle {
            data_path: data_path.to_string(),
            bruker_lib_path: bruker_lib_path.to_string(),
            bruker_lib,
            global_meta_data,
            frame_meta_data,
            acquisition_mode,
            max_scan_count,
            frame_idptr,
            tims_offset_values,
        })
    }

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
    pub fn tof_to_mz(&self, frame_id: u32, tof: &Vec<u32>) -> Vec<f64> {
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

    pub fn mz_to_tof(&self, frame_id: u32, mz: &Vec<f64>) -> Vec<u32> {
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
    pub fn scan_to_inverse_mobility(&self, frame_id: u32, scan: &Vec<i32>) -> Vec<f64> {
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
    pub fn inverse_mobility_to_scan(&self, frame_id: u32, inv_mob: &Vec<f64>) -> Vec<i32> {
        let mut dbl_inv_mob: Vec<f64> = Vec::new();
        dbl_inv_mob.resize(inv_mob.len(), 0.0);

        for (i, &val) in inv_mob.iter().enumerate() {
            dbl_inv_mob[i] = val;
        }

        let mut scan_values: Vec<f64> = Vec::new();
        scan_values.resize(inv_mob.len(),  0.0);

        self.bruker_lib.inv_mob_to_tims_scan(frame_id, &dbl_inv_mob, &mut scan_values).expect("Bruker binary call failed at: tims_oneoverk0_to_scannum;");

        scan_values.iter().map(|&x| x.round() as i32).collect()
    }

    /// helper function to flatten the scan values
    ///
    /// # Arguments
    ///
    /// * `scan` - A vector of u32 that holds the scan values
    /// * `zero_indexed` - A bool that indicates if the scan values are zero indexed
    ///
    /// # Returns
    ///
    /// * `scan_i32` - A vector of i32 that holds the scan values
    ///
    pub fn flatten_scan_values(&self, scan: &Vec<u32>, zero_indexed: bool) -> Vec<i32> {
        let add = if zero_indexed { 0 } else { 1 };
        scan.iter().enumerate()
            .flat_map(|(index, &count)| vec![(index + add) as i32; count as usize]
                .into_iter()).collect()
    }

    pub fn get_raw_frame(&self, frame_id: u32) -> Result<RawTimsFrame, Box<dyn std::error::Error>> {

        let frame_index = (frame_id - 1) as usize;
        let offset = self.tims_offset_values[frame_index] as u64;

        let mut file_path = PathBuf::from(&self.data_path);
        file_path.push("analysis.tdf_bin");
        let mut infile = File::open(&file_path)?;

        infile.seek(SeekFrom::Start(offset))?;

        let mut bin_buffer = [0u8; 4];
        infile.read_exact(&mut bin_buffer)?;
        let bin_size = Cursor::new(bin_buffer).read_i32::<LittleEndian>()?;

        infile.read_exact(&mut bin_buffer)?;

        match self.global_meta_data.tims_compression_type {
            // TODO: implement
            _ if self.global_meta_data.tims_compression_type == 1 => {
                return Err("Decompression Type1 not implemented.".into());
            },

            // Extract from ZSTD compressed binary
            _ if self.global_meta_data.tims_compression_type == 2 => {

                let mut compressed_data = vec![0u8; bin_size as usize - 8];
                infile.read_exact(&mut compressed_data)?;

                let decompressed_bytes = zstd_decompress(&compressed_data)?;

                let (scan, tof, intensity) = parse_decompressed_bruker_binary_data(&decompressed_bytes)?;

                let ms_type_raw = self.frame_meta_data[frame_index].ms_ms_type;

                let ms_type = match ms_type_raw {
                    0 => MsType::Precursor,
                    8 => MsType::FragmentDda,
                    9 => MsType::FragmentDia,
                    _ => MsType::Unknown,
                };

                Ok(RawTimsFrame {
                    frame_id: frame_id as i32,
                    retention_time: self.frame_meta_data[(frame_id - 1) as usize].time,
                    ms_type,
                    scan,
                    tof,
                    intensity: intensity.iter().map(|&x| x as f64).collect(),
                })
            },

            // Error on unknown compression algorithm
            _ => {
                return Err("TimsCompressionType is not 1 or 2.".into());
            }
        }
    }

    /// get a frame from the tims dataset
    ///
    /// # Arguments
    ///
    /// * `frame_id` - A u32 that holds the frame id
    ///
    /// # Returns
    ///
    /// * `frame` - A TimsFrame struct
    ///
    pub fn get_frame(&self, frame_id: u32) -> Result<TimsFrame, Box<dyn std::error::Error>> {

        let frame_index = (frame_id - 1) as usize;
        let offset = self.tims_offset_values[frame_index] as u64;

        let mut file_path = PathBuf::from(&self.data_path);
        file_path.push("analysis.tdf_bin");
        let mut infile = File::open(&file_path)?;

        infile.seek(SeekFrom::Start(offset))?;

        let mut bin_buffer = [0u8; 4];
        infile.read_exact(&mut bin_buffer)?;
        let bin_size = Cursor::new(bin_buffer).read_i32::<LittleEndian>()?;

        infile.read_exact(&mut bin_buffer)?;

        match self.global_meta_data.tims_compression_type {
            // TODO: implement
            _ if self.global_meta_data.tims_compression_type == 1 => {
                return Err("Decompression Type1 not implemented.".into());
            },

            // Extract from ZSTD compressed binary
            _ if self.global_meta_data.tims_compression_type == 2 => {

                let mut compressed_data = vec![0u8; bin_size as usize - 8];
                infile.read_exact(&mut compressed_data)?;

                let decompressed_bytes = zstd_decompress(&compressed_data)?;

                let (scan, tof, intensity) = parse_decompressed_bruker_binary_data(&decompressed_bytes)?;
                let intensity_dbl = intensity.iter().map(|&x| x as f64).collect();
                let tof_i32 = tof.iter().map(|&x| x as i32).collect();
                let scan_i32: Vec<i32> = self.flatten_scan_values(&scan, true);

                let mz = self.tof_to_mz(frame_id, &tof);
                let inv_mobility = self.scan_to_inverse_mobility(frame_id, &scan_i32);

                let ms_type_raw = self.frame_meta_data[frame_index].ms_ms_type;

                let ms_type = match ms_type_raw {
                    0 => MsType::Precursor,
                    8 => MsType::FragmentDda,
                    9 => MsType::FragmentDia,
                    _ => MsType::Unknown,
                };

                Ok(TimsFrame {
                    frame_id: frame_id as i32,
                    ms_type,
                    scan: scan_i32,
                    tof: tof_i32,
                    ims_frame: ImsFrame { retention_time: self.frame_meta_data[(frame_id - 1) as usize].time, mobility: inv_mobility, mz, intensity: intensity_dbl }
                })
            },

            // Error on unknown compression algorithm
            _ => {
                return Err("TimsCompressionType is not 1 or 2.".into());
            }
        }
    }

    /// get a frame from the tims dataset as an ImsFrame
    ///
    /// # Arguments
    ///
    /// * `frame_id` - A u32 that holds the frame id
    ///
    /// # Returns
    ///
    /// * `frame` - An ImsFrame struct
    ///
    pub fn get_ims_frame(&self, frame_id: u32) -> Result<ImsFrame, Box<dyn std::error::Error>> {
        let frame = self.get_frame(frame_id)?;
        Ok(frame.get_ims_frame())
    }

    /// get a slice of frames from the tims dataset
    ///
    /// # Arguments
    ///
    /// * `frame_ids` - A vector of u32 that holds the frame ids
    ///
    /// # Returns
    ///
    /// * `tims_slice` - A TimsSlice struct
    ///
    pub fn get_tims_slice(&self, frame_ids: Vec<u32>) -> TimsSlice {

        let result: Vec<TimsFrame> = frame_ids
            .into_iter()
            .map(|f| {
                match self.get_frame(f) {
                    Ok(frame) => Some(frame),
                    Err(_e) => {
                        // TODO implement error handling
                        None
                    }
                }
            })
            .filter(Option::is_some)
            .map(Option::unwrap)
            .collect();

        TimsSlice { frames: result }
    }

    /// get the number of frames in the dataset
    ///
    /// # Returns
    ///
    /// * the number of frames in the dataset
    ///
    pub fn get_frame_count(&self) -> i32 {
        self.frame_meta_data.len() as i32
    }
}