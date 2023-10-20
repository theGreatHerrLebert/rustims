use std::fmt::Display;
use super::raw::BrukerTimsDataLibrary;
use super::meta::{read_global_meta_sql, read_meta_data_sql, FrameMeta, GlobalMetaData};
use std::fs::write;

use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::fs::File;
use std::io::{Seek, SeekFrom, Cursor};
use byteorder::{LittleEndian, ByteOrder, ReadBytesExt};

use mscore::{TimsFrame, ImsFrame, MsType, TimsSlice};

/// Decompresses a ZSTD compressed byte array
///
/// # Arguments
///
/// * `compressed_data` - A byte slice that holds the compressed data
///
/// # Returns
///
/// * `decompressed_data` - A vector of u8 that holds the decompressed data
///
fn zstd_decompress(compressed_data: &[u8]) -> io::Result<Vec<u8>> {
    let mut decoder = zstd::Decoder::new(compressed_data)?;
    let mut decompressed_data = Vec::new();
    decoder.read_to_end(&mut decompressed_data)?;
    Ok(decompressed_data)
}

/// Parses the decompressed bruker binary data
///
/// # Arguments
///
/// * `decompressed_bytes` - A byte slice that holds the decompressed data
///
/// # Returns
///
/// * `scan_indices` - A vector of u32 that holds the scan indices
/// * `tof_indices` - A vector of u32 that holds the tof indices
/// * `intensities` - A vector of u32 that holds the intensities
///
fn parse_decompressed_bruker_binary_data(decompressed_bytes: &[u8]) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), Box<dyn std::error::Error>> {

    let mut buffer_u32 = Vec::new();

    for i in 0..(decompressed_bytes.len() / 4) {
        let value = LittleEndian::read_u32(&[
            decompressed_bytes[i],
            decompressed_bytes[i + (decompressed_bytes.len() / 4)],
            decompressed_bytes[i + (2 * decompressed_bytes.len() / 4)],
            decompressed_bytes[i + (3 * decompressed_bytes.len() / 4)]
        ]);
        buffer_u32.push(value);
    }

    use std::fs::write;

    let file_name = format!("/home/administrator/Documents/promotion/rust/notebook/rust_buffer_frame.bin");
    let mut file = File::create(file_name)?;
    for &value in buffer_u32.iter() {
        file.write_all(&value.to_le_bytes())?;
    }

    // get the number of scans
    let scan_count = buffer_u32[0] as usize;

    // get the scan indices
    let mut scan_indices: Vec<u32> = buffer_u32[..scan_count].to_vec();
    for index in &mut scan_indices {
        *index /= 2;
    }

    // first scan index is always 0?
    scan_indices[0] = 0;

    // get the tof indices, which are the first half of the buffer after the scan indices
    let mut tof_indices: Vec<u32> = buffer_u32.iter().skip(scan_count).step_by(2).cloned().collect();

    // convert the tof indices to cumulative sums
    let mut index = 0;
    for &size in &scan_indices {
        let mut current_sum = 0;
        for _ in 0..size {
            current_sum += tof_indices[index];
            tof_indices[index] = current_sum;
            index += 1;
        }
    }

    // get the intensities, which are the second half of the buffer
    let intensities: Vec<u32> = buffer_u32.iter().skip(scan_count + 1).step_by(2).cloned().collect();

    // get the last scan index
    let last_scan = intensities.len() as u32 - scan_indices[1..].iter().sum::<u32>();

    // shift the scan indices to the right
    for i in 0..(scan_indices.len() - 1) {
        scan_indices[i] = scan_indices[i + 1];
    }

    // set the last scan index
    let len = scan_indices.len();
    scan_indices[len - 1] = last_scan;

    // adjust the tof indices to be zero-indexed
    let adjusted_tof_indices: Vec<u32> = tof_indices.iter().map(|&val| val - 1).collect();
    Ok((scan_indices, adjusted_tof_indices, intensities))
}



#[derive(Debug)]
pub enum AcquisitionMode {
    DDA,
    DIA,
    MIDIA,
    Unknown,
}

impl AcquisitionMode {
    pub fn to_i32(&self) -> i32 {
        match self {
            AcquisitionMode::DDA => 8,
            AcquisitionMode::DIA => 9,
            AcquisitionMode::MIDIA => 10,
            AcquisitionMode::Unknown => -1,
        }
    }
}

impl Display for AcquisitionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcquisitionMode::DDA => write!(f, "DDA"),
            AcquisitionMode::DIA => write!(f, "DIA"),
            AcquisitionMode::MIDIA => write!(f, "MIDIA"),
            AcquisitionMode::Unknown => write!(f, "UNKNOWN"),
        }
    }
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
        let max_scan_count = frame_meta_data.iter().map(|x| x.num_scans).max().unwrap() + 1;

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
            10 => AcquisitionMode::MIDIA,
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
        mz_values.resize(tof.len() as usize, 0.0);

        self.bruker_lib.tims_index_to_mz(frame_id, &dbl_tofs, &mut mz_values).expect("Bruker binary call failed at: tims_index_to_mz;");

        mz_values
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
        inv_mob.resize(scan.len() as usize, 0.0);

        self.bruker_lib.tims_scan_to_inv_mob(frame_id, &dbl_scans, &mut inv_mob).expect("Bruker binary call failed at: tims_scannum_to_oneoverk0;");

        inv_mob
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
                write("/home/administrator/Documents/promotion/rust/notebook/compressed_rust.bytes", &compressed_data)?;

                let decompressed_bytes = zstd_decompress(&compressed_data)?;
                write("/home/administrator/Documents/promotion/rust/notebook/decompressed_rust.bytes", &decompressed_bytes)?;

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