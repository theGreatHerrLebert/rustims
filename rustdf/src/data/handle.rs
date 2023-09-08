use super::raw::BrukerTimsDataLibrary;
use super::meta::{read_global_meta_sql, read_meta_data_sql, FrameMeta, GlobalMetaData};

use std::io::{self, Read};
use std::path::PathBuf;
use std::fs::File;
use std::io::{Seek, SeekFrom, Cursor};
use byteorder::{LittleEndian, ByteOrder, ReadBytesExt};

fn zstd_decompress(compressed_data: &[u8]) -> io::Result<Vec<u8>> {
    let mut decoder = zstd::Decoder::new(compressed_data)?;
    let mut decompressed_data = Vec::new();
    decoder.read_to_end(&mut decompressed_data)?;
    Ok(decompressed_data)
}

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

    let scan_count = buffer_u32[0] as usize;

    let mut scan_indices: Vec<u32> = buffer_u32[..scan_count].to_vec();
    for index in &mut scan_indices {
        *index /= 2;
    }
    scan_indices[0] = 0;
    
    let mut tof_indices: Vec<u32> = buffer_u32.iter().skip(scan_count).step_by(2).cloned().collect();
    
    let mut index = 0;
    for &size in &scan_indices {
        let mut current_sum = 0;
        for _ in 0..size {
            current_sum += tof_indices[index];
            tof_indices[index] = current_sum;
            index += 1;
        }
    }
    
    let intensities: Vec<u32> = buffer_u32.iter().skip(scan_count + 1).step_by(2).cloned().collect();
    
    let last_scan = intensities.len() as u32 - scan_indices[1..].iter().sum::<u32>();
    
    for i in 0..(scan_indices.len() - 1) {
        scan_indices[i] = scan_indices[i + 1];
    }

    let len = scan_indices.len();
    
    scan_indices[len - 1] = last_scan;
    
    let adjusted_tof_indices: Vec<u32> = tof_indices.iter().map(|&val| val - 1).collect();

    Ok((scan_indices, adjusted_tof_indices, intensities))
}



#[derive(Debug)]
pub enum AquisitionMode {
    DDA,
    DIA,
    MIDIA,
    UNKNOWN
}

pub struct TimsDataset {
    pub data_path: String,
    pub bruker_lib_path: String,
    pub bruker_lib: BrukerTimsDataLibrary,
    pub global_meta_data: GlobalMetaData,
    pub frame_meta_data: Vec<FrameMeta>,
    pub aquisition_mode: AquisitionMode,
    pub max_scan_count: i64,
    pub frame_idptr: Vec<i64>,
    pub tims_offset_values: Vec<i64>,
}

impl TimsDataset {
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Result<TimsDataset, Box<dyn std::error::Error>> {
        
        let bruker_lib = BrukerTimsDataLibrary::new(bruker_lib_path, data_path)?;
        let global_meta_data = read_global_meta_sql(data_path)?;
        let frame_meta_data = read_meta_data_sql(data_path)?;

        let max_scan_count = frame_meta_data.iter().map(|x| x.num_scans).max().unwrap() + 1;

        let mut frame_idptr: Vec<i64> = Vec::new();
        frame_idptr.resize(frame_meta_data.len() + 1, 0);

        for (i, row) in frame_meta_data.iter().enumerate() {
            frame_idptr[i + 1] = row.num_peaks + frame_idptr[i];
        }

        let tims_offset_values = frame_meta_data.iter().map(|x| x.tims_id).collect::<Vec<i64>>();

        let aquisition_mode = match frame_meta_data[0].scan_mode {
            8 => AquisitionMode::DDA,
            9 => AquisitionMode::DIA,
            10 => AquisitionMode::MIDIA,
            _ => AquisitionMode::UNKNOWN,
        };

        Ok(TimsDataset {
            data_path: data_path.to_string(),
            bruker_lib_path: bruker_lib_path.to_string(),
            bruker_lib,
            global_meta_data,
            frame_meta_data,
            aquisition_mode,
            max_scan_count,
            frame_idptr,
            tims_offset_values,
        })
    }

    pub fn get_frame(&self, frame_id: u32) -> Result<(Vec<u32>, Vec<u32>, Vec<f64>, Vec<u32>), Box<dyn std::error::Error>> {

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
                let mut dbl_tofs: Vec<f64> = Vec::new();
                dbl_tofs.resize(self.global_meta_data.digitizer_num_samples as usize, 0.0);

                for (i, &val) in tof.iter().enumerate() {
                    dbl_tofs[i] = val as f64;
                }

                let mut mz_values: Vec<f64> = Vec::new();
                mz_values.resize(self.global_meta_data.digitizer_num_samples as usize, 0.0);

                self.bruker_lib.tims_index_to_mz(frame_id, &dbl_tofs, &mut mz_values, self.global_meta_data.digitizer_num_samples)?;
                mz_values.truncate(tof.len());
                
                Ok((scan, tof, mz_values, intensity))
            },

            // Error on unknown compression algorithm
            _ => {
                return Err("TimsCompressionType is not 1 or 2.".into());
            }
        }
    }
}