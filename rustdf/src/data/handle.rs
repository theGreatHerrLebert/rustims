use super::raw::BrukerTimsDataLibrary;
use super::meta::{read_global_meta_sql, read_meta_data_sql, FrameMeta, GlobalMetaData};

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
    pub frame_idptr: Vec<i64>,
    pub tims_offset_values: Vec<i64>,
}

impl TimsDataset {
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Result<TimsDataset, Box<dyn std::error::Error>> {
        
        let bruker_lib = BrukerTimsDataLibrary::new(bruker_lib_path, data_path)?;
        let global_meta_data = read_global_meta_sql(data_path)?;
        let frame_meta_data = read_meta_data_sql(data_path)?;

        let mut frame_idptr: Vec<i64> = Vec::new();
        frame_idptr.push(0);

        for (i, row) in frame_meta_data.iter().enumerate() {
            frame_idptr.push(row.num_peaks + frame_idptr[i]);
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
            frame_idptr,
            tims_offset_values,
        })
    }

    pub fn get_frame(&self, frame_id: i64) -> Result<i64, Box<dyn std::error::Error>> {
        let frame_start = self.frame_idptr[frame_id as usize];
        let frame_end = self.frame_idptr[frame_id as usize + 1];
        let offset = self.tims_offset_values[frame_id as usize];

        println!("frame_start: {}", frame_start);
        println!("frame_end: {}", frame_end);
        println!("offset: {}", offset);
        Ok(1)
    }
}