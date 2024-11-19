extern crate rusqlite;

use rusqlite::{Connection, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct DiaMsMisInfo {
    pub frame_id: u32,
    pub window_group: u32,
}

#[derive(Debug, Clone)]
pub struct DiaMsMsWindow {
    pub window_group: u32,
    pub scan_num_begin: u32,
    pub scan_num_end: u32,
    pub isolation_mz: f64,
    pub isolation_width: f64,
    pub collision_energy: f64,
}

#[derive(Debug, Clone)]
pub struct PasefMsMsMeta {
    pub frame_id: i64,
    pub scan_num_begin: i64,
    pub scan_num_end: i64,
    pub isolation_mz: f64,
    pub isolation_width: f64,
    pub collision_energy: f64,
    pub precursor_id: i64,
}

#[derive(Debug, Clone)]
pub struct DDAPrecursorMeta {
    pub precursor_id: i64,
    pub precursor_mz_highest_intensity: f64,
    pub precursor_mz_average: f64,
    pub precursor_mz_monoisotopic: Option<f64>,
    pub precursor_charge: Option<i64>,
    pub precursor_average_scan_number: f64,
    pub precursor_total_intensity: f64,
    pub precursor_frame_id: i64,
}

pub struct DDAFragmentInfo {
    pub frame_id: i64,
    pub scan_begin: i64,
    pub scan_end: i64,
    pub isolation_mz: f64,
    pub isolation_width: f64,
    pub collision_energy: f64,
    pub precursor_id: i64,
}

pub struct DIAFragmentFrameInfo {}

pub struct DIAWindowGroupInfo {}

#[derive(Debug)]
pub struct GlobalMetaData {
    pub schema_type: String,
    pub schema_version_major: i64,
    pub schema_version_minor: i64,
    pub acquisition_software_vendor: String,
    pub instrument_vendor: String,
    pub closed_property: i64,
    pub tims_compression_type: i64,
    pub max_num_peaks_per_scan: i64,
    pub mz_acquisition_range_lower: f64,
    pub mz_acquisition_range_upper: f64,
    pub one_over_k0_range_lower: f64,
    pub one_over_k0_range_upper: f64,
    pub tof_max_index: u32,
}

#[derive(Debug)]
pub struct FrameMeta {
    pub id: i64,
    pub time: f64,
    pub polarity: String,
    pub scan_mode: i64,
    pub ms_ms_type: i64,
    pub tims_id: i64,
    pub max_intensity: f64,
    pub sum_intensity: f64,
    pub num_scans: i64,
    pub num_peaks: i64,
    pub mz_calibration: i64,
    pub t_1: f64,
    pub t_2: f64,
    pub tims_calibration: i64,
    pub property_group: i64,
    pub accumulation_time: f64,
    pub ramp_time: f64,
}

struct GlobalMetaInternal {
    key: String,
    value: String,
}

pub fn read_dda_precursor_meta(bruker_d_folder_name: &str) -> Result<Vec<DDAPrecursorMeta>, Box<dyn std::error::Error>> {
    // Connect to the database
    let db_path = Path::new(bruker_d_folder_name).join("analysis.tdf");
    let conn = Connection::open(db_path)?;

    // prepare the query
    let rows: Vec<&str> = vec!["Id", "LargestPeakMz", "AverageMz", "MonoisotopicMz", "Charge", "ScanNumber", "Intensity", "Parent"];
    let query = format!("SELECT {} FROM Precursors", rows.join(", "));

    // execute the query
    let frames_rows: Result<Vec<DDAPrecursorMeta>, _> = conn.prepare(&query)?.query_map([], |row| {
        Ok(DDAPrecursorMeta {
            precursor_id: row.get(0)?,
            precursor_mz_highest_intensity: row.get(1)?,
            precursor_mz_average: row.get(2)?,
            precursor_mz_monoisotopic: row.get(3)?,  // Now using Option<f64>
            precursor_charge: row.get(4)?,           // Now using Option<i64>
            precursor_average_scan_number: row.get(5)?,
            precursor_total_intensity: row.get(6)?,
            precursor_frame_id: row.get(7)?,
        })
    })?.collect();

    // return the frames
    Ok(frames_rows?)
}

pub fn read_pasef_frame_ms_ms_info(bruker_d_folder_name: &str) -> Result<Vec<PasefMsMsMeta>, Box<dyn std::error::Error>> {
    // Connect to the database
    let db_path = Path::new(bruker_d_folder_name).join("analysis.tdf");
    let conn = Connection::open(db_path)?;

    // prepare the query
    let rows: Vec<&str> = vec!["Frame", "ScanNumBegin", "ScanNumEnd", "IsolationMz", "IsolationWidth", "CollisionEnergy", "Precursor"];
    let query = format!("SELECT {} FROM PasefFrameMsMsInfo", rows.join(", "));

    // execute the query
    let frames_rows: Result<Vec<PasefMsMsMeta>, _> = conn.prepare(&query)?.query_map([], |row| {
        Ok(PasefMsMsMeta {
        frame_id: row.get(0)?,
        scan_num_begin: row.get(1)?,
        scan_num_end: row.get(2)?,
        isolation_mz: row.get(3)?,
        isolation_width: row.get(4)?,
        collision_energy: row.get(5)?,
        precursor_id: row.get(6)?, })
        })?.collect();

    // return the frames
    Ok(frames_rows?)
}

// Read the global meta data from the analysis.tdf file
pub fn read_global_meta_sql(bruker_d_folder_name: &str) -> Result<GlobalMetaData, Box<dyn std::error::Error>> {

    // Connect to the database
    let db_path = Path::new(bruker_d_folder_name).join("analysis.tdf");
    let conn = Connection::open(db_path)?;

    // execute the query
    let frames_rows: Result<Vec<GlobalMetaInternal>, _> = conn.prepare("SELECT * FROM GlobalMetadata")?.query_map([], |row| {
        Ok(GlobalMetaInternal {
            key: row.get(0)?,
            value: row.get(1)?,
            })
        })?.collect();

    let mut global_meta = GlobalMetaData {
        schema_type: String::new(),
        schema_version_major: -1,
        schema_version_minor: -1,
        acquisition_software_vendor: String::new(),
        instrument_vendor: String::new(),
        closed_property: -1,
        tims_compression_type: -1,
        max_num_peaks_per_scan: -1,
        mz_acquisition_range_lower: -1.0,
        mz_acquisition_range_upper: -1.0,
        one_over_k0_range_lower: -1.0,
        one_over_k0_range_upper: -1.0,
        tof_max_index: 0,
    };

    // go over the keys and parse values for the global meta data
    for row in frames_rows? {
        match row.key.as_str() {
            "SchemaType" => global_meta.schema_type = row.value,
            "SchemaVersionMajor" => global_meta.schema_version_major = row.value.parse::<i64>().unwrap(),
            "SchemaVersionMinor" => global_meta.schema_version_minor = row.value.parse::<i64>().unwrap(),
            "AcquisitionSoftwareVendor" => global_meta.acquisition_software_vendor = row.value,
            "InstrumentVendor" => global_meta.instrument_vendor = row.value,
            "ClosedProperly" => global_meta.closed_property = row.value.parse::<i64>().unwrap(),
            "TimsCompressionType" => global_meta.tims_compression_type = row.value.parse::<i64>().unwrap(),
            "MaxNumPeaksPerScan" => global_meta.max_num_peaks_per_scan = row.value.parse::<i64>().unwrap(),
            "MzAcqRangeLower" => global_meta.mz_acquisition_range_lower = row.value.parse::<f64>().unwrap(),
            "MzAcqRangeUpper" => global_meta.mz_acquisition_range_upper = row.value.parse::<f64>().unwrap(),
            "OneOverK0AcqRangeLower" => global_meta.one_over_k0_range_lower = row.value.parse::<f64>().unwrap(),
            "OneOverK0AcqRangeUpper" => global_meta.one_over_k0_range_upper = row.value.parse::<f64>().unwrap(),
            "DigitizerNumSamples" => global_meta.tof_max_index = (row.value.parse::<i64>().unwrap() + 1) as u32,
            _ => (),
        }
    }
    // return global_meta
    Ok(global_meta)   
}

// Read the frame meta data from the analysis.tdf file
pub fn read_meta_data_sql(bruker_d_folder_name: &str) -> Result<Vec<FrameMeta>, Box<dyn std::error::Error>> {
    // Connect to the database
    let db_path = Path::new(bruker_d_folder_name).join("analysis.tdf");
    let conn = Connection::open(db_path)?;

    // prepare the query
    let rows: Vec<&str> = vec!["Id", "Time", "ScanMode", "Polarity", "MsMsType", "TimsId", "MaxIntensity", "SummedIntensities", 
    "NumScans", "NumPeaks", "MzCalibration", "T1", "T2", "TimsCalibration", "PropertyGroup", "AccumulationTime", "RampTime"];
    let query = format!("SELECT {} FROM Frames", rows.join(", "));

    // execute the query
    let frames_rows: Result<Vec<FrameMeta>, _> = conn.prepare(&query)?.query_map([], |row| {
    Ok(FrameMeta {
        id: row.get(0)?,
        time: row.get(1)?,
        scan_mode: row.get(2)?,
        polarity: row.get(3)?,
        ms_ms_type: row.get(4)?,
        tims_id: row.get(5)?,
        max_intensity: row.get(6)?,
        sum_intensity: row.get(7)?,
        num_scans: row.get(8)?,
        num_peaks: row.get(9)?,
        mz_calibration: row.get(10)?,
        t_1: row.get(11)?,
        t_2: row.get(12)?,
        tims_calibration: row.get(13)?,
        property_group: row.get(14)?,
        accumulation_time: row.get(15)?,
        ramp_time: row.get(16)?,
        })
    })?.collect();

    // return the frames
    Ok(frames_rows?)
}

pub fn read_dia_ms_ms_info(bruker_d_folder_name: &str) -> Result<Vec<DiaMsMisInfo>, Box<dyn std::error::Error>> {
    // Connect to the database
    let db_path = Path::new(bruker_d_folder_name).join("analysis.tdf");
    let conn = Connection::open(db_path)?;

    // prepare the query
    let rows: Vec<&str> = vec!["Frame", "WindowGroup"];
    let query = format!("SELECT {} FROM DiaFrameMsMsInfo", rows.join(", "));

    // execute the query
    let frames_rows: Result<Vec<DiaMsMisInfo>, _> = conn.prepare(&query)?.query_map([], |row| {
        Ok(DiaMsMisInfo {
            frame_id: row.get(0)?,
            window_group: row.get(1)?,
        })
    })?.collect();

    // return the frames
    Ok(frames_rows?)
}

pub fn read_dia_ms_ms_windows(bruker_d_folder_name: &str) -> Result<Vec<DiaMsMsWindow>, Box<dyn std::error::Error>> {
    // Connect to the database
    let db_path = Path::new(bruker_d_folder_name).join("analysis.tdf");
    let conn = Connection::open(db_path)?;

    // prepare the query
    let rows: Vec<&str> = vec!["WindowGroup", "ScanNumBegin", "ScanNumEnd", "IsolationMz", "IsolationWidth", "CollisionEnergy"];
    let query = format!("SELECT {} FROM DiaFrameMsMsWindows", rows.join(", "));

    // execute the query
    let frames_rows: Result<Vec<DiaMsMsWindow>, _> = conn.prepare(&query)?.query_map([], |row| {
        Ok(DiaMsMsWindow {
            window_group: row.get(0)?,
            scan_num_begin: row.get(1)?,
            scan_num_end: row.get(2)?,
            isolation_mz: row.get(3)?,
            isolation_width: row.get(4)?,
            collision_energy: row.get(5)?,
        })
    })?.collect();

    // return the frames
    Ok(frames_rows?)
}