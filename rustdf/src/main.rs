// use rustdf::data::handle::TimsDataHandle;
use std::env;
use std::path::Path;
use rusqlite::{Connection, Result};
use rustdf::data::meta::{FrameMeta};

fn main() -> Result<()> {
    let _args: Vec<String> = env::args().collect();

    let data_path = "/media/hd01/CCSPred/M210115_001_Slot1-1_1_850.d";
    // Connect to the database
    let db_path = Path::new(data_path).join("analysis.tdf");
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

    for row in frames_rows? {
        println!("FrameMeta: {:?}", row);
    }

    // return the frames
    Ok(())
}
