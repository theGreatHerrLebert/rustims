use mscore::timstof::frame::{RawTimsFrame, TimsFrame};
use mscore::timstof::slice::TimsSlice;
use crate::data::handle::{AcquisitionMode, TimsDataHandle};
use crate::data::meta::{FrameMeta, GlobalMetaData, read_global_meta_sql, read_meta_data_sql};
use crate::data::raw::BrukerTimsDataLibrary;

trait TimsData {
    fn get_frame(&self, frame_id: u32) -> TimsFrame;
    fn get_raw_frame(&self, frame_id: u32) -> RawTimsFrame;
    fn get_slice(&self, frame_ids: Vec<u32>) -> TimsSlice;
    fn get_acquisition_mode(&self) -> AcquisitionMode;
    fn get_frame_count(&self) -> i32;
    fn get_data_path(&self) -> &str;
}

trait IndexConverter {
    fn tof_to_mz(&self, frame_id: u32, tof_values: &Vec<u32>) -> Vec<f64>;
    fn mz_to_tof(&self, frame_id: u32, mz_values: &Vec<f64>) -> Vec<u32>;
    fn scan_to_inverse_mobility(&self, frame_id: u32, scan_values: &Vec<u32>) -> Vec<f64>;
    fn inverse_mobility_to_scan(&self, frame_id: u32, inverse_mobility_values: &Vec<f64>) -> Vec<u32>;
}

pub struct BrukerLibTimsDataConverter {
    pub data_path: String,
    pub bruker_lib_path: String,
    pub bruker_lib: BrukerTimsDataLibrary,
    pub global_meta_data: GlobalMetaData,
}

struct SimpleIndexConverter {

}

impl IndexConverter for SimpleIndexConverter {
    fn tof_to_mz(&self, _frame_id: u32, _tof_values: &Vec<u32>) -> Vec<f64> {
        todo!("Implement this method")
    }

    fn mz_to_tof(&self, _frame_id: u32, _mz_values: &Vec<f64>) -> Vec<u32> {
        todo!("Implement this method")
    }

    fn scan_to_inverse_mobility(&self, _frame_id: u32, _scan_values: &Vec<u32>) -> Vec<f64> {
        todo!("Implement this method")
    }

    fn inverse_mobility_to_scan(&self, _frame_id: u32, _inverse_mobility_values: &Vec<f64>) -> Vec<u32> {
        todo!("Implement this method")
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
    handle: TimsDataHandle,
    global_meta_data: GlobalMetaData,
    frame_meta_data: Vec<FrameMeta>,
}

impl TimsData for TimsLazyLoder {
    fn get_frame(&self, frame_id: u32) -> TimsFrame {
        self.handle.get_frame(frame_id).unwrap()
    }

    fn get_raw_frame(&self, frame_id: u32) -> RawTimsFrame {
        self.handle.get_raw_frame(frame_id).unwrap()
    }

    fn get_slice(&self, frame_ids: Vec<u32>) -> TimsSlice {
        self.handle.get_tims_slice(frame_ids)
    }

    fn get_acquisition_mode(&self) -> AcquisitionMode {
        self.handle.acquisition_mode.clone()
    }

    fn get_frame_count(&self) -> i32 {
        self.handle.get_frame_count()
    }

    fn get_data_path(&self) -> &str {
        &self.handle.data_path
    }
}

pub struct TimsInMemoryLoader {
    handle: TimsDataHandle,
    global_meta_data: GlobalMetaData,
    frame_meta_data: Vec<FrameMeta>,
    compressed_data: Vec<u8>
}

impl TimsData for TimsInMemoryLoader {
    fn get_frame(&self, frame_id: u32) -> TimsFrame {
        self.handle.get_frame(frame_id).unwrap()
    }

    fn get_raw_frame(&self, frame_id: u32) -> RawTimsFrame {
        self.handle.get_raw_frame(frame_id).unwrap()
    }

    fn get_slice(&self, frame_ids: Vec<u32>) -> TimsSlice {
        self.handle.get_tims_slice(frame_ids)
    }

    fn get_acquisition_mode(&self) -> AcquisitionMode {
        self.handle.acquisition_mode.clone()
    }

    fn get_frame_count(&self) -> i32 {
        self.handle.get_frame_count()
    }

    fn get_data_path(&self) -> &str {
        &self.handle.data_path
    }
}

pub enum TimsDataLoader {
    InMemory(TimsInMemoryLoader),
    Lazy(TimsLazyLoder)
}

impl TimsDataLoader {
    pub fn new_lazy(bruker_lib_path: &str, data_path: &str) -> Self {
        let handle = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();
        let global_meta_data = read_global_meta_sql(&handle.data_path).unwrap();
        let frame_meta_data = read_meta_data_sql(&handle.data_path).unwrap();
        TimsDataLoader::Lazy(TimsLazyLoder {
            handle,
            global_meta_data,
            frame_meta_data
        })
    }

    pub fn new_in_memory(bruker_lib_path: &str, data_path: &str) -> Self {
        let handle = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();
        let global_meta_data = read_global_meta_sql(&handle.data_path).unwrap();
        let frame_meta_data = read_meta_data_sql(&handle.data_path).unwrap();
        let compressed_data = handle.read_compressed_data_full();
        TimsDataLoader::InMemory(TimsInMemoryLoader {
            handle,
            global_meta_data,
            frame_meta_data,
            compressed_data
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

    fn get_slice(&self, frame_ids: Vec<u32>) -> TimsSlice {
        match self {
            TimsDataLoader::InMemory(loader) => loader.get_slice(frame_ids),
            TimsDataLoader::Lazy(loader) => loader.get_slice(frame_ids)
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