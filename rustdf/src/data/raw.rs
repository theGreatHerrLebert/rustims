use libloading::{Library, Symbol};
use std::os::raw::{c_char, c_double};

//
// A struct that holds a handle to the raw data
//
// # Example
//
// ```
// let bruker_lib_path = "path/to/libtimsdata.so";
// let data_path = "path/to/data.d";
// let tims_data = BrukerTimsDataLibrary::new(bruker_lib_path, data_path);
// ```
pub struct BrukerTimsDataLibrary {
    pub lib: Library,
    pub handle: u64,
}

impl BrukerTimsDataLibrary {
    //
    // Create a new BrukerTimsDataLibrary struct
    //
    // # Arguments
    //
    // * `bruker_lib_path` - A string slice that holds the path to the bruker library
    // * `data_path` - A string slice that holds the path to the data
    //
    // # Example
    //
    // ```
    // let bruker_lib_path = "path/to/libtimsdata.so";
    // let data_path = "path/to/data.d";
    // let tims_data = BrukerTimsDataLibrary::new(bruker_lib_path, data_path);
    // ```
    pub fn new(bruker_lib_path: &str, data_path: &str) -> Result<BrukerTimsDataLibrary, Box<dyn std::error::Error>> {
        
        // Load the library
        let lib = unsafe {
            Library::new(bruker_lib_path)?
        };
        
        // create a handle to the raw data
        let handle = unsafe {
            let func: Symbol<unsafe extern fn(*const c_char, u32) -> u64> = lib.get(b"tims_open")?;
            let path = std::ffi::CString::new(data_path)?;
            let handle = func(path.as_ptr(), 0);
            handle
        };

        // return the BrukerTimsDataLibrary struct
        Ok(BrukerTimsDataLibrary {
            lib,
            handle,
        })
    }

    //
    // Close the handle to the raw data
    //
    // # Example
    //
    // ```
    // let close = tims_data.tims_close();
    // match close {
    //     Ok(_) => println!("tims_data closed"),
    //     Err(e) => println!("error: {}", e),
    // };
    // ```
    pub fn tims_close(&self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let func: Symbol<unsafe extern fn(u64) -> ()> = self.lib.get(b"tims_close")?;
            func(self.handle);
        }
        Ok(())
    }

    //
    // Convert the given indices to mz values.
    //
    // # Example
    //
    // ```
    // let indices = vec![...];
    // let mz_values_result = tims_data.tims_index_to_mz(estimation, &mut indices, tof_max_index);
    // match mz_values_result {
    //     Ok(mz_values) => println!("{:?}", mz_values),
    //     Err(e) => println!("error: {}", e),
    // };
    // ```
    pub fn tims_index_to_mz(&self, frame_id: u32, dbl_tofs: &[c_double], mzs: &mut [c_double]) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let func: Symbol<unsafe extern "C" fn(u64, u32, *const c_double, *mut c_double, u32)> = self.lib.get(b"tims_index_to_mz")?;
            func(self.handle, frame_id, dbl_tofs.as_ptr(), mzs.as_mut_ptr(), dbl_tofs.len() as u32);
        }
        Ok(())
    }

    //
    // Convert the given mz values to indices.
    //
    // # Example
    //
    // ```
    // let mzs = vec![...];
    // let indices_result = tims_data.tims_mz_to_index(estimation, &mut mzs);
    // match indices_result {
    //     Ok(indices) => println!("{:?}", indices),
    //     Err(e) => println!("error: {}", e),
    // };
    // ```
    pub fn tims_mz_to_index(&self, frame_id: u32, mzs: &[c_double], indices: &mut [c_double]) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let func: Symbol<unsafe extern "C" fn(u64, u32, *const c_double, *mut c_double, u32)> = self.lib.get(b"tims_mz_to_index")?;
            func(self.handle, frame_id, mzs.as_ptr(), indices.as_mut_ptr(), mzs.len() as u32);
        }
        Ok(())
    }

    //
    // Convert the given indices to inverse mobility values.
    //
    // # Example
    //
    // ```
    // let indices = vec![...];
    // let scan_values_result = tims_data.tims_scan_to_inv_mob(estimation, &mut indices);
    // match mz_values_result {
    //     Ok(mz_values) => println!("{:?}", mz_values),
    //     Err(e) => println!("error: {}", e),
    // };
    // ```
    pub fn tims_scan_to_inv_mob(&self, frame_id: u32, dbl_scans: &[c_double], inv_mob: &mut [c_double]) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let func: Symbol<unsafe extern "C" fn(u64, u32, *const c_double, *mut c_double, u32)> = self.lib.get(b"tims_scannum_to_oneoverk0")?;
            func(self.handle, frame_id, dbl_scans.as_ptr(), inv_mob.as_mut_ptr(), dbl_scans.len() as u32);
        }
        Ok(())
    }

    //
    // Convert the given inverse mobility values to scan values.
    //
    // # Example
    //
    // ```
    // let inv_mob = vec![...];
    // let scan_values_result = tims_data.tims_inv_mob_to_scan(estimation, &mut inv_mob);
    // match mz_values_result {
    //     Ok(mz_values) => println!("{:?}", mz_values),
    //     Err(e) => println!("error: {}", e),
    // };
    // ```
    pub fn inv_mob_to_tims_scan(&self, frame_id: u32, inv_mob: &[c_double], scans: &mut [c_double]) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let func: Symbol<unsafe extern "C" fn(u64, u32, *const c_double, *mut c_double, u32)> = self.lib.get(b"tims_oneoverk0_to_scannum")?;
            func(self.handle, frame_id, inv_mob.as_ptr(), scans.as_mut_ptr(), inv_mob.len() as u32);
        }
        Ok(())
    }
}

impl Drop for BrukerTimsDataLibrary {
    fn drop(&mut self) {
        let close = self.tims_close();
        match close {
            Ok(_) => (),
            Err(e) => println!("error: {}", e),
        };
    }
}