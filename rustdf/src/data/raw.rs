use libloading::{Library, Symbol};
use std::os::raw::c_char;

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

        println!("bruker binary successfully loaded library.");
        
        // create a handle to the raw data
        let handle = unsafe {
            let func: Symbol<unsafe extern fn(*const c_char, u32) -> u64> = lib.get(b"tims_open")?;
            let path = std::ffi::CString::new(data_path)?;
            let handle = func(path.as_ptr(), 0);
            handle
        };

        println!("bruker library created handle to TDF data.");

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
}

impl Drop for BrukerTimsDataLibrary {
    fn drop(&mut self) {
        let close = self.tims_close();
        match close {
            Ok(_) => println!("bruker library closed handle to TDF data."),
            Err(e) => println!("error: {}", e),
        };
    }
}