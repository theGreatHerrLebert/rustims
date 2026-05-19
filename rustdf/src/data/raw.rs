use libloading::{Library, Symbol};
use std::os::raw::{c_char, c_double, c_float};
use std::sync::Mutex;

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
    pub fn new(
        bruker_lib_path: &str,
        data_path: &str,
    ) -> Result<BrukerTimsDataLibrary, Box<dyn std::error::Error>> {
        // Load the library
        let lib = unsafe { Library::new(bruker_lib_path)? };

        // create a handle to the raw data
        let handle = unsafe {
            let func: Symbol<unsafe extern "C" fn(*const c_char, u32) -> u64> =
                lib.get(b"tims_open")?;
            let path = std::ffi::CString::new(data_path)?;
            let handle = func(path.as_ptr(), 0);
            handle
        };

        // return the BrukerTimsDataLibrary struct
        Ok(BrukerTimsDataLibrary { lib, handle })
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
            let func: Symbol<unsafe extern "C" fn(u64) -> ()> = self.lib.get(b"tims_close")?;
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
    pub fn tims_index_to_mz(
        &self,
        frame_id: u32,
        dbl_tofs: &[c_double],
        mzs: &mut [c_double],
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let func: Symbol<unsafe extern "C" fn(u64, u32, *const c_double, *mut c_double, u32)> =
                self.lib.get(b"tims_index_to_mz")?;
            func(
                self.handle,
                frame_id,
                dbl_tofs.as_ptr(),
                mzs.as_mut_ptr(),
                dbl_tofs.len() as u32,
            );
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
    pub fn tims_mz_to_index(
        &self,
        frame_id: u32,
        mzs: &[c_double],
        indices: &mut [c_double],
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let func: Symbol<unsafe extern "C" fn(u64, u32, *const c_double, *mut c_double, u32)> =
                self.lib.get(b"tims_mz_to_index")?;
            func(
                self.handle,
                frame_id,
                mzs.as_ptr(),
                indices.as_mut_ptr(),
                mzs.len() as u32,
            );
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
    pub fn tims_scan_to_inv_mob(
        &self,
        frame_id: u32,
        dbl_scans: &[c_double],
        inv_mob: &mut [c_double],
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let func: Symbol<unsafe extern "C" fn(u64, u32, *const c_double, *mut c_double, u32)> =
                self.lib.get(b"tims_scannum_to_oneoverk0")?;
            func(
                self.handle,
                frame_id,
                dbl_scans.as_ptr(),
                inv_mob.as_mut_ptr(),
                dbl_scans.len() as u32,
            );
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
    pub fn inv_mob_to_tims_scan(
        &self,
        frame_id: u32,
        inv_mob: &[c_double],
        scans: &mut [c_double],
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let func: Symbol<unsafe extern "C" fn(u64, u32, *const c_double, *mut c_double, u32)> =
                self.lib.get(b"tims_oneoverk0_to_scannum")?;
            func(
                self.handle,
                frame_id,
                inv_mob.as_ptr(),
                scans.as_mut_ptr(),
                inv_mob.len() as u32,
            );
        }
        Ok(())
    }

    // ----------------------------------------------------------------- //
    // Centroided spectrum extraction (Bruker's built-in peak picker).
    //
    // SDK function:
    //   uint32_t tims_extract_centroided_spectrum_for_frame_v2(
    //       uint64_t handle,
    //       int64_t  frame_id,
    //       uint32_t scan_begin,
    //       uint32_t scan_end,
    //       void (*callback)(int64_t precursor_id, uint32_t num_peaks,
    //                        double* mzs, float* intensities),
    //       void* user_data);
    //
    // Bruker invokes the callback ONCE per scan-range with the centroided
    // (m/z, intensity) arrays. We use a thread-local Mutex<Vec<...>> trick
    // to hand the data back to Rust without dealing with `void*` user_data
    // closures (libloading + C function pointers don't compose with Rust
    // closures cleanly).
    //
    // Signatures recovered from pyTDFSDK (gtluu/pyTDFSDK init_tdf_sdk.py).
    // Same callback shape as Bruker's published `MSMS_SPECTRUM_FUNCTOR`.
    // ----------------------------------------------------------------- //

    /// Extract a centroided spectrum for a (frame, scan-range) tile via
    /// Bruker's built-in peak picker. Returns `(mz, intensity)` pairs.
    pub fn tims_extract_centroided_spectrum_for_frame(
        &self,
        frame_id: i64,
        scan_begin: u32,
        scan_end: u32,
    ) -> Result<(Vec<f64>, Vec<f32>), Box<dyn std::error::Error>> {
        // Stash the result in a thread-local-ish global; only one extract
        // call may run at a time per process. We serialise on a Mutex.
        let mut result_mz: Vec<f64> = Vec::new();
        let mut result_int: Vec<f32> = Vec::new();
        // We trampoline through a global: the C callback writes into
        // EXTRACT_BUF.
        let _guard = EXTRACT_BUF
            .lock()
            .map_err(|_| "EXTRACT_BUF poisoned")?;
        unsafe { EXTRACT_BUF_DATA = Some((Vec::new(), Vec::new())); }
        unsafe {
            let func: Symbol<
                unsafe extern "C" fn(
                    u64, i64, u32, u32,
                    extern "C" fn(i64, u32, *const f64, *const f32),
                    *mut std::ffi::c_void,
                ) -> u32,
            > = self.lib.get(b"tims_extract_centroided_spectrum_for_frame_v2")?;
            let rc = func(
                self.handle,
                frame_id,
                scan_begin,
                scan_end,
                centroid_trampoline,
                std::ptr::null_mut(),
            );
            if rc == 0 {
                return Err("tims_extract_centroided_spectrum_for_frame_v2 returned 0".into());
            }
        }
        if let Some((mz, intens)) = unsafe { EXTRACT_BUF_DATA.take() } {
            result_mz = mz;
            result_int = intens;
        }
        Ok((result_mz, result_int))
    }

    /// PASEF MS/MS centroided peaks for one MS2 frame.
    /// SDK: tims_read_pasef_msms_for_frame_v2(handle, frame_id, callback, void**)
    /// Bruker invokes the callback ONCE PER PRECURSOR found in the frame
    /// (DDA-PASEF). We accumulate all (precursor_id, mz, intensity) hits.
    pub fn tims_read_pasef_msms_for_frame(
        &self,
        frame_id: i64,
    ) -> Result<Vec<(i64, Vec<f64>, Vec<f32>)>, Box<dyn std::error::Error>> {
        let _guard = PASEF_BUF.lock().map_err(|_| "PASEF_BUF poisoned")?;
        unsafe { PASEF_BUF_DATA = Some(Vec::new()); }
        unsafe {
            let func: Symbol<
                unsafe extern "C" fn(
                    u64, i64,
                    extern "C" fn(i64, u32, *const f64, *const f32, *mut *mut std::ffi::c_void),
                    *mut *mut std::ffi::c_void,
                ) -> u32,
            > = self.lib.get(b"tims_read_pasef_msms_for_frame_v2")?;
            let rc = func(
                self.handle,
                frame_id,
                pasef_trampoline,
                std::ptr::null_mut(),
            );
            if rc == 0 {
                return Err("tims_read_pasef_msms_for_frame_v2 returned 0".into());
            }
        }
        let out = unsafe { PASEF_BUF_DATA.take() }.unwrap_or_default();
        Ok(out)
    }
}

// Globals for the C-callback trampolines. Locked on each extract call so
// only one thread runs the SDK at a time per process — same restriction
// pyTDFSDK + alphatims live with.
static EXTRACT_BUF: Mutex<()> = Mutex::new(());
static mut EXTRACT_BUF_DATA: Option<(Vec<f64>, Vec<f32>)> = None;

extern "C" fn centroid_trampoline(
    _precursor_id: i64,
    n_peaks: u32,
    mzs: *const f64,
    intensities: *const f32,
) {
    if n_peaks == 0 || mzs.is_null() || intensities.is_null() { return; }
    let mz_slice = unsafe { std::slice::from_raw_parts(mzs, n_peaks as usize) };
    let in_slice = unsafe { std::slice::from_raw_parts(intensities, n_peaks as usize) };
    unsafe {
        if let Some((ref mut mz_acc, ref mut in_acc)) = EXTRACT_BUF_DATA {
            mz_acc.extend_from_slice(mz_slice);
            in_acc.extend_from_slice(in_slice);
        }
    }
}

static PASEF_BUF: Mutex<()> = Mutex::new(());
static mut PASEF_BUF_DATA: Option<Vec<(i64, Vec<f64>, Vec<f32>)>> = None;

extern "C" fn pasef_trampoline(
    precursor_id: i64,
    n_peaks: u32,
    mzs: *const f64,
    intensities: *const f32,
    _user_data: *mut *mut std::ffi::c_void,
) {
    if n_peaks == 0 || mzs.is_null() || intensities.is_null() { return; }
    let mz_slice = unsafe { std::slice::from_raw_parts(mzs, n_peaks as usize) };
    let in_slice = unsafe { std::slice::from_raw_parts(intensities, n_peaks as usize) };
    unsafe {
        if let Some(ref mut acc) = PASEF_BUF_DATA {
            acc.push((precursor_id, mz_slice.to_vec(), in_slice.to_vec()));
        }
    }
}

// Silence the unused `c_float` warning on platforms where it's only
// touched by callbacks above.
#[allow(dead_code)]
const _: fn() = || { let _: c_float = 0.0; };

impl Drop for BrukerTimsDataLibrary {
    fn drop(&mut self) {
        let close = self.tims_close();
        match close {
            Ok(_) => (),
            Err(e) => println!("error: {}", e),
        };
    }
}
