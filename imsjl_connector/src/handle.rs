use libc::{c_char};
use std::ffi::CString;

use rustdf::data::handle::{TimsDataHandle};
use crate::frame::{convert_to_ctims_frame, CTimsFrame};

#[repr(C)]
pub struct CTimsDataHandle {
    pub inner: TimsDataHandle,
}

// Convert Rust String to C-compatible string
fn to_c_string(rust_string: String) -> *mut c_char {
    CString::new(rust_string).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn tims_data_handle_new(data_path: *const c_char, bruker_lib_path: *const c_char) -> *mut CTimsDataHandle {
    // Convert C strings to Rust strings
    let data_path = unsafe { std::ffi::CStr::from_ptr(data_path) }.to_str().unwrap();
    let bruker_lib_path = unsafe { std::ffi::CStr::from_ptr(bruker_lib_path) }.to_str().unwrap();

    let handle = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();

    // Box structure to keep it on heap and return pointer
    Box::into_raw(Box::new(CTimsDataHandle { inner: handle }))
}

#[no_mangle]
pub extern "C" fn tims_data_handle_get_data_path(handle: *mut CTimsDataHandle) -> *mut c_char {
    let handle = unsafe { &*handle };
    to_c_string(handle.inner.data_path.clone())
}

#[no_mangle]
pub extern "C" fn tims_data_handle_get_bruker_binary_path(handle: *mut CTimsDataHandle) -> *mut c_char {
    let handle = unsafe { &*handle };
    to_c_string(handle.inner.bruker_lib_path.clone())
}

#[no_mangle]
pub extern "C" fn tims_data_handle_get_frame_count(handle: *mut CTimsDataHandle) -> i32 {
    assert!(!handle.is_null());
    let handle = unsafe { &mut *handle };
    handle.inner.get_frame_count()
}

#[no_mangle]
pub extern "C" fn tims_data_handle_destroy(handle: *mut CTimsDataHandle) {
    unsafe {
        let _ = Box::from_raw(handle);
    }
}

#[no_mangle]
pub extern "C" fn tims_data_handle_get_frame(handle: &TimsDataHandle, frame_id: i32) -> CTimsFrame {
    let frame = handle.get_frame(frame_id as u32).unwrap();
    convert_to_ctims_frame(frame)
}
