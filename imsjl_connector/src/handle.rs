
use libc::{c_char};
use std::ffi::CString;

use rustdf::data::handle::{TimsDataHandle};

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

    // Create your structure
    let handle = TimsDataHandle::new(bruker_lib_path, data_path).unwrap();

    // Box the structure to keep it on the heap and return a pointer
    Box::into_raw(Box::new(CTimsDataHandle { inner: handle }))
}

#[no_mangle]
pub extern "C" fn tims_data_handle_get_data_path(handle: *mut CTimsDataHandle) -> *mut c_char {
    let handle = unsafe { &*handle };
    to_c_string(handle.inner.data_path.clone())
}

//... Continue similarly for other methods ...

// Important: You'll also want to create a destructor function for your structure to avoid memory leaks.

#[no_mangle]
pub extern "C" fn tims_data_handle_destroy(handle: *mut CTimsDataHandle) {
    unsafe {
        let _ = Box::from_raw(handle);
    }
}
