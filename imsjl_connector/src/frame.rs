use std::slice;
use mscore::data::tims_frame::TimsFrame;

#[repr(C)]
pub struct CTimsFrame {
    frame_id: i32,
    ms_type: i32,
    retention_time: f64,

    scan: *const i32,
    scan_size: usize,

    mobility: *const f64,
    mobility_size: usize,

    tof: *const i32,
    tof_size: usize,

    mz: *const f64,
    mz_size: usize,

    intensity: *const f64,
    intensity_size: usize,
}

#[no_mangle]
pub extern "C" fn convert_to_ctims_frame(frame: TimsFrame) -> CTimsFrame {
    // Clone the vectors and convert them to boxed slices
    let scan_boxed = frame.scan.clone().into_boxed_slice();
    let mobility_boxed = frame.ims_frame.mobility.clone().into_boxed_slice();
    let tof_boxed = frame.tof.clone().into_boxed_slice();
    let mz_boxed = frame.ims_frame.mz.clone().into_boxed_slice();
    let intensity_boxed = frame.ims_frame.intensity.clone().into_boxed_slice();

    CTimsFrame {
        frame_id: frame.frame_id,
        ms_type: frame.ms_type.ms_type_numeric(),
        retention_time: frame.ims_frame.retention_time,

        scan: Box::into_raw(scan_boxed) as *const i32,
        scan_size: frame.scan.len(),

        mobility: Box::into_raw(mobility_boxed) as *const f64,
        mobility_size: frame.ims_frame.mobility.len(),

        tof: Box::into_raw(tof_boxed) as *const i32,
        tof_size: frame.tof.len(),

        mz: Box::into_raw(mz_boxed) as *const f64,
        mz_size: frame.ims_frame.mz.len(),

        intensity: Box::into_raw(intensity_boxed) as *const f64,
        intensity_size: frame.ims_frame.intensity.len(),
    }
}
#[no_mangle]
pub extern "C" fn free_ctims_frame_data(frame: CTimsFrame) {
    unsafe {
        let _ = Box::from_raw(slice::from_raw_parts_mut(frame.scan as *mut i32, frame.scan_size));
        let _ = Box::from_raw(slice::from_raw_parts_mut(frame.mobility as *mut f64, frame.mobility_size));
        let _ = Box::from_raw(slice::from_raw_parts_mut(frame.tof as *mut i32, frame.tof_size));
        let _ = Box::from_raw(slice::from_raw_parts_mut(frame.mz as *mut f64, frame.mz_size));
        let _ = Box::from_raw(slice::from_raw_parts_mut(frame.intensity as *mut f64, frame.intensity_size));
    }
}
