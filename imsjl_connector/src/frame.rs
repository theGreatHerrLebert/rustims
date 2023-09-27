use mscore::TimsFrame;

#[repr(C)]
pub struct CTimsFrame {
    frame_id: i32,
    ms_type: i32,
    retention_time: f64,

    scan: *const i32,
    scan_size: usize,

    inv_mobility: *const f64,
    inv_mobility_size: usize,

    tof: *const i32,
    tof_size: usize,

    mz: *const f64,
    mz_size: usize,

    intensity: *const f64,
    intensity_size: usize,
}

#[no_mangle]
pub extern "C" fn convert_to_ctims_frame(frame: TimsFrame) -> CTimsFrame {
    CTimsFrame {
        frame_id: frame.frame_id,
        ms_type: frame.ms_type.ms_type_numeric(),
        retention_time: frame.ims_frame.retention_time,

        scan: frame.scan.as_ptr(),
        scan_size: frame.scan.len(),

        inv_mobility: frame.ims_frame.inv_mobility.as_ptr(),
        inv_mobility_size: frame.ims_frame.inv_mobility.len(),

        tof: frame.tof.as_ptr(),
        tof_size: frame.tof.len(),

        mz: frame.ims_frame.mz.as_ptr(),
        mz_size: frame.ims_frame.mz.len(),

        intensity: frame.ims_frame.intensity.as_ptr(),
        intensity_size: frame.ims_frame.intensity.len(),
    }
}