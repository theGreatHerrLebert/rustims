module RustCAPI

include("Data.jl")

struct CTimsFrame
    frame_id::Int32
    ms_type::Int32
    retention_time::Float64

    scan::Ptr{Int32}
    scan_size::UInt64

    inv_mobility::Ptr{Float64}
    inv_mobility_size::UInt64

    tof::Ptr{Int32}
    tof_size::UInt64

    mz::Ptr{Float64}
    mz_size::UInt64

    intensity::Ptr{Float64}
    intensity_size::UInt64
end

const lib = "/home/administrator/Documents/promotion/rustims/imsjl_connector/target/release/libimsjl_connector.so"

TimsDataHandle_new(data_path::String, bruker_lib_path::String) = ccall((:tims_data_handle_new, lib), Ptr{Cvoid}, (Cstring, Cstring), data_path, bruker_lib_path)

TimsDataHandle_get_data_path(handle::Ptr{Cvoid})::String = unsafe_string(ccall((:tims_data_handle_get_data_path, lib), Cstring, (Ptr{Cvoid},), handle))

TimsDataHandle_get_bruker_binary_path(handle::Ptr{Cvoid})::String = unsafe_string(ccall((:tims_data_handle_get_bruker_binary_path, lib), Cstring, (Ptr{Cvoid},), handle))

TimsDataHandle_get_frame_count(handle::Ptr{Cvoid})::Int32 = ccall((:tims_data_handle_get_frame_count, lib), Int32, (Ptr{Cvoid},), handle)

TimsDataHandle_destroy(handle::Ptr{Cvoid}) = ccall((:tims_data_handle_destroy, lib), Cvoid, (Ptr{Cvoid},), handle)

TimsDataHandle_get_frame(handle::Ptr{Cvoid}, frame_id::Int32)::CTimsFrame = ccall((:tims_data_handle_get_frame, lib), CTimsFrame, (Ptr{Cvoid}, Int32), handle, frame_id)

function ctims_frame_to_julia_tims_frame(ctims_frame::CTimsFrame)::TimsFrame

    julia_scan = unsafe_wrap(Array, ctims_frame.scan, ctims_frame.scan_size, own=true)
    julia_inv_mobility = unsafe_wrap(Array, ctims_frame.inv_mobility, ctims_frame.inv_mobility_size, own=true)
    julia_tof = unsafe_wrap(Array, ctims_frame.tof, ctims_frame.tof_size, own=true)
    julia_mz = unsafe_wrap(Array, ctims_frame.mz, ctims_frame.mz_size, own=true)
    julia_intensity = unsafe_wrap(Array, ctims_frame.intensity, ctims_frame.intensity_size, own=true)

    TimsFrame(
        ctims_frame.frame_id,
        ctims_frame.ms_type,
        ctims_frame.retention_time,
        julia_scan,
        julia_inv_mobility,
        julia_tof,
        julia_mz,
        julia_intensity
    )
end

export TimsDataHandle_new, TimsDataHandle_get_data_path, TimsDataHandle_get_bruker_binary_path, TimsDataHandle_destroy, TimsDataHandle_get_frame_count, TimsDataHandle_get_frame, ctims_frame_to_julia_tims_frame

end