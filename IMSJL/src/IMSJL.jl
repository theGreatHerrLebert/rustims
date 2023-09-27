module IMSJL

export TimsDataHandle_new, TimsDataHandle_get_data_path, TimsDataHandle_destroy

const lib = "path/to/libimsjl_connector.so"

function TimsDataHandle_new(data_path::String, bruker_lib_path::String)
    ccall((:tims_data_handle_new, lib), Ptr{Cvoid}, (Cstring, Cstring), data_path, bruker_lib_path)
end

function TimsDataHandle_get_data_path(handle::Ptr{Cvoid})::String
    return unsafe_string(ccall((:tims_data_handle_get_data_path, lib), Cstring, (Ptr{Cvoid},), handle))
end

#... Continue similarly for other methods ...

# Don't forget to clean up to avoid memory leaks
function TimsDataHandle_destroy(handle::Ptr{Cvoid})
    ccall((:tims_data_handle_destroy, lib), Cvoid, (Ptr{Cvoid},), handle)
end

end # module IMSJL
