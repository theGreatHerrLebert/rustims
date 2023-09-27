"""
# module JuliaDataHandle

- Julia version: 
- Author: davidteschner
- Date: 2023-09-27

# Examples

```jldoctest
julia>
```
"""
module JuliaDataHandle

include("RustCAPI.jl")
include("Data.jl")

struct TimsDataHandle
    data_path::String
    bruker_binary_path::String
    handle::Ptr{Void}  # or appropriate type for the handle

    # Constructor that only requires the data_path
    function TimsDataHandle(data_path::String)

        # Dynamically determine the paths for .so files
        bruker_binary_path = determine_bruker_binary_path()

        # Acquire the actual handle from the C-exposed Rust object
        handle = RustcApi.TimsDataHandle_new(data_path, bruker_binary_path)

        # Construct the instance
        new(data_path, bruker_binary_path, handle)
    end
end

function determine_bruker_binary_path()::String
    return "/home/administrator/Documents/promotion/ENV-11/lib/python3.11/site-packages/opentims_bruker_bridge/libtimsdata.so"
end

function determine_imsjl_connector_path()::String
    return "/home/administrator/Documents/promotion/rustims/imsjl_connector/target/release/libimsjl_connector.so"
end

end