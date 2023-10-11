module DataHandle

using DataFrames, FilePathsBase, SQLite
using IMSJL.Data, IMSJL.RustCAPI, IMSJL.Slice

import Base

struct TimsDataHandle
    data_path::String
    bruker_binary_path::String
    frame_meta_data::DataFrame
    num_frames::Int
    handle::Ptr{Cvoid}  # or appropriate type for the handle

    # Constructor that only requires the data_path
    function TimsDataHandle(data_path::String)

        # Dynamically determine the paths for .so files
        bruker_binary_path = determine_bruker_binary_path()

        # Acquire the actual handle from the C-exposed Rust object
        handle = RustCAPI.TimsDataHandle_new(data_path, bruker_binary_path)

        # Acquire the frame meta data
        frame_meta_data = get_frame_meta_data(data_path)

        # Acquire the number of frames
        num_frames = size(frame_meta_data, 1)

        # Construct the instance
        new(data_path, bruker_binary_path, frame_meta_data, num_frames, handle)
    end
end

function determine_bruker_binary_path()::String
    # TODO: find better place to put libtimsdata.so
    return joinpath(pkgdir(@__MODULE__), "lib", "libtimsdata.so")
end

function determine_imsjl_connector_path()::String
    return joinpath(pkgdir(@__MODULE__), "..", "imsjl_connector", "target", "release", "libimsjl_connector.so")
end

function get_tims_frame(handle::TimsDataHandle, frame_id::Number)::TimsFrame
    ctims_frame = RustCAPI.TimsDataHandle_get_frame(handle.handle, Int32(frame_id))
    return RustCAPI.ctims_frame_to_julia_tims_frame(ctims_frame)
end

function get_frame_meta_data(ds_path::String)::DataFrame
    db = SQLite.DB(string(Path(ds_path), Path("/analysis.tdf")))
    return DataFrame(DBInterface.execute(db, "SELECT * FROM Frames"))
end

# Initial iteration state
Base.iterate(td::TimsDataHandle) = iterate(td, 1)

# Producing the next value and iteration state
function Base.iterate(td::TimsDataHandle, state)
    # Check if we're past the last row
    if state > size(td.frame_meta_data, 1)
        return nothing  # Signal the end of iteration
    end

    # Return the current row and the next state (which is the next row index)
    return get_tims_frame(td, state), state + 1
end

# Define length method for convenience
Base.length(td::TimsDataHandle) = size(td.frame_meta_data, 1)

function Base.getindex(td::TimsDataHandle, frame_id::Int)::TimsFrame
    return get_tims_frame(td, frame_id)
end

function Base.getindex(td::TimsDataHandle, ur::UnitRange{Int})::TimsSlice
    return TimsSlice(ur.start, ur.stop)
end

export TimsDataHandle, get_tims_frame

end