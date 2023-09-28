module DataHandle

using DataFrames, FilePathsBase, SQLite
using IMSJL.Data, IMSJL.RustCAPI

struct TimsDataHandle
    data_path::String
    bruker_binary_path::String
    frame_meta_data::DataFrame
    handle::Ptr{Cvoid}  # or appropriate type for the handle

    # Constructor that only requires the data_path
    function TimsDataHandle(data_path::String)

        # Dynamically determine the paths for .so files
        bruker_binary_path = determine_bruker_binary_path()

        # Acquire the actual handle from the C-exposed Rust object
        handle = RustCAPI.TimsDataHandle_new(data_path, bruker_binary_path)

        # Acquire the frame meta data
        frame_meta_data = get_frame_meta_data(data_path)

        # Construct the instance
        new(data_path, bruker_binary_path, frame_meta_data, handle)
    end
end

function determine_bruker_binary_path()::String
    return "/home/administrator/Documents/promotion/ENV-11/lib/python3.11/site-packages/opentims_bruker_bridge/libtimsdata.so"
end

function determine_imsjl_connector_path()::String
    return "/home/administrator/Documents/promotion/rustims/imsjl_connector/target/release/libimsjl_connector.so"
end

function get_tims_frame(handle::TimsDataHandle, frame_id::Number)::TimsFrame
    ctims_frame = RustCAPI.TimsDataHandle_get_frame(handle.handle, Int32(frame_id))
    return RustCAPI.ctims_frame_to_julia_tims_frame(ctims_frame)
end

function get_frame_meta_data(ds_path::String)::DataFrame
    db = SQLite.DB(string(Path(ds_path), Path("/analysis.tdf")))
    return DataFrame(DBInterface.execute(db, "SELECT * FROM Frames"))
end

export TimsDataHandle, get_tims_frame

end