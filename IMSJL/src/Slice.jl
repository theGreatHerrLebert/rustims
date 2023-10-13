module Slice

import Base

import IMSJL.Data.TimsFrame

struct TimsSlice
    first_frame_id::Int32
    last_frame_id::Int32
    data::Vector{TimsFrame}
end



function TimsSlice()
    return TimsSlice(0, 0, Vector())
end

# todo add intensity max
function filter_ranged(slice::TimsSlice, mz_min, mz_max; scan_min=0, scan_max=1000, intensity_min=0.0, intensity_max=200000)::TimsSlice
    new_data = Vector{TimsFrame}()
    for frame in slice.data
        # the mathematical operator is xor
        if sum((frame.mz .>= mz_min) .⊻ (frame.mz .<= mz_max)) == 0 && sum((frame.scan .>= scan_min) .⊻ (frame.scan .<= scan_max)) == 0 && sum(frame.intensity .< intensity_min) == 0
            push!(new_data, frame)
        end
    end
    if isempty(new_data)
        return TimsSlice()
    end
    first_index = new_data[1].frame_id
    last_index = new_data[end].frame_id 
    return TimsSlice(first_index, last_index, new_data)
end

function get_frames(slice::TimsSlice)::Vector
    return slice.data
end


function Base.show(io::IO, slice::TimsSlice)
    print("TimsSlice($(slice.first_frame_id), $(slice.last_frame_id))")
end

export TimsSlice

end