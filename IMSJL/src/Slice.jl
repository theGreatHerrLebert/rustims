module Slice

import Base

struct TimsSlice
    first_frame_id::Int32
    last_frame_id::Int32
end


function Base.show(io::IO, slice::TimsSlice)
    print("TimsSlice($(slice.first_frame_id), $(slice.last_frame_id))")
end


# Todo: function filter_ranged(mz_min, mz_max, scan_min, scan_max, intensity_min, num_threads)

export TimsSlice

end