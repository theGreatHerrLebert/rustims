module Data

import Base.show

export TimsFrame

struct TimsFrame
    frame_id::Int32
    ms_type_numeric::Int32
    retention_time::Float64
    scan::Vector{Int32}
    inv_mobility::Vector{Float64}
    tof::Vector{Int32}
    mz::Vector{Float64}
    intensity::Vector{Float64}
end

function show(io::IO, frame::TimsFrame)
    num_peaks = length(frame.mz)  # or whichever array represents peaks
    print(io, "TimsFrame(frame_id=$(frame.frame_id), ms_type=$(frame.ms_type_numeric), num_peaks=$num_peaks)")
end

end