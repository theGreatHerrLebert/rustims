module Spectrum

import Base

#struct MzSpectrum
#mz::Vector{Float64}
#    intensity::Vector{Float64}

#    @assert length(mz) == length(intensity) 
#end
@enum MsType PRECURSOR=0 FRAGMENT_DDA=8 FRAGMENT_DIA=9 UNKNOWN=-1

struct TimsSpectrum
    frame_id::Int
    scan::Int
    retention_time::Float64
    mobility::Float64
    ms_type::MsType
    tof::Vector{Int}
    mz::Vector{Float64}
    intensity::Vector{Float64}
    # @assert length(index) == length(mz) == length(intensity)
end

# split TimsFrame by scan values

function TimsSpectrum()
    return TimsSpectrum(0, 0, 0.0, 0.0, MsType(-1), Vector(), Vector(), Vector())
end

function Base.show(io::IO, spectrum::TimsSpectrum)
    print("TimsSpectrum(id=$(spectrum.frame_id), retention_time=$(round(spectrum.retention_time; digits=2)), 
    scan=$(spectrum.scan), mobility=$(round(spectrum.mobility; digits=2)), ms_type=$(spectrum.ms_type), 
    num_peaks=$(length(spectrum.tof)))")
end

export TimsSpectrum

end