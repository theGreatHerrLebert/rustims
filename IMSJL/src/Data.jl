module Data

import Base

import IMSJL.Spectrum.TimsSpectrum
import IMSJL.Spectrum.MsType
# @enum MsType PRECURSOR=0 FRAGMENT_DDA=8 FRAGMENT_DIA=9 UNKNOWN=-1

struct TimsFrame
    frame_id::Int32
    ms_type::MsType
    retention_time::Float64
    scan::Vector{Int32}
    mobility::Vector{Float64}
    tof::Vector{Int32}
    mz::Vector{Float64}
    intensity::Vector{Float64}
end

function TimsFrame()
    return TimsFrame(0, MsType(-1), 0.0, Vector(), Vector(), Vector(), Vector(), Vector())
end


function get_spectra(tf::TimsFrame)::Vector{TimsSpectrum}
    # guard agains empty frame!
    prev_scan = tf.scan[1]

    # generate empty vector of TimsSpectrum of length unique scans
    spectra = Vector{TimsSpectrum}(undef, length(unique(tf.scan)))
    
    # generate empty vectors for Spectrum tof, mz, intensity
    v_tof = Vector{Int}()
    v_mz = Vector{Float64}()
    v_intensity = Vector{Float64}()
    
    # init array index
    spectra_idx = 1
    for (idx, sc) in enumerate(tf.scan)
        println(sc)
        if prev_scan == sc
            append!(v_tof, tf.tof)
            append!(v_mz, tf.mz)
            append!(v_intensity, tf.intensity)
        elseif prev_scan != sc
            spectra[spectra_idx] = TimsSpectrum(tf.frame_id, prev_scan, tf.retention_time, tf.mobility[idx], tf.ms_type, v_tof, v_mz, v_intensity)
            empty!(v_tof)
            empty!(v_mz)
            empty!(v_intensity)
            append!(v_tof, tf.tof)
            append!(v_mz, tf.mz)
            append!(v_intensity, tf.intensity)
            prev_scan = sc
            spectra_idx += 1
        end
    end
    # append last spectrum to spectra
    spectra[end] = TimsSpectrum(tf.frame_id, prev_scan, tf.retention_time, tf.mobility[end], tf.ms_type, v_tof, v_mz, v_intensity)
    return spectra
end


function Base.show(io::IO, frame::TimsFrame)
    num_peaks = length(frame.mz)
    print(io, "TimsFrame(frame_id=$(frame.frame_id), ms_type=$(frame.ms_type), num_peaks=$num_peaks)")
end

export TimsFrame, MsType

end