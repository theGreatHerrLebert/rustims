use byteorder::{ByteOrder, LittleEndian};
use mscore::timstof::frame::TimsFrame;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::io;
use std::io::{Read, Write};

/// Decompresses a ZSTD compressed byte array
///
/// # Arguments
///
/// * `compressed_data` - A byte slice that holds the compressed data
///
/// # Returns
///
/// * `decompressed_data` - A vector of u8 that holds the decompressed data
///
pub fn zstd_decompress(compressed_data: &[u8]) -> io::Result<Vec<u8>> {
    let mut decoder = zstd::Decoder::new(compressed_data)?;
    let mut decompressed_data = Vec::new();
    decoder.read_to_end(&mut decompressed_data)?;
    Ok(decompressed_data)
}

/// Compresses a byte array using ZSTD
///
/// # Arguments
///
/// * `decompressed_data` - A byte slice that holds the decompressed data
///
/// # Returns
///
/// * `compressed_data` - A vector of u8 that holds the compressed data
///
pub fn zstd_compress(decompressed_data: &[u8], compression_level: i32) -> io::Result<Vec<u8>> {
    let mut encoder = zstd::Encoder::new(Vec::new(), compression_level)?;
    encoder.write_all(decompressed_data)?;
    let compressed_data = encoder.finish()?;
    Ok(compressed_data)
}

/// Deduplicate `(scan, tof)` pairs (summing their intensities) and return the
/// arrays sorted ascending by `(scan, tof)`.
///
/// The Bruker `tdf_bin` layout requires this ordering: [`modify_tofs`] delta-
/// encodes TOF within each scan (so TOFs must ascend within a scan) and
/// [`get_peak_cnts`] walks scans assuming they ascend. The Python writer
/// enforces the same invariant via an `np.unique` dedup + `np.lexsort((tof,
/// scan))` before encoding; the Rust write path previously fed raw, unsorted
/// frame data straight into the encoder, producing negative/garbage TOF deltas
/// that vendor readers (e.g. DiaNN) reject. Mirroring the Python preprocessing
/// here keeps the two writers byte-for-byte identical.
pub fn sort_dedup_scan_tof(
    scans: &[u32],
    tofs: &[u32],
    intensities: &[u32],
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    use std::collections::HashMap;
    let mut acc: HashMap<(u32, u32), u64> = HashMap::with_capacity(scans.len());
    for i in 0..scans.len() {
        *acc.entry((scans[i], tofs[i])).or_insert(0) += intensities[i] as u64;
    }
    let mut pairs: Vec<((u32, u32), u64)> = acc.into_iter().collect();
    // Sort by scan, then tof — matches numpy's lexsort((tof, scan)).
    pairs.sort_unstable_by_key(|&((s, t), _)| (s, t));

    let n = pairs.len();
    let mut out_scan = Vec::with_capacity(n);
    let mut out_tof = Vec::with_capacity(n);
    let mut out_int = Vec::with_capacity(n);
    for ((s, t), inten) in pairs {
        out_scan.push(s);
        out_tof.push(t);
        out_int.push(inten.min(u32::MAX as u64) as u32);
    }
    (out_scan, out_tof, out_int)
}

pub fn reconstruct_compressed_data(
    scans: Vec<u32>,
    tofs: Vec<u32>,
    intensities: Vec<u32>,
    total_scans: u32,
    compression_level: i32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Ensuring all vectors have the same length
    assert_eq!(scans.len(), tofs.len());
    assert_eq!(scans.len(), intensities.len());

    // Dedup + sort by (scan, tof) so TOF delta-encoding stays monotonic.
    let (scans, mut tofs, intensities) = sort_dedup_scan_tof(&scans, &tofs, &intensities);

    // Modify TOFs based on scans
    modify_tofs(&mut tofs, &scans);

    // Get peak counts from total scans and scans
    let peak_cnts = get_peak_cnts(total_scans, &scans);

    // Interleave TOFs and intensities
    let mut interleaved = Vec::new();
    for (&tof, &intensity) in tofs.iter().zip(intensities.iter()) {
        interleaved.push(tof);
        interleaved.push(intensity);
    }

    // Get real data using the custom loop logic
    let real_data = get_realdata(&peak_cnts, &interleaved);

    // Compress real_data using zstd_compress
    let compressed_data = zstd_compress(&real_data, compression_level)?;

    // Final data preparation with compressed data
    let mut final_data = Vec::new();

    // Include the length of the compressed data as a header (4 bytes)
    final_data.extend_from_slice(&(compressed_data.len() as u32 + 8).to_le_bytes());

    // Include total_scans as part of the header
    final_data.extend_from_slice(&total_scans.to_le_bytes());

    // Include the compressed data itself
    final_data.extend_from_slice(&compressed_data);

    Ok(final_data)
}

pub fn compress_collection(
    frames: Vec<TimsFrame>,
    max_scan_count: u32,
    compression_level: i32,
    num_threads: usize,
) -> Vec<Vec<u8>> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let result = pool.install(|| {
        frames
            .par_iter()
            .map(|frame| {
                let compressed_data = reconstruct_compressed_data(
                    frame.scan.iter().map(|&x| x as u32).collect(),
                    frame.tof.iter().map(|&x| x as u32).collect(),
                    frame
                        .ims_frame
                        .intensity
                        .iter()
                        .map(|&x| x as u32)
                        .collect(),
                    max_scan_count,
                    compression_level,
                )
                .unwrap();
                compressed_data
            })
            .collect()
    });
    result
}

/// Parses the decompressed bruker binary data
///
/// # Arguments
///
/// * `decompressed_bytes` - A byte slice that holds the decompressed data
///
/// # Returns
///
/// * `scan_indices` - A vector of u32 that holds the scan indices
/// * `tof_indices` - A vector of u32 that holds the tof indices
/// * `intensities` - A vector of u32 that holds the intensities
///
pub fn parse_decompressed_bruker_binary_data(
    decompressed_bytes: &[u8],
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), Box<dyn std::error::Error>> {
    let mut buffer_u32 = Vec::new();

    for i in 0..(decompressed_bytes.len() / 4) {
        let value = LittleEndian::read_u32(&[
            decompressed_bytes[i],
            decompressed_bytes[i + (decompressed_bytes.len() / 4)],
            decompressed_bytes[i + (2 * decompressed_bytes.len() / 4)],
            decompressed_bytes[i + (3 * decompressed_bytes.len() / 4)],
        ]);
        buffer_u32.push(value);
    }

    // get the number of scans
    let scan_count = buffer_u32[0] as usize;

    // get the scan indices
    let mut scan_indices: Vec<u32> = buffer_u32[..scan_count].to_vec();
    for index in &mut scan_indices {
        *index /= 2;
    }

    // first scan index is always 0?
    scan_indices[0] = 0;

    // get the tof indices, which are the first half of the buffer after the scan indices
    let mut tof_indices: Vec<u32> = buffer_u32
        .iter()
        .skip(scan_count)
        .step_by(2)
        .cloned()
        .collect();

    // get the intensities, which are the second half of the buffer
    let intensities: Vec<u32> = buffer_u32
        .iter()
        .skip(scan_count + 1)
        .step_by(2)
        .cloned()
        .collect();

    // calculate the last scan before moving scan indices
    let last_scan = intensities.len() as u32 - scan_indices[1..].iter().sum::<u32>();

    // shift the scan indices to the right
    for i in 0..(scan_indices.len() - 1) {
        scan_indices[i] = scan_indices[i + 1];
    }

    // set the last scan index
    let len = scan_indices.len();
    scan_indices[len - 1] = last_scan;

    // convert the tof indices to cumulative sums
    let mut index = 0;
    for &size in &scan_indices {
        let mut current_sum = 0;
        for _ in 0..size {
            current_sum += tof_indices[index];
            tof_indices[index] = current_sum;
            index += 1;
        }
    }

    // adjust the tof indices to be zero-indexed
    let adjusted_tof_indices: Vec<u32> = tof_indices.iter().map(|&val| val - 1).collect();
    Ok((scan_indices, adjusted_tof_indices, intensities))
}

pub fn get_peak_cnts(total_scans: u32, scans: &[u32]) -> Vec<u32> {
    let mut peak_cnts = vec![total_scans];
    let mut ii = 0;
    for scan_id in 1..total_scans {
        let mut counter = 0;
        while ii < scans.len() && scans[ii] < scan_id {
            ii += 1;
            counter += 1;
        }
        peak_cnts.push(counter * 2);
    }
    peak_cnts
}

pub fn modify_tofs(tofs: &mut [u32], scans: &[u32]) {
    let mut last_tof = -1i32; // Using i32 to allow -1
    let mut last_scan = 0;
    for ii in 0..tofs.len() {
        if last_scan != scans[ii] {
            last_tof = -1;
            last_scan = scans[ii];
        }
        let val = tofs[ii] as i32; // Cast to i32 for calculation
        tofs[ii] = (val - last_tof) as u32; // Cast back to u32
        last_tof = val;
    }
}

pub fn get_realdata(peak_cnts: &[u32], interleaved: &[u32]) -> Vec<u8> {
    let mut back_data = Vec::new();

    // Convert peak counts to bytes and add to back_data
    for &cnt in peak_cnts {
        back_data.extend_from_slice(&cnt.to_le_bytes());
    }

    // Convert interleaved data to bytes and add to back_data
    for &value in interleaved {
        back_data.extend_from_slice(&value.to_le_bytes());
    }

    // Call get_realdata_loop for data rearrangement
    get_realdata_loop(&back_data)
}

pub fn get_realdata_loop(back_data: &[u8]) -> Vec<u8> {
    let mut real_data = vec![0u8; back_data.len()];
    let mut reminder = 0;
    let mut bd_idx = 0;
    for rd_idx in 0..back_data.len() {
        if bd_idx >= back_data.len() {
            reminder += 1;
            bd_idx = reminder;
        }
        real_data[rd_idx] = back_data[bd_idx];
        bd_idx += 4;
    }
    real_data
}

pub fn get_data_for_compression(
    tofs: &Vec<u32>,
    scans: &Vec<u32>,
    intensities: &Vec<u32>,
    max_scans: u32,
) -> Vec<u8> {
    // Dedup + sort by (scan, tof) so TOF delta-encoding stays monotonic.
    let (scans, tofs, intensities) = sort_dedup_scan_tof(scans, tofs, intensities);

    let mut tof_copy = tofs.clone();
    modify_tofs(&mut tof_copy, &scans);
    let peak_cnts = get_peak_cnts(max_scans, &scans);
    // Interleave the delta-encoded TOFs (`tof_copy`), not the raw `tofs`.
    let interleaved: Vec<u32> = tof_copy
        .iter()
        .zip(intensities.iter())
        .flat_map(|(tof, intensity)| vec![*tof, *intensity])
        .collect();

    get_realdata(&peak_cnts, &interleaved)
}

pub fn get_data_for_compression_par(
    tofs: Vec<Vec<u32>>,
    scans: Vec<Vec<u32>>,
    intensities: Vec<Vec<u32>>,
    max_scans: u32,
    num_threads: usize,
) -> Vec<Vec<u8>> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let result = pool.install(|| {
        tofs.par_iter()
            .zip(scans.par_iter())
            .zip(intensities.par_iter())
            .map(|((tof, scan), intensity)| {
                get_data_for_compression(tof, scan, intensity, max_scans)
            })
            .collect()
    });

    result
}

pub fn flatten_scan_values(scan: &Vec<u32>, zero_indexed: bool) -> Vec<u32> {
    let add = if zero_indexed { 0 } else { 1 };
    scan.iter()
        .enumerate()
        .flat_map(|(index, &count)| vec![(index + add) as u32; count as usize].into_iter())
        .collect()
}

// Merge and sort inclusive integer ranges like [(3,7), (8,12), (20,25)].
pub fn merge_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if ranges.is_empty() { return ranges; }
    ranges.sort_unstable_by_key(|x| x.0);
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    let mut cur = ranges[0];
    for (l, r) in ranges.into_iter().skip(1) {
        if l <= cur.1 + 1 {
            cur.1 = cur.1.max(r);
        } else {
            out.push(cur);
            cur = (l, r);
        }
    }
    out.push(cur);
    out
}

/// One frame's worth of writer output: the statistics the TDF `Frames` row needs, plus the
/// zstd-compressed payload ready to append to `analysis.tdf_bin`.
pub struct CompressedFrame {
    pub num_peaks: u32,
    pub max_intensity: u32,
    pub summed_intensity: u64,
    pub data: Vec<u8>,
}

/// The whole per-frame TDF writer pipeline — m/z → TOF and 1/K0 → scan conversion, `(scan, tof)`
/// dedup, interleave into the Bruker layout, zstd — for a batch of frames, over a rayon pool.
///
/// This used to run one frame at a time in Python because the conversion went through the Bruker
/// SDK, which is not safe to call concurrently on one handle. Given a `Sync` converter (the SDK-free
/// `BrukerFormulaConverter` reproduces the SDK's integer output exactly in both of these directions)
/// the whole pipeline is per-frame independent, so only the final append has to stay ordered.
///
/// `mz`, `mobility` and `intensity` are parallel slices, one entry per frame, matching `frame_ids`.
/// Results come back in input order.
pub fn build_compressed_frames(
    converter: &(dyn crate::data::handle::IndexConverter + Sync),
    frame_ids: &[u32],
    mz: &[&[f64]],
    mobility: &[&[f64]],
    intensity: &[&[f64]],
    max_scans: u32,
    compression_level: i32,
    num_threads: usize,
) -> Result<Vec<CompressedFrame>, String> {
    if frame_ids.len() != mz.len() || frame_ids.len() != mobility.len() || frame_ids.len() != intensity.len() {
        return Err(format!(
            "frame_ids ({}), mz ({}), mobility ({}) and intensity ({}) must have the same length",
            frame_ids.len(), mz.len(), mobility.len(), intensity.len()
        ));
    }
    // Per frame the three arrays index each other in `sort_dedup_scan_tof`, so a ragged triplet
    // would be an out-of-bounds panic in Rust rather than an error the caller can handle.
    for i in 0..frame_ids.len() {
        if mz[i].len() != mobility[i].len() || mz[i].len() != intensity[i].len() {
            return Err(format!(
                "frame {} (id {}): mz ({}), mobility ({}) and intensity ({}) must have the same length",
                i, frame_ids[i], mz[i].len(), mobility[i].len(), intensity[i].len()
            ));
        }
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.max(1))
        .build()
        .map_err(|e| format!("could not build the writer thread pool: {e}"))?;

    pool.install(|| {
        (0..frame_ids.len())
            .into_par_iter()
            .map(|i| {
                let frame_id = frame_ids[i];
                let mz_values: Vec<f64> = mz[i].to_vec();
                let im_values: Vec<f64> = mobility[i].to_vec();

                let tof = converter.mz_to_tof(frame_id, &mz_values);
                let scan = converter.inverse_mobility_to_scan(frame_id, &im_values);
                let raw_intensity: Vec<u32> = intensity[i].iter().map(|&x| x as u32).collect();

                // m/z → TOF is not injective, so several peaks can land on one cell; sum them and
                // sort by (scan, tof), which is the order the encoder's delta coding needs.
                let (scan, tof, intensity) = sort_dedup_scan_tof(&scan, &tof, &raw_intensity);

                let num_peaks = intensity.len() as u32;
                let max_intensity = intensity.iter().copied().max().unwrap_or(0);
                let summed_intensity: u64 = intensity.iter().map(|&x| x as u64).sum();

                let mut tof_delta = tof.clone();
                modify_tofs(&mut tof_delta, &scan);
                let peak_cnts = get_peak_cnts(max_scans, &scan);
                let interleaved: Vec<u32> = tof_delta
                    .iter()
                    .zip(intensity.iter())
                    .flat_map(|(t, i)| [*t, *i])
                    .collect();
                let real_data = get_realdata(&peak_cnts, &interleaved);

                // `zstd::bulk::compress` (ZSTD_compress) writes the decompressed size into the
                // frame header; `encode_all` does not, and readers that use the simple API —
                // including the Python `zstd` module every existing .d was written with — refuse
                // a frame without it. Keep the header shape the format already has.
                let data = zstd::bulk::compress(real_data.as_slice(), compression_level)
                    .map_err(|e| format!("frame id {frame_id}: zstd compression failed: {e}"))?;

                Ok(CompressedFrame { num_peaks, max_intensity, summed_intensity, data })
            })
            .collect()
    })
}
