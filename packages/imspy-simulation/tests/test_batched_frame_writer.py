"""The batched writer must produce exactly what the per-frame writer produced.

`TDFWriter.write_frames` moves conversion, dedup, interleave and zstd into one parallel Rust call.
The output is a vendor file format, so "equivalent" is not good enough: the compressed payloads must
be byte-identical to the per-frame path, and the Frames statistics must match.
"""
import os

import numpy as np
import pytest
import zstd

from imspy_simulation.tdf import dedup_scan_tof
from imspy_simulation.utility import get_compressible_data

REF_D = "/scratch/timsim-demo/TIMSIM-HeLa-RUSTW-SMOKE/TIMSIM-HeLa-RUSTW-SMOKE.d"
pytestmark = pytest.mark.skipif(not os.path.exists(REF_D), reason="local reference .d not available")


def _reference(ds, frames, num_scans):
    """Exactly what TDFWriter.compress_frame did, frame by frame, through the SDK."""
    out = []
    for f in frames:
        tof = np.array(ds.mz_to_tof(f.frame_id, f.mz)).astype(np.uint32)
        scan = np.array(ds.inverse_mobility_to_scan(f.frame_id, f.mobility)).astype(np.uint32)
        intensity = f.intensity.astype(np.uint32)
        scan, tof, intensity = dedup_scan_tof(scan, tof, intensity)
        blob = zstd.ZSTD_compress(bytes(get_compressible_data(tof, scan, intensity, num_scans)), 0)
        out.append((len(intensity),
                    int(intensity.max()) if len(intensity) else 0,
                    int(intensity.sum()), blob))
    return out


def _load(n=120):
    from imspy_core.timstof import TimsDatasetDIA
    ds = TimsDatasetDIA(REF_D, in_memory=False, use_bruker_sdk=True)
    frames = [f for f in (ds.get_tims_frame(i) for i in range(1, n + 1)) if len(f.mz) > 0]
    assert frames, "reference produced no non-empty frames"
    return ds, frames, int(ds.num_scans)


@pytest.mark.parametrize("num_threads", [1, 4, 16])
def test_batched_writer_is_byte_identical_to_per_frame(num_threads):
    ds, frames, num_scans = _load()
    if not hasattr(ds, "build_compressed_frames"):
        pytest.skip("connector without build_compressed_frames")
    ref = _reference(ds, frames, num_scans)
    got = ds.build_compressed_frames(
        [f.frame_id for f in frames],
        [np.ascontiguousarray(f.mz, dtype=np.float64) for f in frames],
        [np.ascontiguousarray(f.mobility, dtype=np.float64) for f in frames],
        [np.ascontiguousarray(f.intensity, dtype=np.float64) for f in frames],
        num_scans, 0, num_threads,
    )
    assert len(got) == len(ref)
    for i, ((n1, m1, s1, b1), (n2, m2, s2, b2)) in enumerate(zip(ref, got)):
        assert (n1, m1, s1) == (n2, m2, s2), f"frame {i}: Frames statistics differ"
        # A zstd frame written without the decompressed size in its header cannot be read back by
        # the simple API every existing .d was written with, so the header shape matters too.
        assert b1 == bytes(b2), f"frame {i}: compressed payload differs"


def test_batched_writer_output_still_decompresses_with_the_simple_api():
    ds, frames, num_scans = _load(n=30)
    if not hasattr(ds, "build_compressed_frames"):
        pytest.skip("connector without build_compressed_frames")
    got = ds.build_compressed_frames(
        [f.frame_id for f in frames],
        [np.ascontiguousarray(f.mz, dtype=np.float64) for f in frames],
        [np.ascontiguousarray(f.mobility, dtype=np.float64) for f in frames],
        [np.ascontiguousarray(f.intensity, dtype=np.float64) for f in frames],
        num_scans, 0, 4,
    )
    for _, _, _, blob in got:
        assert len(zstd.ZSTD_uncompress(bytes(blob))) > 0


def test_ragged_arrays_raise_instead_of_panicking():
    """A ragged triplet used to index out of bounds inside Rust; it must surface as an error."""
    ds, frames, num_scans = _load(n=10)
    if not hasattr(ds, "build_compressed_frames"):
        pytest.skip("connector without build_compressed_frames")
    mz = [np.ascontiguousarray(f.mz, dtype=np.float64) for f in frames]
    mobility = [np.ascontiguousarray(f.mobility, dtype=np.float64) for f in frames]
    intensity = [np.ascontiguousarray(f.intensity, dtype=np.float64) for f in frames]
    mobility[0] = mobility[0][:-1]          # one value short
    with pytest.raises(RuntimeError, match="must have the same length"):
        ds.build_compressed_frames([f.frame_id for f in frames], mz, mobility, intensity,
                                   num_scans, 0, 4)


def test_mismatched_outer_lengths_raise():
    ds, frames, num_scans = _load(n=10)
    if not hasattr(ds, "build_compressed_frames"):
        pytest.skip("connector without build_compressed_frames")
    mz = [np.ascontiguousarray(f.mz, dtype=np.float64) for f in frames]
    mobility = [np.ascontiguousarray(f.mobility, dtype=np.float64) for f in frames]
    intensity = [np.ascontiguousarray(f.intensity, dtype=np.float64) for f in frames]
    with pytest.raises(RuntimeError, match="same length"):
        ds.build_compressed_frames([f.frame_id for f in frames][:-1], mz, mobility, intensity,
                                   num_scans, 0, 4)
