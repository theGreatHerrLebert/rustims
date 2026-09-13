"""Parity tests: batched sage digest == per-protein digest; seeded parallel noise overlay is
deterministic and equals the serial per-frame sampling in distribution."""
import os
import numpy as np
import pandas as pd
import pytest

from imspy_simulation.timsim.jobs.simulate_proteins import (
    protein_to_peptides, proteins_to_peptides_batched, generate_single_fasta, parse_fasta_to_dataframe,
)

FASTA = "/scratch/timsim-demo/timsim-necro/flow/hela_subset.fasta"
REF_D = "/scratch/timsim-demo/TIMSIM-HeLa-RUSTW-SMOKE/TIMSIM-HeLa-RUSTW-SMOKE.d"

VAR = {"M": ["[UNIMOD:35]"], "[": ["[UNIMOD:1]"]}
STAT = {"C": "[UNIMOD:4]"}


@pytest.mark.skipif(not os.path.exists(FASTA), reason="local HeLa subset fasta not available")
@pytest.mark.parametrize("decoys", [False, True])
def test_batched_digest_matches_per_protein(decoys):
    tbl = parse_fasta_to_dataframe(FASTA).sample(n=400, random_state=7)
    batched = proteins_to_peptides_batched(
        tbl.index, tbl.sequence, generate_decoys=decoys, variable_mods=VAR, static_mods=STAT,
        chunk_size=150,  # force several chunks
    )
    n_nonempty = 0
    for (idx, row), got in zip(tbl.iterrows(), batched):
        ref = protein_to_peptides(
            generate_single_fasta(idx, row.sequence), generate_decoys=decoys,
            variable_mods=VAR, static_mods=STAT, cleave_at='KR', restrict='P',
            missed_cleavages=2, min_len=7, max_len=30, digest=True,
        )
        if ref is None:
            assert got == set(), f"protein {idx}: per-protein digest empty, batched gave {len(got)}"
        else:
            assert got == ref, f"protein {idx}: {len(got ^ ref)} differing peptides"
            n_nonempty += 1
    assert n_nonempty > 300


def test_batched_digest_empty_input():
    assert proteins_to_peptides_batched([], []) == []


@pytest.mark.skipif(not os.path.exists(REF_D), reason="local reference .d not available")
def test_overlay_reference_noise_is_seeded_and_matches_serial_statistics():
    from imspy_core.timstof import TimsDatasetDIA
    ds = TimsDatasetDIA(REF_D, in_memory=False)
    if not hasattr(ds, "overlay_reference_noise"):
        pytest.skip("imspy-core without overlay_reference_noise")
    info = ds.dia_ms_ms_info
    wg_of = dict(zip(info.Frame, info.WindowGroup))
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    frames = [ds.get_tims_frame(i) for i in ids]
    groups = [int(wg_of[i]) if i in wg_of else None for i in ids]
    kw = dict(num_precursor_frames=5, num_fragment_frames=5, max_intensity_precursor=150000.0,
              max_intensity_fragment=150000.0, take_precursor=0.2, take_fragment=0.2, num_threads=4)
    a = ds.overlay_reference_noise(frames, groups, seed=41, **kw)
    b = ds.overlay_reference_noise(frames, groups, seed=41, **kw)
    c = ds.overlay_reference_noise(frames, groups, seed=42, **kw)
    assert len(a) == len(frames)
    for fa, fb, fc, f0 in zip(a, b, c, frames):
        assert fa.frame_id == f0.frame_id
        np.testing.assert_array_equal(fa.mz, fb.mz)
        np.testing.assert_array_equal(fa.intensity, fb.intensity)
        np.testing.assert_array_equal(fa.scan, fb.scan)
        assert len(fa.mz) >= len(f0.mz)          # noise only adds peaks
        assert not (len(fa.mz) == len(fc.mz) and np.array_equal(fa.mz, fc.mz))  # seed matters
    # serial reference path (unseeded) must give the same added-peak count within sampling noise
    added_par = np.array([len(x.mz) - len(f.mz) for x, f in zip(a, frames)], dtype=float)
    added_ser = []
    for f, g in zip(frames, groups):
        if g is None:
            noise = ds.sample_precursor_signal(5, 150000.0, 0.2)
        else:
            noise = ds.sample_fragment_signal(5, g, 150000.0, 0.2)
        added_ser.append(len((f + noise).mz) - len(f.mz))
    added_ser = np.array(added_ser, dtype=float)
    assert added_par.sum() > 0 and added_ser.sum() > 0
    ratio = added_par.sum() / added_ser.sum()
    assert 0.5 < ratio < 2.0, f"parallel/serial added-peak ratio {ratio:.2f}"


# ---------------------------------------------------------------------------
# Bruker SDK thread-safety regression guards.
#
# `tims_index_to_mz` is not safe to call concurrently on one SDK handle: before the
# fix, the same seed produced m/z differing by up to 4e-5 Th on ~8 % of peaks, in
# 3/8 runs at 4 threads and 7/8 at 32. rustdf now reads through the SDK-free formula
# converter inside parallel regions instead of serialising, so these must hold with
# the SDK enabled (the default).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(REF_D), reason="local reference .d not available")
@pytest.mark.parametrize("threads", [1, 4, 32])
def test_overlay_is_bit_identical_across_thread_counts_with_sdk(threads):
    from imspy_core.timstof import TimsDatasetDIA
    ds = TimsDatasetDIA(REF_D, in_memory=False, use_bruker_sdk=True)
    info = ds.dia_ms_ms_info
    wg_of = dict(zip(info.Frame, info.WindowGroup))
    ids = list(range(1, 19))
    frames = [ds.get_tims_frame(i) for i in ids]
    groups = [int(wg_of[i]) if i in wg_of else None for i in ids]
    kw = dict(num_precursor_frames=5, num_fragment_frames=5, max_intensity_precursor=150000.0,
              max_intensity_fragment=150000.0, take_precursor=0.2, take_fragment=0.2)
    ref = ds.overlay_reference_noise(frames, groups, seed=41, num_threads=1, **kw)
    for rep in range(5):
        got = ds.overlay_reference_noise(frames, groups, seed=41, num_threads=threads, **kw)
        for a, b in zip(ref, got):
            assert len(a.mz) == len(b.mz), f"rep {rep}: peak count differs on frame {a.frame_id}"
            np.testing.assert_array_equal(a.mz, b.mz)
            np.testing.assert_array_equal(a.intensity, b.intensity)


@pytest.mark.skipif(not os.path.exists(REF_D), reason="local reference .d not available")
def test_sdk_free_converter_matches_the_sdk():
    """The general fix rests on this: the formula converter reproduces the SDK, so parallel
    regions may use it instead of dropping to one thread."""
    from imspy_core.timstof import TimsDatasetDIA
    sdk = TimsDatasetDIA(REF_D, in_memory=False, use_bruker_sdk=True)
    fml = TimsDatasetDIA(REF_D, in_memory=False, use_bruker_sdk=False)
    checked = 0
    for fid in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
        a, b = sdk.get_tims_frame(fid), fml.get_tims_frame(fid)
        if len(a.mz) == 0:
            continue
        assert len(a.mz) == len(b.mz)
        np.testing.assert_array_equal(a.scan, b.scan)
        np.testing.assert_array_equal(a.intensity, b.intensity)
        # last-bit agreement, orders of magnitude below any search tolerance
        assert np.abs(a.mz - b.mz).max() < 1e-8, "m/z drift between SDK and formula converter"
        assert np.abs(a.mobility - b.mobility).max() < 1e-12
        checked += 1
    assert checked >= 5
