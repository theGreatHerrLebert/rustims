"""Block->spectrum alignment for the SCIEX template characterizer.

Physical `.wiff.scan` blocks and pwiz spectra are NOT 1:1 in file order — pwiz reports spectra
that have no block behind them, and on real templates the divergence starts mid-run (block ~28885
of 49741 on the ZenoTOF 7600 60VW CC0 template; block ~75882 of 95372 on K562). Pairing block `i`
with spectrum `i` is therefore right up to that point and silently wrong after it, which mislabels
the MS level and isolation window of every later block while still producing a structurally valid
profile. These tests pin the alignment that replaces index-pairing.
"""
import numpy as np

from imspy_simulation.timsim.jobs.sciex_characterize import (
    align_blocks_to_spectra,
    fill_offsets,
)


def test_identity_when_blocks_and_spectra_match():
    mz = [100.0 + i for i in range(200)]
    amap = align_blocks_to_spectra(mz, mz)
    assert list(amap) == list(range(200))


def test_recovers_offset_when_spectra_are_inserted():
    """Spectra with no block behind them push the offset forward; it must follow, not desync."""
    blocks = [100.0 + i for i in range(400)]
    scans = blocks[:200] + [9000.0 + i for i in range(37)] + blocks[200:]
    amap = align_blocks_to_spectra(blocks, scans)
    assert list(amap[:200]) == list(range(200))
    assert list(amap[200:]) == list(range(237, 437))


def test_offset_never_runs_backwards():
    blocks = [100.0 + i for i in range(300)]
    scans = blocks[:100] + [500.0] * 20 + blocks[100:]
    amap = align_blocks_to_spectra(blocks, scans)
    d = [j - i for i, j in enumerate(amap) if j >= 0]
    assert d == sorted(d)


def test_isolated_mismatch_does_not_move_the_offset():
    """One block pwiz reports differently must not trigger a re-acquisition."""
    blocks = [100.0 + i for i in range(200)]
    scans = list(blocks)
    scans[50] = 7777.0
    amap = align_blocks_to_spectra(blocks, scans)
    assert amap[50] == -1
    assert list(amap[51:]) == list(range(51, 200))


def test_fill_spans_gaps_only_between_agreeing_anchors():
    """A gap bracketed by a CHANGE of offset is left unassigned rather than guessed."""
    amap = np.array([0, -1, 2, -1, 13], dtype=np.int64)  # offsets 0,_,0,_,9
    filled = fill_offsets(amap, 5)
    assert filled[1] == 1  # between two offset-0 anchors -> inherited
    assert filled[3] == -1  # brackets disagree (0 then 9) -> not guessed
    assert filled[0] == 0 and filled[2] == 2 and filled[4] == 13


def test_unmatched_blocks_stay_unmatched_without_anchors():
    amap = np.full(4, -1, dtype=np.int64)
    assert list(fill_offsets(amap, 4)) == [-1, -1, -1, -1]


def test_parse_spectrum_decodes_32_and_64_bit_mz_arrays():
    """Regression: the width test used to be `"d" in fmt`, which is always true for "<%df" (the %d
    placeholder), so every 32-bit (`--32`) mzML mis-sized its arrays and characterize() refused."""
    import base64, struct, zlib
    from imspy_simulation.timsim.jobs import sciex_characterize as sc
    mzs = [350.25, 400.5, 812.125]
    def block(acc_width, packed, compress):
        raw = zlib.compress(packed) if compress else packed
        comp = '<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression"/>' if compress else ''
        return (f'<binaryDataArray><cvParam cvRef="MS" accession="{acc_width}"/>{comp}'
                f'<cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>'
                f'<binary>{base64.b64encode(raw).decode()}</binary></binaryDataArray>')
    for acc, fmt, comp in [("MS:1000521", "<3f", True), ("MS:1000521", "<3f", False),
                           ("MS:1000523", "<3d", True), ("MS:1000523", "<3d", False)]:
        head = ('<spectrum><cvParam accession="MS:1000511" value="1"/>'
                + block(acc, struct.pack(fmt, *mzs), comp) + '</spectrum>')
        ml, iso, mz_min, tic, bmz = sc._parse_spectrum(head)
        assert abs(mz_min - 350.25) < 1e-3, (acc, comp, mz_min)
