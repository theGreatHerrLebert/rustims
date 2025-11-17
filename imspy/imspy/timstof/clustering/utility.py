from collections.abc import Iterable
from typing import List, Sequence

import imspy_connector
ims = imspy_connector.py_dia  # <- the Rust PyO3 module you compiled

def stitch_im_peaks(
    peaks: Sequence[Sequence[Sequence["ImPeak1D"]]] | Sequence["ImPeak1D"],
    min_overlap_frames: int = 1,
    max_scan_delta: int = 1,
    jaccard_min: float = 0.0,
    max_mz_row_delta: int = 0,
    allow_cross_groups: bool = False,
    min_im_overlap_scans: int = 1,
    im_jaccard_min: float = 0.0,
    require_mutual_apex_inside: bool = True,
) -> List["ImPeak1D"]:
    """
    Stitch IM 1D peaks across overlapping RT windows.

    Accepts either:
      - flat:   list[ImPeak1D]
      - nested: list[list[list[ImPeak1D]]] (windows × rows × peaks)

    Returns:
      flat, stitched list[ImPeak1D] in TOF/scan/RT space.
    """

    from imspy.timstof.clustering.data import ImPeak1D

    # normalize to flat Vec<PyImPeak1D> under the hood
    if not peaks:
        return []

    if isinstance(peaks[0], ImPeak1D):
        flat = list(peaks)  # type: ignore[arg-type]
    else:
        # windows × rows × peaks → flat
        flat = [p for win in peaks for row in win for p in row]  # type: ignore[arg-type]

    py_peaks = [p.get_py_ptr() for p in flat]

    stitched_py = ims.stitch_im_peaks_flat_unordered(
        py_peaks,
        min_overlap_frames=min_overlap_frames,
        max_scan_delta=max_scan_delta,
        jaccard_min=jaccard_min,
        max_tof_row_delta=max_mz_row_delta,
        allow_cross_groups=allow_cross_groups,
        min_im_overlap_scans=min_im_overlap_scans,
        im_jaccard_min=im_jaccard_min,
        require_mutual_apex_inside=require_mutual_apex_inside,
    )

    # Wrap back into high-level Python objects
    return [ImPeak1D.from_py_ptr(p) for p in stitched_py]