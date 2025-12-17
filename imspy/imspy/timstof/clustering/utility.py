from collections.abc import Iterable
from typing import List, Sequence

import numpy as np
from sagepy.core import Precursor, RawSpectrum, SpectrumProcessor

import imspy_connector
ims = imspy_connector.py_dia  # <- the Rust PyO3 module you compiled

def stitch_im_peaks(
    peaks: Sequence[Sequence[Sequence["ImPeak1D"]]] | Sequence["ImPeak1D"],
    min_overlap_frames: int = 1,
    max_scan_delta: int = 1,
    jaccard_min: float = 0.0,
    max_tof_row_delta: int = 0,
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
        max_tof_row_delta=max_tof_row_delta,
        allow_cross_groups=allow_cross_groups,
        min_im_overlap_scans=min_im_overlap_scans,
        im_jaccard_min=im_jaccard_min,
        require_mutual_apex_inside=require_mutual_apex_inside,
    )

    # Wrap back into high-level Python objects
    return [ImPeak1D.from_py_ptr(p) for p in stitched_py]

def build_sagepy_queries_from_pseudo_spectra(
    spectra: Sequence["PseudoSpectrum"],
    *,
    file_id: int = 1,
    take_top_n: int = 150,
    deisotope: bool = True,
    min_fragments: int = 5,
    use_charge: bool = True,
    # fragment merging controls
    merge_fragments: bool = False,
    merge_max_ppm: float = 10.0,
    merge_allow_cross_window_group: bool = False,
    # NEW: MS1-derived precursor metadata
    ms1_index=None,
    ds=None,
):
    """
    Turn a list of PseudoSpectrum objects into SAGE Py query spectra.

    - Uses precursor_mz and precursor_charge from the pseudo spectrum
      (charge <= 0 -> None).
    - Uses fragment m/z and intensity either directly from PseudoFragment,
      or from PseudoSpectrum.merged_peaks(...) if `merge_fragments=True`.
    - Sorts fragments by m/z.
    - Skips spectra with fewer than `min_fragments` peaks.
    - If `ms1_index` and `ds` are given, attaches:
        * Precursor.intensity    <- cluster.raw_sum
        * Precursor.inverse_ion_mobility <- ds.scan_to_inverse_mobility(...)
        * RawSpectrum.scan_start_time     <- cluster.rt_mu (converted to minutes)

    Parameters
    ----------
    spectra : Sequence[PseudoSpectrum]
        Input pseudo spectra.
    file_id : int, default 1
        File ID to attach to all RawSpectrum objects.
    take_top_n : int, default 150
        Passed to SpectrumProcessor.
    deisotope : bool, default True
        Passed to SpectrumProcessor.
    min_fragments : int, default 5
        Minimum number of fragment peaks after filtering/merging.
    use_charge : bool, default True
        If False, precursor charge is set to None.
    merge_fragments : bool, default False
        If True, use PseudoSpectrum.merged_peaks(...) instead of raw fragments.
    merge_max_ppm : float, default 10.0
        PPM tolerance for merging when `merge_fragments=True`.
        If <= 0, merging degenerates to a pure sort-and-return.
    merge_allow_cross_window_group : bool, default False
        If True, merged_peaks is allowed to merge peaks across window_groups.
    ms1_index : optional
        Indexable structure mapping precursor_cluster_id -> MS1 cluster object
        with im_mu, rt_mu, raw_sum.
    ds : optional
        Dataset object providing scan_to_inverse_mobility(frame, [scan]) -> [1/K0].
    """

    from imspy.timstof.clustering.pseudo import PseudoSpectrum

    spec_processor = SpectrumProcessor(
        take_top_n=take_top_n,
        deisotope=deisotope,
    )

    queries = []
    lengths: list[int] = []

    for idx, spec in enumerate(spectra):
        # ---------- precursor charge ----------
        charge = int(spec.precursor_charge)
        if charge <= 0:
            charge = None

        # ---------- MS1-derived precursor info ----------
        inv_mob = None
        rt_mu = None
        raw_sum = None

        if ms1_index is not None and ds is not None:
            try:
                inv_mob, rt_mu, raw_sum = get_precursor_info(spec, ms1_index, ds)
            except Exception:
                # fail gracefully, keep them as None
                inv_mob, rt_mu, raw_sum = None, None, None

        precursor = Precursor(
            charge=charge if use_charge else None,
            mz=float(spec.precursor_mz),
            intensity=raw_sum,                 # None if we couldn't look it up
            inverse_ion_mobility=inv_mob,      # None if unknown
        )

        # ---------- fragments: raw vs merged ----------
        if merge_fragments:
            frag_mz, frag_int = spec.merged_peaks(
                max_ppm=float(merge_max_ppm),
                allow_cross_window_group=bool(merge_allow_cross_window_group),
            )
            frag_mz = np.asarray(frag_mz, dtype=np.float32)
            frag_int = np.asarray(frag_int, dtype=np.float32)
        else:
            frag_mz = np.array(
                [float(f.mz) for f in spec.fragments],
                dtype=np.float32,
            )
            frag_int = np.array(
                [float(f.intensity) for f in spec.fragments],
                dtype=np.float32,
            )

        # drop NaNs / non-positive intensities
        mask = np.isfinite(frag_mz) & np.isfinite(frag_int) & (frag_int > 0.0)
        frag_mz = frag_mz[mask]
        frag_int = frag_int[mask]

        if frag_mz.size < min_fragments:
            continue

        # sort by m/z
        order = np.argsort(frag_mz)
        frag_mz = frag_mz[order]
        frag_int = frag_int[order]

        lengths.append(frag_mz.size)

        # spectrum ID
        if spec.feature_id is not None:
            spec_id = f"F-{spec.feature_id}"
        else:
            if getattr(spec, "precursor_cluster_ids", None):
                spec_id = f"C-{spec.precursor_cluster_ids[0]}"
            else:
                spec_id = f"UNK-{idx}"

        # ---------- RT -> RawSpectrum.scan_start_time ----------
        # Rust side expects minutes. timsTOF usually has rt_mu in seconds,
        # so convert here; if your rt_mu is already minutes, drop the / 60.
        scan_start_time = 0.0
        if rt_mu is not None:
            scan_start_time = float(rt_mu) / 60.0

        raw_spectrum = RawSpectrum(
            file_id=file_id,
            spec_id=spec_id,
            total_ion_current=float(frag_int.sum()),
            precursors=[precursor],
            mz=frag_mz,
            intensity=frag_int,
            scan_start_time=scan_start_time,
            ion_injection_time=0.0,
        )

        query = spec_processor.process(raw_spectrum)
        queries.append(query)

    return queries

def get_precursor_info(spec, ms1_index, ds):
    """
    From a PseudoSpectrum, look up the corresponding MS1 cluster
    and derive:
      - inverse ion mobility (1/K0)
      - retention time (rt_mu)
      - precursor intensity (raw_sum)

    Assumptions:
    -----------
    - spec.precursor_cluster_ids[0] is a key into ms1_index
    - ms1_index[idx] has attributes: im_mu (scan index), rt_mu, raw_sum
    - ds.scan_to_inverse_mobility(frame, [scan]) -> [inv_mob]
    """
    # be defensive: no precursor cluster
    if not getattr(spec, "precursor_cluster_ids", None):
        return None, None, None

    idx = spec.precursor_cluster_ids[0]
    p = ms1_index[idx]

    # mobility: convert scan index -> inverse ion mobility
    inv_mob = None
    try:
        inv_mob = float(ds.scan_to_inverse_mobility(1, [int(p.im_mu)])[0])
    except Exception:
        # if something goes wrong, just leave it None
        inv_mob = None

    # RT and intensity directly from the cluster
    rt_mu = float(p.rt_mu)
    raw_sum = float(p.raw_sum)

    return inv_mob, rt_mu, raw_sum