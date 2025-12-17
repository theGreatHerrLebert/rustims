from typing import List, Sequence

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
    merge_fragments: bool = False,
    merge_max_ppm: float = 10.0,
    merge_allow_cross_window_group: bool = False,

    # NEW: MS1-derived precursor metadata
    ms1_index=None,
    ds=None,
    feature_index=None,
    intensity_source: str = "volume_proxy_then_raw_sum",
    feature_agg: str = "most_intense_member",
    feature_top_k: int = 3,

    # forward compat (optional)
    **_ignored,
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
                inv_mob, rt_mu, raw_sum = get_precursor_info(
                    spec, ms1_index, ds,
                    feature_index=feature_index,
                    intensity_source=intensity_source,
                    feature_agg=feature_agg,
                    feature_top_k=feature_top_k,
                )
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

# --------------------------- quant helpers -----------------------------------
import numpy as np

def _finite_pos(x) -> float | None:
    try:
        x = float(x)
        if np.isfinite(x) and x > 0.0:
            return x
    except Exception:
        pass
    return None


def cluster_intensity(cluster, source: str = "volume_proxy_then_raw_sum") -> float | None:
    """
    Choose a stable intensity proxy for a single MS1 cluster.

    source:
      - "raw_sum"
      - "volume_proxy"
      - "volume_proxy_then_raw_sum"  (default)
      - "rt_area_then_raw_sum"       (optional, only if rt_area is meaningful for you)
    """
    if source == "raw_sum":
        return _finite_pos(getattr(cluster, "raw_sum", None))

    if source == "volume_proxy":
        return _finite_pos(getattr(cluster, "volume_proxy", None))

    if source == "rt_area_then_raw_sum":
        v = _finite_pos(getattr(cluster, "rt_area", None))
        return v if v is not None else _finite_pos(getattr(cluster, "raw_sum", None))

    # default
    v = _finite_pos(getattr(cluster, "volume_proxy", None))
    return v if v is not None else _finite_pos(getattr(cluster, "raw_sum", None))


def feature_intensity(
    feature,
    ms1_index: dict[int, object] | None,
    *,
    intensity_source: str = "volume_proxy_then_raw_sum",
    agg: str = "most_intense_member",
    top_k: int = 3,
) -> float | None:
    """
    Aggregate intensity over isotopes (feature members) or pick the best member.

    agg:
      - "most_intense_member"  (robust default)
      - "sum_members"
      - "sum_top_k_members"    (nice compromise)
    """
    ids = getattr(feature, "member_cluster_ids", None)
    if not ids or ms1_index is None:
        return None

    vals: list[float] = []
    for cid in ids:
        c = ms1_index.get(int(cid))
        if c is None:
            continue
        v = cluster_intensity(c, intensity_source)
        if v is not None:
            vals.append(float(v))

    if not vals:
        return None

    vals.sort(reverse=True)

    if agg == "sum_members":
        return float(sum(vals))

    if agg == "sum_top_k_members":
        k = max(1, int(top_k))
        return float(sum(vals[:k]))

    # default: most_intense_member
    return float(vals[0])


def get_precursor_info(
    spec,
    ms1_index: dict[int, object] | None,
    ds,
    *,
    feature_index: dict[int, object] | None = None,
    intensity_source: str = "volume_proxy_then_raw_sum",
    feature_agg: str = "most_intense_member",
    feature_top_k: int = 3,
):
    """
    Returns (inv_mob, rt_mu_sec, intensity).

    Preference order:
      1) If spec.feature_id + feature_index available:
           - intensity from aggregated feature members
           - rt/im from most_intense_precursor (stable)
      2) Else: from spec.precursor_cluster_ids[0] via ms1_index
    """
    inv_mob = rt_mu = inten = None

    # --- feature-based path
    fid = getattr(spec, "feature_id", None)
    if fid is not None and feature_index is not None and ms1_index is not None:
        feat = feature_index.get(int(fid))
        if feat is not None:
            inten = feature_intensity(
                feat,
                ms1_index,
                intensity_source=intensity_source,
                agg=feature_agg,
                top_k=feature_top_k,
            )
            try:
                c0 = feat.most_intense_precursor
                rt_mu = _finite_pos(getattr(c0, "rt_mu", None))
                # note: rt_mu is in seconds in your cluster wrapper
                if rt_mu is not None:
                    rt_mu = float(getattr(c0, "rt_mu"))
                inv_mob = None
                try:
                    inv_mob = float(ds.scan_to_inverse_mobility(1, [int(getattr(c0, "im_mu"))])[0])
                except Exception:
                    inv_mob = None
                return inv_mob, rt_mu, inten
            except Exception:
                # fall through to cluster-level if feature object is missing fields
                pass

    # --- cluster-based path
    cids = getattr(spec, "precursor_cluster_ids", None)
    if cids and ms1_index is not None:
        c = ms1_index.get(int(cids[0]))
        if c is not None:
            inten = cluster_intensity(c, intensity_source)
            try:
                rt_mu = float(getattr(c, "rt_mu"))
            except Exception:
                rt_mu = None
            inv_mob = None
            try:
                inv_mob = float(ds.scan_to_inverse_mobility(1, [int(getattr(c, "im_mu"))])[0])
            except Exception:
                inv_mob = None

    return inv_mob, rt_mu, inten