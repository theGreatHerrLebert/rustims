# imspy/cluster.py
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

import imspy_connector
ims = imspy_connector.py_cluster

from imspy.simulation.annotation import RustWrapperObject

# --- Thin wrappers -----------------------------------------------------------

class ClusterSpec(RustWrapperObject):
    """Thin wrapper for Rust PyClusterSpec."""
    def __init__(self, *args, **kwargs):
        # prefer from_py_ptr; constructing directly uses the rust ctor:
        rt_left      = kwargs.get("rt_left")
        rt_right     = kwargs.get("rt_right")
        im_left      = kwargs.get("im_left")
        im_right     = kwargs.get("im_right")
        mz_center    = kwargs.get("mz_center_hint")
        mz_ppm       = kwargs.get("mz_ppm_window")
        extra_rt_pad = kwargs.get("extra_rt_pad", 0)
        extra_im_pad = kwargs.get("extra_im_pad", 0)
        mz_bins      = kwargs.get("mz_hist_bins", 64)
        if None in (rt_left, rt_right, im_left, im_right, mz_center, mz_ppm):
            raise RuntimeError("Use ClusterSpec.from_py_ptr(...) or provide all required args.")
        self.__py_ptr = ims.PyClusterSpec(
            rt_left, rt_right, im_left, im_right,
            float(mz_center), float(mz_ppm),
            int(extra_rt_pad), int(extra_im_pad), int(mz_bins)
        )

    @classmethod
    def from_py_ptr(cls, p: "ims.PyClusterSpec") -> "ClusterSpec":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    # properties (read back from Rust object)
    @property
    def rt_left(self) -> int: return self.__py_ptr.rt_left
    @property
    def rt_right(self) -> int: return self.__py_ptr.rt_right
    @property
    def im_left(self) -> int: return self.__py_ptr.im_left
    @property
    def im_right(self) -> int: return self.__py_ptr.im_right
    @property
    def mz_center_hint(self) -> float: return self.__py_ptr.mz_center_hint
    @property
    def mz_ppm_window(self) -> float: return self.__py_ptr.mz_ppm_window
    @property
    def extra_rt_pad(self) -> int: return self.__py_ptr.extra_rt_pad
    @property
    def extra_im_pad(self) -> int: return self.__py_ptr.extra_im_pad
    @property
    def mz_hist_bins(self) -> int: return self.__py_ptr.mz_hist_bins

    def __repr__(self) -> str:
        return (f"ClusterSpec(rt=[{self.rt_left},{self.rt_right}], "
                f"im=[{self.im_left},{self.im_right}], "
                f"mz≈{self.mz_center_hint:.5f}±{self.mz_ppm_window}ppm, "
                f"pads(rt={self.extra_rt_pad}, im={self.extra_im_pad}), "
                f"mz_bins={self.mz_hist_bins})")


class AttachOptions(RustWrapperObject):
    """Thin wrapper for Rust PyAttachOptions."""
    def __init__(
        self,
        attach_frames: bool = True,
        attach_scans: bool = True,
        attach_mz_axis: bool = True,
        attach_patch_2d: bool = False,
    ):
        self.__py_ptr = ims.PyAttachOptions(
            bool(attach_frames),
            bool(attach_scans),
            bool(attach_mz_axis),
            bool(attach_patch_2d),
        )

    @classmethod
    def from_py_ptr(cls, p: "ims.PyAttachOptions") -> "AttachOptions":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def attach_frames(self) -> bool: return self.__py_ptr.attach_frames
    @property
    def attach_scans(self) -> bool: return self.__py_ptr.attach_scans
    @property
    def attach_mz_axis(self) -> bool: return self.__py_ptr.attach_mz_axis
    @property
    def attach_patch_2d(self) -> bool: return self.__py_ptr.attach_patch_2d

    def __repr__(self) -> str:
        return (f"AttachOptions(frames={self.attach_frames}, scans={self.attach_scans}, "
                f"mz_axis={self.attach_mz_axis}, patch_2d={self.attach_patch_2d})")


class EvalOptions(RustWrapperObject):
    """Thin wrapper for Rust PyEvalOptions."""
    def __init__(
        self,
        attach: AttachOptions | None = None,
        refine_mz_once: bool = False,
        refine_k_sigma: float = 3.0,
    ):
        if attach is None:
            attach = AttachOptions()
        self.__py_ptr = ims.PyEvalOptions(attach.get_py_ptr(), bool(refine_mz_once), float(refine_k_sigma))

    @classmethod
    def from_py_ptr(cls, p: "ims.PyEvalOptions") -> "EvalOptions":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def attach(self) -> AttachOptions:
        # PyEvalOptions exposes fields via getters on the Rust side (assumed).
        # If not, keep as stored object.
        # Here we return a *view* by wrapping the same pointer.
        # If PyEvalOptions doesn't have a getter, omit this property.
        raise AttributeError("AttachOptions view not exposed; use the object you constructed.")

    @property
    def refine_mz_once(self) -> bool: return self.__py_ptr.refine_mz_once
    @property
    def refine_k_sigma(self) -> float: return self.__py_ptr.refine_k_sigma

    def __repr__(self) -> str:
        return f"EvalOptions(refine_mz_once={self.refine_mz_once}, k={self.refine_k_sigma})"


class ClusterFit1D(RustWrapperObject):
    """Thin wrapper for Rust ClusterFit1D (read-only)."""
    def __init__(self, *a, **k):
        raise RuntimeError("ClusterFit1D is created in Rust; use ClusterFit1D.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyClusterFit1D") -> "ClusterFit1D":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    # fields
    @property
    def mu(self) -> float: return self.__py_ptr.mu
    @property
    def sigma(self) -> float: return self.__py_ptr.sigma
    @property
    def height(self) -> float: return self.__py_ptr.height
    @property
    def baseline(self) -> float: return self.__py_ptr.baseline
    @property
    def area(self) -> float: return self.__py_ptr.area
    @property
    def r2(self) -> float: return self.__py_ptr.r2
    @property
    def n(self) -> int: return self.__py_ptr.n

    def __repr__(self) -> str:
        return f"Fit1D(mu={self.mu:.3f}, σ={self.sigma:.3f}, h={self.height:.1f}, base={self.baseline:.1f}, area={self.area:.1f})"

    def get_py_ptr(self):
        return self.__py_ptr


class ClusterResult(RustWrapperObject):
    """Thin wrapper for Rust ClusterResult (read-only)."""
    def __init__(self, *a, **k):
        raise RuntimeError("ClusterResult is created in Rust; use ClusterResult.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyClusterResult") -> "ClusterResult":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    # windows / meta
    @property
    def rt_window(self) -> Tuple[int, int]: return tuple(self.__py_ptr.rt_window)
    @property
    def im_window(self) -> Tuple[int, int]: return tuple(self.__py_ptr.im_window)
    @property
    def mz_window_da(self) -> Tuple[float, float]: return tuple(self.__py_ptr.mz_window_da)

    # fits
    @property
    def rt_fit(self) -> ClusterFit1D: return ClusterFit1D.from_py_ptr(self.__py_ptr.rt_fit)
    @property
    def im_fit(self) -> ClusterFit1D: return ClusterFit1D.from_py_ptr(self.__py_ptr.im_fit)
    @property
    def mz_fit(self) -> ClusterFit1D: return ClusterFit1D.from_py_ptr(self.__py_ptr.mz_fit)

    # intensity summaries
    @property
    def raw_sum(self) -> float: return self.__py_ptr.raw_sum
    @property
    def fit_volume(self) -> float: return self.__py_ptr.fit_volume

    # provenance
    @property
    def rt_peak_id(self) -> int: return self.__py_ptr.rt_peak_id
    @property
    def im_peak_id(self) -> int: return self.__py_ptr.im_peak_id
    @property
    def mz_center_hint(self) -> float: return self.__py_ptr.mz_center_hint

    # attached axes / patch (optional)
    @property
    def frame_ids_used(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.frame_ids_used, dtype=np.uint32)

    @property
    def frames_axis(self) -> Optional[np.ndarray]:
        fa = self.__py_ptr.frames_axis
        return None if fa is None else np.asarray(fa, dtype=np.uint32)

    @property
    def scans_axis(self) -> Optional[np.ndarray]:
        sa = self.__py_ptr.scans_axis
        return None if sa is None else np.asarray(sa, dtype=np.int64)

    @property
    def mz_axis(self) -> Optional[np.ndarray]:
        ma = self.__py_ptr.mz_axis
        return None if ma is None else np.asarray(ma, dtype=np.float32)

    @property
    def patch_2d(self) -> Optional[np.ndarray]:
        """Return (frames×scans) matrix as Fortran-order view if attached, else None."""
        buf = self.__py_ptr.patch_2d_colmajor
        if buf is None:
            return None
        frames, scans = self.__py_ptr.patch_shape
        # column-major layout → reshape with order="F"
        arr = np.asarray(buf, dtype=np.float32)
        return arr.reshape((frames, scans), order="F")

    @property
    def patch_shape(self) -> Tuple[int, int]:
        return tuple(self.__py_ptr.patch_shape)

    def __repr__(self) -> str:
        (rt_l, rt_r) = self.rt_window
        (im_l, im_r) = self.im_window
        return (f"ClusterResult(rt=[{rt_l},{rt_r}], im=[{im_l},{im_r}], "
                f"mz=({self.mz_window_da[0]:.4f},{self.mz_window_da[1]:.4f}), "
                f"sum={self.raw_sum:.1f})")


# --- Convenience function: build specs from peaks (fully wrapped) ------------

def make_cluster_specs_from_peaks(
    rt_peaks: Sequence["RtPeak1D"],
    im_rows: Sequence[Sequence["ImPeak1D"]],
    *,
    mz_ppm_window: float = 15.0,
    extra_rt_pad: int = 0,
    extra_im_pad: int = 0,
    mz_hist_bins: int = 64,
) -> List[ClusterSpec]:
    """
    Build ClusterSpec objects from aligned RT peaks and their IM peak rows.

    Args
    ----
    rt_peaks : sequence of RtPeak1D, length = R
    im_rows  : sequence (length R) of sequences of ImPeak1D (one list per RT row)
    mz_ppm_window : ±ppm window around each rt_peak.mz_center_hint
    extra_rt_pad  : extra frames to include around the RT padded window
    extra_im_pad  : extra scans to include around the IM window
    mz_hist_bins  : number of bins for the m/z marginal histogram inside the cluster

    Returns
    -------
    List[ClusterSpec]
    """
    if len(rt_peaks) != len(im_rows):
        raise ValueError("rt_peaks and im_rows must have the same length (row-aligned).")

    # Convert to raw PyO3 objects (no user poking at pointers)
    rt_py = [p.get_py_ptr() for p in rt_peaks]
    im_py = [[q.get_py_ptr() for q in row] for row in im_rows]

    out_py = ims.make_cluster_specs_from_peaks(
        rt_py, im_py,
        mz_ppm_window, extra_rt_pad, extra_im_pad, mz_hist_bins
    )
    return [ClusterSpec.from_py_ptr(p) for p in out_py]