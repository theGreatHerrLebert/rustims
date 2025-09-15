# imspy/cluster.py
from __future__ import annotations
from typing import List, Iterable, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

import imspy_connector
ims = imspy_connector.py_cluster

from imspy.simulation.annotation import RustWrapperObject
from .dia import RtPeak1D, ImPeak1D  # <- adjust import path to where you defined them


# -------------------- Core wrappers --------------------

class ClusterSpec(RustWrapperObject):
    """Python wrapper around Rust PyClusterSpec (read-only properties)."""
    def __init__(
        self,
        rt_row: int,
        rt_left: int,
        rt_right: int,
        scan_left: int,
        scan_right: int,
        mz_ppm: float,
        resolution: int,
    ):
        self.__py_ptr = ims.PyClusterSpec(
            rt_row, rt_left, rt_right, scan_left, scan_right, float(mz_ppm), int(resolution)
        )

    @property
    def rt_row(self) -> int: return self.__py_ptr.rt_row
    @property
    def rt_left(self) -> int: return self.__py_ptr.rt_left
    @property
    def rt_right(self) -> int: return self.__py_ptr.rt_right
    @property
    def scan_left(self) -> int: return self.__py_ptr.scan_left
    @property
    def scan_right(self) -> int: return self.__py_ptr.scan_right
    @property
    def mz_ppm(self) -> float: return self.__py_ptr.mz_ppm
    @property
    def resolution(self) -> int: return self.__py_ptr.resolution

    def get_py_ptr(self):
        return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, spec: ims.PyClusterSpec) -> "ClusterSpec":
        inst = cls.__new__(cls)
        inst.__py_ptr = spec
        return inst

    def __repr__(self) -> str:
        return (f"ClusterSpec(rt_row={self.rt_row}, rt_left={self.rt_left}, rt_right={self.rt_right}, "
                f"scan_left={self.scan_left}, scan_right={self.scan_right}, "
                f"mz_ppm={self.mz_ppm}, resolution={self.resolution})")


class Gaussian1D(RustWrapperObject):
    def __init__(self, py: ims.PyGaussian1D):
        self.__py_ptr = py

    @property
    def mu(self) -> float: return self.__py_ptr.mu
    @property
    def sigma(self) -> float: return self.__py_ptr.sigma
    @property
    def fwhm(self) -> float: return self.__py_ptr.fwhm

    def get_py_ptr(self): return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, g: ims.PyGaussian1D) -> "Gaussian1D":
        return cls(g)

    def __repr__(self) -> str:
        return f"Gaussian1D(mu={self.mu:.3f}, sigma={self.sigma:.3f}, fwhm={self.fwhm:.3f})"


class Separable2DFit(RustWrapperObject):
    def __init__(self, py: ims.PySeparable2DFit):
        self.__py_ptr = py

    @property
    def rt(self) -> Gaussian1D: return Gaussian1D.from_py_ptr(self.__py_ptr.rt)
    @property
    def im(self) -> Gaussian1D: return Gaussian1D.from_py_ptr(self.__py_ptr.im)
    @property
    def A(self) -> float: return self.__py_ptr.A      # amplitude
    @property
    def B(self) -> float: return self.__py_ptr.B      # background

    def get_py_ptr(self): return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, f: ims.PySeparable2DFit) -> "Separable2DFit":
        return cls(f)

    def __repr__(self) -> str:
        return f"Separable2DFit(rt={self.rt}, im={self.im}, A={self.A:.2f}, B={self.B:.2f})"


class ClusterQuality(RustWrapperObject):
    def __init__(self, py: ims.PyClusterQuality):
        self.__py_ptr = py

    @property
    def r2(self) -> float: return self.__py_ptr.r2
    @property
    def mse(self) -> float: return self.__py_ptr.mse
    @property
    def snr_local(self) -> float: return self.__py_ptr.snr_local
    @property
    def edge_mass_frac(self) -> float: return self.__py_ptr.edge_mass_frac

    def get_py_ptr(self): return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, q: ims.PyClusterQuality) -> "ClusterQuality":
        return cls(q)

    def __repr__(self) -> str:
        return f"ClusterQuality(r2={self.r2:.4f}, mse={self.mse:.3f}, snr={self.snr_local:.2f}, edge={self.edge_mass_frac:.3f})"


class ClusterPatch(RustWrapperObject):
    """Carries the raw 2D RT×IM data for a cluster selection (rows=frames, cols=scans)."""
    def __init__(self, py: ims.PyClusterPatch):
        self.__py_ptr = py

    @property
    def rows(self) -> int: return self.__py_ptr.rows
    @property
    def cols(self) -> int: return self.__py_ptr.cols

    def rt_frames(self) -> np.ndarray:
        return self.__py_ptr.rt_frames()
    def scans(self) -> np.ndarray:
        return self.__py_ptr.scans()
    def rt_trace(self) -> np.ndarray:
        return self.__py_ptr.rt_trace()
    def im_trace(self) -> np.ndarray:
        return self.__py_ptr.im_trace()
    def patch(self) -> np.ndarray:
        return self.__py_ptr.patch()

    @property
    def total_area(self) -> float: return self.__py_ptr.total_area
    @property
    def apex_value(self) -> float: return self.__py_ptr.apex_value
    @property
    def apex_pos(self) -> Tuple[int, int]: return self.__py_ptr.apex_pos

    def get_py_ptr(self): return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, p: ims.PyClusterPatch) -> "ClusterPatch":
        return cls(p)


class ClusterResult(RustWrapperObject):
    """Bundle of spec, patch, fit and quality."""
    def __init__(self, py: ims.PyClusterResult):
        self.__py_ptr = py

    @property
    def spec(self) -> ClusterSpec:
        return ClusterSpec.from_py_ptr(self.__py_ptr.spec)
    @property
    def patch(self) -> ClusterPatch:
        return ClusterPatch.from_py_ptr(self.__py_ptr.patch)
    @property
    def fit(self) -> Separable2DFit:
        return Separable2DFit.from_py_ptr(self.__py_ptr.fit)
    @property
    def quality(self) -> ClusterQuality:
        return ClusterQuality.from_py_ptr(self.__py_ptr.quality)

    def to_dict(self) -> Dict[str, Any]:
        """Compact dict for DataFrame creation."""
        d = self.__py_ptr.to_dict()
        # d is a Python dict returned from Rust, so just return it
        return dict(d)

    def get_py_ptr(self): return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, r: ims.PyClusterResult) -> "ClusterResult":
        return cls(r)

    def __repr__(self) -> str:
        q = self.quality
        return f"ClusterResult({q})"


# -------------------- Helpers --------------------
from typing import Dict, List, Tuple, Union

IMMapType = Union[
    Dict[int, List["ImPeak1D"]],                    # key = mz_row
    Dict[Tuple[int, int], List["ImPeak1D"]],        # key = (mz_row, rt_col)
    List[List["ImPeak1D"]],                         # row-aligned with rt_peaks
]

def _clamp_bounds_nonneg(left: int, right: int) -> tuple[int, int]:
    # ensure non-negative and left <= right
    if left < 0:  left = 0
    if right < 0: right = 0
    if right < left:
        right = left
    return left, right

def _make_cluster_spec(rp, scan_left, scan_right, mz_ppm: float, resolution: int):
    """
    Create ClusterSpec with safe integer bounds (non-negative) and correct signature:
    ClusterSpec(rt_row, rt_left, rt_right, scan_left, scan_right, mz_ppm, resolution)
    """
    # Prefer padded RT bounds if present; else fall back to raw
    rt_left  = int(getattr(rp, "left_padded",  getattr(rp, "left",  0)))
    rt_right = int(getattr(rp, "right_padded", getattr(rp, "right", 0)))
    scan_left  = int(scan_left)
    scan_right = int(scan_right)

    # Clamp everything to be non-negative and left<=right
    rt_left,  rt_right  = _clamp_bounds_nonneg(rt_left,  rt_right)
    scan_left, scan_right = _clamp_bounds_nonneg(scan_left, scan_right)

    # rp.mz_row is the RT row index (your Python ClusterSpec calls it rt_row)
    return ClusterSpec(
        int(rp.mz_row),
        rt_left, rt_right,
        scan_left, scan_right,
        float(mz_ppm), int(resolution),
    )

IMMapType = Union[
    Dict[int, List["ImPeak1D"]],             # key = mz_row
    Dict[Tuple[int, int], List["ImPeak1D"]], # key = (mz_row, rt_col)
    List[List["ImPeak1D"]],                  # aligned with rt_peaks order
]

def _get_im_list_for_peak(im_peaks_by_rtrow: IMMapType, peak, idx: int) -> List["ImPeak1D"]:
    if isinstance(im_peaks_by_rtrow, dict):
        if (peak.mz_row, peak.rt_col) in im_peaks_by_rtrow:
            return im_peaks_by_rtrow[(peak.mz_row, peak.rt_col)]
        if peak.mz_row in im_peaks_by_rtrow:  # type: ignore[operator]
            return im_peaks_by_rtrow[peak.mz_row]  # type: ignore[index]
        return []
    if isinstance(im_peaks_by_rtrow, list):
        if 0 <= idx < len(im_peaks_by_rtrow):
            return im_peaks_by_rtrow[idx]
        return []
    return []

def build_specs_from_peaks(
    rt_peaks: List["RtPeak1D"],
    im_peaks_by_rtrow: IMMapType,
    *,
    mz_ppm: float = 20.0,
    resolution: int = 2,
    default_scan_span: int = 30,
) -> List["ClusterSpec"]:
    specs: List["ClusterSpec"] = []

    # precompute half span for fallback
    half = max(1, default_scan_span // 2)

    for i, rp in enumerate(rt_peaks):
        im_list = _get_im_list_for_peak(im_peaks_by_rtrow, rp, i)

        if im_list:
            best = max(im_list, key=lambda p: p.area_raw)
            scan_left, scan_right = int(best.left), int(best.right)
        else:
            # Fallback: symmetric window around an approximate center
            # (use subcol+rt_col if available; else rt_col)
            center = int(round(getattr(rp, "subcol", 0.0) + rp.rt_col))
            scan_left  = center - half
            scan_right = center + half

        specs.append(_make_cluster_spec(rp, scan_left, scan_right, mz_ppm, resolution))
    return specs


def evaluate_clusters_separable(
    ds_py_ptr,           # ds.get_py_ptr()
    specs: List[ClusterSpec],
    bins: np.ndarray,    # from get_dense_mz_vs_rt...
    frames: np.ndarray,  # from get_dense_mz_vs_rt...
    num_threads: int = 4
) -> List[ClusterResult]:
    """Thin Python bridge to the Rust method; returns high-level Python wrappers."""
    specs_py = [s.get_py_ptr() for s in specs]
    res_py = ds_py_ptr.evaluate_clusters_separable(specs_py, bins, frames, num_threads)
    return [ClusterResult.from_py_ptr(r) for r in res_py]


def cluster_results_to_pandas_df(results: List[ClusterResult]) -> pd.DataFrame:
    """Flatten ClusterResult into a tidy DataFrame (uses Rust-side to_dict for speed/consistency)."""
    return pd.DataFrame([r.to_dict() for r in results])


# -------------------- (Optional) Quick sanity visual helpers --------------------
def plot_cluster_patch(result: ClusterResult, ax=None, cmap="viridis"):
    import matplotlib.pyplot as plt
    p = result.patch
    mat = p.patch()
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title("Cluster patch (rows=frames, cols=scans)")
    ax.set_xlabel("scan")
    ax.set_ylabel("frame")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax

def plot_traces(result: ClusterResult, ax_rt=None, ax_im=None):
    import matplotlib.pyplot as plt
    p = result.patch
    if ax_rt is None or ax_im is None:
        fig, (ax_rt, ax_im) = plt.subplots(1, 2, figsize=(10, 3))
    ax_rt.plot(p.rt_trace()); ax_rt.set_title("RT trace (sum over scans)")
    ax_im.plot(p.im_trace()); ax_im.set_title("IM trace (sum over frames)")
    return ax_rt, ax_im

class ClusterCloud(RustWrapperObject):
    """
    Python wrapper around Rust `PyClusterCloud`.

    Exposes read-only metadata (rt_left/right, scan_left/right) and
    returns numpy arrays for frame_ids, scans, tofs, intensities.
    """
    def __init__(
        self,
        rt_left: int,
        rt_right: int,
        scan_left: int,
        scan_right: int,
        frame_ids: "np.ndarray[np.uint32]",
        scans: "np.ndarray[np.uint32]",
        tofs: "np.ndarray[np.uint32]",
        intensities: "np.ndarray[np.float32]",
    ):
        # In practice you won't construct from Python; keeping parity with your pattern.
        # This ctor is only here for symmetry; prefer `from_py_ptr`.
        self.__py_ptr = ims.PyClusterCloud()  # not actually used; real instances come from Rust

    @classmethod
    def from_py_ptr(cls, cloud_py: ims.PyClusterCloud) -> "ClusterCloud":
        inst = cls.__new__(cls)
        inst.__py_ptr = cloud_py
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    # ---- scalar metadata ----
    @property
    def rt_left(self) -> int:
        return self.__py_ptr.rt_left

    @property
    def rt_right(self) -> int:
        return self.__py_ptr.rt_right

    @property
    def scan_left(self) -> int:
        return self.__py_ptr.scan_left

    @property
    def scan_right(self) -> int:
        return self.__py_ptr.scan_right

    # ---- vector accessors (as numpy arrays) ----
    @property
    def frame_ids(self) -> np.ndarray:
        # Rust returns a PyArray1[u32]; np.asarray gives a view/copy as needed.
        return np.asarray(self.__py_ptr.frame_ids(), dtype=np.uint32)

    @property
    def scans(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.scans(), dtype=np.uint32)

    @property
    def tofs(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.tofs(), dtype=np.uint32)

    @property
    def intensities(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.intensities(), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.intensities.size)

    def __repr__(self):
        return (f"ClusterCloud(rt=[{self.rt_left},{self.rt_right}], "
                f"scan=[{self.scan_left},{self.scan_right}], "
                f"points={len(self)})")

    # --- convenience: quick 2D rasterization (RT col × Scan) ---
    def rasterize(self, rt_frame_ids: np.ndarray) -> np.ndarray:
        """
        Make a dense RT×Scan image (sum of intensities per (frame, scan)).
        Returns array shape (scan_right-scan_left+1, rt_right-rt_left+1).
        """
        import numpy as _np
        rt_left, rt_right = self.rt_left, self.rt_right
        sc_left, sc_right = self.scan_left, self.scan_right
        nx = rt_right - rt_left + 1
        ny = sc_right - sc_left + 1

        # map frame_id -> column
        idx = {int(fid): i for i, fid in enumerate(_np.asarray(rt_frame_ids, dtype=_np.uint32))}
        cols = _np.fromiter((idx.get(int(fid), -1) for fid in self.frame_ids), count=len(self), dtype=_np.int32)
        mask = (cols >= rt_left) & (cols <= rt_right)
        cols = cols[mask]
        scans = self.scans[mask]
        ints = self.intensities[mask]

        # shift into local window coordinates
        cols = cols - rt_left
        scans = scans - sc_left

        img = _np.zeros((ny, nx), dtype=_np.float32)
        # guard against out-of-range due to any mapping mismatch
        valid = (cols >= 0) & (cols < nx) & (scans >= 0) & (scans < ny)
        _np.add.at(img, (scans[valid], cols[valid]), ints[valid])
        return img
