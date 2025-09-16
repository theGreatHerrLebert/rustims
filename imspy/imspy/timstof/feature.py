# imspy/feature.py
from __future__ import annotations
from typing import List, Sequence, Optional, Dict, Any
import numpy as np
import pandas as pd

import imspy_connector
ims_feat = imspy_connector.py_feature  # PyAveragineLut, PyBuildOpts, PyFeature
from imspy.simulation.annotation import RustWrapperObject

class AveragineLut(RustWrapperObject):
    """Thin wrapper over Rust PyAveragineLut."""
    def __init__(
        self,
        mass_min: float,
        mass_max: float,
        step: float,
        z_min: int,
        z_max: int,
        k: int = 6,
        resolution: int = 3,
        num_threads: int = 4,
    ) -> None:
        self.__py_ptr = ims_feat.PyAveragineLut(
            float(mass_min), float(mass_max), float(step),
            int(z_min), int(z_max), int(k), int(resolution), int(num_threads)
        )

    @classmethod
    def from_py_ptr(cls, p) -> "AveragineLut":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def masses(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.masses, dtype=np.float32)

    @property
    def z_min(self) -> int: return self.__py_ptr.z_min
    @property
    def z_max(self) -> int: return self.__py_ptr.z_max
    @property
    def k(self) -> int: return self.__py_ptr.k

    def lookup(self, neutral_mass: float, z: int) -> np.ndarray:
        return np.asarray(self.__py_ptr.lookup(float(neutral_mass), int(z)), dtype=np.float32)

    def __repr__(self) -> str:
        return f"AveragineLut(grid={len(self.masses)}, z=[{self.z_min}..{self.z_max}], k={self.k})"


class BuildOpts(RustWrapperObject):
    """Thin wrapper over Rust PyBuildOpts."""
    def __init__(
        self,
        ppm_narrow: float = 10.0,
        k_max: int = 6,
        min_raw_sum: float = 0.0,
        num_threads: int = 4,
        charge_hist_ppm_window: Optional[float] = None,
        charge_hist_bins: Optional[int] = None,
    ) -> None:
        self.__py_ptr = ims_feat.PyBuildOpts(
            float(ppm_narrow), int(k_max), float(min_raw_sum), int(num_threads),
            charge_hist_ppm_window, charge_hist_bins
        )

    @classmethod
    def from_py_ptr(cls, p) -> "BuildOpts":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def ppm_narrow(self) -> float: return self.__py_ptr.ppm_narrow
    @property
    def k_max(self) -> int: return self.__py_ptr.k_max
    @property
    def min_raw_sum(self) -> float: return self.__py_ptr.min_raw_sum
    @property
    def num_threads(self) -> int: return self.__py_ptr.num_threads
    @property
    def charge_hist_ppm_window(self) -> Optional[float]: return self.__py_ptr.charge_hist_ppm_window
    @property
    def charge_hist_bins(self) -> Optional[int]: return self.__py_ptr.charge_hist_bins

    def __repr__(self) -> str:
        return (f"BuildOpts(ppm_narrow={self.ppm_narrow}, k_max={self.k_max}, "
                f"min_raw_sum={self.min_raw_sum}, threads={self.num_threads}, "
                f"hist_ppm={self.charge_hist_ppm_window}, hist_bins={self.charge_hist_bins})")


class Feature(RustWrapperObject):
    """Thin wrapper over Rust PyFeature (read-only)."""
    def __init__(self, *a, **k):
        raise RuntimeError("Features are constructed in Rust; use Feature.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: "ims_feat.PyFeature") -> "Feature":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    # scalar fields
    @property
    def rt_mu(self) -> float: return self.__py_ptr.rt_mu
    @property
    def rt_sigma(self) -> float: return self.__py_ptr.rt_sigma
    @property
    def im_mu(self) -> float: return self.__py_ptr.im_mu
    @property
    def im_sigma(self) -> float: return self.__py_ptr.im_sigma
    @property
    def mz_mono(self) -> float: return self.__py_ptr.mz_mono
    @property
    def z(self) -> int: return self.__py_ptr.z
    @property
    def avg_score(self) -> float: return self.__py_ptr.avg_score
    @property
    def z_conf(self) -> float: return self.__py_ptr.z_conf
    @property
    def raw_sum(self) -> float: return self.__py_ptr.raw_sum
    @property
    def fit_volume(self) -> float: return self.__py_ptr.fit_volume
    @property
    def source_cluster_id(self) -> int: return self.__py_ptr.source_cluster_id

    # isotopic vector
    @property
    def iso_i(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.iso_i, dtype=np.float32)

    def __repr__(self) -> str:
        return (f"Feature(rt=μ{self.rt_mu:.2f} σ{self.rt_sigma:.2f}, "
                f"im=μ{self.im_mu:.2f} σ{self.im_sigma:.2f}, "
                f"mz_mono={self.mz_mono:.5f}, z={self.z}, "
                f"avg={self.avg_score:.3f}, raw={self.raw_sum:.1f})")


def features_to_dataframe(feats: Sequence[Feature]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, f in enumerate(feats):
        iso = f.iso_i
        row = dict(
            feature_id=i,
            rt_mu=f.rt_mu, rt_sigma=f.rt_sigma,
            im_mu=f.im_mu, im_sigma=f.im_sigma,
            mz_mono=f.mz_mono, z=f.z,
            avg_score=f.avg_score, z_conf=f.z_conf,
            raw_sum=f.raw_sum, fit_volume=f.fit_volume,
            source_cluster_id=f.source_cluster_id,
        )
        # expand iso_i (up to 8)
        for k in range(min(8, len(iso))):
            row[f"iso{k}"] = float(iso[k])
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["log1p_raw_sum"] = np.log1p(df["raw_sum"])
        df["log1p_fit_volume"] = np.log1p(df["fit_volume"])
    return df