# imspy/feature.py
from __future__ import annotations
from typing import List, Sequence, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

import imspy_connector
ims = imspy_connector.py_feature  # PyAveragineLut, PyFeature
from imspy.simulation.annotation import RustWrapperObject


class GroupingParams(RustWrapperObject):
    def __init__(
        self,
        *,
        rt_pad_overlap: int,
        im_pad_overlap: int,
        mz_ppm_tol: float,
        iso_ppm_tol: float,
        z_min: int,
        z_max: int,
    ):
        self.__py_ptr = ims.PyGroupingParams(
            int(rt_pad_overlap),
            int(im_pad_overlap),
            float(mz_ppm_tol),
            float(iso_ppm_tol),
            int(z_min),
            int(z_max),
        )

    @classmethod
    def from_py_ptr(cls, p: "ims.PyGroupingParams") -> "GroupingParams":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    def __repr__(self) -> str:
        return repr(self.__py_ptr)


class Envelope(RustWrapperObject):
    def __init__(self, *a, **k):
        raise RuntimeError("Envelope is created in Rust; use Envelope.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyEnvelope") -> "Envelope":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def id(self) -> int: return self.__py_ptr.id
    @property
    def cluster_ids(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.cluster_ids, dtype=np.int64)
    @property
    def rt_bounds(self) -> Tuple[int,int]: return tuple(self.__py_ptr.rt_bounds)
    @property
    def im_bounds(self) -> Tuple[int,int]: return tuple(self.__py_ptr.im_bounds)
    @property
    def mz_center(self) -> float: return self.__py_ptr.mz_center
    @property
    def mz_span_da(self) -> float: return self.__py_ptr.mz_span_da
    @property
    def charge_hint(self) -> Optional[int]:
        z = self.__py_ptr.charge_hint
        return z if z >= 0 else None

    def __repr__(self) -> str:
        return repr(self.__py_ptr)


class GroupingOutput(RustWrapperObject):
    def __init__(self, *a, **k):
        raise RuntimeError("Use group_clusters_into_envelopes(...).")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyGroupingOutput") -> "GroupingOutput":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def envelopes(self) -> List[Envelope]:
        return [Envelope.from_py_ptr(e) for e in self.__py_ptr.envelopes]

    @property
    def assignment(self) -> List[Optional[int]]:
        return list(self.__py_ptr.assignment)

    @property
    def provisional(self) -> List[List[int]]:
        return [list(x) for x in self.__py_ptr.provisional]

    def __repr__(self) -> str:
        return repr(self.__py_ptr)


def group_clusters_into_envelopes(
    clusters: Sequence["ClusterResult"],
    params: GroupingParams,
) -> GroupingOutput:
    clusters_py = [c.get_py_ptr() for c in clusters]
    out_py = ims.group_clusters_into_envelopes_py(clusters_py, params.get_py_ptr())
    return GroupingOutput.from_py_ptr(out_py)

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
        self.__py_ptr = ims.PyAveragineLut(
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


class Feature(RustWrapperObject):
    """Thin wrapper over Rust PyFeature (read-only)."""
    def __init__(self, *a, **k):
        raise RuntimeError("Features are constructed in Rust; use Feature.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyFeature") -> "Feature":
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