
from __future__ import annotations
from typing import Sequence, Optional
import numpy as np

import imspy_connector
ims = imspy_connector.py_feature
from imspy.simulation.annotation import RustWrapperObject


class FeatureBuildParams(RustWrapperObject):
    def __init__(
            self,
            ppm_narrow: float,
            k_max: int,
            min_cosine: float,
            min_members: int,
            max_points_per_slice: int = 0,
            min_hist_conf: float = 0.0,
            allow_unknown_charge: bool = True,
            recover_missing: bool = False,
            recover_ppm: Optional[float] = None,
            min_iso_abs: float = 0.0,
            min_iso_frac_of_sum: float = 0.0):
        self.__py_ptr = ims.PyFeatureBuildParams(ppm_narrow, k_max, min_cosine,
                                                 min_members, max_points_per_slice,
                                                 min_hist_conf, allow_unknown_charge,
                                                    recover_missing, recover_ppm,
                                                    min_iso_abs, min_iso_frac_of_sum)

    @property
    def ppm_narrow(self) -> float:   return self.__py_ptr.ppm_narrow
    @property
    def k_max(self) -> int:          return self.__py_ptr.k_max
    @property
    def min_cosine(self) -> float:   return self.__py_ptr.min_cosine
    @property
    def min_members(self) -> int:    return self.__py_ptr.min_members
    @property
    def max_points_per_slice(self) -> int: return self.__py_ptr.max_points_per_slice
    @property
    def min_hist_conf(self) -> float: return self.__py_ptr.min_hist_conf
    @property
    def allow_unknown_charge(self) -> bool: return self.__py_ptr.allow_unknown_charge
    @property
    def recover_missing(self) -> bool: return self.__py_ptr.recover_missing
    @property
    def recover_ppm(self) -> Optional[float]: return self.__py_ptr.recover_ppm
    @property
    def min_iso_abs(self) -> float: return self.__py_ptr.min_iso_abs
    @property
    def min_iso_frac_of_sum(self) -> float: return self.__py_ptr.min_iso_frac_of_sum

    def get_py_ptr(self) -> "ims.PyFeatureBuildParams": return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, obj: "ims.PyFeatureBuildParams") -> "FeatureBuildParams":
        inst = cls.__new__(cls)
        inst.__py_ptr = obj
        return inst

    def __repr__(self) -> str: return repr(self.__py_ptr)


class Feature(RustWrapperObject):
    def __init__(self, *a, **k):
        raise RuntimeError("Feature is created in Rust; use Feature.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyFeature") -> "Feature":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self) -> "ims.PyFeature": return self.__py_ptr

    @property
    def envelope_id(self) -> int: return self.__py_ptr.envelope_id
    @property
    def charge(self) -> int:      return self.__py_ptr.charge
    @property
    def mz_mono(self) -> float:   return float(self.__py_ptr.mz_mono)
    @property
    def rt_bounds(self) -> tuple[int,int]: return (self.__py_ptr.rt_left, self.__py_ptr.rt_right)
    @property
    def im_bounds(self) -> tuple[int,int]: return (self.__py_ptr.im_left, self.__py_ptr.im_right)
    @property
    def cosine(self) -> float:    return float(self.__py_ptr.cosine)
    @property
    def n_members(self) -> int:   return self.__py_ptr.n_members
    @property
    def cluster_ids(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.cluster_ids, dtype=np.int64)
    @property
    def repr_cluster_id(self) -> int: return self.__py_ptr.repr_cluster_id
    @property
    def iso(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.iso, dtype=np.float32)
    @property
    def present_mask(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.present_mask, dtype=np.uint8)
    @property
    def k_detected(self) -> int: return self.__py_ptr.k_detected

    def __repr__(self) -> str: return repr(self.__py_ptr)

    def to_dict(self) -> dict:
        """Flat, DataFrame-friendly representation."""
        return {
            "envelope_id": int(self.envelope_id),
            "charge": int(self.charge),
            "mz_mono": float(self.mz_mono),
            "neutral_mass": float(getattr(self, "neutral_mass", float("nan"))),
            "mz_center": float(getattr(self, "mz_center", float("nan"))),
            "rt_left": int(self.rt_bounds[0]),
            "rt_right": int(self.rt_bounds[1]),
            "im_left": int(self.im_bounds[0]),
            "im_right": int(self.im_bounds[1]),
            "cosine": float(self.cosine),
            "n_members": int(self.n_members),
            "raw_sum": float(getattr(self, "raw_sum", float("nan"))),
            "repr_cluster_id": int(self.repr_cluster_id),
            # arrays as plain Python lists for easy JSON/DF serialization
            "cluster_ids": [int(x) for x in self.cluster_ids.tolist()],
            "iso": [float(x) for x in getattr(self, "iso", np.asarray([], np.float32)).tolist()],
            "present_mask": [int(x) for x in self.present_mask.tolist()],
            "k_detected": int(self.k_detected),
        }


class Envelope(RustWrapperObject):
    def __init__(self, *a, **k):
        raise RuntimeError("Envelope is created in Rust; use Envelope.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyEnvelope") -> "Envelope":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self) -> "ims.PyEnvelope": return self.__py_ptr

    @property
    def id(self) -> int: return self.__py_ptr.id
    @property
    def cluster_ids(self) -> np.ndarray:
        return np.asarray(self.__py_ptr.cluster_ids, dtype=np.int64)
    @property
    def rt_bounds(self) -> tuple[int,int]: return tuple(self.__py_ptr.rt_bounds)
    @property
    def im_bounds(self) -> tuple[int,int]: return tuple(self.__py_ptr.im_bounds)
    @property
    def mz_center(self) -> float: return float(self.__py_ptr.mz_center)
    @property
    def mz_span_da(self) -> float: return float(self.__py_ptr.mz_span_da)
    @property
    def charge_hint(self) -> Optional[int]:
        return self.__py_ptr.charge_hint

    def __repr__(self) -> str: return repr(self.__py_ptr)

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "cluster_ids": [int(x) for x in self.cluster_ids.tolist()],
            "n_members": int(len(self.cluster_ids)),
            "rt_left": int(self.rt_bounds[0]),
            "rt_right": int(self.rt_bounds[1]),
            "im_left": int(self.im_bounds[0]),
            "im_right": int(self.im_bounds[1]),
            "mz_center": float(self.mz_center),
            "mz_span_da": float(self.mz_span_da),
            "charge_hint": (None if self.charge_hint is None else int(self.charge_hint)),
        }


class GroupingOutput(RustWrapperObject):
    def __init__(self, *a, **k):
        raise RuntimeError("Use group_clusters_into_envelopes(...).")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyGroupingOutput") -> "GroupingOutput":
        inst = cls.__new__(cls)
        inst.__py_ptr = p
        return inst

    def get_py_ptr(self) -> "ims.PyGroupingOutput": return self.__py_ptr

    @property
    def envelopes(self) -> list[Envelope]:
        return [Envelope.from_py_ptr(e) for e in self.__py_ptr.envelopes]

    @property
    def assignment(self) -> list[Optional[int]]:
        return list(self.__py_ptr.assignment)

    @property
    def provisional(self) -> list[list[int]]:
        return [list(x) for x in self.__py_ptr.provisional]

    def __repr__(self) -> str: return repr(self.__py_ptr)

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
        iso_abs_da: float = 0.03
    ):
        self.__py_ptr = ims.PyGroupingParams(
            int(rt_pad_overlap),
            int(im_pad_overlap),
            float(mz_ppm_tol),
            float(iso_ppm_tol),
            int(z_min),
            int(z_max),
            float(iso_abs_da)
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

def build_features_from_envelopes(
    frames: Sequence["TimsFrame"],    # your existing frame wrapper
    envelopes: Sequence[Envelope],
    clusters: Sequence["ClusterResult"],
    lut: AveragineLut,
    gp: GroupingParams,
    fp: FeatureBuildParams,
) -> list[Feature]:
    frames_py = [f.get_py_ptr() for f in frames]
    envs_py   = [e.get_py_ptr() for e in envelopes]
    cl_py     = [c.get_py_ptr() for c in clusters]
    feats_py = ims.build_features_from_envelopes_py(
        frames_py, envs_py, cl_py, lut.get_py_ptr(), gp.get_py_ptr(), fp.get_py_ptr()
    )
    return [Feature.from_py_ptr(f) for f in feats_py]

def integrate_isotope_series(
    frames, rt_bounds, im_bounds, mz_mono, z, ppm_narrow, k_max, max_points_per_slice=0
) -> np.ndarray:
    frames_py = [f.get_py_ptr() for f in frames]
    v = ims.integrate_isotope_series_py(
        frames_py, rt_bounds, im_bounds, float(mz_mono), int(z),
        float(ppm_narrow), int(k_max), int(max_points_per_slice)
    )
    return np.asarray(v, dtype=np.float32)

def build_local_mz_histogram(frames, rt_bounds, im_bounds, mz_center, win_ppm, bins):
    frames_py = [f.get_py_ptr() for f in frames]
    axis, y = ims.build_local_mz_histogram_py(
        frames_py, rt_bounds, im_bounds, float(mz_center), float(win_ppm), int(bins)
    )
    return np.asarray(axis, dtype=np.float32), np.asarray(y, dtype=np.float32)

def estimate_charge_from_hist(mz_axis: np.ndarray, mz_hist: np.ndarray):
    return ims.estimate_charge_from_hist_py(list(map(float, mz_axis)), list(map(float, mz_hist)))

