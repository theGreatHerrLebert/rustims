
from __future__ import annotations
from typing import List, Sequence, Tuple, Optional
import numpy as np

import imspy_connector
ims = imspy_connector.py_feature
from imspy.simulation.annotation import RustWrapperObject

class FeatureBuildParams(RustWrapperObject):
    def __init__(self, ppm_narrow: float, k_max: int, min_cosine: float, min_members: int):
        self.__py_ptr = ims.PyFeatureBuildParams(ppm_narrow, k_max, min_cosine, min_members)

    @property
    def ppm_narrow(self) -> float:   return self.__py_ptr.ppm_narrow
    @property
    def k_max(self) -> int:          return self.__py_ptr.k_max
    @property
    def min_cosine(self) -> float:   return self.__py_ptr.min_cosine
    @property
    def min_members(self) -> int:    return self.__py_ptr.min_members

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

    def __repr__(self) -> str: return repr(self.__py_ptr)

    def __repr__(self) -> str: return repr(self.__py_ptr)

class AveragineLut(RustWrapperObject):
    def __init__(self, mass_min: float, mass_max: float, step: float,
                 z_min: int, z_max: int, k: int = 6, resolution: int = 3, num_threads: int = 4):
        self.__py_ptr = ims.PyAveragineLut(mass_min, mass_max, step, z_min, z_max, k, resolution, num_threads)

    @property
    def masses(self) -> np.ndarray: return np.asarray(self.__py_ptr.masses, dtype=np.float32)
    @property
    def z_min(self) -> int:         return self.__py_ptr.z_min
    @property
    def z_max(self) -> int:         return self.__py_ptr.z_max
    @property
    def k(self)    -> int:          return self.__py_ptr.k

    def lookup(self, neutral_mass: float, z: int) -> np.ndarray:
        return np.asarray(self.__py_ptr.lookup(neutral_mass, z), dtype=np.float32)

    def __repr__(self) -> str: return repr(self.__py_ptr)

class Envelope(RustWrapperObject):
    def __init__(self, *a, **k):
        raise RuntimeError("Envelope is created in Rust; use Envelope.from_py_ptr().")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyEnvelope") -> "Envelope":
        inst = cls.__new__(cls)
        inst._RustWrapperObject__py_ptr = p
        return inst

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
    def mz_center(self) -> float: return float(self.__py_ptr.mz_center)
    @property
    def mz_span_da(self) -> float: return float(self.__py_ptr.mz_span_da)
    @property
    def charge_hint(self) -> Optional[int]:
        # Option[u8] maps to None or int in Python already:
        return self.__py_ptr.charge_hint

    def __repr__(self) -> str: return repr(self.__py_ptr)

class GroupingOutput(RustWrapperObject):
    def __init__(self, *a, **k):
        raise RuntimeError("GroupingOutput is created in Rust; use from_py_ptr.")

    @classmethod
    def from_py_ptr(cls, p: "ims.PyGroupingOutput") -> "GroupingOutput":
        inst = cls.__new__(cls)
        inst._RustWrapperObject__py_ptr = p
        return inst

    @property
    def envelopes(self) -> List[Envelope]:
        return [Envelope.from_py_ptr(e) for e in self.__py_ptr.envelopes]
    @property
    def assignment(self) -> np.ndarray:
        # Vec<Option<usize>> shows up as list[Optional[int]]
        # Convert to object dtype to keep None values
        return np.asarray(self.__py_ptr.assignment, dtype=object)
    @property
    def provisional(self) -> List[np.ndarray]:
        return [np.asarray(v, dtype=np.int64) for v in self.__py_ptr.provisional]

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

