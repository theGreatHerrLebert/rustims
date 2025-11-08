# imspy/feature.py
from __future__ import annotations
from typing import Optional, Sequence

import imspy_connector
imsf = imspy_connector.py_feature  # <- the Rust PyO3 module you compiled

# If you already have a PyClusterResult1D wrapper in Python, you can accept that.
# Otherwise we assume you pass the raw PyO3 instances (from imspy_connector).

# ---------------------------
# Averagine LUT (ergonomic)
# ---------------------------
class AveragineLut:
    def __init__(
        self,
        mass_min: float = 200.0,
        mass_max: float = 6000.0,
        step: float = 25.0,
        z_min: int = 1,
        z_max: int = 6,
        k: int = 6,
        resolution: int = 60000,
        num_threads: Optional[int] = None,
    ):
        if num_threads is None:
            # Let Rust decide based on available_parallelism
            # We don’t pass it; instead use the default_grid as a shortcut when exact defaults.
            if (
                mass_min, mass_max, step, z_min, z_max, k, resolution
            ) == (200.0, 6000.0, 25.0, 1, 6, 6, 60000):
                self.__py_ptr = imsf.PyAveragineLut.default_grid()
            else:
                # Fall back to 1 thread if not specified; feel free to wire a CPython
                # CPU count if you want it dynamic.
                self.__py_ptr = imsf.PyAveragineLut(
                    mass_min, mass_max, step, z_min, z_max, k, resolution, 1
                )
        else:
            self.__py_ptr = imsf.PyAveragineLut(
                mass_min, mass_max, step, z_min, z_max, k, resolution, int(num_threads)
            )

    @classmethod
    def from_py_ptr(cls, py_ptr):
        self = cls.__new__(cls)
        self.__py_ptr = py_ptr
        return self

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def z_min(self) -> int:
        return self.__py_ptr.z_min

    @property
    def z_max(self) -> int:
        return self.__py_ptr.z_max

    @property
    def k(self) -> int:
        return self.__py_ptr.k

    def lookup(self, neutral_mass: float, z: int):
        return self.__py_ptr.lookup(neutral_mass, z)

    def __repr__(self) -> str:
        return f"AveragineLut(z={self.z_min}..{self.z_max}, k={self.k})"


# ---------------------------
# Params
# ---------------------------
class GroupingParams:
    def __init__(
        self,
        rt_pad_overlap: int,
        im_pad_overlap: int,
        mz_ppm_tol: float,
        iso_ppm_tol: float,
        iso_abs_da: float,
        z_min: int,
        z_max: int,
    ):
        self.__py_ptr = imsf.PyGroupingParams(
            rt_pad_overlap, im_pad_overlap, mz_ppm_tol, iso_ppm_tol, iso_abs_da, z_min, z_max
        )

    @classmethod
    def from_py_ptr(cls, py_ptr):
        self = cls.__new__(cls)
        self.__py_ptr = py_ptr
        return self

    def get_py_ptr(self):
        return self.__py_ptr

    def __repr__(self) -> str:
        return (
            f"GroupingParams(rt_pad={self.__py_ptr.inner.rt_pad_overlap}, "
            f"im_pad={self.__py_ptr.inner.im_pad_overlap}, "
            f"mz_ppm_tol={self.__py_ptr.inner.mz_ppm_tol}, "
            f"iso_ppm_tol={self.__py_ptr.inner.iso_ppm_tol}, "
            f"iso_abs_da={self.__py_ptr.inner.iso_abs_da}, "
            f"z={self.__py_ptr.inner.z_min}..{self.__py_ptr.inner.z_max})"
        )


class FeatureBuildParams:
    def __init__(
        self,
        k_max: int,
        min_members: int,
        min_cosine: float,
        w_spacing: float,
        w_coelute: float,
        w_monotonic: float,
        penalty_skip_one: float,
        steal_delta: float,
        require_lowest_is_mono: bool,
    ):
        self.__py_ptr = imsf.PyFeatureBuildParams(
            k_max,
            min_members,
            min_cosine,
            w_spacing,
            w_coelute,
            w_monotonic,
            penalty_skip_one,
            steal_delta,
            require_lowest_is_mono,
        )

    @classmethod
    def from_py_ptr(cls, py_ptr):
        self = cls.__new__(cls)
        self.__py_ptr = py_ptr
        return self

    def get_py_ptr(self):
        return self.__py_ptr

    def __repr__(self) -> str:
        return (
            f"FeatureBuildParams(k_max={self.__py_ptr.inner.k_max}, "
            f"min_members={self.__py_ptr.inner.min_members}, "
            f"min_cosine={self.__py_ptr.inner.min_cosine})"
        )


# ---------------------------
# Outputs
# ---------------------------
class Feature:
    def __init__(self, py_ptr):
        self.__py_ptr = py_ptr

    @classmethod
    def from_py_ptr(cls, py_ptr):
        return cls(py_ptr)

    def get_py_ptr(self):
        return self.__py_ptr

    @property
    def envelope_id(self) -> int:
        return self.__py_ptr.envelope_id

    @property
    def charge(self) -> int:
        return self.__py_ptr.charge

    @property
    def mz_mono(self) -> float:
        return self.__py_ptr.mz_mono

    @property
    def neutral_mass(self) -> float:
        return self.__py_ptr.neutral_mass

    @property
    def rt_bounds(self) -> tuple[int, int]:
        return self.__py_ptr.rt_bounds

    @property
    def im_bounds(self) -> tuple[int, int]:
        return self.__py_ptr.im_bounds

    @property
    def mz_center(self) -> float:
        return self.__py_ptr.mz_center

    @property
    def n_members(self) -> int:
        return self.__py_ptr.n_members

    @property
    def member_cluster_ids(self) -> list[int]:
        return self.__py_ptr.member_cluster_ids

    @property
    def iso_raw(self):
        return self.__py_ptr.iso_raw  # numpy array

    @property
    def iso_l2(self):
        return self.__py_ptr.iso_l2  # numpy array

    @property
    def cos_averagine(self) -> float:
        return self.__py_ptr.cos_averagine

    def __repr__(self) -> str:
        return (
            f"Feature(eid={self.envelope_id}, z={self.charge}, "
            f"mono={self.mz_mono:.4f}, members={self.n_members})"
        )


class GroupingOutput:
    def __init__(self, py_ptr):
        self.__py_ptr = py_ptr

    @property
    def envelopes(self) -> list[tuple[int, tuple[int, int], tuple[int, int], float, Optional[int]]]:
        return self.__py_ptr.envelopes

    @property
    def assignment(self) -> list[Optional[int]]:
        return self.__py_ptr.assignment

    def __repr__(self) -> str:
        return f"GroupingOutput(n_env={len(self.envelopes)})"


class BuildResult:
    def __init__(self, py_ptr):
        self.__py_ptr = py_ptr

    @property
    def features(self) -> list[Feature]:
        return [Feature.from_py_ptr(f) for f in self.__py_ptr.features]

    @property
    def grouping(self) -> GroupingOutput:
        return GroupingOutput(self.__py_ptr.grouping)

    def __repr__(self) -> str:
        return f"BuildResult(n_features={len(self.features)})"


# ---------------------------
# Top-level convenience API
# ---------------------------
def build_features_from_clusters(
    clusters: Sequence,  # list of PyClusterResult1D wrappers (your Python side) or raw PyO3 objects
    gp: GroupingParams,
    fp: FeatureBuildParams,
    lut: Optional[AveragineLut] = None,
) -> BuildResult:
    """
    clusters: e.g. [py_cluster1, py_cluster2, ...] where each element is the Python wrapper
              around your PyO3 `PyClusterResult1D`. Both wrapper and raw PyO3 object work,
              as long as they expose the underlying `PyClusterResult1D` to Rust.

    gp, fp: parameter wrappers defined above.
    lut: optional AveragineLut wrapper for cosine gating.
    """
    # unwrap to raw PyO3 objects if the caller passed Python-side wrappers
    def to_py_ptr(x):
        return getattr(x, "get_py_ptr", lambda: x)()

    raw_list = [to_py_ptr(c) for c in clusters]

    res = imsf.build_features_from_clusters_py(
        raw_list,
        gp.get_py_ptr(),
        fp.get_py_ptr(),
        None if lut is None else lut.get_py_ptr(),
    )
    return BuildResult(res)