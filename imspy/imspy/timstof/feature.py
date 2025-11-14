# imspy/feature.py
from __future__ import annotations

from typing import Optional, Sequence, Any

import imspy_connector

imsf = imspy_connector.py_feature  # Rust PyO3 module

class AveragineLut:
    """
    Thin Python wrapper around PyAveragineLut.

    Currently not used by `build_simple_features_from_clusters`, but we keep it
    for future scoring / diagnostics (e.g. cosine vs averagine).
    """

    def __init__(
        self,
        mass_min: float = 150.0,
        mass_max: float = 6000.0,
        step: float = 25.0,
        z_min: int = 1,
        z_max: int = 6,
        k: int = 6,
        resolution: int = 1,
        num_threads: Optional[int] = 4,
    ) -> None:
        if num_threads is None:
            num_threads = 4

        self.__py_ptr = imsf.PyAveragineLut(
            float(mass_min),
            float(mass_max),
            float(step),
            int(z_min),
            int(z_max),
            int(k),
            int(resolution),
            int(num_threads),
        )

    @classmethod
    def from_py_ptr(cls, py_ptr: Any) -> "AveragineLut":
        self = cls.__new__(cls)
        self.__py_ptr = py_ptr
        return self

    def get_py_ptr(self) -> Any:
        return self.__py_ptr

    @property
    def z_min(self) -> int:
        return int(self.__py_ptr.z_min)

    @property
    def z_max(self) -> int:
        return int(self.__py_ptr.z_max)

    @property
    def k(self) -> int:
        return int(self.__py_ptr.k)

    def lookup(self, neutral_mass: float, z: int):
        """
        Returns a numpy array (len=8) with the averagine envelope (L2-normalized).
        """
        return self.__py_ptr.lookup(float(neutral_mass), int(z))

    def __repr__(self) -> str:
        return repr(self.__py_ptr)


# -----------------------------------------------------------------------------
# SimpleFeatureParams – controls the greedy isotopic grouping
# -----------------------------------------------------------------------------


class SimpleFeatureParams:
    """
    Python wrapper around PySimpleFeatureParams.

    Parameters mirror the Rust struct:

        z_min, z_max: allowed charge range
        iso_ppm_tol: ppm tolerance for isotopic spacing
        iso_abs_da: absolute m/z tolerance (safety floor)
        min_members: minimal number of isotopes in a chain
        max_members: maximal number of isotopes in a chain
        min_raw_sum: minimal cluster raw_sum
        min_mz: minimal m/z to consider
        min_rt_overlap_frac: minimal fractional RT overlap between neighboring isotopes
        min_im_overlap_frac: minimal fractional IM overlap between neighboring isotopes
    """

    def __init__(
        self,
        z_min: int = 1,
        z_max: int = 5,
        iso_ppm_tol: float = 10.0,
        iso_abs_da: float = 0.003,
        min_members: int = 2,
        max_members: int = 5,
        min_raw_sum: float = 0.0,
        min_mz: float = 100.0,
        min_rt_overlap_frac: float = 0.3,
        min_im_overlap_frac: float = 0.3,
    ) -> None:
        self.__py_ptr = imsf.PySimpleFeatureParams(
            int(z_min),
            int(z_max),
            float(iso_ppm_tol),
            float(iso_abs_da),
            int(min_members),
            int(max_members),
            float(min_raw_sum),
            float(min_mz),
            float(min_rt_overlap_frac),
            float(min_im_overlap_frac),
        )

    @classmethod
    def default(cls) -> "SimpleFeatureParams":
        """
        Convenience: Rust-side Default.
        """
        return cls.from_py_ptr(imsf.PySimpleFeatureParams.default())

    @classmethod
    def from_py_ptr(cls, py_ptr: Any) -> "SimpleFeatureParams":
        self = cls.__new__(cls)
        self.__py_ptr = py_ptr
        return self

    def get_py_ptr(self) -> Any:
        return self.__py_ptr

    def __repr__(self) -> str:
        return repr(self.__py_ptr)


# -----------------------------------------------------------------------------
# SimpleFeature – output of the simple builder
# -----------------------------------------------------------------------------


class SimpleFeature:
    """
    Thin wrapper around the Rust `SimpleFeature`.

    We deliberately expose only the things you actually care about for downstream:
    - feature_id, charge, mz_mono, neutral_mass
    - rt_bounds, im_bounds
    - mz_center, n_members, member_cluster_ids
    - raw_sum
    """

    def __init__(self, py_ptr: Any) -> None:
        self.__py_ptr = py_ptr

    @classmethod
    def from_py_ptr(cls, py_ptr: Any) -> "SimpleFeature":
        return cls(py_ptr)

    def get_py_ptr(self) -> Any:
        return self.__py_ptr

    @property
    def feature_id(self) -> int:
        return int(self.__py_ptr.feature_id)

    @property
    def charge(self) -> int:
        return int(self.__py_ptr.charge)

    @property
    def mz_mono(self) -> float:
        return float(self.__py_ptr.mz_mono)

    @property
    def neutral_mass(self) -> float:
        return float(self.__py_ptr.neutral_mass)

    @property
    def rt_bounds(self) -> tuple[int, int]:
        return tuple(self.__py_ptr.rt_bounds)

    @property
    def im_bounds(self) -> tuple[int, int]:
        return tuple(self.__py_ptr.im_bounds)

    @property
    def mz_center(self) -> float:
        return float(self.__py_ptr.mz_center)

    @property
    def n_members(self) -> int:
        return int(self.__py_ptr.n_members)

    @property
    def member_cluster_ids(self) -> list[int]:
        return list(self.__py_ptr.member_cluster_ids)

    @property
    def raw_sum(self) -> float:
        return float(self.__py_ptr.raw_sum)

    def __repr__(self) -> str:
        return (
            f"SimpleFeature(id={self.feature_id}, z={self.charge}, "
            f"mz_mono={self.mz_mono:.4f}, members={self.n_members}, "
            f"raw_sum={self.raw_sum:.1f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the SimpleFeature to a dictionary representation.
        """
        return {
            "feature_id": self.feature_id,
            "charge": self.charge,
            "mz_mono": self.mz_mono,
            "neutral_mass": self.neutral_mass,
            "rt_bounds": self.rt_bounds,
            "im_bounds": self.im_bounds,
            "mz_center": self.mz_center,
            "n_members": self.n_members,
            "member_cluster_ids": self.member_cluster_ids,
            "raw_sum": self.raw_sum,
            "first_member_cluster_id": self.member_cluster_ids[0] if self.n_members > 0 else None,
        }


# -----------------------------------------------------------------------------
# Top-level convenience API
# -----------------------------------------------------------------------------


def _to_py_cluster_ptr(x: Any) -> Any:
    """
    Helper: unwrap Python-side wrappers to the underlying PyO3 handle.

    - If `x` has a `get_py_ptr()` method (like your usual RustWrapperObject),
      that is used.
    - Otherwise `x` is assumed to already be a PyO3 object from imspy_connector.
    """
    return getattr(x, "get_py_ptr", lambda: x)()


def build_simple_features_from_clusters(
    clusters: Sequence[Any],
    params: SimpleFeatureParams | None = None,
) -> list[SimpleFeature]:
    """
    Build simple isotopic features from a list of MS1 clusters.

    Parameters
    ----------
    clusters:
        Sequence of cluster objects. Each element can be:
        - a Python wrapper of `PyClusterResult1D` exposing `.get_py_ptr()`, or
        - the raw `PyClusterResult1D` PyO3 object from `imspy_connector`.

    params:
        SimpleFeatureParams instance. If None, Rust defaults are used.

    Returns
    -------
    list[SimpleFeature]
        One SimpleFeature per grouped isotopic envelope.
    """
    if params is None:
        params = SimpleFeatureParams.default()

    raw_clusters = [_to_py_cluster_ptr(c) for c in clusters]

    py_feats = imsf.build_simple_features_from_clusters_py(
        raw_clusters,
        params.get_py_ptr(),
    )

    return [SimpleFeature.from_py_ptr(f) for f in py_feats]