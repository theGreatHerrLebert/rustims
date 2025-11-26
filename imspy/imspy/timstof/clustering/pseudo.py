from __future__ import annotations
from typing import Any
import numpy as np

import imspy_connector
ims_pseudo = imspy_connector.py_pseudo


class PseudoFragment:
    def __init__(self, py_ptr: Any) -> None:
        self._p = py_ptr

    @property
    def mz(self) -> float:
        return float(self._p.mz)

    @property
    def intensity(self) -> float:
        return float(self._p.intensity)

    @property
    def ms2_cluster_id(self) -> int:
        return int(self._p.ms2_cluster_id)

    @property
    def window_group(self):
        return int(self._p.window_group)


class PseudoSpectrum:
    def __init__(self, py_ptr: Any) -> None:
        self._p = py_ptr

    @classmethod
    def from_clusters(cls, precursor, fragments):
        precursor_ptr = precursor.get_py_ptr()
        fragment_ptrs = [f.get_py_ptr() for f in fragments]

        py_ps = ims_pseudo.PyPseudoSpectrum(
            precursor=precursor_ptr,
            feature=None,
            fragments=fragment_ptrs,
        )
        return cls(py_ps)

    @classmethod
    def from_feature(cls, feature: "SimpleFeature", fragments):
        """
        Build a PseudoSpectrum from a SimpleFeature and a list of
        fragment ClusterResult1D wrappers.
        """
        feature_ptr = feature.get_py_ptr()
        fragment_ptrs = [f.get_py_ptr() for f in fragments]

        py_ps = ims_pseudo.PyPseudoSpectrum(
            precursor=None,
            feature=feature_ptr,
            fragments=fragment_ptrs,
        )
        return cls(py_ps)

    @property
    def precursor_mz(self) -> float:
        return float(self._p.precursor_mz)

    @property
    def precursor_charge(self) -> int:
        return int(self._p.precursor_charge)

    @property
    def rt_apex(self) -> float:
        return float(self._p.rt_apex)

    @property
    def im_apex(self) -> float:
        return float(self._p.im_apex)

    @property
    def feature_id(self) -> int | None:
        fid = self._p.feature_id
        return int(fid) if fid is not None else None

    @property
    def precursor_cluster_ids(self) -> list[int]:
        return list(self._p.precursor_cluster_ids)

    @property
    def fragments(self) -> list[PseudoFragment]:
        return [PseudoFragment(f) for f in self._p.fragments]

    @property
    def window_groups(self) -> list[int] | None:
        wgid = self._p.window_groups
        if wgid is None:
            return None
        return list(wgid)

    @property
    def fragment_mz_array(self) -> np.ndarray:
        return np.array([f.mz for f in self.fragments], dtype=np.float32)

    @property
    def fragment_intensity_array(self) -> np.ndarray:
        return np.array([f.intensity for f in self.fragments], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "precursor_mz": self.precursor_mz,
            "precursor_charge": self.precursor_charge,
            "rt_apex": self.rt_apex,
            "im_apex": self.im_apex,
            "feature_id": self.feature_id,
            "window_groups": self.window_groups,
            "precursor_cluster_ids": self.precursor_cluster_ids,
            "fragments": [
                {
                    "mz": f.mz,
                    "intensity": f.intensity,
                    "ms2_cluster_id": f.ms2_cluster_id,
                }
                for f in self.fragments
            ],
        }

    def __repr__(self):
        return (
            f"PseudoSpectrum(precursor_mz={self.precursor_mz:.4f}, "
            f"charge={self.precursor_charge}, rt_apex={self.rt_apex:.2f}, "
            f"im_apex={self.im_apex:.2f}, n_fragments={len(self.fragments)}, "
            f"feature_id={self.feature_id}, window_groups={self.window_groups})"
        )