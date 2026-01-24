"""
Pseudo spectrum construction for DIA clustering.

This module provides Python wrappers for pseudo spectrum data structures.
"""
from __future__ import annotations
from typing import Any
import numpy as np

import imspy_connector
ims_pseudo = imspy_connector.py_pseudo


class PseudoFragment:
    """Represents a single fragment peak in a pseudo spectrum."""

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

    def __repr__(self):
        return (
            f"PseudoFragment(mz={self.mz:.4f}, intensity={self.intensity:.1f}, "
            f"ms2_cluster_id={self.ms2_cluster_id}, "
            f"window_group={self.window_group})"
        )


class PseudoSpectrum:
    """
    Represents a pseudo spectrum constructed from DIA clustering.

    A pseudo spectrum contains a precursor (or isotopic feature) and its
    associated fragment peaks, enabling peptide identification from DIA data.
    """

    def __init__(self, py_ptr: Any) -> None:
        self._p = py_ptr

    @classmethod
    def from_clusters(cls, precursor, fragments):
        """
        Build a PseudoSpectrum from a precursor cluster and fragment clusters.

        Parameters
        ----------
        precursor : ClusterResult1D
            The precursor cluster.
        fragments : list[ClusterResult1D]
            List of fragment clusters.

        Returns
        -------
        PseudoSpectrum
        """
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

        Parameters
        ----------
        feature : SimpleFeature
            The isotopic feature.
        fragments : list[ClusterResult1D]
            List of fragment clusters.

        Returns
        -------
        PseudoSpectrum
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

    def merged_peaks(
            self,
            max_ppm: float,
            allow_cross_window_group: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return merged fragment peaks as (mz, intensity) arrays.

        Parameters
        ----------
        max_ppm : float
            m/z tolerance in ppm for merging. If <= 0, no merging is done and
            you just get the (sorted) raw fragment peaks back.
        allow_cross_window_group : bool, default False
            If False, only fragments from the same window_group are merged
            together. If True, fragments from different window_groups may be
            merged if they are within `max_ppm`.

        Returns
        -------
        mz : np.ndarray, shape (N,), dtype=float32
        intensity : np.ndarray, shape (N,), dtype=float32
        """
        mz, intensity = self._p.merged_peaks(
            float(max_ppm),
            bool(allow_cross_window_group),
        )
        return (
            np.asarray(mz, dtype=np.float32),
            np.asarray(intensity, dtype=np.float32),
        )

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


# Type hints for forward references
SimpleFeature = "SimpleFeature"
