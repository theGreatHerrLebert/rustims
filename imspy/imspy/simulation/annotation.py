from typing import Union
from abc import ABC, abstractmethod
import imspy_connector
import numpy as np
import pandas as pd
from numpy.typing import NDArray

ims = imspy_connector.py_annotation


# TODO: Add docstrings, add interface to all other classes that bind rust code
class RustWrapperObject(ABC):
    @classmethod
    @abstractmethod
    def from_py_ptr(cls, obj):
        pass

    @abstractmethod
    def get_py_ptr(self):
        pass


class SourceType(RustWrapperObject):
    def __init__(self, source_type: str):
        self.known_source_types = ['signal', 'chemical_noise', 'random_noise', 'unknown']
        source_type = source_type.lower()
        assert source_type in self.known_source_types, (f"Unknown source type: {source_type}. Known source types: "
                                                        f"{self.known_source_types}")
        index = self.known_source_types.index(source_type)
        self.__py_ptr = ims.PySourceType(index)

    @property
    def source_type(self) -> str:
        return self.__py_ptr.source_type

    def __repr__(self) -> str:
        return f"SourceType(source_type={self.source_type})"

    @classmethod
    def from_py_ptr(cls, source_type: ims.PySourceType) -> 'SourceType':
        instance = cls.__new__(cls)
        instance.__py_ptr = source_type
        return instance

    def get_py_ptr(self) -> ims.PySourceType:
        return self.__py_ptr


# charge_state: i32, peptide_id: i32, isotope_peak: i32
class SignalAttributes(RustWrapperObject):
    def __init__(self, charge_state: int, peptide_id: int, isotope_peak: int, description: Union[None, str] = None):
        self.__py_ptr = ims.PySignalAttributes(charge_state, peptide_id, isotope_peak, description)

    @property
    def charge_state(self):
        return self.__py_ptr.charge_state

    @property
    def peptide_id(self):
        return self.__py_ptr.peptide_id

    @property
    def isotope_peak(self):
        return self.__py_ptr.isotope_peak

    @property
    def description(self) -> Union[None, str]:
        return self.__py_ptr.description

    def __repr__(self):
        return (f"SignalAnnotation(charge_state={self.charge_state}, peptide_id={self.peptide_id}, "
                f"isotope_peak={self.isotope_peak}, description={self.description})")

    @classmethod
    def from_py_ptr(cls, signal_attributes: ims.PySignalAttributes) -> 'SignalAttributes':
        instance = cls.__new__(cls)
        instance.__py_ptr = signal_attributes
        return instance

    def get_py_ptr(self) -> ims.PySignalAttributes:
        return self.__py_ptr


class ContributionSource(RustWrapperObject):
    def __init__(self, intensity_contribution: float, source_type: SourceType,
                 signal_attributes: Union[None, SignalAttributes] = None):
        self.__py_ptr = ims.PyContributionSource(
            intensity_contribution,
            source_type.get_py_ptr(),
            signal_attributes.get_py_ptr() if signal_attributes else None
        )

    @property
    def intensity_contribution(self) -> float:
        return self.__py_ptr.intensity_contribution

    @property
    def source_type(self) -> SourceType:
        return SourceType.from_py_ptr(self.__py_ptr.source_type)

    @property
    def signal_attributes(self) -> Union[None, SignalAttributes]:
        return SignalAttributes.from_py_ptr(
            self.__py_ptr.signal_attributes) if self.__py_ptr.signal_attributes else None

    def __repr__(self) -> str:
        return (f"ContributionSource(intensity_contribution={self.intensity_contribution}, "
                f"source_type={self.source_type}, signal_attributes={self.signal_attributes})")

    @classmethod
    def from_py_ptr(cls, contribution_source: ims.PyContributionSource) -> 'ContributionSource':
        instance = cls.__new__(cls)
        instance.__py_ptr = contribution_source
        return instance

    def get_py_ptr(self) -> ims.PyContributionSource:
        return self.__py_ptr


class PeakAnnotation(RustWrapperObject):
    def __init__(self, contributions: list[ContributionSource]):
        assert len(contributions) > 0, "At least one contribution is required."
        self.__py_ptr = ims.PyPeakAnnotation(
            [c.get_py_ptr() for c in contributions]
        )

    @property
    def contributions(self) -> list[ContributionSource]:
        return [ContributionSource.from_py_ptr(c) for c in self.__py_ptr.contributions]

    def __repr__(self) -> str:
        return f"PeakAnnotation(contributions={self.contributions})"

    @classmethod
    def from_py_ptr(cls, peak_annotation: ims.PyPeakAnnotation) -> 'PeakAnnotation':
        instance = cls.__new__(cls)
        instance.__py_ptr = peak_annotation
        return instance

    def get_py_ptr(self) -> ims.PyPeakAnnotation:
        return self.__py_ptr


class MzSpectrumAnnotated(RustWrapperObject):
    def __init__(self, mz: list[float], intensity: list[float], annotations: list[PeakAnnotation]):
        assert len(mz) == len(intensity) == len(annotations), "Length of mz, intensity and annotations must be equal."
        self.__py_ptr = ims.PyMzSpectrumAnnotated(
            mz,
            intensity,
            [a.get_py_ptr() for a in annotations]
        )

    @property
    def mz(self) -> list[float]:
        return self.__py_ptr.mz

    @property
    def intensity(self) -> list[float]:
        return self.__py_ptr.intensity

    @property
    def annotations(self) -> list[PeakAnnotation]:
        return [PeakAnnotation.from_py_ptr(a) for a in self.__py_ptr.annotations]

    def __add__(self, other: 'MzSpectrumAnnotated') -> 'MzSpectrumAnnotated':
        return MzSpectrumAnnotated.from_py_ptr(
            self.__py_ptr + other.__py_ptr)

    def __repr__(self) -> str:
        return f"MzSpectrumAnnotated(mz={self.mz}, intensity={self.intensity}, annotations={self.annotations})"

    @classmethod
    def from_py_ptr(cls, mz_spectrum_annotated: ims.PyMzSpectrumAnnotated) -> 'MzSpectrumAnnotated':
        instance = cls.__new__(cls)
        instance.__py_ptr = mz_spectrum_annotated
        return instance

    def get_py_ptr(self) -> ims.PyMzSpectrumAnnotated:
        return self.__py_ptr

    def filter(self,
               mz_min: float = 0.0,
               mz_max: float = 1700.0,
               intensity_min: float = 0.0,
               intensity_max: float = 1e9) -> 'MzSpectrumAnnotated':
        return MzSpectrumAnnotated.from_py_ptr(
            self.__py_ptr.filter_ranged(mz_min, mz_max, intensity_min, intensity_max))


class TimsFrameAnnotated(RustWrapperObject):
    def __init__(self,
                 frame_id: int,
                 retention_time: float,
                 ms_type: int,
                 tof: NDArray[int],
                 mz: NDArray[float],
                 scan: NDArray[int],
                 inv_mobility: NDArray[float],
                 intensity: NDArray[float],
                 annotations: NDArray[PeakAnnotation]):
        assert len(tof) == len(mz) == len(scan) == len(inv_mobility) == len(intensity) == len(
            annotations), "Length of tof, mz, scan, inv_mobility, intensity and annotations must be equal."

        self.__py_ptr = ims.PyTimsFrameAnnotated(
            frame_id,
            retention_time,
            ms_type,
            tof,
            mz,
            scan,
            inv_mobility,
            intensity,
            [a.get_py_ptr() for a in annotations])

    @property
    def frame_id(self) -> int:
        return self.__py_ptr.frame_id

    @property
    def retention_time(self) -> float:
        return self.__py_ptr.retention_time

    @property
    def ms_type(self) -> int:
        return self.__py_ptr.ms_type_numeric

    @property
    def tof(self) -> list[int]:
        return self.__py_ptr.tof

    @property
    def mz(self) -> list[float]:
        return self.__py_ptr.mz

    @property
    def scan(self) -> list[int]:
        return self.__py_ptr.scan

    @property
    def inv_mobility(self) -> list[float]:
        return self.__py_ptr.inv_mobility

    @property
    def intensity(self) -> list[float]:
        return self.__py_ptr.intensity

    @property
    def annotations(self) -> list[PeakAnnotation]:
        return [PeakAnnotation.from_py_ptr(a) for a in self.__py_ptr.annotations]

    @property
    def peptide_ids_first_only(self) -> NDArray[int]:
        return self.__py_ptr.peptide_ids_first_only

    @property
    def charge_states_first_only(self) -> NDArray[int]:
        return self.__py_ptr.charge_states_first_only

    @property
    def isotope_peaks_first_only(self) -> NDArray[int]:
        return self.__py_ptr.isotope_peaks_first_only

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame({
            'tof': self.tof,
            'mz': self.mz,
            'scan': self.scan,
            'inv_mobility': self.inv_mobility,
            'intensity': self.intensity,
            'peptide_id': self.peptide_ids_first_only,
            'charge_state': self.charge_states_first_only,
            'isotope_peak': self.isotope_peaks_first_only
        })

    def filter(self,
               mz_min: float = 0.0,
               mz_max: float = 1700.0,
               inv_mobility_min: float = 0.0,
               inv_mobility_max: float = 4.0,
               scan_min: int = 0,
               scan_max: int = 1000,
               intensity_min: float = 0.0,
               intensity_max: float = 1e9,
               ) -> 'TimsFrameAnnotated':
        return TimsFrameAnnotated.from_py_ptr(
            self.__py_ptr.filter_ranged(mz_min, mz_max, inv_mobility_min,
                                        inv_mobility_max, scan_min, scan_max, intensity_min, intensity_max))

    @property
    def ms_type_numeric(self) -> int:
        return self.__py_ptr.ms_type_numeric

    def __add__(self, other: 'TimsFrameAnnotated') -> 'TimsFrameAnnotated':
        return TimsFrameAnnotated.from_py_ptr(self.__py_ptr +
                                              other.__py_ptr)

    def __repr__(self) -> str:
        return (f"TimsFrameAnnotated("
                f"frame_id={self.frame_id}, "
                f"retention_time={np.round(self.retention_time / 60, 2)}, "
                f"ms_type={self.ms_type}, num_peaks={len(self.mz)}, "
                f"sum_intensity={sum(np.round(self.intensity))})")

    @classmethod
    def from_py_ptr(cls, tims_frame_annotated: ims.PyTimsFrameAnnotated) -> 'TimsFrameAnnotated':
        instance = cls.__new__(cls)
        instance.__py_ptr = tims_frame_annotated
        return instance

    def get_py_ptr(self) -> ims.PyTimsFrameAnnotated:
        return self.__py_ptr
