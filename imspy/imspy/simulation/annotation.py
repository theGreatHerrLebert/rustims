from typing import Union

import imspy_connector
ims = imspy_connector.py_annotation


class SourceType:
    def __init__(self, source_type: str):
        self.known_source_types = ['signal', 'chemical_noise', 'random_noise', 'unknown']
        source_type = source_type.lower()
        assert source_type in self.known_source_types, f"Unknown source type: {source_type}. Known source types: {self.known_source_types}"
        index = self.known_source_types.index(source_type)
        self.__source_type = ims.PySourceType(index)

    @property
    def source_type(self) -> str:
        return self.__source_type.source_type

    def __repr__(self) -> str:
        return f"SourceType(source_type={self.source_type})"

    @classmethod
    def from_py_source_type(cls, source_type: ims.PySourceType) -> 'SourceType':
        instance = cls.__new__(cls)
        instance.__source_type = source_type
        return instance

    def get_py_ptr(self) -> ims.PySourceType:
        return self.__source_type


# charge_state: i32, peptide_id: i32, isotope_peak: i32
class SignalAttributes:
    def __init__(self, charge_state: int, peptide_id: int, isotope_peak: int, description: Union[None, str] = None):
        self.__signal_attributes = ims.PySignalAttributes(charge_state, peptide_id, isotope_peak, description)

    @property
    def charge_state(self):
        return self.__signal_attributes.charge_state

    @property
    def peptide_id(self):
        return self.__signal_attributes.peptide_id

    @property
    def isotope_peak(self):
        return self.__signal_attributes.isotope_peak

    @property
    def description(self) -> Union[None, str]:
        return self.__signal_attributes.description

    def __repr__(self):
        return f"SignalAnnotation(charge_state={self.charge_state}, peptide_id={self.peptide_id}, isotope_peak={self.isotope_peak}, description={self.description})"

    @classmethod
    def from_py_signal_annotation(cls, signal_attributes: ims.PySignalAttributes) -> 'SignalAttributes':
        instance = cls.__new__(cls)
        instance.__signal_attributes = signal_attributes
        return instance

    def get_py_ptr(self) -> ims.PySignalAttributes:
        return self.__signal_attributes


class ContributionSource:
    def __init__(self, intensity_contribution: float, source_type: SourceType, signal_attributes: Union[None, SignalAttributes] = None):
        self.__contribution_source = ims.PyContributionSource(
            intensity_contribution,
            source_type.get_py_ptr(),
            signal_attributes.get_py_ptr() if signal_attributes else None
        )

    @property
    def intensity_contribution(self) -> float:
        return self.__contribution_source.intensity_contribution

    @property
    def source_type(self) -> SourceType:
        return SourceType.from_py_source_type(self.__contribution_source.source_type)

    @property
    def signal_attributes(self) -> Union[None, SignalAttributes]:
        return SignalAttributes.from_py_signal_annotation(self.__contribution_source.signal_attributes) if self.__contribution_source.signal_attributes else None

    def __repr__(self) -> str:
        return f"ContributionSource(intensity_contribution={self.intensity_contribution}, source_type={self.source_type}, signal_attributes={self.signal_attributes})"

    @classmethod
    def from_py_contribution_source(cls, contribution_source: ims.PyContributionSource) -> 'ContributionSource':
        instance = cls.__new__(cls)
        instance._contribution_source = contribution_source
        return instance

    def get_py_ptr(self) -> ims.PyContributionSource:
        return self.__contribution_source


class PeakAnnotation:
    def __init__(self, contributions: list[ContributionSource]):
        assert len(contributions) > 0, "At least one contribution is required."
        self.__peak_annotation = ims.PyPeakAnnotation(
            [c.get_py_ptr() for c in contributions]
        )

    @property
    def contributions(self) -> list[ContributionSource]:
        return [ContributionSource.from_py_contribution_source(c) for c in self.__peak_annotation.contributions]

    def __repr__(self) -> str:
        return f"PeakAnnotation(contributions={self.contributions})"

    @classmethod
    def from_py_peak_annotation(cls, peak_annotation: ims.PyPeakAnnotation) -> 'PeakAnnotation':
        instance = cls.__new__(cls)
        instance.__peak_annotation = peak_annotation
        return instance

    def get_py_ptr(self) -> ims.PyPeakAnnotation:
        return self.__peak_annotation


class MzSpectrumAnnotated:
    def __init__(self, mz: list[float], intensity: list[float], annotations: list[PeakAnnotation]):
        assert len(mz) == len(intensity) == len(annotations), "Length of mz, intensity and annotations must be equal."
        self.__mz_spectrum_annotated = ims.PyMzSpectrumAnnotated(
            mz,
            intensity,
            [a.get_py_ptr() for a in annotations]
        )

    @property
    def mz(self) -> list[float]:
        return self.__mz_spectrum_annotated.mz

    @property
    def intensity(self) -> list[float]:
        return self.__mz_spectrum_annotated.intensity

    @property
    def annotations(self) -> list[PeakAnnotation]:
        return [PeakAnnotation.from_py_peak_annotation(a) for a in self.__mz_spectrum_annotated.annotations]

    def __add__(self, other: 'MzSpectrumAnnotated') -> 'MzSpectrumAnnotated':
        return MzSpectrumAnnotated.from_py_mz_spectrum_annotated(self.__mz_spectrum_annotated + other.__mz_spectrum_annotated)

    def __repr__(self) -> str:
        return f"MzSpectrumAnnotated(mz={self.mz}, intensity={self.intensity}, annotations={self.annotations})"

    @classmethod
    def from_py_mz_spectrum_annotated(cls, mz_spectrum_annotated: ims.PyMzSpectrumAnnotated) -> 'MzSpectrumAnnotated':
        instance = cls.__new__(cls)
        instance.__mz_spectrum_annotated = mz_spectrum_annotated
        return instance

    def get_py_ptr(self) -> ims.PyMzSpectrumAnnotated:
        return self.__mz_spectrum_annotated
