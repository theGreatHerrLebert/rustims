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
        return self.__source_type.source_type_as_str

    def __repr__(self) -> str:
        return f"SourceType(source_type={self.source_type})"

    @classmethod
    def from_py_source_type(cls, source_type: ims.PySourceType) -> 'SourceType':
        instance = cls.__new__(cls)
        instance._source_type = source_type
        return instance

    def get_py_ptr(self) -> ims.PySourceType:
        return self.__source_type


# charge_state: i32, peptide_id: i32, isotope_peak: i32
class SignalAttributes:
    def __init__(self, charge_state: int, peptide_id: int, isotope_peak: int):
        self.__signal_attributes = ims.PySignalAttributes(charge_state, peptide_id, isotope_peak)

    @property
    def charge_state(self):
        return self.__signal_attributes.charge_state

    @property
    def peptide_id(self):
        return self.__signal_attributes.peptide_id

    @property
    def isotope_peak(self):
        return self.__signal_attributes.isotope_peak

    def __repr__(self):
        return f"SignalAnnotation(charge_state={self.charge_state}, peptide_id={self.peptide_id}, isotope_peak={self.isotope_peak})"

    @classmethod
    def from_py_signal_annotation(cls, signal_attributes: ims.PySignalAttributes) -> 'SignalAttributes':
        instance = cls.__new__(cls)
        instance._signal_annotation = signal_attributes
        return instance

    def get_py_ptr(self) -> ims.PySignalAttributes:
        return self.__signal_attributes


class ContributionSource:
    def __init__(self, intensity_contribution: float, source_type: SourceType, signal_attributes: SignalAttributes = None):
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
        return SourceType.from_py_source_type(self.__contribution_source.py_source_type)

    @property
    def signal_attributes(self) -> Union[None, SignalAttributes]:
        return SignalAttributes.from_py_signal_annotation(self.__contribution_source.py_signal_attributes) if self.__contribution_source.py_signal_attributes else None

    def __repr__(self) -> str:
        return f"ContributionSource(intensity_contribution={self.intensity_contribution}, source_type={self.source_type}, signal_attributes={self.signal_attributes})"

    @classmethod
    def from_py_contribution_source(cls, contribution_source: ims.PyContributionSource) -> 'ContributionSource':
        instance = cls.__new__(cls)
        instance._contribution_source = contribution_source
        return instance

    def get_py_ptr(self) -> ims.PyContributionSource:
        return self.__contribution_source
