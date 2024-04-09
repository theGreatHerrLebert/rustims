import imspy_connector
ims = imspy_connector.py_annotation


class SourceType:
    def __init__(self, source_type: str):
        source_type = source_type.lower()
        known_source_types = ['signal', 'chemical_noise', 'random_noise', 'unknown']
        assert source_type in known_source_types, f"Unknown source type: {source_type}. Known source types: {known_source_types}"
        index = known_source_types.index(source_type)
        self.__source_type = ims.PySourceType(index)

    @property
    def source_type(self):
        return self.__source_type.source_type

    def __repr__(self):
        return f"SourceType(source_type={self.source_type})"

    @classmethod
    def from_py_source_type(cls, source_type: ims.PySourceType):
        instance = cls.__new__(cls)
        instance._source_type = source_type
        return instance

    def get_py_ptr(self):
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
    def from_py_signal_annotation(cls, signal_attributes: ims.PySignalAnnotation):
        instance = cls.__new__(cls)
        instance._signal_annotation = signal_attributes
        return instance

    def get_py_ptr(self):
        return self.__signal_attributes
