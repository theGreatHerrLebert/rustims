import imspy_connector
ims = imspy_connector.py_annotation


class SourceType:
    def __init__(self, source_type: str):
        source_type = source_type.lower()
        known_source_types = ['signal', 'chemical_noise', 'random_noise']
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
