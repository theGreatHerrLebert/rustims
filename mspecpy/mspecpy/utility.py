from abc import ABC, abstractmethod


class RustWrapper(ABC):
    @classmethod
    @abstractmethod
    def from_py_ptr(cls, obj):
        pass

    @abstractmethod
    def get_py_ptr(self):
        pass
