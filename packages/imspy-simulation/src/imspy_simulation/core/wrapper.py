"""Base class for PyO3-wrapped Rust objects.

This module provides a clean abstraction for Python wrappers around
Rust objects exposed via PyO3. It replaces the ad-hoc RustWrapperObject
pattern used throughout the codebase.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

T = TypeVar("T")


class PyO3Wrapper(ABC, Generic[T]):
    """Base class for Python wrappers around PyO3-exposed Rust objects.

    This provides a consistent interface for accessing the underlying
    Rust pointer and creating wrapper instances from existing pointers.

    Type parameter T represents the PyO3 class being wrapped.

    Example usage:
        class MyWrapper(PyO3Wrapper[PyMyRustClass]):
            def __init__(self, param: int):
                self._py_ptr = PyMyRustClass(param)

            @property
            def _ptr(self) -> PyMyRustClass:
                return self._py_ptr

            @classmethod
            def _from_ptr(cls, ptr: PyMyRustClass) -> "MyWrapper":
                instance = cls.__new__(cls)
                instance._py_ptr = ptr
                return instance
    """

    @property
    @abstractmethod
    def _ptr(self) -> T:
        """Get the underlying PyO3 pointer.

        Returns:
            The wrapped PyO3 object.
        """
        ...

    @classmethod
    @abstractmethod
    def _from_ptr(cls, ptr: T) -> "PyO3Wrapper[T]":
        """Create a wrapper instance from an existing PyO3 pointer.

        Args:
            ptr: The PyO3 object to wrap.

        Returns:
            A new wrapper instance containing the pointer.
        """
        ...

    # Legacy compatibility methods
    def get_py_ptr(self) -> T:
        """Get the underlying PyO3 pointer (legacy method).

        Deprecated: Use the `_ptr` property instead.

        Returns:
            The wrapped PyO3 object.
        """
        return self._ptr

    @classmethod
    def from_py_ptr(cls, ptr: T) -> "PyO3Wrapper[T]":
        """Create a wrapper from a PyO3 pointer (legacy method).

        Deprecated: Use the `_from_ptr` class method instead.

        Args:
            ptr: The PyO3 object to wrap.

        Returns:
            A new wrapper instance.
        """
        return cls._from_ptr(ptr)
