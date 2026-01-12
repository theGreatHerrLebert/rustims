"""
Base classes for imspy.

This module contains base classes used across the package,
separated to avoid circular imports.
"""

from abc import ABC, abstractmethod


class RustWrapperObject(ABC):
    """Abstract base class for Python wrappers around Rust objects.

    All classes that wrap PyO3 Rust objects should inherit from this
    and implement from_py_ptr and get_py_ptr methods.
    """

    @classmethod
    @abstractmethod
    def from_py_ptr(cls, obj):
        """Create a Python wrapper from a PyO3 pointer.

        Args:
            obj: The PyO3 Rust object pointer.

        Returns:
            An instance of the wrapper class.
        """
        pass

    @abstractmethod
    def get_py_ptr(self):
        """Get the underlying PyO3 Rust object pointer.

        Returns:
            The PyO3 Rust object pointer.
        """
        pass
