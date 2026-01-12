from abc import ABC, abstractmethod
from typing import List, Tuple, Union, TYPE_CHECKING

import imspy_connector

from imspy.data.spectrum import MzSpectrum
from imspy.core.base import RustWrapperObject

if TYPE_CHECKING:
    from imspy.simulation.annotation import MzSpectrumAnnotated

ims = imspy_connector.py_peptide


def _get_mz_spectrum_annotated():
    """Lazy import of MzSpectrumAnnotated to avoid circular imports."""
    from imspy.simulation.annotation import MzSpectrumAnnotated
    return MzSpectrumAnnotated


class PeptideProductIonSeriesCollection(RustWrapperObject):
    def __init__(self, series: List['PeptideProductIonSeries']):
        """Create a new product ion series collection.

        Args:
            series: The product ion series.
        """
        self.__py_ptr = ims.PyPeptideProductIonSeriesCollection([s.get_py_ptr() for s in series])

    @property
    def series(self) -> List['PeptideProductIonSeries']:
        return [PeptideProductIonSeries.from_py_ptr(series) for series in self.__py_ptr.series]

    def find_series(self, charge: int) -> Union[None, 'PeptideProductIonSeries']:
        """Find the product ion series with the given charge.

        Args:
            charge: The charge of the product ion series.

        Returns:
            The product ion series with the given charge, or None if not found.
        """
        series = self.__py_ptr.find_ion_series(charge)
        return PeptideProductIonSeries.from_py_ptr(series) if series else None

    def to_json(self) -> str:
        return self.__py_ptr.to_json()

    def get_py_ptr(self):
        return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, collection: ims.PyPeptideProductIonSeriesCollection):
        instance = cls.__new__(cls)
        instance.__py_ptr = collection
        return instance

    def __repr__(self):
        return f"PeptideProductIonSeriesCollection(series={self.series})"

    # mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64
    def generate_isotopic_spectrum(
            self,
            mass_tolerance: float = 1e-3,
            abundance_threshold: float = 1e-6,
            max_result: int = 2000,
            intensity_min: float = 1e-6
    ) -> MzSpectrum:
        """Calculate the isotope distribution of the product ion series collection.

        Args:
            mass_tolerance: The mass tolerance for the isotope distribution calculation.
            abundance_threshold: The abundance threshold for the isotope distribution calculation.
            max_result: The maximum number of results to return.
            intensity_min: The minimum intensity of the isotope distribution.

        Returns:
            The isotope distribution of the product ion series collection.
        """
        py_spec = self.__py_ptr.generate_isotopic_spectrum(mass_tolerance, abundance_threshold, max_result, intensity_min)
        return MzSpectrum.from_py_ptr(py_spec)

    def generate_isotopic_spectrum_annotated(
            self,
            mass_tolerance: float = 1e-3,
            abundance_threshold: float = 1e-6,
            max_result: int = 2000,
            intensity_min: float = 1e-6
    ) -> 'MzSpectrumAnnotated':
        """Calculate the isotope distribution of the product ion series collection.

        Args:
            mass_tolerance: The mass tolerance for the isotope distribution calculation.
            abundance_threshold: The abundance threshold for the isotope distribution calculation.
            max_result: The maximum number of results to return.
            intensity_min: The minimum intensity of the isotope distribution.

        Returns:
            The isotope distribution of the product ion series collection.
        """
        MzSpectrumAnnotated = _get_mz_spectrum_annotated()
        py_spec = self.__py_ptr.generate_isotopic_spectrum_annotated(mass_tolerance, abundance_threshold, max_result, intensity_min)
        return MzSpectrumAnnotated.from_py_ptr(py_spec)


class PeptideProductIonSeries(RustWrapperObject):
    def __init__(
            self,
            charge: int,
            n_ions: List['PeptideProductIon'],
            c_ions: List['PeptideProductIon']
    ):
        """Create a new product ion series.

        Args:
            charge: The charge of the product ions.
            n_ions: The N-terminal product ions.
            c_ions: The C-terminal product ions.
        """
        self.__py_ptr = ims.PyPeptideProductIonSeries(
            charge=charge,
            n_ions=[ion.get_py_ptr() for ion in n_ions],
            c_ions=[ion.get_py_ptr() for ion in c_ions]
        )

    @property
    def n_ions(self) -> List['PeptideProductIon']:
        return [PeptideProductIon.from_py_ptr(ion) for ion in self.__py_ptr.n_ions]

    @property
    def c_ions(self) -> List['PeptideProductIon']:
        return [PeptideProductIon.from_py_ptr(ion) for ion in self.__py_ptr.c_ions]

    @property
    def charge(self) -> int:
        return self.__py_ptr.charge

    def get_py_ptr(self):
        return self.__py_ptr

    def to_json(self) -> str:
        return self.__py_ptr.to_json()

    @classmethod
    def from_py_ptr(cls, series: ims.PyPeptideProductIonSeries):
        instance = cls.__new__(cls)
        instance.__py_ptr = series
        return instance

    def __repr__(self):
        return (f"PeptideProductIonSeries(charge={self.charge}, n_ions={self.n_ions}, "
                f"c_ions={self.c_ions})")


class PeptideProductIon(RustWrapperObject):
    def __init__(self, kind: str, sequence: str, charge: int = 1, intensity: float = 1.0, peptide_id: Union[None, int] = None):
        """Create a new product ion.

        Args:
            kind: The kind of product ion, must be one of 'a', 'b', 'c', 'x', 'y', 'z'.
            sequence: The sequence of the product ion.
            charge: The charge of the product ion.
            intensity: The intensity of the product ion.
        """
        assert kind in ['a', 'b', 'c', 'x', 'y', 'z'], (f"Invalid kind: {kind}, "
                                                        f"must be one of 'a', 'b', 'c', 'x', 'y', 'z'")
        self.__py_ptr = ims.PyPeptideProductIon(
            kind=kind,
            sequence=sequence,
            charge=charge,
            intensity=intensity,
            peptide_id=peptide_id
        )

    @property
    def kind(self) -> str:
        return self.__py_ptr.kind

    @property
    def sequence(self) -> str:
        return self.__py_ptr.sequence

    @property
    def charge(self) -> int:
        return self.__py_ptr.charge

    @property
    def intensity(self) -> float:
        return self.__py_ptr.intensity

    @property
    def mono_isotopic_mass(self) -> float:
        return self.__py_ptr.mono_isotopic_mass()

    @property
    def mz(self) -> float:
        return self.__py_ptr.mz

    def atomic_composition(self):
        return self.__py_ptr.atomic_composition()

    def to_json(self) -> str:
        return self.__py_ptr.to_json()

    def get_py_ptr(self):
        return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, product_ion: ims.PyPeptideProductIon):
        instance = cls.__new__(cls)
        instance.__py_ptr = product_ion
        return instance

    def isotope_distribution(
            self,
            mass_tolerance: float = 1e-3,
            abundance_threshold: float = 1e-8,
            max_result: int = 200,
            intensity_min: float = 1e-4
    ) -> List[Tuple[float, float]]:
        """Calculate the isotope distribution of the product ion.

        Args:
            mass_tolerance: The mass tolerance for the isotope distribution calculation.
            abundance_threshold: The abundance threshold for the isotope distribution calculation.
            max_result: The maximum number of results to return.
            intensity_min: The minimum intensity of the isotope distribution.

        Returns:
            The isotope distribution of the product ion.
        """
        return self.__py_ptr.isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min)

    def __repr__(self):
        return (f"ProductIon(kind={self.kind}, sequence={self.sequence}, charge={self.charge}, mz={self.mz}, "
                f" intensity={self.intensity})")


class PeptideSequence(RustWrapperObject):
    def __init__(self, sequence: str, peptide_id: Union[None, int] = None):
        """Create a new peptide sequence.

        Args:
            sequence: The sequence of the peptide.
        """
        self.known_fragment_types = ['a', 'b', 'c', 'x', 'y', 'z']
        self.__py_ptr = ims.PyPeptideSequence(sequence, peptide_id=peptide_id)

    @property
    def sequence(self) -> str:
        return self.__py_ptr.sequence

    @property
    def peptide_id(self) -> Union[None, int]:
        return self.__py_ptr.peptide_id

    @property
    def mono_isotopic_mass(self) -> float:
        return self.__py_ptr.mono_isotopic_mass

    @property
    def atomic_composition(self):
        return self.__py_ptr.atomic_composition()

    @property
    def amino_acid_count(self) -> int:
        return self.__py_ptr.amino_acid_count()

    def to_tokens(self, group_modifications: bool = True) -> List[str]:
        return self.__py_ptr.to_tokens(group_modifications)

    def to_sage_representation(self) -> Tuple[str, List[float]]:
        return self.__py_ptr.to_sage_representation()

    def to_json(self) -> str:
        return self.__py_ptr.to_json()

    def get_py_ptr(self):
        return self.__py_ptr

    def calculate_product_ion_series(
            self,
            charge: int = 1,
            fragment_type:
            str = 'b'
    ) -> Tuple[List[PeptideProductIon], List[PeptideProductIon]]:
        """Calculate the b and y product ion series of the peptide sequence.

        Args:
            charge: The charge of the product ions.
            fragment_type: The type of the product ions, must be one of 'a', 'b', 'c', 'x', 'y', 'z'.

        Returns:
            The b and y product ion series of the peptide sequence.
        """
        fragment_type = fragment_type.lower()
        assert fragment_type in self.known_fragment_types, (f"Invalid fragment type: {fragment_type}, "
                                                            f"must be one of {self.known_fragment_types}")

        n_ions, c_ions = self.__py_ptr.calculate_product_ion_series(charge, fragment_type)

        return ([PeptideProductIon.from_py_ptr(ion) for ion in n_ions],
                [PeptideProductIon.from_py_ptr(ion) for ion in c_ions][::-1])

    def calculate_mono_isotopic_product_ion_spectrum(
            self,
            charge:
            int = 1,
            fragment_type: str = 'b'
    ) -> MzSpectrum:
        """Calculate the mono-isotopic product ion spectrum of the peptide sequence.
        
        Args:
            charge: The charge of the product ions.
            fragment_type: The type of the product ions, must be one of 'a', 'b', 'c', 'x', 'y', 'z'.
            
        Returns:
            The mono-isotopic product ion spectrum of the peptide sequence.
        """
        fragment_type = fragment_type.lower()
        assert fragment_type in ['a', 'b', 'c', 'x', 'y', 'z'], (f"Invalid fragment type: {fragment_type}, "
                                                                    f"must be one of 'a', 'b', 'c', 'x', 'y', 'z'")

        py_spec = self.__py_ptr.calculate_mono_isotopic_product_ion_spectrum(charge, fragment_type)
        return MzSpectrum.from_py_ptr(py_spec)

    def calculate_mono_isotopic_product_ion_spectrum_annotated(self, charge: int = 1,
                                                               fragment_type: str = 'b') -> 'MzSpectrumAnnotated':
        """Calculate the mono-isotopic product ion spectrum of the peptide sequence.

        Args:
            charge: The charge of the product ions.
            fragment_type: The type of the product ions, must be one of 'a', 'b', 'c', 'x', 'y', 'z'.

        Returns:
            The mono-isotopic product ion spectrum of the peptide sequence.
        """
        fragment_type = fragment_type.lower()
        assert fragment_type in ['a', 'b', 'c', 'x', 'y', 'z'], (f"Invalid fragment type: {fragment_type}, "
                                                                    f"must be one of 'a', 'b', 'c', 'x', 'y', 'z'")

        MzSpectrumAnnotated = _get_mz_spectrum_annotated()
        py_spec = self.__py_ptr.calculate_mono_isotopic_product_ion_spectrum_annotated(charge, fragment_type)
        return MzSpectrumAnnotated.from_py_ptr(py_spec)

    # mass_tolerance: f64, abundance_threshold: f64, max_result: i32, intensity_min: f64
    def calculate_isotopic_product_ion_spectrum_annotated(self, charge: int = 1,
                                                          fragment_type: str = 'b',
                                                          mass_tolerance: float = 1e-3,
                                                          abundance_threshold: float = 1e-8,
                                                          max_result: int = 200,
                                                          intensity_min: float = 1e-4
                                                          ) -> 'MzSpectrumAnnotated':
        """Calculate the isotopic product ion spectrum of the peptide sequence.

        Args:
            intensity_min:
            max_result:
            abundance_threshold:
            mass_tolerance:
            charge: The charge of the product ions.
            fragment_type: The type of the product ions, must be one of 'a', 'b', 'c', 'x', 'y', 'z'.

        Returns:
            The isotopic product ion spectrum of the peptide sequence.
        """
        fragment_type = fragment_type.lower()
        assert fragment_type in ['a', 'b', 'c', 'x', 'y', 'z'], (f"Invalid fragment type: {fragment_type}, "
                                                                    f"must be one of 'a', 'b', 'c', 'x', 'y', 'z'")

        MzSpectrumAnnotated = _get_mz_spectrum_annotated()
        py_spec = self.__py_ptr.calculate_isotopic_product_ion_spectrum_annotated(charge, fragment_type, mass_tolerance, abundance_threshold, max_result, intensity_min)
        return MzSpectrumAnnotated.from_py_ptr(py_spec)

    def associate_fragment_ion_series_with_prosit_intensities(
            self, flat_intensities: List[float],
            charge: int,
            fragment_type: str = "b",
            normalize: bool = True,
            half_charge_one: bool = True
    ) -> PeptideProductIonSeriesCollection:
        """Associate the peptide sequence with predicted intensities from Prosit intensity prediction.

        Args:
            flat_intensities: The flat intensities.
            # TODO: check how charge should be handled
            charge: The charge of the product ions.
            fragment_type: The type of the product ions, must be one of 'a', 'b', 'c', 'x', 'y', 'z'.
            normalize: Whether to normalize the intensities.
            half_charge_one: Whether to use half charge one.

        Returns:
            The b and y product ion series of the peptide sequence.
        """
        fragment_type = fragment_type.lower()
        assert fragment_type in self.known_fragment_types, (f"Invalid fragment type: {fragment_type}, "
                                                            f"must be one of {self.known_fragment_types}")

        result = self.__py_ptr.associate_with_predicted_intensities(flat_intensities,
                                                                    charge, fragment_type, normalize, half_charge_one)

        return PeptideProductIonSeriesCollection.from_py_ptr(result)

    @classmethod
    def from_py_ptr(cls, obj: ims.PyPeptideSequence):
        instance = cls.__new__(cls)
        instance.__py_ptr = obj
        return instance

    def __repr__(self):
        return (f"PeptideSequence(sequence={self.sequence}, "
                f"mono_isotopic_mass={self.mono_isotopic_mass}, peptide_id={self.peptide_id})")


class PeptideIon(RustWrapperObject):
    def __init__(self, sequence: str, charge: int, intensity: float, peptide_id: Union[None, int] = None):
        """Create a new peptide ion.

        Args:
            sequence: The sequence of the peptide ion.
            charge: The charge of the peptide ion.
            intensity: The intensity of the peptide ion.
        """
        self.__py_ptr = ims.PyPeptideIon(sequence, charge, intensity, peptide_id)

    @property
    def sequence(self) -> PeptideSequence:
        return PeptideSequence.from_py_ptr(self.__py_ptr.sequence)

    @property
    def peptide_id(self) -> Union[None, int]:
        return self.__py_ptr.peptide_id

    @property
    def charge(self) -> int:
        return self.__py_ptr.charge

    @property
    def intensity(self) -> float:
        return self.__py_ptr.intensity

    @property
    def mz(self) -> float:
        return self.__py_ptr.mz

    @property
    def atomic_composition(self):
        return self.__py_ptr.atomic_composition()

    def calculate_isotopic_spectrum(
            self,
            mass_tolerance: float = 1e-3,
            abundance_threshold: float = 1e-8,
            max_result: int = 200,
            intensity_min: float = 1e-4
    ) -> MzSpectrum:
        """Calculate the isotopic spectrum of the peptide ion.

        Args:
            mass_tolerance: The mass tolerance for the isotopic spectrum calculation.
            abundance_threshold: The abundance threshold for the isotopic spectrum calculation.
            max_result: The maximum number of results to return.
            intensity_min: The minimum intensity of the isotopic spectrum.

        Returns:
            The isotopic spectrum of the peptide ion.
        """
        assert 0 <= abundance_threshold <= 1, f"Abundance threshold must be between 0 and 1, was: {abundance_threshold}"
        py_spec = self.__py_ptr.calculate_isotopic_spectrum(mass_tolerance, abundance_threshold, max_result, intensity_min)
        return MzSpectrum.from_py_ptr(py_spec)

    def calculate_isotopic_spectrum_annotated(
            self,
            mass_tolerance: float = 1e-3,
            abundance_threshold: float = 1e-8,
            max_result: int = 200,
            intensity_min: float = 1e-4,
    ) -> 'MzSpectrumAnnotated':
        """Calculate the isotopic spectrum of the peptide ion.

        Args:
            mass_tolerance: The mass tolerance for the isotopic spectrum calculation.
            abundance_threshold: The abundance threshold for the isotopic spectrum calculation.
            max_result: The maximum number of results to return.
            intensity_min: The minimum intensity of the isotopic spectrum.

        Returns:
            The isotopic spectrum of the peptide ion.
        """
        assert 0 <= abundance_threshold <= 1, f"Abundance threshold must be between 0 and 1, was: {abundance_threshold}"
        MzSpectrumAnnotated = _get_mz_spectrum_annotated()
        py_spec = self.__py_ptr.calculate_isotopic_spectrum_annotated(mass_tolerance, abundance_threshold, max_result, intensity_min)
        return MzSpectrumAnnotated.from_py_ptr(py_spec)

    def get_py_ptr(self):
        return self.__py_ptr

    def to_json(self) -> str:
        return self.__py_ptr.to_json()

    @classmethod
    def from_py_ptr(cls, ion: ims.PyPeptideIon):
        instance = cls.__new__(cls)
        instance.__py_ptr = ion
        return instance

    def __repr__(self):
        return f"PeptideIon(sequence={self.sequence}, charge={self.charge}, mz={self.mz}, intensity={self.intensity})"
