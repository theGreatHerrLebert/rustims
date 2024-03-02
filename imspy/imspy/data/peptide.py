from typing import List, Tuple, Dict, Union

import imspy_connector
ims = imspy_connector.py_peptide


class PeptideProductIonSeriesCollection:
    def __init__(self, series: List['PeptideProductIonSeries']):
        """Create a new product ion series collection.

        Args:
            series: The product ion series.
        """
        self.__ptr = ims.PyPeptideProductIonSeriesCollection([s.get_ptr() for s in series])

    @property
    def series(self) -> List['PeptideProductIonSeries']:
        return [PeptideProductIonSeries.from_py_ptr(series) for series in self.__ptr.series]

    def find_series(self, charge: int) -> Union[None, 'PeptideProductIonSeries']:
        """Find the product ion series with the given charge.

        Args:
            charge: The charge of the product ion series.

        Returns:
            The product ion series with the given charge, or None if not found.
        """
        series = self.__ptr.find_ion_series(charge)
        return PeptideProductIonSeries.from_py_ptr(series) if series else None

    def get_ptr(self):
        return self.__ptr

    @classmethod
    def from_py_ptr(cls, collection: ims.PyPeptideProductIonSeriesCollection):
        instance = cls.__new__(cls)
        instance.__ptr = collection
        return instance

    def __repr__(self):
        return f"PeptideProductIonSeriesCollection(series={self.series})"


class PeptideProductIonSeries:
    def __init__(self, charge: int, n_ions: List['PeptideProductIon'], c_ions: List['PeptideProductIon']):
        """Create a new product ion series.

        Args:
            charge: The charge of the product ions.
            n_ions: The N-terminal product ions.
            c_ions: The C-terminal product ions.
        """
        self.__ptr = ims.PyPeptideProductIonSeries(
            charge=charge,
            n_ions=[ion.get_ptr() for ion in n_ions],
            c_ions=[ion.get_ptr() for ion in c_ions]
        )

    @property
    def n_ions(self) -> List['PeptideProductIon']:
        return [PeptideProductIon.from_py_ptr(ion) for ion in self.__ptr.n_ions]

    @property
    def c_ions(self) -> List['PeptideProductIon']:
        return [PeptideProductIon.from_py_ptr(ion) for ion in self.__ptr.c_ions]

    @property
    def charge(self) -> int:
        return self.__ptr.charge

    def get_ptr(self):
        return self.__ptr

    @classmethod
    def from_py_ptr(cls, series: ims.PyPeptideProductIonSeries):
        instance = cls.__new__(cls)
        instance.__ptr = series
        return instance

    def __repr__(self):
        return (f"PeptideProductIonSeries(charge={self.charge}, n_ions={self.n_ions}, "
                f"c_ions={self.c_ions})")


class PeptideProductIon:
    def __init__(self, kind: str, sequence: str, charge: int = 1, intensity: float = 1.0):
        """Create a new product ion.

        Args:
            kind: The kind of product ion, must be one of 'a', 'b', 'c', 'x', 'y', 'z'.
            sequence: The sequence of the product ion.
            charge: The charge of the product ion.
            intensity: The intensity of the product ion.
        """
        assert kind in ['a', 'b', 'c', 'x', 'y', 'z'], (f"Invalid kind: {kind}, "
                                                        f"must be one of 'a', 'b', 'c', 'x', 'y', 'z'")
        self.__ptr = ims.PyPeptideProductIon(
            kind=kind,
            sequence=sequence,
            charge=charge,
            intensity=intensity
        )

    @property
    def kind(self) -> str:
        return self.__ptr.kind

    @property
    def sequence(self) -> str:
        return self.__ptr.sequence

    @property
    def charge(self) -> int:
        return self.__ptr.charge

    @property
    def intensity(self) -> float:
        return self.__ptr.intensity

    @property
    def mono_isotopic_mass(self) -> float:
        return self.__ptr.mono_isotopic_mass()

    @property
    def mz(self) -> float:
        return self.__ptr.mz

    def atomic_composition(self):
        return self.__ptr.atomic_composition()

    def get_ptr(self):
        return self.__ptr

    @classmethod
    def from_py_ptr(cls, product_ion: ims.PyPeptideProductIon):
        instance = cls.__new__(cls)
        instance.__ptr = product_ion
        return instance

    def isotope_distribution(self, mass_tolerance: float = 1e-3, abundance_threshold: float = 1e-8,
                             max_result: int = 200, intensity_min: float = 1e-4) -> List[Tuple[float, float]]:
        """Calculate the isotope distribution of the product ion.

        Args:
            mass_tolerance: The mass tolerance for the isotope distribution calculation.
            abundance_threshold: The abundance threshold for the isotope distribution calculation.
            max_result: The maximum number of results to return.
            intensity_min: The minimum intensity of the isotope distribution.

        Returns:
            The isotope distribution of the product ion.
        """
        return self.__ptr.isotope_distribution(mass_tolerance, abundance_threshold, max_result, intensity_min)

    def __repr__(self):
        return (f"ProductIon(kind={self.kind}, sequence={self.sequence}, charge={self.charge}, mz={self.mz}, "
                f" intensity={self.intensity})")


class PeptideSequence:
    def __init__(self, sequence: str):
        """Create a new peptide sequence.

        Args:
            sequence: The sequence of the peptide.
        """
        self.__ptr = ims.PyPeptideSequence(sequence)

    @property
    def sequence(self) -> str:
        return self.__ptr.sequence

    @property
    def mono_isotopic_mass(self) -> float:
        return self.__ptr.mono_isotopic_mass

    @property
    def atomic_composition(self):
        return self.__ptr.atomic_composition()

    @property
    def amino_acid_count(self) -> int:
        return self.__ptr.amino_acid_count()

    def to_tokens(self, group_modifications: bool = True) -> List[str]:
        return self.__ptr.to_tokens(group_modifications)

    def to_sage_representation(self) -> Tuple[str, List[float]]:
        return self.__ptr.to_sage_representation()

    def get_ptr(self):
        return self.__ptr

    def calculate_product_ion_series(self, charge: int = 1, fragment_type: str = 'b') -> Tuple[List[PeptideProductIon], List[PeptideProductIon]]:
        """Calculate the b and y product ion series of the peptide sequence.

        Args:
            charge: The charge of the product ions.
            fragment_type: The type of the product ions, must be one of 'a', 'b', 'c', 'x', 'y', 'z'.

        Returns:
            The b and y product ion series of the peptide sequence.
        """
        fragment_type = fragment_type.lower()
        assert fragment_type in ['a', 'b', 'c', 'x', 'y', 'z'], (f"Invalid fragment type: {fragment_type}, "
                                                                 f"must be one of 'a', 'b', 'c', 'x', 'y', 'z'")

        n_ions, c_ions = self.__ptr.calculate_product_ion_series(charge, fragment_type)
        return [PeptideProductIon.from_py_ptr(ion) for ion in n_ions], [PeptideProductIon.from_py_ptr(ion) for ion in c_ions][::-1]

    def associate_fragment_ion_series_with_prosit_intensities(
            self, flat_intensities: List[float],
            charge: int = 2,
            fragment_type: str = "b",
            normalize: bool = True,
            half_charge_one: bool = True) \
            -> PeptideProductIonSeriesCollection:
        """Associate the peptide sequence with predicted intensities.

        Args:
            flat_intensities: The flat intensities.
            charge: The charge of the product ions.
            fragment_type: The type of the product ions, must be one of 'a', 'b', 'c', 'x', 'y', 'z'.
            normalize: Whether to normalize the intensities.
            half_charge_one: Whether to use half charge one.

        Returns:
            The b and y product ion series of the peptide sequence.
        """
        fragment_type = fragment_type.lower()
        assert fragment_type in ['a', 'b', 'c', 'x', 'y', 'z'], (f"Invalid fragment type: {fragment_type}, "
                                                                 f"must be one of 'a', 'b', 'c', 'x', 'y', 'z'")

        result = self.__ptr.associate_with_predicted_intensities(flat_intensities,
                                                                 charge, fragment_type, normalize, half_charge_one)

        return PeptideProductIonSeriesCollection.from_py_ptr(result)

    @classmethod
    def fom_py_ptr(cls, seq: ims.PyPeptideSequence):
        instance = cls.__new__(cls)
        instance.__ptr = seq
        return instance

    def __repr__(self):
        return f"AminoAcidSequence(sequence={self.sequence}, mono_isotopic_mass={self.mono_isotopic_mass})"