import pandas as pd

from imspy_simulation.isotopes import generate_isotope_patterns_rust, simulate_precursor_spectra


def simulate_precursor_spectra_sequence(
        ions: pd.DataFrame,
        num_threads: int = 16,
        verbose: bool = False
) -> pd.DataFrame:
    """Simulate sequence specific precursor isotopic distributions.

    Args:
        ions: DataFrame containing ions.
        num_threads: Number of threads.
        verbose: Verbosity.

    Returns:
        pd.DataFrame: DataFrame containing ions with simulated spectra.
    """

    if verbose:
        print("Simulating sequence specific precursor isotopic distributions ...")

    specs = simulate_precursor_spectra(
        ions['sequence'].values,
        ions['charge'].values,
        num_threads=num_threads,
    )

    if verbose:
        print("Serializing simulated spectra to json ...")

    specs = [spec.to_jsons() for spec in specs]
    ions.insert(5, 'simulated_spectrum', specs)

    return ions


def simulate_precursor_spectra_averagine(
        ions: pd.DataFrame,
        isotope_min_intensity: float,
        isotope_k: int,
        num_threads: int,
        verbose: bool = False) -> pd.DataFrame:

    if verbose:
        print("Simulating precursor isotopic distributions ...")

    specs = generate_isotope_patterns_rust(
        ions['monoisotopic-mass'], ions.charge,
        min_intensity=isotope_min_intensity,
        k=isotope_k,
        centroid=True,
        num_threads=num_threads,
    )

    specs = [spec.to_jsons() for spec in specs]
    ions.insert(5, 'simulated_spectrum', specs)

    return ions
