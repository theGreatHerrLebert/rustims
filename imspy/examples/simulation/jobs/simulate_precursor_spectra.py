import pandas as pd

from imspy.simulation.isotopes import generate_isotope_patterns_rust


def simulate_precursor_spectra(
        ions: pd.DataFrame,
        isotope_min_intensity: float,
        isotope_k: int,
        num_threads: int,
        verbose: bool = False) -> pd.DataFrame:


    if verbose:
        print("Simulating precursor isotopic distributions...")

    specs = generate_isotope_patterns_rust(
        ions['monoisotopic-mass'], ions.charge,
        min_intensity=isotope_min_intensity,
        k=isotope_k,
        centroid=True,
        num_threads=num_threads,
    )

    specs = [spec.to_jsons() for spec in specs]
    ions.insert(6, 'simulated_spectrum', specs)

    return ions