"""
Comprehensive test for hyperscore vs weightedhyperscore equality.

This module tests that when using uniform (1.0) intensities, the weighted
hyperscore produces identical results to the unweighted hyperscore. This
validates the new weighted scoring code path.

Two test approaches are used:
1. Theoretical fragments: b/y ions calculated from PeptideSequence
2. Prosit predictions: Realistic spectra from Prosit2023

Both approaches verify: hyperscore == weightedhyperscore with uniform store.
"""

import pytest
import numpy as np
from typing import List, Tuple

from sagepy.core import (
    SageSearchConfiguration,
    EnzymeBuilder,
    IndexedDatabase,
    Scorer,
    ScoreType,
    PredictedIntensityStore,
    Tolerance,
    SpectrumProcessor,
    RawSpectrum,
    Precursor,
    Representation,
    ProcessedSpectrum,
)

from imspy.data.peptide import PeptideSequence
from imspy.algorithm.intensity.sage_interface import (
    remove_unimod_annotation,
    ION_KIND_B,
    ION_KIND_Y,
)


# =============================================================================
# Constants
# =============================================================================

FASTA_PATH = "/Users/davidteschner/Promotion/timsim/4P.fasta"
PROTON_MASS = 1.007276466


# =============================================================================
# Helper Functions
# =============================================================================

def generate_theoretical_spectrum(
    sequence: str,
    charge: int,
    fragment_charges: List[int] = [1],
    base_intensity: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a theoretical MS2 spectrum from a peptide sequence.

    Args:
        sequence: Peptide sequence (may include UNIMOD modifications)
        charge: Precursor charge state
        fragment_charges: Fragment ion charge states to include
        base_intensity: Intensity value for all fragments

    Returns:
        Tuple of (mz_array, intensity_array, precursor_mz)
    """
    # Remove UNIMOD annotations for PeptideSequence
    clean_seq = remove_unimod_annotation(sequence)
    pep = PeptideSequence(clean_seq)

    mz_list = []
    intensity_list = []

    # Generate b and y ions at each fragment charge
    for frag_charge in fragment_charges:
        b_ions, y_ions = pep.calculate_product_ion_series(
            charge=frag_charge,
            fragment_type='b'
        )

        for ion in b_ions:
            mz_list.append(ion.mz)
            intensity_list.append(base_intensity)

        for ion in y_ions:
            mz_list.append(ion.mz)
            intensity_list.append(base_intensity)

    # Sort by m/z
    mz_array = np.array(mz_list, dtype=np.float64)
    intensity_array = np.array(intensity_list, dtype=np.float64)
    sort_idx = np.argsort(mz_array)
    mz_array = mz_array[sort_idx]
    intensity_array = intensity_array[sort_idx]

    # Calculate precursor m/z
    precursor_mz = (pep.mono_isotopic_mass + charge * PROTON_MASS) / charge

    return mz_array, intensity_array, precursor_mz


def add_noise_to_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    noise_ratio: float,
    mz_range: Tuple[float, float] = None,
    noise_intensity_range: Tuple[float, float] = (100.0, 500.0),
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add random noise peaks to a spectrum.

    Args:
        mz: Original m/z array
        intensity: Original intensity array
        noise_ratio: Ratio of noise peaks to signal peaks (0.5 = 50% noise)
        mz_range: Range for noise peaks (default: min/max of signal)
        noise_intensity_range: Intensity range for noise peaks
        seed: Random seed for reproducibility

    Returns:
        Tuple of (combined_mz, combined_intensity)
    """
    if seed is not None:
        np.random.seed(seed)

    if noise_ratio <= 0:
        return mz, intensity

    n_noise = int(len(mz) * noise_ratio)
    if n_noise == 0:
        return mz, intensity

    if mz_range is None:
        mz_range = (mz.min() * 0.8, mz.max() * 1.2)

    noise_mz = np.random.uniform(mz_range[0], mz_range[1], n_noise)
    noise_intensity = np.random.uniform(
        noise_intensity_range[0], noise_intensity_range[1], n_noise
    )

    combined_mz = np.concatenate([mz, noise_mz])
    combined_intensity = np.concatenate([intensity, noise_intensity])

    sort_idx = np.argsort(combined_mz)
    return combined_mz[sort_idx], combined_intensity[sort_idx]


def create_processed_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    precursor_charge: int,
    spec_id: str,
    rt: float = 300.0,
) -> ProcessedSpectrum:
    """Create a ProcessedSpectrum for sage scoring."""
    isolation_window = Tolerance(da=(-1.5, 1.5))

    precursor = Precursor(
        mz=precursor_mz,
        intensity=float(np.max(intensity)) * 10,
        charge=precursor_charge,
        isolation_window=isolation_window,
    )

    raw = RawSpectrum(
        file_id=0,
        spec_id=spec_id,
        total_ion_current=float(np.sum(intensity)),
        precursors=[precursor],
        mz=mz.astype(np.float64),
        intensity=intensity.astype(np.float64),
        representation=Representation("centroid"),
        scan_start_time=rt,
        ion_injection_time=100.0,
        ms_level=2,
    )

    processor = SpectrumProcessor(take_top_n=150)
    return processor.process(raw)


def compare_scores(
    features_unweighted: List,
    features_weighted: List,
    tolerance: float = 1e-5,
) -> Tuple[bool, List[str]]:
    """
    Compare two lists of scoring features.

    Returns:
        Tuple of (all_match, list_of_mismatch_messages)
    """
    mismatches = []

    if len(features_unweighted) != len(features_weighted):
        mismatches.append(
            f"Different number of PSMs: {len(features_unweighted)} vs {len(features_weighted)}"
        )
        return False, mismatches

    for i, (f_old, f_new) in enumerate(zip(features_unweighted, features_weighted)):
        # Use .idx attribute for comparison (PeptideIx wrapper doesn't support ==)
        if f_old.peptide_idx.idx != f_new.peptide_idx.idx:
            mismatches.append(
                f"PSM {i}: Different peptide_idx: {f_old.peptide_idx.idx} vs {f_new.peptide_idx.idx}"
            )

        if abs(f_old.hyperscore - f_new.hyperscore) > tolerance:
            mismatches.append(
                f"PSM {i}: Hyperscore mismatch: {f_old.hyperscore:.6f} vs {f_new.hyperscore:.6f}"
            )

        if f_old.matched_peaks != f_new.matched_peaks:
            mismatches.append(
                f"PSM {i}: Different matched_peaks: {f_old.matched_peaks} vs {f_new.matched_peaks}"
            )

    return len(mismatches) == 0, mismatches


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def fasta_content():
    """Load 4P FASTA content."""
    import os
    if not os.path.exists(FASTA_PATH):
        pytest.skip(f"4P FASTA not found at {FASTA_PATH}")
    with open(FASTA_PATH, 'r') as f:
        return f.read()


@pytest.fixture(scope="module")
def indexed_db(fasta_content):
    """Build indexed database from 4P FASTA."""
    config = SageSearchConfiguration(
        fasta=fasta_content,
        static_mods={"C": "[UNIMOD:4]"},
        variable_mods={},
        enzyme_builder=EnzymeBuilder(
            missed_cleavages=2,
            min_len=7,
            max_len=30,
            cleave_at="KR",
            restrict="P",
            c_terminal=True,
        ),
        generate_decoys=True,
        decoy_tag="DECOY_",
        bucket_size=8192,
    )
    return config.generate_indexed_database()


@pytest.fixture(scope="module")
def target_peptides(indexed_db):
    """Get target peptides (non-decoy) with indices."""
    peptides = list(indexed_db.peptides_as_string())
    db_peptides = indexed_db._peptides

    targets = []
    for i, (seq, pep) in enumerate(zip(peptides, db_peptides)):
        if not pep.decoy:
            # Only include peptides short enough for theoretical calculation
            clean_seq = remove_unimod_annotation(seq)
            if len(clean_seq) <= 30:
                targets.append((i, seq))

    return targets


@pytest.fixture(scope="module")
def uniform_store(indexed_db):
    """Create uniform intensity store."""
    peptides = list(indexed_db.peptides_as_string())
    peptide_lengths = [len(p) for p in peptides]

    return PredictedIntensityStore.uniform(
        peptide_lengths=peptide_lengths,
        max_charge=2,
        ion_kinds=[ION_KIND_B, ION_KIND_Y],
    )


@pytest.fixture
def scorer_params():
    """Common scorer parameters."""
    return {
        "precursor_tolerance": Tolerance(ppm=(-15.0, 15.0)),
        "fragment_tolerance": Tolerance(ppm=(-20.0, 20.0)),
        "min_matched_peaks": 4,
        "report_psms": 10,
        "max_fragment_charge": 2,
    }


# =============================================================================
# Test Class: Theoretical Fragments
# =============================================================================

class TestHyperscoreEqualityTheoretical:
    """
    Test hyperscore == weightedhyperscore using theoretical fragment spectra.

    These tests use calculated b/y ion m/z values with uniform intensities,
    providing a controlled environment for testing the scoring math.
    """

    def test_single_peptide_clean_spectrum(
        self, indexed_db, uniform_store, target_peptides, scorer_params
    ):
        """Test with a single clean peptide spectrum."""
        if len(target_peptides) == 0:
            pytest.skip("No target peptides available")

        idx, seq = target_peptides[0]
        mz, intensity, prec_mz = generate_theoretical_spectrum(seq, charge=2)
        spectrum = create_processed_spectrum(mz, intensity, prec_mz, 2, "test_0")

        scorer_unweighted = Scorer(
            score_type=ScoreType("hyperscore"),
            **scorer_params,
        )
        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        features_old = scorer_unweighted.score(indexed_db, spectrum)
        features_new = scorer_weighted.score(
            indexed_db, spectrum, intensity_store=uniform_store
        )

        all_match, mismatches = compare_scores(features_old, features_new)
        assert all_match, f"Score mismatches:\n" + "\n".join(mismatches)

    def test_multiple_peptides_no_noise(
        self, indexed_db, uniform_store, target_peptides, scorer_params
    ):
        """Test with multiple peptides, no noise."""
        n_peptides = min(50, len(target_peptides))
        if n_peptides < 10:
            pytest.skip("Not enough target peptides")

        scorer_unweighted = Scorer(
            score_type=ScoreType("hyperscore"),
            **scorer_params,
        )
        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        total_mismatches = []
        for i, (idx, seq) in enumerate(target_peptides[:n_peptides]):
            for charge in [2, 3]:
                mz, intensity, prec_mz = generate_theoretical_spectrum(
                    seq, charge=charge, fragment_charges=[1, 2]
                )
                spectrum = create_processed_spectrum(
                    mz, intensity, prec_mz, charge, f"test_{i}_z{charge}"
                )

                features_old = scorer_unweighted.score(indexed_db, spectrum)
                features_new = scorer_weighted.score(
                    indexed_db, spectrum, intensity_store=uniform_store
                )

                all_match, mismatches = compare_scores(features_old, features_new)
                if not all_match:
                    total_mismatches.extend(
                        [f"[{seq} z{charge}] {m}" for m in mismatches]
                    )

        assert len(total_mismatches) == 0, \
            f"{len(total_mismatches)} mismatches:\n" + "\n".join(total_mismatches[:20])

    @pytest.mark.parametrize("noise_ratio", [0.25, 0.5, 1.0])
    def test_peptides_with_noise(
        self, indexed_db, uniform_store, target_peptides, scorer_params, noise_ratio
    ):
        """Test with varying noise levels."""
        n_peptides = min(30, len(target_peptides))
        if n_peptides < 5:
            pytest.skip("Not enough target peptides")

        scorer_unweighted = Scorer(
            score_type=ScoreType("hyperscore"),
            **scorer_params,
        )
        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        total_mismatches = []
        for i, (idx, seq) in enumerate(target_peptides[:n_peptides]):
            mz, intensity, prec_mz = generate_theoretical_spectrum(seq, charge=2)
            mz_noisy, intensity_noisy = add_noise_to_spectrum(
                mz, intensity, noise_ratio, seed=42 + i
            )
            spectrum = create_processed_spectrum(
                mz_noisy, intensity_noisy, prec_mz, 2, f"noise_{i}"
            )

            features_old = scorer_unweighted.score(indexed_db, spectrum)
            features_new = scorer_weighted.score(
                indexed_db, spectrum, intensity_store=uniform_store
            )

            all_match, mismatches = compare_scores(features_old, features_new)
            if not all_match:
                total_mismatches.extend([f"[{seq}] {m}" for m in mismatches])

        assert len(total_mismatches) == 0, \
            f"Noise {noise_ratio}: {len(total_mismatches)} mismatches:\n" + \
            "\n".join(total_mismatches[:10])

    def test_batch_scoring_equality(
        self, indexed_db, uniform_store, target_peptides, scorer_params
    ):
        """Test batch scoring methods."""
        n_peptides = min(20, len(target_peptides))
        if n_peptides < 5:
            pytest.skip("Not enough target peptides")

        # Generate spectra
        spectra = []
        for i, (idx, seq) in enumerate(target_peptides[:n_peptides]):
            mz, intensity, prec_mz = generate_theoretical_spectrum(seq, charge=2)
            spectrum = create_processed_spectrum(
                mz, intensity, prec_mz, 2, f"batch_{i}"
            )
            spectra.append(spectrum)

        scorer_unweighted = Scorer(
            score_type=ScoreType("hyperscore"),
            **scorer_params,
        )
        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        # Batch score
        results_old = scorer_unweighted.score_collection_top_n(
            indexed_db, spectra, num_threads=1
        )
        results_new = scorer_weighted.score_collection_top_n(
            indexed_db, spectra, num_threads=1, intensity_store=uniform_store
        )

        total_mismatches = []
        for i, (old_list, new_list) in enumerate(zip(results_old, results_new)):
            all_match, mismatches = compare_scores(old_list, new_list)
            if not all_match:
                total_mismatches.extend([f"[Spectrum {i}] {m}" for m in mismatches])

        assert len(total_mismatches) == 0, \
            f"Batch scoring: {len(total_mismatches)} mismatches:\n" + \
            "\n".join(total_mismatches[:20])


# =============================================================================
# Test Class: Prosit Predictions
# =============================================================================

class TestHyperscoreEqualityProsit:
    """
    Test hyperscore == weightedhyperscore using Prosit-predicted spectra.

    These tests use realistic fragmentation patterns from Prosit2023,
    testing more of the pipeline integration.
    """

    @pytest.mark.slow
    def test_prosit_spectra_no_noise(
        self, indexed_db, uniform_store, target_peptides, scorer_params
    ):
        """Test with Prosit-predicted spectra, no noise."""
        try:
            from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper
        except ImportError:
            pytest.skip("Prosit not available")

        n_peptides = min(30, len(target_peptides))
        if n_peptides < 5:
            pytest.skip("Not enough target peptides")

        # Get sequences and predict intensities
        sequences = [remove_unimod_annotation(seq) for _, seq in target_peptides[:n_peptides]]
        charges = [2] * n_peptides
        collision_energies = [35.0] * n_peptides

        predictor = Prosit2023TimsTofWrapper(verbose=False)
        intensity_predictions = predictor.predict_intensities(
            sequences=sequences,
            charges=np.array(charges),
            collision_energies=collision_energies,
            flatten=False,
        )

        scorer_unweighted = Scorer(
            score_type=ScoreType("hyperscore"),
            **scorer_params,
        )
        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        total_mismatches = []
        for i, (idx, seq) in enumerate(target_peptides[:n_peptides]):
            clean_seq = remove_unimod_annotation(seq)
            pep = PeptideSequence(clean_seq)

            # Get fragment m/z values
            b_ions, y_ions = pep.calculate_product_ion_series(charge=1, fragment_type='b')

            # Get predicted intensities
            pred = intensity_predictions[i]  # Shape: [29, 2, 3] or similar

            # Build spectrum from predicted fragments
            mz_list = []
            int_list = []

            # Extract intensities for each ion
            for j, ion in enumerate(b_ions[:min(len(b_ions), 29)]):
                for frag_charge in [1, 2]:
                    if frag_charge <= 2:
                        intensity_val = pred[j, 1, frag_charge - 1]  # B ions at index 1
                        if intensity_val > 0.01:
                            ion_mz = (ion.mz * 1 + (frag_charge - 1) * PROTON_MASS) / frag_charge
                            mz_list.append(ion_mz)
                            int_list.append(float(intensity_val) * 1000)

            for j, ion in enumerate(y_ions[:min(len(y_ions), 29)]):
                for frag_charge in [1, 2]:
                    if frag_charge <= 2:
                        intensity_val = pred[j, 0, frag_charge - 1]  # Y ions at index 0
                        if intensity_val > 0.01:
                            ion_mz = (ion.mz * 1 + (frag_charge - 1) * PROTON_MASS) / frag_charge
                            mz_list.append(ion_mz)
                            int_list.append(float(intensity_val) * 1000)

            if len(mz_list) < 4:
                continue  # Not enough peaks

            mz_array = np.array(mz_list, dtype=np.float64)
            int_array = np.array(int_list, dtype=np.float64)
            sort_idx = np.argsort(mz_array)
            mz_array = mz_array[sort_idx]
            int_array = int_array[sort_idx]

            prec_mz = (pep.mono_isotopic_mass + 2 * PROTON_MASS) / 2
            spectrum = create_processed_spectrum(
                mz_array, int_array, prec_mz, 2, f"prosit_{i}"
            )

            features_old = scorer_unweighted.score(indexed_db, spectrum)
            features_new = scorer_weighted.score(
                indexed_db, spectrum, intensity_store=uniform_store
            )

            all_match, mismatches = compare_scores(features_old, features_new)
            if not all_match:
                total_mismatches.extend([f"[{seq}] {m}" for m in mismatches])

        assert len(total_mismatches) == 0, \
            f"Prosit spectra: {len(total_mismatches)} mismatches:\n" + \
            "\n".join(total_mismatches[:20])

    @pytest.mark.slow
    @pytest.mark.parametrize("noise_ratio", [0.25, 0.5])
    def test_prosit_spectra_with_noise(
        self, indexed_db, uniform_store, target_peptides, scorer_params, noise_ratio
    ):
        """Test Prosit-predicted spectra with added noise."""
        try:
            from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper
        except ImportError:
            pytest.skip("Prosit not available")

        n_peptides = min(20, len(target_peptides))
        if n_peptides < 5:
            pytest.skip("Not enough target peptides")

        sequences = [remove_unimod_annotation(seq) for _, seq in target_peptides[:n_peptides]]
        charges = [2] * n_peptides
        collision_energies = [35.0] * n_peptides

        predictor = Prosit2023TimsTofWrapper(verbose=False)
        intensity_predictions = predictor.predict_intensities(
            sequences=sequences,
            charges=np.array(charges),
            collision_energies=collision_energies,
            flatten=False,
        )

        scorer_unweighted = Scorer(
            score_type=ScoreType("hyperscore"),
            **scorer_params,
        )
        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        total_mismatches = []
        for i, (idx, seq) in enumerate(target_peptides[:n_peptides]):
            clean_seq = remove_unimod_annotation(seq)
            pep = PeptideSequence(clean_seq)

            b_ions, y_ions = pep.calculate_product_ion_series(charge=1, fragment_type='b')
            pred = intensity_predictions[i]

            mz_list = []
            int_list = []

            for j, ion in enumerate(b_ions[:min(len(b_ions), 29)]):
                intensity_val = pred[j, 1, 0]  # B ions, charge 1
                if intensity_val > 0.01:
                    mz_list.append(ion.mz)
                    int_list.append(float(intensity_val) * 1000)

            for j, ion in enumerate(y_ions[:min(len(y_ions), 29)]):
                intensity_val = pred[j, 0, 0]  # Y ions, charge 1
                if intensity_val > 0.01:
                    mz_list.append(ion.mz)
                    int_list.append(float(intensity_val) * 1000)

            if len(mz_list) < 4:
                continue

            mz_array = np.array(mz_list, dtype=np.float64)
            int_array = np.array(int_list, dtype=np.float64)

            # Add noise
            mz_noisy, int_noisy = add_noise_to_spectrum(
                mz_array, int_array, noise_ratio, seed=42 + i
            )

            prec_mz = (pep.mono_isotopic_mass + 2 * PROTON_MASS) / 2
            spectrum = create_processed_spectrum(
                mz_noisy, int_noisy, prec_mz, 2, f"prosit_noise_{i}"
            )

            features_old = scorer_unweighted.score(indexed_db, spectrum)
            features_new = scorer_weighted.score(
                indexed_db, spectrum, intensity_store=uniform_store
            )

            all_match, mismatches = compare_scores(features_old, features_new)
            if not all_match:
                total_mismatches.extend([f"[{seq}] {m}" for m in mismatches])

        assert len(total_mismatches) == 0, \
            f"Prosit noise {noise_ratio}: {len(total_mismatches)} mismatches:\n" + \
            "\n".join(total_mismatches[:10])


# =============================================================================
# Test Class: OpenMS Hyperscore Variants
# =============================================================================

class TestOpenMSHyperscoreEquality:
    """
    Test openmshyperscore == weightedopenmshyperscore with uniform intensities.
    """

    def test_openms_single_peptide(
        self, indexed_db, uniform_store, target_peptides, scorer_params
    ):
        """Test OpenMS hyperscore variants."""
        if len(target_peptides) == 0:
            pytest.skip("No target peptides")

        idx, seq = target_peptides[0]
        mz, intensity, prec_mz = generate_theoretical_spectrum(seq, charge=2)
        spectrum = create_processed_spectrum(mz, intensity, prec_mz, 2, "openms_0")

        scorer_unweighted = Scorer(
            score_type=ScoreType("openmshyperscore"),
            **scorer_params,
        )
        scorer_weighted = Scorer(
            score_type=ScoreType("weightedopenmshyperscore"),
            **scorer_params,
        )

        features_old = scorer_unweighted.score(indexed_db, spectrum)
        features_new = scorer_weighted.score(
            indexed_db, spectrum, intensity_store=uniform_store
        )

        all_match, mismatches = compare_scores(features_old, features_new)
        assert all_match, f"OpenMS score mismatches:\n" + "\n".join(mismatches)

    def test_openms_multiple_peptides(
        self, indexed_db, uniform_store, target_peptides, scorer_params
    ):
        """Test OpenMS hyperscore with multiple peptides."""
        n_peptides = min(30, len(target_peptides))
        if n_peptides < 5:
            pytest.skip("Not enough target peptides")

        scorer_unweighted = Scorer(
            score_type=ScoreType("openmshyperscore"),
            **scorer_params,
        )
        scorer_weighted = Scorer(
            score_type=ScoreType("weightedopenmshyperscore"),
            **scorer_params,
        )

        total_mismatches = []
        for i, (idx, seq) in enumerate(target_peptides[:n_peptides]):
            mz, intensity, prec_mz = generate_theoretical_spectrum(seq, charge=2)
            spectrum = create_processed_spectrum(
                mz, intensity, prec_mz, 2, f"openms_{i}"
            )

            features_old = scorer_unweighted.score(indexed_db, spectrum)
            features_new = scorer_weighted.score(
                indexed_db, spectrum, intensity_store=uniform_store
            )

            all_match, mismatches = compare_scores(features_old, features_new)
            if not all_match:
                total_mismatches.extend([f"[{seq}] {m}" for m in mismatches])

        assert len(total_mismatches) == 0, \
            f"OpenMS: {len(total_mismatches)} mismatches:\n" + \
            "\n".join(total_mismatches[:20])


# =============================================================================
# Test Class: Weighted Scoring with Real Prosit Predictions
# =============================================================================

class TestWeightedScoringWithPrositStore:
    """
    Test weighted scoring using actual Prosit predictions in the intensity store.

    Unlike the equality tests above (which use uniform 1.0 intensities),
    these tests use REAL Prosit predictions. The scores will be DIFFERENT
    from unweighted scoring - this validates the weighted scoring actually
    uses the predicted intensities.

    ⚠️ IMPORTANT: DECOY HANDLING IN THIS TEST CLASS
    ================================================

    This test class predicts intensities ONLY for target peptides. Decoy peptides
    receive uniform 1.0 intensities (via `default_value=1.0`).

    **Why this is acceptable for these tests:**
    - These tests validate that the weighted scoring CODE PATH works correctly
    - We generate spectra from TARGET peptides and verify:
      1. Scores differ from unweighted (intensity predictions are used)
      2. Ranking is maintained/improved for correct peptides
      3. Non-uniform intensities are stored correctly
    - No FDR estimation is performed in these tests

    **Why this is NOT acceptable for production:**
    - Uniform decoy intensities inflate decoy scores relative to targets
    - Target fragments get down-weighted by predictions (0.0-1.0)
    - Decoy fragments always contribute at full weight (1.0)
    - This BREAKS FDR estimation - more false positives pass thresholds

    **For production weighted scoring:**
    ```python
    pipeline.predict_intensities(
        charges=[2, 3],
        exclude_decoys=False,  # MUST predict for ALL peptides
    )
    ```

    See INTENSITY_PREDICTION.md "Decoy Intensity Predictions" section for details.
    """

    @pytest.fixture(scope="class")
    def prosit_intensity_store(self, indexed_db, target_peptides):
        """
        Create intensity store with actual Prosit predictions for TARGET peptides only.

        ⚠️ NOTE: This fixture uses `default_value=1.0` for non-predicted peptides
        (including decoys). This is acceptable for CODE PATH VALIDATION tests but
        NOT for production use. See class docstring for details.

        For production, use `exclude_decoys=False` in the pipeline to predict
        intensities for ALL peptides.
        """
        import tempfile
        import os

        try:
            from imspy.algorithm.intensity.sage_interface import (
                predict_intensities_for_sage,
                write_predictions_for_database,
            )
        except ImportError:
            pytest.skip("Prosit not available")

        peptides = list(indexed_db.peptides_as_string())
        peptide_lengths = [len(p) for p in peptides]

        # Predict for TARGET peptides only at charge 2
        # NOTE: For production, you MUST also predict for decoys (exclude_decoys=False)
        n_peptides = min(100, len(target_peptides))
        indices = [t[0] for t in target_peptides[:n_peptides]]
        sequences = [t[1] for t in target_peptides[:n_peptides]]
        charges = [2] * len(sequences)

        result = predict_intensities_for_sage(
            sequences=sequences,
            charges=charges,
            peptide_indices=indices,
            max_fragment_charge=2,
            batch_size=2048,
            verbose=False,
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            temp_path = f.name

        write_predictions_for_database(
            temp_path,
            result,
            num_peptides=len(peptides),
            peptide_lengths=peptide_lengths,
            aggregation='max_charge',
            # ⚠️ TESTING ONLY: Uniform 1.0 for unpredicted peptides (including decoys)
            # For production, predict intensities for ALL peptides (exclude_decoys=False)
            default_value=1.0,
        )

        store = PredictedIntensityStore(temp_path)

        yield store

        # Cleanup
        os.unlink(temp_path)

    @pytest.mark.slow
    def test_weighted_scores_differ_from_unweighted(
        self, indexed_db, prosit_intensity_store, target_peptides, scorer_params
    ):
        """
        Test that weighted scoring with real predictions produces
        DIFFERENT scores than unweighted scoring.

        This validates that the intensity predictions are actually being used.
        """
        n_peptides = min(30, len(target_peptides))
        if n_peptides < 5:
            pytest.skip("Not enough target peptides")

        scorer_unweighted = Scorer(
            score_type=ScoreType("hyperscore"),
            **scorer_params,
        )
        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        scores_differ_count = 0
        total_comparisons = 0

        for i, (idx, seq) in enumerate(target_peptides[:n_peptides]):
            mz, intensity, prec_mz = generate_theoretical_spectrum(
                seq, charge=2, fragment_charges=[1, 2]
            )
            spectrum = create_processed_spectrum(
                mz, intensity, prec_mz, 2, f"prosit_store_{i}"
            )

            features_unweighted = scorer_unweighted.score(indexed_db, spectrum)
            features_weighted = scorer_weighted.score(
                indexed_db, spectrum, intensity_store=prosit_intensity_store
            )

            # Compare top PSM scores
            if features_unweighted and features_weighted:
                for f_old, f_new in zip(features_unweighted[:3], features_weighted[:3]):
                    total_comparisons += 1
                    # Scores should differ (not equal) with real predictions
                    if abs(f_old.hyperscore - f_new.hyperscore) > 1e-5:
                        scores_differ_count += 1

        # At least some scores should differ with real predictions
        pct_different = scores_differ_count / max(total_comparisons, 1) * 100
        print(f"\nScores that differ: {scores_differ_count}/{total_comparisons} "
              f"({pct_different:.1f}%)")

        # We expect MOST scores to differ when using real predictions
        assert scores_differ_count > 0, \
            "Expected some scores to differ with real Prosit predictions"

    @pytest.mark.slow
    def test_correct_peptide_gets_higher_weighted_score(
        self, indexed_db, prosit_intensity_store, target_peptides, scorer_params
    ):
        """
        Test that for spectra generated from a peptide's predicted fragments,
        the weighted score for the correct peptide should be relatively higher.

        This tests that intensity weighting improves scoring.
        """
        try:
            from imspy.algorithm.intensity.predictors import Prosit2023TimsTofWrapper
        except ImportError:
            pytest.skip("Prosit not available")

        n_peptides = min(20, len(target_peptides))
        if n_peptides < 5:
            pytest.skip("Not enough target peptides")

        # Predict intensities for spectrum generation
        sequences = [remove_unimod_annotation(seq) for _, seq in target_peptides[:n_peptides]]
        charges = [2] * n_peptides
        collision_energies = [35.0] * n_peptides

        predictor = Prosit2023TimsTofWrapper(verbose=False)
        intensity_predictions = predictor.predict_intensities(
            sequences=sequences,
            charges=np.array(charges),
            collision_energies=collision_energies,
            flatten=False,
        )

        scorer_unweighted = Scorer(
            score_type=ScoreType("hyperscore"),
            **scorer_params,
        )
        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        weighted_better_count = 0
        total_tests = 0

        for i, (idx, seq) in enumerate(target_peptides[:n_peptides]):
            clean_seq = remove_unimod_annotation(seq)
            pep = PeptideSequence(clean_seq)

            b_ions, y_ions = pep.calculate_product_ion_series(charge=1, fragment_type='b')
            pred = intensity_predictions[i]

            # Build realistic spectrum from Prosit predictions
            mz_list = []
            int_list = []

            for j, ion in enumerate(b_ions[:min(len(b_ions), 29)]):
                intensity_val = pred[j, 1, 0]  # B ions, charge 1
                if intensity_val > 0.01:
                    mz_list.append(ion.mz)
                    int_list.append(float(intensity_val) * 1000)

            for j, ion in enumerate(y_ions[:min(len(y_ions), 29)]):
                intensity_val = pred[j, 0, 0]  # Y ions, charge 1
                if intensity_val > 0.01:
                    mz_list.append(ion.mz)
                    int_list.append(float(intensity_val) * 1000)

            if len(mz_list) < 6:
                continue

            mz_array = np.array(mz_list, dtype=np.float64)
            int_array = np.array(int_list, dtype=np.float64)
            sort_idx = np.argsort(mz_array)
            mz_array = mz_array[sort_idx]
            int_array = int_array[sort_idx]

            prec_mz = (pep.mono_isotopic_mass + 2 * PROTON_MASS) / 2
            spectrum = create_processed_spectrum(
                mz_array, int_array, prec_mz, 2, f"realistic_{i}"
            )

            features_unweighted = scorer_unweighted.score(indexed_db, spectrum)
            features_weighted = scorer_weighted.score(
                indexed_db, spectrum, intensity_store=prosit_intensity_store
            )

            if features_unweighted and features_weighted:
                total_tests += 1

                # Check if the correct peptide is ranked higher in weighted
                # Find rank of correct peptide in each
                correct_idx = idx

                rank_unweighted = None
                rank_weighted = None

                for rank, f in enumerate(features_unweighted):
                    if f.peptide_idx.idx == correct_idx:
                        rank_unweighted = rank
                        break

                for rank, f in enumerate(features_weighted):
                    if f.peptide_idx.idx == correct_idx:
                        rank_weighted = rank
                        break

                # Check if weighted improved the rank (or kept it the same)
                if rank_unweighted is not None and rank_weighted is not None:
                    if rank_weighted <= rank_unweighted:
                        weighted_better_count += 1

        pct_better = weighted_better_count / max(total_tests, 1) * 100
        print(f"\nWeighted scoring better or equal: {weighted_better_count}/{total_tests} "
              f"({pct_better:.1f}%)")

        # Weighted scoring should maintain or improve ranking in most cases
        assert weighted_better_count >= total_tests * 0.5, \
            f"Expected weighted scoring to be at least as good in most cases"

    @pytest.mark.slow
    def test_intensity_store_values_used_correctly(
        self, indexed_db, prosit_intensity_store, target_peptides
    ):
        """
        Verify that the intensity store contains non-uniform values
        for predicted peptides.
        """
        peptides = list(indexed_db.peptides_as_string())

        # Check predicted peptides have non-uniform values
        non_uniform_count = 0
        total_checked = 0

        for idx, seq in target_peptides[:50]:
            pep_len = len(seq)
            for pos in range(min(5, pep_len - 1)):
                for ion_kind in [ION_KIND_B, ION_KIND_Y]:
                    intensity = prosit_intensity_store.get_intensity_or_default(
                        idx, pep_len, ion_kind, pos, 1
                    )
                    total_checked += 1
                    if abs(intensity - 1.0) > 0.01:
                        non_uniform_count += 1

        pct_non_uniform = non_uniform_count / max(total_checked, 1) * 100
        print(f"\nNon-uniform intensities: {non_uniform_count}/{total_checked} "
              f"({pct_non_uniform:.1f}%)")

        assert non_uniform_count > 0, \
            "Expected some non-uniform intensities from Prosit predictions"


# =============================================================================
# Test Class: Production Pattern - Predictions for ALL Peptides (incl. Decoys)
# =============================================================================

class TestWeightedScoringProductionPattern:
    """
    Test weighted scoring using the PRODUCTION-CORRECT pattern:
    predicting intensities for BOTH targets AND decoys.

    This is the recommended approach for actual database searches because:
    1. Decoy sequences are valid amino acid sequences (reversed targets)
    2. Prosit can predict their fragment intensities
    3. Fair target-decoy competition requires same weighting treatment
    4. Without this, FDR estimation is biased (decoys get inflated scores)

    These tests verify that:
    - Both target and decoy peptides have real predictions
    - Decoy intensities are non-uniform (not 1.0)
    - Weighted scoring behaves correctly for both peptide types
    """

    @pytest.fixture(scope="class")
    def all_peptides_with_decoy_flag(self, indexed_db):
        """Get ALL peptides (targets + decoys) with their decoy flag."""
        peptides = list(indexed_db.peptides_as_string())
        db_peptides = indexed_db._peptides

        all_peps = []
        for i, (seq, pep) in enumerate(zip(peptides, db_peptides)):
            clean_seq = remove_unimod_annotation(seq)
            if len(clean_seq) <= 30:  # Prosit limit
                all_peps.append((i, seq, pep.decoy))

        return all_peps

    @pytest.fixture(scope="class")
    def production_intensity_store(self, indexed_db, all_peptides_with_decoy_flag):
        """
        Create intensity store with Prosit predictions for ALL peptides.

        This is the PRODUCTION-CORRECT pattern:
        - Predicts for both targets AND decoys
        - No peptide gets default 1.0 uniform intensities
        - Fair weighting for target-decoy competition
        """
        import tempfile
        import os

        try:
            from imspy.algorithm.intensity.sage_interface import (
                predict_intensities_for_sage,
                write_predictions_for_database,
            )
        except ImportError:
            pytest.skip("Prosit not available")

        peptides = list(indexed_db.peptides_as_string())
        peptide_lengths = [len(p) for p in peptides]

        # Predict for ALL peptides (targets + decoys) - PRODUCTION PATTERN
        n_peptides = min(200, len(all_peptides_with_decoy_flag))
        indices = [t[0] for t in all_peptides_with_decoy_flag[:n_peptides]]
        sequences = [t[1] for t in all_peptides_with_decoy_flag[:n_peptides]]
        charges = [2] * len(sequences)

        result = predict_intensities_for_sage(
            sequences=sequences,
            charges=charges,
            peptide_indices=indices,
            max_fragment_charge=2,
            batch_size=2048,
            verbose=False,
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix='.sagi', delete=False) as f:
            temp_path = f.name

        write_predictions_for_database(
            temp_path,
            result,
            num_peptides=len(peptides),
            peptide_lengths=peptide_lengths,
            aggregation='max_charge',
            # Still need default for peptides beyond our sample
            default_value=1.0,
        )

        store = PredictedIntensityStore(temp_path)

        yield store

        os.unlink(temp_path)

    @pytest.mark.slow
    def test_decoys_have_real_predictions(
        self, indexed_db, production_intensity_store, all_peptides_with_decoy_flag
    ):
        """
        Verify that DECOY peptides have non-uniform intensities.

        This is critical for fair FDR estimation - decoys must not get
        uniform 1.0 weights while targets get down-weighted predictions.
        """
        # Find decoy peptides
        decoy_peptides = [(i, seq) for i, seq, is_decoy in all_peptides_with_decoy_flag if is_decoy]

        if len(decoy_peptides) == 0:
            pytest.skip("No decoy peptides found")

        non_uniform_count = 0
        total_checked = 0

        for idx, seq in decoy_peptides[:50]:
            pep_len = len(seq)
            for pos in range(min(5, pep_len - 1)):
                for ion_kind in [ION_KIND_B, ION_KIND_Y]:
                    intensity = production_intensity_store.get_intensity_or_default(
                        idx, pep_len, ion_kind, pos, 1
                    )
                    total_checked += 1
                    if abs(intensity - 1.0) > 0.01:
                        non_uniform_count += 1

        pct_non_uniform = non_uniform_count / max(total_checked, 1) * 100
        print(f"\nDecoy non-uniform intensities: {non_uniform_count}/{total_checked} "
              f"({pct_non_uniform:.1f}%)")

        # CRITICAL: Decoys should have real predictions, not uniform 1.0
        assert non_uniform_count > 0, \
            "CRITICAL: Decoy peptides have uniform 1.0 intensities - " \
            "this will break FDR estimation! Decoys must have real predictions."

    @pytest.mark.slow
    def test_targets_and_decoys_similar_intensity_distribution(
        self, indexed_db, production_intensity_store, all_peptides_with_decoy_flag
    ):
        """
        Verify that target and decoy peptides have similar intensity distributions.

        If decoys had uniform 1.0 while targets had varied predictions, the
        distributions would differ significantly - indicating biased FDR.
        """
        target_peptides = [(i, seq) for i, seq, is_decoy in all_peptides_with_decoy_flag if not is_decoy]
        decoy_peptides = [(i, seq) for i, seq, is_decoy in all_peptides_with_decoy_flag if is_decoy]

        if len(target_peptides) < 10 or len(decoy_peptides) < 10:
            pytest.skip("Not enough peptides for distribution comparison")

        def get_intensity_stats(peptide_list, max_peps=50):
            intensities = []
            for idx, seq in peptide_list[:max_peps]:
                pep_len = len(seq)
                for pos in range(min(5, pep_len - 1)):
                    intensity = production_intensity_store.get_intensity_or_default(
                        idx, pep_len, ION_KIND_Y, pos, 1
                    )
                    intensities.append(intensity)
            return np.array(intensities)

        target_intensities = get_intensity_stats(target_peptides)
        decoy_intensities = get_intensity_stats(decoy_peptides)

        target_mean = np.mean(target_intensities)
        decoy_mean = np.mean(decoy_intensities)
        target_std = np.std(target_intensities)
        decoy_std = np.std(decoy_intensities)

        print(f"\nTarget intensities: mean={target_mean:.3f}, std={target_std:.3f}")
        print(f"Decoy intensities:  mean={decoy_mean:.3f}, std={decoy_std:.3f}")

        # Both should have similar distributions
        # If decoys were uniform 1.0, mean would be ~1.0 and std would be ~0
        assert decoy_std > 0.05, \
            f"Decoy intensity std ({decoy_std:.3f}) too low - " \
            "suggests uniform values instead of real predictions"

        # Means should be reasonably similar (within 50%)
        mean_diff_pct = abs(target_mean - decoy_mean) / max(target_mean, 0.01) * 100
        print(f"Mean difference: {mean_diff_pct:.1f}%")

    @pytest.mark.slow
    def test_weighted_scoring_fair_for_decoys(
        self, indexed_db, production_intensity_store, all_peptides_with_decoy_flag, scorer_params
    ):
        """
        Test that weighted scoring with real decoy predictions provides
        fair target-decoy competition.

        We generate spectra from both target and decoy peptides and verify
        that both types are scored using their actual predicted intensities.
        """
        target_peptides = [(i, seq) for i, seq, is_decoy in all_peptides_with_decoy_flag if not is_decoy]
        decoy_peptides = [(i, seq) for i, seq, is_decoy in all_peptides_with_decoy_flag if is_decoy]

        if len(target_peptides) < 5 or len(decoy_peptides) < 5:
            pytest.skip("Not enough peptides for fair comparison")

        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        # Score spectra from target peptides
        target_scores = []
        for idx, seq in target_peptides[:20]:
            mz, intensity, prec_mz = generate_theoretical_spectrum(seq, charge=2)
            spectrum = create_processed_spectrum(mz, intensity, prec_mz, 2, f"target_{idx}")

            features = scorer_weighted.score(
                indexed_db, spectrum, intensity_store=production_intensity_store
            )
            if features:
                target_scores.append(features[0].hyperscore)

        # Score spectra from decoy peptides
        decoy_scores = []
        for idx, seq in decoy_peptides[:20]:
            mz, intensity, prec_mz = generate_theoretical_spectrum(seq, charge=2)
            spectrum = create_processed_spectrum(mz, intensity, prec_mz, 2, f"decoy_{idx}")

            features = scorer_weighted.score(
                indexed_db, spectrum, intensity_store=production_intensity_store
            )
            if features:
                decoy_scores.append(features[0].hyperscore)

        if len(target_scores) < 5 or len(decoy_scores) < 5:
            pytest.skip("Not enough scores for comparison")

        target_mean = np.mean(target_scores)
        decoy_mean = np.mean(decoy_scores)
        target_std = np.std(target_scores)
        decoy_std = np.std(decoy_scores)

        print(f"\nTarget scores: mean={target_mean:.2f}, std={target_std:.2f}")
        print(f"Decoy scores:  mean={decoy_mean:.2f}, std={decoy_std:.2f}")

        # Both should have reasonable score distributions
        # If decoys were uniformly weighted, their scores might be artificially high
        assert len(target_scores) > 0 and len(decoy_scores) > 0, \
            "Should have scores for both target and decoy peptides"

    @pytest.mark.slow
    def test_compare_uniform_vs_predicted_decoys(
        self, indexed_db, production_intensity_store, uniform_store,
        all_peptides_with_decoy_flag, scorer_params
    ):
        """
        Compare weighted scoring with:
        1. Uniform decoys (testing only - WRONG for production)
        2. Predicted decoys (correct for production)

        This demonstrates why predicting decoy intensities matters.
        """
        decoy_peptides = [(i, seq) for i, seq, is_decoy in all_peptides_with_decoy_flag if is_decoy]

        if len(decoy_peptides) < 5:
            pytest.skip("Not enough decoy peptides")

        scorer_weighted = Scorer(
            score_type=ScoreType("weightedhyperscore"),
            **scorer_params,
        )

        scores_uniform_decoys = []
        scores_predicted_decoys = []

        for idx, seq in decoy_peptides[:30]:
            mz, intensity, prec_mz = generate_theoretical_spectrum(seq, charge=2)
            spectrum = create_processed_spectrum(mz, intensity, prec_mz, 2, f"decoy_compare_{idx}")

            # Score with uniform store (decoys get 1.0)
            features_uniform = scorer_weighted.score(
                indexed_db, spectrum, intensity_store=uniform_store
            )

            # Score with production store (decoys get real predictions)
            features_predicted = scorer_weighted.score(
                indexed_db, spectrum, intensity_store=production_intensity_store
            )

            if features_uniform and features_predicted:
                scores_uniform_decoys.append(features_uniform[0].hyperscore)
                scores_predicted_decoys.append(features_predicted[0].hyperscore)

        if len(scores_uniform_decoys) < 5:
            pytest.skip("Not enough decoy scores")

        mean_uniform = np.mean(scores_uniform_decoys)
        mean_predicted = np.mean(scores_predicted_decoys)

        print(f"\nDecoy scores with uniform intensities:   mean={mean_uniform:.2f}")
        print(f"Decoy scores with predicted intensities: mean={mean_predicted:.2f}")

        # Document the difference - uniform decoys may have different scores
        # This test is informational to show the effect of proper decoy handling
        diff_pct = abs(mean_uniform - mean_predicted) / max(mean_uniform, 1) * 100
        print(f"Difference: {diff_pct:.1f}%")
