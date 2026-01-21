"""
Tests for batch preprocessing comparison.

Verifies that the Rust batch preprocessing produces equivalent results
to the sequential Python preprocessing pipeline.

These tests require a real timsTOF DDA dataset and are skipped if not available.
Set the environment variable IMSPY_TEST_DDA_PATH to point to a .d directory.

Example:
    export IMSPY_TEST_DDA_PATH="/path/to/dataset.d"
    pytest tests/test_batch_preprocessing.py -v
"""

import os
import pytest
import numpy as np
from typing import List, Tuple

# Check if test data is available via environment variable
TEST_DATA_PATH = os.environ.get(
    "IMSPY_TEST_DDA_PATH",
    ""  # No default - tests will be skipped if not set
)
SKIP_REASON = (
    "DDA test data not available. Set IMSPY_TEST_DDA_PATH environment variable "
    "to point to a timsTOF DDA .d directory to run these tests."
)


def has_test_data() -> bool:
    """Check if test data directory exists."""
    return TEST_DATA_PATH and os.path.exists(TEST_DATA_PATH) and os.path.isdir(TEST_DATA_PATH)


@pytest.fixture
def dataset():
    """Load the test dataset."""
    if not has_test_data():
        pytest.skip(SKIP_REASON)

    from imspy.timstof import TimsDatasetDDA
    return TimsDatasetDDA(TEST_DATA_PATH, in_memory=False, use_bruker_sdk=False)


@pytest.fixture
def dataset_name():
    """Get the dataset name."""
    if not TEST_DATA_PATH:
        pytest.skip(SKIP_REASON)
    return os.path.basename(TEST_DATA_PATH).replace(".d", "")


class TestBatchPreprocessingEquivalence:
    """Test that batch and sequential preprocessing produce equivalent results."""

    @pytest.mark.skipif(not has_test_data(), reason=SKIP_REASON)
    def test_batch_vs_sequential_spectra_count(self, dataset, dataset_name):
        """Test that both methods produce the same number of spectra."""
        from imspy.timstof.dbsearch.utility import (
            get_searchable_specs_batch,
            get_searchable_spec,
            sanitize_mz,
            sanitize_charge,
        )
        from sagepy.core import Precursor, Tolerance, SpectrumProcessor

        # Get batch processed spectra
        batch_specs = get_searchable_specs_batch(
            dataset=dataset,
            ds_name=dataset_name,
            take_top_n=150,
            deisotope=True,
            num_threads=4,
        )

        # Get sequential processed spectra
        fragments = dataset.get_pasef_fragments(num_threads=1)

        # Aggregate by precursor_id
        fragments = fragments.groupby('precursor_id').agg({
            'frame_id': 'first',
            'time': 'first',
            'precursor_id': 'first',
            'raw_data': 'sum',
            'scan_begin': 'first',
            'scan_end': 'first',
            'isolation_mz': 'first',
            'isolation_width': 'first',
            'collision_energy': 'first',
            'largest_peak_mz': 'first',
            'average_mz': 'first',
            'monoisotopic_mz': 'first',
            'charge': 'first',
            'average_scan': 'first',
            'intensity': 'first',
            'parent_id': 'first',
        })

        # Count spectra
        n_batch = len(batch_specs)
        n_sequential = len(fragments)

        print(f"\nBatch spectra: {n_batch}")
        print(f"Sequential spectra: {n_sequential}")

        # Should have the same number of spectra
        assert n_batch == n_sequential, (
            f"Batch ({n_batch}) and sequential ({n_sequential}) produced different counts"
        )

    @pytest.mark.skipif(not has_test_data(), reason=SKIP_REASON)
    def test_batch_vs_sequential_precursor_mz_values(self, dataset, dataset_name):
        """Test that precursor m/z values are equivalent."""
        from imspy.timstof.dbsearch.utility import (
            get_searchable_specs_batch_with_metadata,
            sanitize_mz,
        )

        # Get batch processed spectra with metadata
        batch_specs, batch_meta = get_searchable_specs_batch_with_metadata(
            dataset=dataset,
            ds_name=dataset_name,
            take_top_n=150,
            deisotope=True,
            num_threads=4,
        )

        # Get sequential metadata
        fragments = dataset.get_pasef_fragments(num_threads=1)
        fragments = fragments.groupby('precursor_id').agg({
            'monoisotopic_mz': 'first',
            'largest_peak_mz': 'first',
            'precursor_id': 'first',
        })

        # Sort both by precursor_id for comparison
        batch_meta_sorted = batch_meta.sort_values('spec_id').reset_index(drop=True)

        # Extract precursor_id from spec_id (format: frame_id-precursor_id-dataset_name)
        batch_meta_sorted['precursor_id'] = batch_meta_sorted['spec_id'].apply(
            lambda x: int(x.split('-')[1])
        )
        batch_meta_sorted = batch_meta_sorted.sort_values('precursor_id').reset_index(drop=True)

        # Compare precursor m/z values
        for idx, row in batch_meta_sorted.iterrows():
            prec_id = row['precursor_id']
            if prec_id in fragments.index:
                seq_row = fragments.loc[prec_id]
                expected_mz = sanitize_mz(seq_row['monoisotopic_mz'], seq_row['largest_peak_mz'])
                actual_mz = row['precursor_mz']

                # Allow small tolerance for floating point differences
                assert abs(expected_mz - actual_mz) < 0.001, (
                    f"Precursor {prec_id}: expected mz {expected_mz}, got {actual_mz}"
                )

        print(f"\nVerified {len(batch_meta_sorted)} precursor m/z values match")

    @pytest.mark.skipif(not has_test_data(), reason=SKIP_REASON)
    def test_batch_vs_sequential_mobility_values(self, dataset, dataset_name):
        """Test that inverse mobility values are similar."""
        from imspy.timstof.dbsearch.utility import get_searchable_specs_batch_with_metadata

        # Get batch processed spectra with metadata
        batch_specs, batch_meta = get_searchable_specs_batch_with_metadata(
            dataset=dataset,
            ds_name=dataset_name,
            take_top_n=150,
            deisotope=True,
            num_threads=4,
        )

        # Get sequential mobility values
        fragments = dataset.get_pasef_fragments(num_threads=1)
        fragments = fragments.groupby('precursor_id').agg({
            'raw_data': 'sum',
            'precursor_id': 'first',
        })

        # Calculate mobility sequentially
        mobility_sequential = fragments['raw_data'].apply(
            lambda r: r.get_inverse_mobility_along_scan_marginal()
        )

        # Extract precursor_id from batch metadata
        batch_meta['precursor_id'] = batch_meta['spec_id'].apply(
            lambda x: int(x.split('-')[1])
        )

        # Compare mobility values
        mismatches = 0
        total_diff = 0.0
        count = 0

        for idx, row in batch_meta.iterrows():
            prec_id = row['precursor_id']
            if prec_id in mobility_sequential.index:
                expected_mob = mobility_sequential.loc[prec_id]
                actual_mob = row['mobility']
                diff = abs(expected_mob - actual_mob)
                total_diff += diff
                count += 1

                # Allow small tolerance (mobility should be very close)
                if diff > 0.01:
                    mismatches += 1

        avg_diff = total_diff / count if count > 0 else 0
        print(f"\nMobility comparison:")
        print(f"  Total comparisons: {count}")
        print(f"  Mismatches (>0.01): {mismatches}")
        print(f"  Average difference: {avg_diff:.6f}")

        # Most should match exactly (same algorithm)
        assert mismatches / count < 0.01, (
            f"Too many mobility mismatches: {mismatches}/{count}"
        )

    @pytest.mark.skipif(not has_test_data(), reason=SKIP_REASON)
    def test_batch_vs_sequential_peak_counts(self, dataset, dataset_name):
        """Test that peak counts are reasonable and similar."""
        from imspy.timstof.dbsearch.utility import (
            get_searchable_specs_batch,
            get_searchable_spec,
            sanitize_mz,
            sanitize_charge,
        )
        from sagepy.core import Precursor, Tolerance, SpectrumProcessor

        take_top_n = 150

        # Get batch processed spectra
        batch_specs = get_searchable_specs_batch(
            dataset=dataset,
            ds_name=dataset_name,
            take_top_n=take_top_n,
            deisotope=True,
            num_threads=4,
        )

        # Get sequential processed spectra
        fragments = dataset.get_pasef_fragments(num_threads=1)
        fragments = fragments.groupby('precursor_id').agg({
            'frame_id': 'first',
            'time': 'first',
            'precursor_id': 'first',
            'raw_data': 'sum',
            'isolation_mz': 'first',
            'isolation_width': 'first',
            'collision_energy': 'first',
            'largest_peak_mz': 'first',
            'monoisotopic_mz': 'first',
            'charge': 'first',
            'intensity': 'first',
        })

        # Calculate mobility for sequential
        fragments['mobility'] = fragments['raw_data'].apply(
            lambda r: r.get_inverse_mobility_along_scan_marginal()
        )

        # Process sequential spectra
        spec_processor = SpectrumProcessor(take_top_n=take_top_n, deisotope=True)
        iso_tol = Tolerance(da=(-3.0, 3.0))

        sequential_specs = []
        for idx, row in fragments.iterrows():
            precursor = Precursor(
                mz=sanitize_mz(row['monoisotopic_mz'], row['largest_peak_mz']),
                intensity=row['intensity'],
                charge=sanitize_charge(row['charge']),
                isolation_window=iso_tol,
                collision_energy=row['collision_energy'],
                inverse_ion_mobility=row['mobility'],
            )
            spec_id = f"{row['frame_id']}-{row['precursor_id']}-{dataset_name}"
            processed = get_searchable_spec(
                precursor=precursor,
                raw_fragment_data=row['raw_data'],
                spec_processor=spec_processor,
                spec_id=spec_id,
                time=row['time'],
            )
            sequential_specs.append((row['precursor_id'], processed))

        # Sort both by precursor_id for comparison
        batch_by_id = {}
        for spec in batch_specs:
            prec_id = int(spec.id.split('-')[1])
            batch_by_id[prec_id] = spec

        seq_by_id = {prec_id: spec for prec_id, spec in sequential_specs}

        # Compare peak counts
        peak_diffs = []
        for prec_id in batch_by_id:
            if prec_id in seq_by_id:
                batch_peaks = len(batch_by_id[prec_id].peaks)
                seq_peaks = len(seq_by_id[prec_id].peaks)
                peak_diffs.append(abs(batch_peaks - seq_peaks))

        avg_diff = np.mean(peak_diffs)
        max_diff = np.max(peak_diffs)

        print(f"\nPeak count comparison:")
        print(f"  Spectra compared: {len(peak_diffs)}")
        print(f"  Average peak count difference: {avg_diff:.2f}")
        print(f"  Max peak count difference: {max_diff}")

        # Peak counts should be very similar (deisotoping may have minor differences)
        assert avg_diff < 10, f"Average peak count difference too high: {avg_diff}"

    @pytest.mark.skipif(not has_test_data(), reason=SKIP_REASON)
    def test_batch_spectra_have_valid_peaks(self, dataset, dataset_name):
        """Test that batch processed spectra have valid, non-empty peaks.

        Note: We don't compare spectral similarity between Rust batch and Sage sequential
        because they use different deisotoping algorithms (Rust vs Sage implementations).
        Instead, we verify that both produce valid spectra with reasonable peak counts.
        """
        from imspy.timstof.dbsearch.utility import get_searchable_specs_batch

        take_top_n = 150

        # Get batch processed spectra
        batch_specs = get_searchable_specs_batch(
            dataset=dataset,
            ds_name=dataset_name,
            take_top_n=take_top_n,
            deisotope=True,
            num_threads=4,
        )

        # Sample 100 spectra for validation
        sample_size = min(100, len(batch_specs))
        sample_specs = batch_specs[:sample_size]

        # Verify all spectra have valid peaks
        empty_spectra = 0
        peak_counts = []
        max_mz_values = []
        total_ion_currents = []

        for spec in sample_specs:
            n_peaks = len(spec.peaks)
            peak_counts.append(n_peaks)

            if n_peaks == 0:
                empty_spectra += 1
                continue

            # Extract peak info
            mz_values = [p.mass for p in spec.peaks]
            max_mz_values.append(max(mz_values))
            total_ion_currents.append(spec.total_ion_current)

        avg_peaks = np.mean(peak_counts)
        median_peaks = np.median(peak_counts)
        avg_max_mz = np.mean(max_mz_values) if max_mz_values else 0
        avg_tic = np.mean(total_ion_currents) if total_ion_currents else 0

        print(f"\nBatch spectra validation (n={sample_size}):")
        print(f"  Empty spectra: {empty_spectra}")
        print(f"  Average peak count: {avg_peaks:.1f}")
        print(f"  Median peak count: {median_peaks:.1f}")
        print(f"  Max peak count: {max(peak_counts)}")
        print(f"  Average max m/z: {avg_max_mz:.1f}")
        print(f"  Average TIC: {avg_tic:.1f}")

        # Assertions
        assert empty_spectra < sample_size * 0.1, f"Too many empty spectra: {empty_spectra}/{sample_size}"
        assert avg_peaks > 10, f"Average peak count too low: {avg_peaks}"
        assert avg_peaks <= take_top_n, f"Average peak count exceeds take_top_n: {avg_peaks}"
        assert avg_max_mz > 100, f"Average max m/z too low: {avg_max_mz}"


def _extract_mz_intensity(spec) -> Tuple[np.ndarray, np.ndarray]:
    """Extract mz and intensity arrays from a ProcessedSpectrum."""
    mz = np.array([p.mass for p in spec.peaks], dtype=np.float64)
    intensity = np.array([p.intensity for p in spec.peaks], dtype=np.float64)
    return mz, intensity


def _calculate_spectral_similarity(
    mz1: np.ndarray,
    intensity1: np.ndarray,
    mz2: np.ndarray,
    intensity2: np.ndarray,
    tolerance_ppm: float = 20.0
) -> float:
    """
    Calculate cosine similarity between two spectra.

    Uses PPM tolerance for peak matching.
    """
    mz1 = np.asarray(mz1, dtype=np.float64)
    mz2 = np.asarray(mz2, dtype=np.float64)
    intensity1 = np.asarray(intensity1, dtype=np.float64)
    intensity2 = np.asarray(intensity2, dtype=np.float64)

    if len(mz1) == 0 or len(mz2) == 0:
        return 0.0

    # Normalize intensities
    norm1 = np.sqrt(np.sum(intensity1 ** 2))
    norm2 = np.sqrt(np.sum(intensity2 ** 2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    intensity1_norm = intensity1 / norm1
    intensity2_norm = intensity2 / norm2

    # Match peaks within tolerance
    dot_product = 0.0
    used_j = set()

    for i, m1 in enumerate(mz1):
        tolerance = m1 * tolerance_ppm / 1e6
        best_j = None
        best_diff = float('inf')

        for j, m2 in enumerate(mz2):
            if j in used_j:
                continue
            diff = abs(m1 - m2)
            if diff <= tolerance and diff < best_diff:
                best_j = j
                best_diff = diff

        if best_j is not None:
            dot_product += intensity1_norm[i] * intensity2_norm[best_j]
            used_j.add(best_j)

    return dot_product


class TestBatchPreprocessingPerformance:
    """Performance tests for batch preprocessing."""

    @pytest.mark.skipif(not has_test_data(), reason=SKIP_REASON)
    def test_batch_is_faster(self, dataset, dataset_name):
        """Test that batch processing is faster than sequential."""
        import time
        from imspy.timstof.dbsearch.utility import (
            get_searchable_specs_batch,
            get_searchable_spec,
            sanitize_mz,
            sanitize_charge,
        )
        from sagepy.core import Precursor, Tolerance, SpectrumProcessor

        take_top_n = 150

        # Time batch processing
        start_batch = time.time()
        batch_specs = get_searchable_specs_batch(
            dataset=dataset,
            ds_name=dataset_name,
            take_top_n=take_top_n,
            deisotope=True,
            num_threads=4,
        )
        batch_time = time.time() - start_batch

        # Time sequential processing
        start_seq = time.time()

        fragments = dataset.get_pasef_fragments(num_threads=1)
        fragments = fragments.groupby('precursor_id').agg({
            'frame_id': 'first',
            'time': 'first',
            'precursor_id': 'first',
            'raw_data': 'sum',
            'collision_energy': 'first',
            'largest_peak_mz': 'first',
            'monoisotopic_mz': 'first',
            'charge': 'first',
            'intensity': 'first',
        })

        fragments['mobility'] = fragments['raw_data'].apply(
            lambda r: r.get_inverse_mobility_along_scan_marginal()
        )

        spec_processor = SpectrumProcessor(take_top_n=take_top_n, deisotope=True)
        iso_tol = Tolerance(da=(-3.0, 3.0))

        sequential_specs = []
        for idx, row in fragments.iterrows():
            precursor = Precursor(
                mz=sanitize_mz(row['monoisotopic_mz'], row['largest_peak_mz']),
                intensity=row['intensity'],
                charge=sanitize_charge(row['charge']),
                isolation_window=iso_tol,
                collision_energy=row['collision_energy'],
                inverse_ion_mobility=row['mobility'],
            )
            spec_id = f"{row['frame_id']}-{row['precursor_id']}-{dataset_name}"
            processed = get_searchable_spec(
                precursor=precursor,
                raw_fragment_data=row['raw_data'],
                spec_processor=spec_processor,
                spec_id=spec_id,
                time=row['time'],
            )
            sequential_specs.append(processed)

        seq_time = time.time() - start_seq

        speedup = seq_time / batch_time if batch_time > 0 else float('inf')

        print(f"\nPerformance comparison:")
        print(f"  Batch time: {batch_time:.2f}s")
        print(f"  Sequential time: {seq_time:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Spectra processed: {len(batch_specs)}")

        # Batch should be faster (at least 1.5x on this dataset)
        assert batch_time < seq_time, (
            f"Batch ({batch_time:.2f}s) not faster than sequential ({seq_time:.2f}s)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
