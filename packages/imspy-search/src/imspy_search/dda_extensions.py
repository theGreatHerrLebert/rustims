"""Extensions to TimsDatasetDDA for sagepy integration.

This module provides methods that were removed from imspy-core's dda.py
to eliminate the sagepy dependency from the core package.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from sagepy.core import (
    Precursor, ProcessedSpectrum, SpectrumProcessor, Tolerance
)

from imspy_core.timstof import TimsDatasetDDA
from imspy_core.timstof.frame import TimsFrame

from imspy_search.utility import sanitize_mz, sanitize_charge, get_searchable_spec


def to_sage_precursor(
    row: pd.Series,
    isolation_window_lower: float = -3.0,
    isolation_window_upper: float = 3.0,
) -> Precursor:
    """Convert a PASEF fragment row to a sagepy Precursor.

    Args:
        row: A pandas Series containing PASEF fragment data
        isolation_window_lower: Lower bound for isolation window (Da)
        isolation_window_upper: Upper bound for isolation window (Da)

    Returns:
        A sagepy Precursor object
    """
    return Precursor(
        mz=sanitize_mz(row['monoisotopic_mz'], row['largest_peak_mz']),
        intensity=row['intensity'],
        charge=sanitize_charge(row['charge']),
        isolation_window=Tolerance(da=(isolation_window_lower, isolation_window_upper)),
        collision_energy=row.collision_energy,
        inverse_ion_mobility=row.mobility if 'mobility' in row.index else None,
    )


def get_sage_processed_precursors(
    dataset: TimsDatasetDDA,
    num_threads: int = 16,
    take_top_n: int = 150,
    isolation_window_lower: float = -3.0,
    isolation_window_upper: float = 3.0,
    ds_name: Optional[str] = None,
) -> pd.DataFrame:
    """Extract and process PASEF fragments as sagepy ProcessedSpectrum objects.

    This function extracts PASEF fragments from a TimsDatasetDDA, aggregates
    them by precursor ID, and converts them to sagepy ProcessedSpectrum objects
    suitable for database search.

    Args:
        dataset: TimsDatasetDDA object
        num_threads: Number of threads for extraction
        take_top_n: Number of top peaks to keep
        isolation_window_lower: Lower bound for isolation window (Da)
        isolation_window_upper: Upper bound for isolation window (Da)
        ds_name: Dataset name for spec_id generation (defaults to dataset path basename)

    Returns:
        DataFrame with columns: precursor_id, mobility, spec_id, sage_precursor, processed_spec
    """
    import os

    if ds_name is None:
        ds_name = os.path.basename(str(dataset.data_path))

    # Get PASEF fragments
    fragments = dataset.get_pasef_fragments(num_threads=num_threads)

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

    # Calculate mobility
    mobility = fragments.apply(
        lambda r: r.raw_data.get_inverse_mobility_along_scan_marginal(),
        axis=1
    )
    fragments['mobility'] = mobility

    # Generate spec_id
    spec_id = fragments.apply(
        lambda r: str(r['frame_id']) + '-' + str(r['precursor_id']) + '-' + ds_name,
        axis=1
    )
    fragments['spec_id'] = spec_id

    # Create sage precursors
    sage_precursor = fragments.apply(
        lambda r: to_sage_precursor(
            r,
            isolation_window_lower=isolation_window_lower,
            isolation_window_upper=isolation_window_upper
        ),
        axis=1
    )
    fragments['sage_precursor'] = sage_precursor

    # Create spectrum processor
    spec_processor = SpectrumProcessor(take_top_n=take_top_n)

    # Process spectra
    processed_spec = fragments.apply(
        lambda r: get_searchable_spec(
            precursor=r.sage_precursor,
            raw_fragment_data=r.raw_data,
            spec_processor=spec_processor,
            spec_id=r.spec_id,
            time=r['time'],
        ),
        axis=1
    )
    fragments['processed_spec'] = processed_spec

    return fragments


def get_processed_spectra_for_search(
    dataset: TimsDatasetDDA,
    num_threads: int = 16,
    take_top_n: int = 150,
    isolation_window_lower: float = -3.0,
    isolation_window_upper: float = 3.0,
) -> List[ProcessedSpectrum]:
    """Get list of ProcessedSpectrum objects for database search.

    This is a convenience wrapper around get_sage_processed_precursors
    that returns just the list of ProcessedSpectrum objects.

    Args:
        dataset: TimsDatasetDDA object
        num_threads: Number of threads for extraction
        take_top_n: Number of top peaks to keep
        isolation_window_lower: Lower bound for isolation window (Da)
        isolation_window_upper: Upper bound for isolation window (Da)

    Returns:
        List of ProcessedSpectrum objects
    """
    fragments = get_sage_processed_precursors(
        dataset=dataset,
        num_threads=num_threads,
        take_top_n=take_top_n,
        isolation_window_lower=isolation_window_lower,
        isolation_window_upper=isolation_window_upper,
    )

    return fragments['processed_spec'].tolist()


def search_timstof_dda(
    dataset: TimsDatasetDDA,
    scorer,
    indexed_db,
    num_threads: int = 16,
    take_top_n: int = 150,
    isolation_window_lower: float = -3.0,
    isolation_window_upper: float = 3.0,
):
    """Search a TimsDatasetDDA against a database using sagepy.

    Args:
        dataset: TimsDatasetDDA object
        scorer: sagepy Scorer object
        indexed_db: Indexed database from sagepy
        num_threads: Number of threads for extraction and search
        take_top_n: Number of top peaks to keep
        isolation_window_lower: Lower bound for isolation window (Da)
        isolation_window_upper: Upper bound for isolation window (Da)

    Returns:
        Dictionary of PSMs from scorer.score_collection_psm
    """
    spectra = get_processed_spectra_for_search(
        dataset=dataset,
        num_threads=num_threads,
        take_top_n=take_top_n,
        isolation_window_lower=isolation_window_lower,
        isolation_window_upper=isolation_window_upper,
    )

    return scorer.score_collection_psm(
        db=indexed_db,
        spectrum_collection=spectra,
        num_threads=num_threads,
    )
