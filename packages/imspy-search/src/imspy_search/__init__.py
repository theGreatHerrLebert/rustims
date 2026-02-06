"""
imspy_search - Database search functionality for timsTOF proteomics data using sagepy.

This package provides database search, PSM rescoring, and FDR control for timsTOF DDA data.

Requires imspy-core and imspy-predictors for core data structures and ML predictors.

Core functionality:
- SAGE-based database search for timsTOF DDA data
- Machine learning-based PSM rescoring
- Target-decoy competition and q-value estimation
- MGF file parsing for Bruker DataAnalysis output
"""

__version__ = "0.4.0"

# Core utility functions
from imspy_search.utility import (
    # Data extraction and preprocessing
    extract_timstof_dda_data,
    get_searchable_spec,
    get_ms1_ims_spectrum,
    # FASTA handling
    split_fasta,
    # PSM handling
    generate_training_data,
    split_psms,
    generate_balanced_rt_dataset,
    generate_balanced_im_dataset,
    # Helper functions
    linear_map,
    map_to_domain,
    sanitize_charge,
    sanitize_mz,
    write_psms_binary,
    merge_dicts_with_merge_dict,
    check_memory,
    # Output formatting
    transform_psm_to_pin,
    parse_to_tims2rescore,
)

# SAGE output processing
from imspy_search.sage_output_utility import (
    re_score_psms as re_score_psms_lda,
    generate_training_data as generate_training_data_df,
    split_dataframe_randomly,
    row_to_fragment,
    remove_substrings,
    PatternReplacer,
    replace_tokens,
    cosim_from_dict,
    fragments_to_dict,
    plot_summary,
)

# MGF parsing
from imspy_search.mgf import (
    mgf_to_sagepy_query,
    iter_spectra,
    parse_spectrum,
)

# Rescoring with deep learning features
from imspy_search.rescoring import (
    re_score_psms,
    create_feature_space,
)

# DDA extensions for sagepy integration
from imspy_search.dda_extensions import (
    to_sage_precursor,
    get_sage_processed_precursors,
    get_processed_spectra_for_search,
    search_timstof_dda,
)

__all__ = [
    # Version
    '__version__',
    # Data extraction
    'extract_timstof_dda_data',
    'get_searchable_spec',
    'get_ms1_ims_spectrum',
    # FASTA handling
    'split_fasta',
    # PSM handling
    'generate_training_data',
    'split_psms',
    'generate_balanced_rt_dataset',
    'generate_balanced_im_dataset',
    # Helper functions
    'linear_map',
    'map_to_domain',
    'sanitize_charge',
    'sanitize_mz',
    'write_psms_binary',
    'merge_dicts_with_merge_dict',
    'check_memory',
    # Output formatting
    'transform_psm_to_pin',
    'parse_to_tims2rescore',
    # SAGE output processing
    're_score_psms_lda',
    'generate_training_data_df',
    'split_dataframe_randomly',
    'row_to_fragment',
    'remove_substrings',
    'PatternReplacer',
    'replace_tokens',
    'cosim_from_dict',
    'fragments_to_dict',
    'plot_summary',
    # MGF parsing
    'mgf_to_sagepy_query',
    'iter_spectra',
    'parse_spectrum',
    # Rescoring
    're_score_psms',
    'create_feature_space',
    # DDA extensions
    'to_sage_precursor',
    'get_sage_processed_precursors',
    'get_processed_spectra_for_search',
    'search_timstof_dda',
]
