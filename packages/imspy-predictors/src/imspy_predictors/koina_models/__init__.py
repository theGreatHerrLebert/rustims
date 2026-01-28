"""Koina model access module for remote prediction services.

Note: Requires koinapy optional dependency. Install with:
    pip install imspy-predictors[koina]

Features:
- Server availability checking
- Configurable timeouts
- Automatic retry with exponential backoff
- Global enable/disable flag
- Detailed error logging
"""

from imspy_predictors.koina_models.access_models import (
    ModelFromKoina,
    # Server management
    check_koina_server,
    disable_koina,
    enable_koina,
    is_koina_disabled,
    # Exception classes
    KoinaError,
    KoinaDisabledError,
    KoinaConnectionError,
    KoinaTimeoutError,
    KoinaPredictionError,
    # Configuration
    DEFAULT_KOINA_HOST,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
)
from imspy_predictors.koina_models.input_filters import (
    filter_input_by_model,
    filter_peptide_length,
    filter_peptide_modifications,
    filter_precursor_charges,
    filter_instrument_types,
    # Validation utilities
    get_model_type,
    get_supported_models,
    get_model_restrictions,
    validate_model_compatibility,
    MODEL_FILTERS,
    MODEL_DESCRIPTIONS,
)

__all__ = [
    # Main class
    'ModelFromKoina',
    # Server management
    'check_koina_server',
    'disable_koina',
    'enable_koina',
    'is_koina_disabled',
    # Exceptions
    'KoinaError',
    'KoinaDisabledError',
    'KoinaConnectionError',
    'KoinaTimeoutError',
    'KoinaPredictionError',
    # Configuration
    'DEFAULT_KOINA_HOST',
    'DEFAULT_TIMEOUT_SECONDS',
    'DEFAULT_MAX_RETRIES',
    'DEFAULT_RETRY_DELAY',
    # Input filters
    'filter_input_by_model',
    'filter_peptide_length',
    'filter_peptide_modifications',
    'filter_precursor_charges',
    'filter_instrument_types',
    # Validation utilities
    'get_model_type',
    'get_supported_models',
    'get_model_restrictions',
    'validate_model_compatibility',
    'MODEL_FILTERS',
    'MODEL_DESCRIPTIONS',
]
