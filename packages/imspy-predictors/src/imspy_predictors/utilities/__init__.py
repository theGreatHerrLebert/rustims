"""ML utilities and tokenizers module."""

from imspy_predictors.utilities.tokenizers import ProformaTokenizer

# HFProformaTokenizer requires transformers (and transitively torch)
# Make it optional to avoid import errors when these are not installed
try:
    from imspy_predictors.utilities.hf_tokenizers import HFProformaTokenizer
    _HAS_HF = True
except ImportError:
    HFProformaTokenizer = None
    _HAS_HF = False

__all__ = [
    'ProformaTokenizer',
    'HFProformaTokenizer',
]
