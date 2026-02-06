"""ML utilities and tokenizers module."""

# SimpleTokenizer is pure Python - always available
from imspy_predictors.utilities.simple_tokenizer import SimpleTokenizer, get_tokenizer

# ProformaTokenizer requires imspy_connector (Rust bindings)
# Make it optional for environments without the Rust toolchain
try:
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer
    _HAS_RUST_TOKENIZER = True
except ImportError:
    ProformaTokenizer = None
    _HAS_RUST_TOKENIZER = False

# HFProformaTokenizer requires transformers (and transitively torch)
# Make it optional to avoid import errors when these are not installed
try:
    from imspy_predictors.utilities.hf_tokenizers import HFProformaTokenizer
    _HAS_HF = True
except ImportError:
    HFProformaTokenizer = None
    _HAS_HF = False

__all__ = [
    'SimpleTokenizer',
    'get_tokenizer',
    'ProformaTokenizer',
    'HFProformaTokenizer',
]
