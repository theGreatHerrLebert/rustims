"""CLI entry points for imspy-search."""

from imspy_search.cli.imspy_dda import main as dda_main
from imspy_search.cli.imspy_ccs import main as ccs_main
from imspy_search.cli.imspy_rescore_sage import main as rescore_sage_main

__all__ = [
    'dda_main',
    'ccs_main',
    'rescore_sage_main',
]
