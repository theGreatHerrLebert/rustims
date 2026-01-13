"""
Integration test infrastructure for timsim.

This module provides tools for running automated integration tests
that simulate datasets and validate them against analysis tools.

The integration tests are split into two phases:
1. Simulation (sim.py): Generate synthetic datasets
2. Evaluation (eval.py): Run analysis tools and validate results
"""

__all__ = ["TESTS", "TEST_CONFIGS"]

# Available integration tests
TESTS = [
    "IT-DIA-HELA",
    "IT-DIA-HYE",
    "IT-DIA-PHOS",
    "IT-DDA-TOPN",
    "IT-DDA-HLA",
]

# Test configurations with metadata
TEST_CONFIGS = {
    "IT-DIA-HELA": {
        "description": "Standard HeLa DIA-PASEF identification benchmark",
        "acquisition": "DIA",
        "sample": "hela",
        "complexity": 25000,
        "analysis_tools": ["diann", "fragpipe"],
    },
    "IT-DIA-HYE": {
        "description": "HYE mixed species DIA-PASEF quantification benchmark",
        "acquisition": "DIA",
        "sample": "hye",
        "complexity": 25000,
        "analysis_tools": ["diann", "fragpipe"],
    },
    "IT-DIA-PHOS": {
        "description": "Phosphoproteomics DIA-PASEF PTM localization benchmark",
        "acquisition": "DIA",
        "sample": "phospho",
        "complexity": 25000,
        "analysis_tools": ["diann", "fragpipe"],
    },
    "IT-DDA-TOPN": {
        "description": "Standard HeLa DDA-PASEF TopN benchmark",
        "acquisition": "DDA",
        "sample": "hela",
        "complexity": 25000,
        "analysis_tools": ["diann", "fragpipe"],
    },
    "IT-DDA-HLA": {
        "description": "HLA immunopeptidomics thunder-PASEF benchmark",
        "acquisition": "DDA",
        "sample": "hla",
        "complexity": 10000,
        "analysis_tools": ["diann", "fragpipe"],
    },
}
