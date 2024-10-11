# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'imspy'
copyright = '2024, David Teschner'
author = 'David Teschner'
release = '0.2.33'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.napoleon',  # for Google style docstrings
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',  # to include type hints in docs
    'myst_parser',  # only if you're using Markdown
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']

# Add base URL for GitHub Pages deployment in subfolder
html_baseurl = "https://thegreatherrlebert.github.io/rustims/imspy/"

# Ensure Sphinx generates an index and module index
html_use_index = True
html_use_modindex = True

# If not using custom static files, comment this out to avoid warnings
html_static_path = ['_static']
