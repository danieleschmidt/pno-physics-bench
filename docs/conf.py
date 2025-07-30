# Configuration file for Sphinx documentation builder

import os
import sys
from pathlib import Path

# Add source directory to path
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'PNO Physics Bench'
copyright = '2025, Daniel Schmidt'
author = 'Daniel Schmidt'
release = '0.1.0'
version = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'sphinx_rtd_theme',
]

# Templates and paths
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Source file settings
source_suffix = '.rst'
master_doc = 'index'
language = 'en'

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

# Napoleon settings for Google/NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Intersphinx mappings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# nbsphinx settings
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# Math settings
mathjax3_config = {
    'tex': {'tags': 'ams', 'useLabelIds': True},
}