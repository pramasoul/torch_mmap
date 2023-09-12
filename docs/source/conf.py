# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
#sys.path.insert(0, os.path.abspath('../../torch_mmap/'))
#sys.path.insert(0, os.path.abspath('/pond/home/soul/local/Projects/torch_mmap/'))
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'torch_mmap'
copyright = '2023, Tom Soulanille'
author = 'Tom Soulanille'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
             'sphinx.ext.intersphinx']

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

autodoc_mock_imports = ["torch"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['.ipynb_checkpoints/*']
