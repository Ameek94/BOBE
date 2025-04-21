# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath("../../"))   # adjust so your package root is on PYTHONPATH

project = 'JaxBo'
copyright = '2025, Ameek Malhotra, Nathan Cohen and Jan Hamann'
author = 'Ameek Malhotra, Nathan Cohen and Jan Hamann'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",            # if you use NumPy or Google–style docstrings
    "sphinx_autodoc_typehints",       # for nicer signature rendering
    # "sphinx.ext.autosummary",       # optional: generate summaries
]

templates_path = ['_templates']
exclude_patterns = []

language = '[en]'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
