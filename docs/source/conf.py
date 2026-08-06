# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError, version

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'rtcog'
copyright = '2026, Javier Gonzalez-Castillo, Marly Rubin'
author = 'Javier Gonzalez-Castillo, Marly Rubin'
try:
    release = version('rtcog')
except PackageNotFoundError:
    release = '3.0.0.dev0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = []
autodoc_mock_imports = [
    'numpy', 'psychopy', 'nibabel', 'yaml',
    'pyaudio', 'playsound', 'whisper', 'pandas',
    'scipy', 'holoviews', 'panel', 'sklearn',
    'hvplot', 'nilearn', 'matplotlib', 'bokeh'
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
