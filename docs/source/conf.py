# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Make the project importable for autodoc without requiring installation.
# conf.py lives in docs/source, so the repo root is two levels up.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

sys.path.append(os.path.abspath("sphinxext"))

# -- Project information -----------------------------------------------------

project = "pivpy"
copyright = "2019, Turbulence Structure Laboratory"
author = "Turbulence Structure Laboratory"

# The full version, including alpha/beta/rc tags
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("pivpy")
except Exception:
    release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "nbsphinx", 
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser"
    ]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

master_doc = "index"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

# Add a small cross-link on the rendered tutorial notebook page without
# modifying the notebook file itself.
nbsphinx_prolog = r"""
{% if env.docname == 'tutorial' %}
.. note::

   Looking for a written, copy/paste friendly version? See :doc:`pivpy_tutorial`.
{% endif %}
"""

# The docs are often built with Sphinx's nitpicky mode (-n), which turns all
# unresolved cross-references into warnings. Many of our docstrings include
# external types (numpy/xarray/matplotlib, etc.) that aren't resolvable without
# intersphinx inventories. To keep -n builds usable (and compatible with -W),
# ignore these common external/type-hint references.
nitpick_ignore_regex = [
    (r"py:class", r"collections\.abc\..*"),
    (r"py:class", r"(?:numpy|np)\..*"),
    (r"py:class", r"(?:xarray|xr)\..*"),
    (r"py:class", r"matplotlib\..*"),
    (r"py:class", r"ArrayLike"),
    (r"py:class", r"Quiver"),
    (r"py:class", r"plt\.Axes"),
    (r"py:class", r"optional"),
    (r"py:class", r"-"),
    (r"py:class", r"2\*n\+1"),
    (r"py:class", r"If return_std_rms is False"),
    (r"py:class", r".*moving window.*"),
    (r"py:class", r"the dataset"),
    (r"py:class", r"fWin = field rolling window"),
    (r"py:class", r"n=1 means a 3x3 rolling window"),
    (r"py:class", r"vortexfitting\.VelocityField"),
]
