# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date
from importlib import metadata

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------
_TEMPLATE_VERSION = "1.0.0"

project = "waveresponse"
copyright = f"{date.today().year}, 4Subsea"
author = "4Subsea"

# The full version, including alpha/beta/rc tags
version = "0.0.1"  # metadata.version(project)
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Intershpinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "datareservoirio": ("https://www.datareservoir.io/python/docs/latest", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = (
    "_static/4insight-logo.svg"  # "_static/Logo 4Subsea horisontal negative.png"
)
html_context = {"default_mode": "light"}
html_favicon = "_static/favicon.png"

html_theme_options = {
    "navbar_end": ["navbar-icon-links"],
    "external_links": [
        {"name": "4Insight.io", "url": "https://4insight.io"},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/4Subsea/waveresponse-python",
            "icon": "fab fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/waveresponse",
            "icon": "fas fa-box",
        },
    ],
}
