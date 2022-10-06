# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# type: ignore
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
from pathlib import Path
from typing import Any, List

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "xFormers"
copyright = "Copyright © 2021 Meta Platforms, Inc"
author = "Facebook AI Research"

root_dir = Path(__file__).resolve().parent.parent.parent
# The full version, including alpha/beta/rc tags
release = (root_dir / "version.txt").read_text().strip()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",  # support NumPy and Google style docstrings
    "recommonmark",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
]

# autosectionlabel throws warnings if section names are duplicated.
# The following tells autosectionlabel to not throw a warning for
# duplicated section names that are in different documents.
autosectionlabel_prefix_document = True

# -- Configurations for plugins ------------
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.6", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/master", None),
}
# -------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[Any] = []

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------


html_theme = "pytorch_sphinx_theme"
templates_path = ["_templates"]


# Add any paths that contain custom static files (such as style sheets) here,
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "includehidden": True,
    "canonical_url": "https://facebookresearch.github.io/xformers",
    "pytorch_project": "docs",
    "logo_only": True,  # default = False
}

# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# setting custom stylesheets https://stackoverflow.com/a/34420612
html_context = {"css_files": ["_static/css/customize.css"]}

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "xformersdocs"
github_doc_root = "https://github.com/facebookresearch/xformers/tree/main/docs/"


# Over-ride PyTorch Sphinx css
def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {
            "url_resolver": lambda url: github_doc_root + url,
            "auto_toc_tree_section": "Contents",
            "enable_math": True,
            "enable_inline_math": True,
            "enable_eval_rst": True,
            "enable_auto_toc_tree": True,
        },
        True,
    )
    app.add_transform(AutoStructify)
    app.add_css_file("css/customize.css")
