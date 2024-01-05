# Copyright © 2023 Apple Inc.

# -*- coding: utf-8 -*-

import os
import subprocess

author = "MLX Contributors"
autosummary_generate = True
copyright = "2023, MLX Contributors"
# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]
highlight_language = "python"
html_logo = "_static/mlx_logo.png"
html_static_path = ["_static"]
# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/ml-explore/mlx",
    "use_repository_button": True,
    "navigation_with_keys": False,
}
# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "mlx_doc"
intersphinx_mapping = {
    "https://docs.python.org/3": None,
    "https://numpy.org/doc/stable/": None,
}
master_doc = "index"
# -- Project information -----------------------------------------------------

project = "MLX"
pygments_style = "sphinx"
python_use_unqualified_type_names = True
release = "0.0.6"
source_suffix = ".rst"
templates_path = ["_templates"]
version = "0.0.6"
