# Copyright © 2023 Apple Inc.

# -*- coding: utf-8 -*-

import os
import subprocess

# -- Project information -----------------------------------------------------

project = "MLX"
copyright = "2023, MLX Contributors"
author = "MLX Contributors"
version = "0.0.6"
release = "0.0.6"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

python_use_unqualified_type_names = True
autosummary_generate = True

intersphinx_mapping = {
    "https://docs.python.org/3": None,
    "https://numpy.org/doc/stable/": None,
}

templates_path = ["_templates"]
html_static_path = ["_static"]
source_suffix = ".rst"
master_doc = "index"
highlight_language = "python"
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/ml-explore/mlx",
    "use_repository_button": True,
    "navigation_with_keys": False,
}

html_logo = "_static/mlx_logo.png"


# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "mlx_doc"
