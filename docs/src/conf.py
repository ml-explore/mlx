# Copyright © 2023 Apple Inc.

# -*- coding: utf-8 -*-

import os
import subprocess

import mlx.core as mx

# -- Project information -----------------------------------------------------

project = "MLX"
copyright = "2023, MLX Contributors"
author = "MLX Contributors"
version = ".".join(mx.__version__.split(".")[:3])
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

python_use_unqualified_type_names = True
autosummary_generate = True
autosummary_filename_map = {"mlx.core.Stream": "stream_class"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

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
    "logo": {
        "image_light": "_static/mlx_logo.png",
        "image_dark": "_static/mlx_logo_dark.png",
    },
}


# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "mlx_doc"


def setup(app):
    wrapped = app.registry.documenters["function"].can_document_member

    def nanobind_function_patch(member: Any, *args, **kwargs) -> bool:
        return "nanobind.nb_func" in str(type(member)) or wrapped(
            member, *args, **kwargs
        )

    app.registry.documenters["function"].can_document_member = nanobind_function_patch
