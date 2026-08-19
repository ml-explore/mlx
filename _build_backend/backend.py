import os
import sysconfig

from setuptools import build_meta as _setuptools_build_meta
from setuptools.build_meta import *  # type: ignore # noqa: F403


def _with_nanobind_backend(requires):
    build_backend = int(os.environ.get("MLX_BUILD_BACKEND_PACKAGE", 0))
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    if not build_backend and not free_threaded:
        requires.append("nanobind-backend>=1.0.0.dev2")
    return requires


def get_requires_for_build_wheel(config_settings=None):
    requires = _setuptools_build_meta.get_requires_for_build_wheel(config_settings)
    return _with_nanobind_backend(requires)


def get_requires_for_build_editable(config_settings=None):
    requires = _setuptools_build_meta.get_requires_for_build_editable(config_settings)
    return _with_nanobind_backend(requires)
