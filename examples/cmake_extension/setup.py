# Copyright © 2023-2024 Apple Inc.

from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="mlx_sample_extensions",
        version="0.0.0",
        description="Sample C++ and Metal extensions for MLX primitives.",
        ext_modules=[extension.CMakeExtension("mlx_sample_extensions")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=[],
        install_requires=["nanobind-backend>=1.0.0"],
        options={"bdist_wheel": {"py_limited_api": "cp310"}},
        zip_safe=False,
        python_requires=">=3.10",
    )
