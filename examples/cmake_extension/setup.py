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
        zip_safe=False,
        python_requires=">=3.10",
    )
