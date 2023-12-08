# Copyright © 2023 Apple Inc.

from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="mlx_sample_extensions",
        version="0.0.0",
        description="Sample C++ and Metal extensions for MLX primitives.",
        ext_modules=[extension.CMakeExtension("mlx_sample_extensions")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mlx_sample_extensions"],
        package_dir={"": "."},
        package_data={"mlx_sample_extensions": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.8",
    )
