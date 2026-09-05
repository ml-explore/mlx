# Copyright © 2026 Apple Inc.

from importlib.metadata import version as package_version

from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        ext_modules=[
            extension.MetalExtension(
                "mlx_sample_metal_extension._ext",
                sources=[
                    "activation.cpp",
                    "silu_and_mul.metal",
                    "gelu.metal",
                    "fatrelu_and_mul.metal",
                ],
                extra_compile_args={"cxx": ["-O3"], "metal": ["-O3"]},
            )
        ],
        cmdclass={"build_ext": extension.BuildExtension},
        install_requires=[f"mlx=={package_version('mlx')}"],
    )
