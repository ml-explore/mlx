# Copyright © 2023 Apple Inc.

import datetime
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from subprocess import run

from setuptools import Command, Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext


def get_version(version):
    if "PYPI_RELEASE" not in os.environ:
        today = datetime.date.today()
        version = f"{version}.dev{today.year}{today.month}{today.day}"

        if "DEV_RELEASE" not in os.environ:
            git_hash = (
                run(
                    "git rev-parse --short HEAD".split(),
                    capture_output=True,
                    check=True,
                )
                .stdout.strip()
                .decode()
            )
            version = f"{version}+{git_hash}"

    return version


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_INSTALL_PREFIX={extdir}{os.sep}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_SHARED_LIBS=ON",
            "-DMLX_BUILD_PYTHON_BINDINGS=ON",
            "-DMLX_BUILD_TESTS=OFF",
            "-DMLX_BUILD_BENCHMARKS=OFF",
            "-DMLX_BUILD_EXAMPLES=OFF",
            f"-DMLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY={extdir}{os.sep}",
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Pass version to C++
        cmake_args += [f"-DMLX_VERSION={self.distribution.get_version()}"]  # type: ignore[attr-defined]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", "--target", "install", *build_args],
            cwd=build_temp,
            check=True,
        )

    # Make sure to copy mlx.metallib for inplace builds
    def run(self):
        super().run()

        # Based on https://github.com/pypa/setuptools/blob/main/setuptools/command/build_ext.py#L102
        if self.inplace:
            for ext in self.extensions:
                if ext.name == "mlx.core":
                    # Resolve inplace package dir
                    build_py = self.get_finalized_command("build_py")
                    inplace_file, regular_file = self._get_inplace_equivalent(
                        build_py, ext
                    )

                    inplace_dir = str(Path(inplace_file).parent.resolve())
                    regular_dir = str(Path(regular_file).parent.resolve())

                    self.copy_tree(regular_dir, inplace_dir)


class GenerateStubs(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self) -> None:
        subprocess.run(["pybind11-stubgen", "mlx.core", "-o", "python"])
        # Note, sed inplace on macos requires a backup prefix, delete the file after its generated
        # this sed is needed to replace references from py::cpp_function to a generic Callable
        subprocess.run(
            [
                "sed",
                "-i",
                "''",
                "s/cpp_function/typing.Callable/g",
                "python/mlx/core/__init__.pyi",
            ]
        )
        subprocess.run(["rm", "python/mlx/core/__init__.pyi''"])


# Read the content of README.md
with open(Path(__file__).parent / "README.md", encoding="utf-8") as f:
    long_description = f.read()

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
if __name__ == "__main__":
    packages = find_namespace_packages(
        where="python", exclude=["src", "tests", "tests.*"]
    )
    package_dir = {"": "python"}
    package_data = {"mlx": ["lib/*", "include/*", "share/*"]}

    setup(
        name="mlx",
        version=get_version("0.0.6"),
        author="MLX Contributors",
        author_email="mlx@group.apple.com",
        description="A framework for machine learning on Apple silicon.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=packages,
        package_dir=package_dir,
        package_data=package_data,
        include_package_data=True,
        extras_require={
            "testing": ["numpy", "torch"],
            "dev": ["pre-commit", "pybind11-stubgen"],
        },
        ext_modules=[CMakeExtension("mlx.core")],
        cmdclass={"build_ext": CMakeBuild, "generate_stubs": GenerateStubs},
        zip_safe=False,
        python_requires=">=3.8",
    )
