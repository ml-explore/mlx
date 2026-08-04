# Copyright © 2023 Apple Inc.

import logging
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from setuptools import Extension
from setuptools.command.build_ext import build_ext

import mlx

_MLX_PATH = str(mlx.__path__[0])
_MLX_PACKAGE_PATH = str(Path(__file__).resolve().parent)
_HOST_SOURCE_SUFFIXES = {".cc", ".cpp", ".cxx"}
_METAL_SOURCE_SUFFIX = ".metal"
_LOGGER = logging.getLogger(__name__)


def _is_ninja_available() -> bool:
    try:
        subprocess.check_output(["ninja", "--version"])
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _cmake_quote(value: os.PathLike[str] | str) -> str:
    value = os.fspath(value)
    delimiter = "="
    while f"]{delimiter}]" in value:
        delimiter += "="
    return f"[{delimiter}[{value}]{delimiter}]"


def _cmake_arguments(values: Sequence[os.PathLike[str] | str]) -> list[str]:
    return [f"  {_cmake_quote(value)}" for value in values]


class MetalExtension(Extension):
    """A setuptools extension built from C++ and Metal sources."""

    def __init__(
        self,
        name: str,
        sources: Sequence[os.PathLike[str] | str],
        *args,
        **kwargs,
    ) -> None:
        source_strings = [os.fspath(source) for source in sources]
        suffixes = {Path(source).suffix.lower() for source in source_strings}
        supported_suffixes = _HOST_SOURCE_SUFFIXES | {_METAL_SOURCE_SUFFIX}
        unsupported_sources = [
            source
            for source in source_strings
            if Path(source).suffix.lower() not in supported_suffixes
        ]
        if unsupported_sources:
            raise ValueError(
                "MetalExtension received unsupported source files: "
                + ", ".join(unsupported_sources)
            )
        if not suffixes.intersection(_HOST_SOURCE_SUFFIXES):
            raise ValueError("MetalExtension requires at least one C++ source file.")
        if _METAL_SOURCE_SUFFIX not in suffixes:
            raise ValueError("MetalExtension requires at least one Metal source file.")

        compile_args = kwargs.get("extra_compile_args")
        if compile_args is None:
            compile_args = {"cxx": [], "metal": []}
        elif isinstance(compile_args, (list, tuple)):
            compile_args = {"cxx": list(compile_args), "metal": []}
        elif isinstance(compile_args, dict):
            unknown_groups = set(compile_args).difference({"cxx", "metal"})
            if unknown_groups:
                raise ValueError(
                    "extra_compile_args only supports the cxx and metal groups."
                )
            if any(
                not isinstance(compile_args.get(group, []), (list, tuple))
                for group in ("cxx", "metal")
            ):
                raise TypeError("cxx and metal compile arguments must be lists.")
            compile_args = {
                "cxx": list(compile_args.get("cxx", [])),
                "metal": list(compile_args.get("metal", [])),
            }
        else:
            raise TypeError("extra_compile_args must be a list or a dictionary.")

        kwargs["extra_compile_args"] = compile_args
        kwargs.setdefault("language", "c++")
        super().__init__(name, sources, *args, **kwargs)
        self.metal_library_name = name.rsplit(".", 1)[-1]


def _metal_extension_cmake(
    ext: MetalExtension, output_dir: Path, *, generate_stubs: bool = True
) -> str:
    host_sources = [
        str(Path(source).resolve())
        for source in ext.sources
        if Path(source).suffix.lower() in _HOST_SOURCE_SUFFIXES
    ]
    metal_sources = [
        str(Path(source).resolve())
        for source in ext.sources
        if Path(source).suffix.lower() == _METAL_SOURCE_SUFFIX
    ]
    include_dirs = [str(Path(path).resolve()) for path in ext.include_dirs]
    depends = [str(Path(path).resolve()) for path in ext.depends]
    library_dirs = [str(Path(path).resolve()) for path in ext.library_dirs]
    extra_objects = [str(Path(path).resolve()) for path in ext.extra_objects]
    languages = ["CXX"]

    lines = [
        "cmake_minimum_required(VERSION 3.25)",
        "",
        f"project(mlx_extension LANGUAGES {' '.join(languages)})",
        "",
        "set(CMAKE_CXX_STANDARD 20)",
        "set(CMAKE_CXX_STANDARD_REQUIRED ON)",
        "set(CMAKE_POSITION_INDEPENDENT_CODE ON)",
        "",
        "find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)",
        "if(NOT nanobind_ROOT)",
        "  execute_process(",
        '    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir',
        "    OUTPUT_STRIP_TRAILING_WHITESPACE",
        "    OUTPUT_VARIABLE nanobind_ROOT",
        "    COMMAND_ERROR_IS_FATAL ANY)",
        "endif()",
        "find_package(nanobind CONFIG REQUIRED)",
        "if(NOT MLX_ROOT)",
        f"  set(MLX_ROOT {_cmake_quote(_MLX_PACKAGE_PATH)})",
        "endif()",
        "find_package(MLX CONFIG REQUIRED)",
        "if(NOT MLX_BUILD_METAL)",
        '  message(FATAL_ERROR "MetalExtension requires an MLX build with Metal support.")',
        "endif()",
        "",
        "nanobind_add_module(",
        "  mlx_extension",
        "  NB_STATIC",
        "  STABLE_ABI",
        "  LTO",
        "  NOMINSIZE",
        "  NB_DOMAIN",
        "  mlx",
        *_cmake_arguments(host_sources),
        ")",
        f"set_target_properties(mlx_extension PROPERTIES OUTPUT_NAME {_cmake_quote(ext.metal_library_name)})",
        "set_target_properties(",
        "  mlx_extension",
        "  PROPERTIES",
        f"  LIBRARY_OUTPUT_DIRECTORY {_cmake_quote(output_dir)}",
        f"  LIBRARY_OUTPUT_DIRECTORY_RELEASE {_cmake_quote(output_dir)}",
        f"  LIBRARY_OUTPUT_DIRECTORY_DEBUG {_cmake_quote(output_dir)}",
        f"  LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO {_cmake_quote(output_dir)}",
        f"  LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL {_cmake_quote(output_dir)})",
        "target_link_libraries(mlx_extension PRIVATE mlx)",
    ]

    if include_dirs:
        lines.extend(
            [
                "target_include_directories(mlx_extension PRIVATE",
                *_cmake_arguments(include_dirs),
                ")",
            ]
        )

    definitions = [
        name if value is None else f"{name}={value}"
        for name, value in ext.define_macros
    ]
    if definitions:
        lines.extend(
            [
                "target_compile_definitions(mlx_extension PRIVATE",
                *_cmake_arguments(definitions),
                ")",
            ]
        )

    cxx_compile_args = list(ext.extra_compile_args["cxx"])  # type: ignore
    cxx_compile_args.extend(f"-U{name}" for name in ext.undef_macros)
    if cxx_compile_args:
        lines.extend(
            [
                "target_compile_options(mlx_extension PRIVATE",
                *_cmake_arguments(cxx_compile_args),
                ")",
            ]
        )

    if library_dirs:
        lines.extend(
            [
                "target_link_directories(mlx_extension PRIVATE",
                *_cmake_arguments(library_dirs),
                ")",
            ]
        )

    link_libraries = [*ext.libraries, *extra_objects]
    if link_libraries:
        lines.extend(
            [
                "target_link_libraries(mlx_extension PRIVATE",
                *_cmake_arguments(link_libraries),
                ")",
            ]
        )

    if ext.extra_link_args:
        lines.extend(
            [
                "target_link_options(mlx_extension PRIVATE",
                *_cmake_arguments(ext.extra_link_args),
                ")",
            ]
        )

    rpaths = ["@loader_path", *ext.runtime_library_dirs]
    lines.extend(
        [
            "set_target_properties(",
            "  mlx_extension",
            "  PROPERTIES",
            f"  BUILD_RPATH {_cmake_quote(';'.join(rpaths))}",
            f"  INSTALL_RPATH {_cmake_quote(';'.join(rpaths))})",
            "",
            "mlx_build_metallib(",
            "  TARGET mlx_extension_metallib",
            f"  TITLE {_cmake_quote(ext.metal_library_name)}",
            "  SOURCES",
            *_cmake_arguments(metal_sources),
            "  INCLUDE_DIRS",
            *_cmake_arguments(include_dirs),
            "  ${MLX_INCLUDE_DIRS}",
            "  DEPS",
            *_cmake_arguments(depends),
        ]
    )
    if ext.extra_compile_args["metal"]:  # type: ignore
        lines.extend(
            [
                "  COMPILE_OPTIONS",
                *_cmake_arguments(ext.extra_compile_args["metal"]),  # type: ignore
            ]
        )
    lines.extend(
        [
            f"  OUTPUT_DIRECTORY {_cmake_quote(output_dir)})",
            "add_dependencies(mlx_extension mlx_extension_metallib)",
            "",
        ]
    )
    if generate_stubs:
        stub_path = output_dir / f"{ext.metal_library_name}.pyi"
        stubgen_code = "import mlx.core; from nanobind.stubgen import main; main()"
        lines.extend(
            [
                "add_custom_command(",
                f"  OUTPUT {_cmake_quote(stub_path)}",
                '  COMMAND "${Python_EXECUTABLE}"',
                "          -c",
                f"          {_cmake_quote(stubgen_code)}",
                "          -q",
                "          -m",
                f"          {_cmake_quote(ext.metal_library_name)}",
                "          -i",
                f"          {_cmake_quote(output_dir)}",
                "          -o",
                f"          {_cmake_quote(stub_path)}",
                "  DEPENDS mlx_extension",
                f'  COMMENT "Generating {ext.metal_library_name}.pyi"',
                "  VERBATIM)",
                f"add_custom_target(mlx_extension_stub ALL DEPENDS {_cmake_quote(stub_path)})",
                "",
            ]
        )
    return "\n".join(lines)


class BuildExtension(build_ext):
    """Build :class:`MetalExtension` modules with CMake and Metal.

    CMake uses the Ninja generator by default. Pass ``use_ninja=False`` with
    :meth:`with_options` to use CMake's platform default generator instead.
    If Ninja is unavailable, the build warns and falls back to the platform
    default generator.
    Nanobind stubs are generated by default. Pass ``generate_stubs=False`` to
    disable stub generation.
    """

    @classmethod
    def with_options(cls, **options):
        """Return a command subclass initialized with the given options."""

        class cls_with_options(cls):  # type: ignore
            def __init__(self, *args, **kwargs) -> None:
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        self.use_ninja = kwargs.pop("use_ninja", True)
        self.generate_stubs = kwargs.pop("generate_stubs", True)
        super().__init__(*args, **kwargs)
        if self.use_ninja and not _is_ninja_available():
            _LOGGER.warning(
                "Attempted to use Ninja as the BuildExtension CMake generator, "
                "but Ninja was not found. Falling back to CMake's platform "
                "default generator."
            )
            self.use_ninja = False

    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, MetalExtension):
            super().build_extension(ext)
            return
        if platform.system() != "Darwin":
            raise RuntimeError("MetalExtension is only supported on macOS.")

        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        output_dir = ext_fullpath.parent.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        build_root = Path(self.build_temp) / ext.name
        source_dir = build_root / "source"
        binary_dir = build_root / "build"
        source_dir.mkdir(parents=True, exist_ok=True)
        binary_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "CMakeLists.txt").write_text(
            _metal_extension_cmake(ext, output_dir, generate_stubs=self.generate_stubs),
            encoding="utf-8",
        )

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"
        cmake_args = [
            "-S",
            str(source_dir),
            "-B",
            str(binary_dir),
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DBUILD_SHARED_LIBS=ON",
        ]
        extra_cmake_args = shlex.split(os.environ.get("CMAKE_ARGS", ""))
        generator_is_configured = any(
            argument.startswith("-G")
            or argument == "--generator"
            or argument.startswith("--generator=")
            for argument in extra_cmake_args
        )
        if self.use_ninja and not generator_is_configured:
            cmake_args.extend(["-G", "Ninja"])
        cmake_args.extend(extra_cmake_args)

        archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
        if archs:
            cmake_args.append(f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}")

        build_args = ["--build", str(binary_dir), "--config", cfg]
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            jobs = self.parallel
            if not isinstance(jobs, int) or isinstance(jobs, bool):
                jobs = os.cpu_count()
            build_args.append(f"-j{jobs}")

        subprocess.run(["cmake", *cmake_args], check=True)
        subprocess.run(["cmake", *build_args], check=True)

    def run(self) -> None:
        super().run()
        if not self.inplace:
            return

        build_py = self.get_finalized_command("build_py")
        for ext in self.extensions:
            if not isinstance(ext, MetalExtension):
                continue
            inplace_file, regular_file = self._get_inplace_equivalent(  # type: ignore
                build_py, ext
            )
            sidecar_suffixes = (
                (".metallib", ".pyi") if self.generate_stubs else (".metallib",)
            )
            for suffix in sidecar_suffixes:
                regular_sidecar = Path(regular_file).parent / (
                    ext.metal_library_name + suffix
                )
                inplace_sidecar = Path(inplace_file).parent / regular_sidecar.name
                self.copy_file(str(regular_sidecar), str(inplace_sidecar))


# A CMakeExtension needs a sourcedir instead of a file list.
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

        # Point CMake at the interpreter running the build. Otherwise
        # find_package(Python) picks whichever interpreter it finds first,
        # which is not the one nanobind and mlx are installed into.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_SHARED_LIBS=ON",
            f"-DPython_EXECUTABLE={sys.executable}",
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += [f"-j{os.cpu_count()}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        # Make sure cmake can find MLX
        os.environ["MLX_DIR"] = _MLX_PATH

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )

    def run(self) -> None:
        super().run()

        # Based on https://github.com/pypa/setuptools/blob/main/setuptools/command/build_ext.py#L102
        if self.inplace:
            for ext in self.extensions:
                if isinstance(ext, CMakeExtension):
                    # Resolve inplace package dir
                    build_py = self.get_finalized_command("build_py")
                    inplace_file, regular_file = self._get_inplace_equivalent(
                        build_py, ext
                    )

                    inplace_dir = str(Path(inplace_file).parent.resolve())
                    regular_dir = str(Path(regular_file).parent.resolve())

                    self.copy_tree(regular_dir, inplace_dir)
