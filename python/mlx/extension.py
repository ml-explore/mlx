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
_MIN_MACOS_DEPLOYMENT_TARGET = "14.0"
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


def _extension_paths(command: build_ext, build_py, ext: Extension) -> tuple[Path, Path]:
    fullname = command.get_ext_fullname(ext.name)
    filename = command.get_ext_filename(fullname)
    package = ".".join(fullname.split(".")[:-1])
    package_dir = build_py.get_package_dir(package)
    inplace_file = Path(package_dir) / Path(filename).name
    regular_file = Path(command.build_lib) / filename
    return inplace_file, regular_file


def _macos_deployment_target(target: str) -> str:
    if not re.fullmatch(r"\d+(?:\.\d+)*", target):
        raise ValueError(f"Invalid macOS deployment target: {target!r}.")
    if int(target.partition(".")[0]) < 14:
        raise ValueError(
            "macOS deployment target must be at least "
            f"{_MIN_MACOS_DEPLOYMENT_TARGET}, got {target!r}."
        )
    return target


def _macos_deployment_target_key(target: str) -> tuple[int, ...]:
    components = [int(component) for component in target.split(".")]
    while len(components) > 1 and components[-1] == 0:
        components.pop()
    return tuple(components)


def _mlx_macos_deployment_target() -> str:
    libmlx = Path(_MLX_PACKAGE_PATH) / "lib" / "libmlx.dylib"
    try:
        output = subprocess.check_output(
            ["xcrun", "vtool", "-show-build", str(libmlx)],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"Failed to read the macOS deployment target from {libmlx}."
        ) from error

    targets = re.findall(r"^\s*minos\s+(\d+(?:\.\d+)*)\s*$", output, re.MULTILINE)
    if not targets:
        raise RuntimeError(f"Could not find a macOS deployment target in {libmlx}.")
    return max(targets, key=_macos_deployment_target_key)


def _cmake_generator(
    extra_cmake_args: Sequence[str], use_ninja: bool
) -> tuple[str, bool]:
    generator = None
    generator_is_configured = False
    for index, argument in enumerate(extra_cmake_args):
        if argument in ("-G", "--generator"):
            generator_is_configured = True
            if index + 1 < len(extra_cmake_args):
                generator = extra_cmake_args[index + 1]
        elif argument.startswith("-G") and argument != "-G":
            generator_is_configured = True
            generator = argument[2:]
        elif argument.startswith("--generator="):
            generator_is_configured = True
            generator = argument.partition("=")[2]

    if generator_is_configured:
        return generator or "explicit", False
    if use_ninja:
        return "Ninja", True
    return os.environ.get("CMAKE_GENERATOR") or "default", False


def _cmake_generator_build_directory(generator: str) -> str:
    generator = re.sub(r"[^a-z0-9]+", "_", generator.lower()).strip("_")
    return f"build-{generator or 'default'}"


class MetalLibrary:
    """A named Metal library built as part of a :class:`MetalExtension`."""

    def __init__(
        self,
        name: str,
        sources: Sequence[os.PathLike[str] | str],
        *,
        extra_compile_args: Sequence[str] = (),
        deployment_target: str | None = None,
    ) -> None:
        if not name or Path(name).name != name:
            raise ValueError(f"Invalid Metal library name: {name!r}.")
        source_strings = [os.fspath(source) for source in sources]
        if not source_strings:
            raise ValueError("MetalLibrary requires at least one Metal source file.")
        if any(
            Path(source).suffix.lower() != _METAL_SOURCE_SUFFIX
            for source in source_strings
        ):
            raise ValueError("MetalLibrary sources must be Metal source files.")
        if isinstance(extra_compile_args, (str, bytes)):
            raise TypeError("MetalLibrary extra_compile_args must be a sequence.")

        self.name = name
        self.sources = source_strings
        self.extra_compile_args = list(extra_compile_args)
        self.deployment_target = (
            _macos_deployment_target(deployment_target)
            if deployment_target is not None
            else None
        )


class MetalExtension(Extension):
    """A setuptools extension built from C++ and Metal sources."""

    def __init__(
        self,
        name: str,
        sources: Sequence[os.PathLike[str] | str],
        *args,
        metal_libraries: Sequence[MetalLibrary] | None = None,
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
        if metal_libraries is None and _METAL_SOURCE_SUFFIX not in suffixes:
            raise ValueError("MetalExtension requires at least one Metal source file.")
        if metal_libraries is not None and _METAL_SOURCE_SUFFIX in suffixes:
            raise ValueError(
                "Metal sources must be specified in metal_libraries when it is used."
            )
        if kwargs.get("py_limited_api", False):
            raise ValueError("MetalExtension does not support py_limited_api.")

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
        if metal_libraries is None:
            self.metal_libraries = [
                MetalLibrary(
                    self.metal_library_name,
                    [
                        source
                        for source in source_strings
                        if Path(source).suffix.lower() == _METAL_SOURCE_SUFFIX
                    ],
                )
            ]
        else:
            self.metal_libraries = list(metal_libraries)
            if not self.metal_libraries:
                raise ValueError("metal_libraries must contain at least one library.")
            if not all(
                isinstance(library, MetalLibrary) for library in self.metal_libraries
            ):
                raise TypeError("metal_libraries must contain MetalLibrary objects.")
            names = [library.name for library in self.metal_libraries]
            if len(names) != len(set(names)):
                raise ValueError("Metal library names must be unique.")


def _metal_extension_cmake(
    ext: MetalExtension, output_dir: Path, *, generate_stubs: bool = True
) -> str:
    host_sources = [
        str(Path(source).resolve())
        for source in ext.sources
        if Path(source).suffix.lower() in _HOST_SOURCE_SUFFIXES
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
        f"MLX_EXTENSION_NAME={ext.metal_library_name}",
        f'MLX_METAL_LIBRARY_NAME="{ext.metal_library_name}"',
        *(
            name if value is None else f"{name}={value}"
            for name, value in ext.define_macros
        ),
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

    mlx_rpath = "@loader_path/" + "../" * ext.name.count(".") + "mlx/lib"
    rpaths = ["@loader_path", mlx_rpath, *ext.runtime_library_dirs]
    lines.extend(
        [
            "set_target_properties(",
            "  mlx_extension",
            "  PROPERTIES",
            "  BUILD_WITH_INSTALL_RPATH TRUE",
            f"  INSTALL_RPATH {_cmake_quote(';'.join(rpaths))})",
            "",
        ]
    )
    metallib_targets = []
    for index, library in enumerate(ext.metal_libraries):
        target = (
            "mlx_extension_metallib"
            if len(ext.metal_libraries) == 1
            else f"mlx_extension_metallib_{index}"
        )
        metallib_targets.append(target)
        lines.extend(
            [
                "mlx_build_metallib(",
                f"  TARGET {target}",
                f"  TITLE {_cmake_quote(library.name)}",
                "  SOURCES",
                *_cmake_arguments(
                    list(str(Path(source).resolve()) for source in library.sources)
                ),
                "  INCLUDE_DIRS",
                *_cmake_arguments(include_dirs),
                "  ${MLX_INCLUDE_DIRS}",
                "  DEPS",
                *_cmake_arguments(depends),
            ]
        )
        compile_options = [
            *ext.extra_compile_args["metal"],  # type: ignore
            *library.extra_compile_args,
        ]
        if compile_options:
            lines.extend(
                [
                    "  COMPILE_OPTIONS",
                    *_cmake_arguments(compile_options),
                ]
            )
        if library.deployment_target is not None:
            lines.append(
                f"  DEPLOYMENT_TARGET {_cmake_quote(library.deployment_target)}"
            )
        lines.extend(
            [
                f"  OUTPUT_DIRECTORY {_cmake_quote(output_dir)})",
                "",
            ]
        )
    lines.extend(
        [
            f"add_dependencies(mlx_extension {' '.join(metallib_targets)})",
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

        extra_cmake_args = shlex.split(os.environ.get("CMAKE_ARGS", ""))
        generator, add_ninja_generator = _cmake_generator(
            extra_cmake_args, self.use_ninja
        )
        build_root = Path(self.build_temp) / ext.name
        source_dir = build_root / "source"
        binary_dir = build_root / _cmake_generator_build_directory(generator)
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
            f"-DMLX_METAL_DEBUG={'ON' if debug else 'OFF'}",
        ]

        deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        if deployment_target is None:
            deployment_target = _mlx_macos_deployment_target()
        else:
            deployment_target = _macos_deployment_target(deployment_target)
        cmake_args.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}")

        deployment_target_override = None
        for argument in extra_cmake_args:
            match = re.fullmatch(
                r"(?:-D)?CMAKE_OSX_DEPLOYMENT_TARGET(?::[^=]+)?=(.*)", argument
            )
            if match:
                deployment_target_override = match.group(1)
        if deployment_target_override is not None:
            _macos_deployment_target(deployment_target_override)
        if add_ninja_generator:
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
            inplace_file, regular_file = _extension_paths(self, build_py, ext)
            sidecar_names = [
                f"{library.name}.metallib" for library in ext.metal_libraries
            ]
            if self.generate_stubs:
                sidecar_names.append(f"{ext.metal_library_name}.pyi")
            for name in sidecar_names:
                regular_sidecar = regular_file.parent / name
                inplace_sidecar = inplace_file.parent / regular_sidecar.name
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
        loader_path = None
        if sys.platform.startswith("darwin"):
            loader_path = "@loader_path"
        elif sys.platform.startswith("linux"):
            loader_path = "$ORIGIN"
        if loader_path is not None:
            mlx_rpath = loader_path + "/" + "../" * ext.name.count(".") + "mlx/lib"
            cmake_args += [
                "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                f"-DCMAKE_INSTALL_RPATH={loader_path};{mlx_rpath}",
            ]

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
                    inplace_file, regular_file = _extension_paths(self, build_py, ext)

                    inplace_dir = str(inplace_file.parent.resolve())
                    regular_dir = str(regular_file.parent.resolve())

                    self.copy_tree(regular_dir, inplace_dir)
