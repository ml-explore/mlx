# Copyright © 2026 Apple Inc.

import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mlx.extension import (
    BuildExtension,
    MetalExtension,
    MetalLibrary,
    _extension_paths,
    _metal_extension_cmake,
    _mlx_macos_deployment_target,
)
from setuptools import Distribution


@unittest.skipUnless(platform.system() == "Darwin", "MetalExtension requires macOS")
class TestMetalExtension(unittest.TestCase):
    def test_extension_paths_use_public_build_metadata(self):
        extension = MetalExtension("sample._ext", ["bindings.cpp", "kernel.metal"])
        command = Mock()
        command.build_lib = "/tmp/build"
        command.get_ext_fullname.return_value = extension.name
        command.get_ext_filename.return_value = "sample/_ext.so"
        build_py = Mock()
        build_py.get_package_dir.return_value = "/tmp/source/sample"

        inplace_file, regular_file = _extension_paths(command, build_py, extension)

        self.assertEqual(inplace_file, Path("/tmp/source/sample/_ext.so"))
        self.assertEqual(regular_file, Path("/tmp/build/sample/_ext.so"))
        command.get_ext_fullname.assert_called_once_with(extension.name)
        command.get_ext_filename.assert_called_once_with(extension.name)
        build_py.get_package_dir.assert_called_once_with("sample")

    def _cmake_configure_command(
        self, command_type, cmake_args="", debug=False, mlx_target="14.0"
    ):
        extension = MetalExtension("sample._ext", ["bindings.cpp", "kernel.metal"])
        distribution = Distribution({"ext_modules": [extension]})
        with tempfile.TemporaryDirectory() as directory:
            command = command_type(distribution)
            command.build_lib = str(Path(directory) / "lib")
            command.build_temp = str(Path(directory) / "temp")
            command.debug = debug
            command.parallel = 1
            with patch(
                "mlx.extension._mlx_macos_deployment_target",
                return_value=mlx_target,
            ):
                with patch("mlx.extension.subprocess.run") as run:
                    with patch.dict(os.environ, {"CMAKE_ARGS": cmake_args}):
                        command.build_extension(extension)
            return run.call_args_list[0].args[0]

    def test_reads_mlx_macos_deployment_target(self):
        vtool_output = """
Load command 10
      cmd LC_BUILD_VERSION
 platform MACOS
    minos 14.0
Load command 10
      cmd LC_BUILD_VERSION
 platform MACOS
    minos 15.2
"""
        with patch("mlx.extension._MLX_PACKAGE_PATH", "/tmp/mlx"), patch(
            "mlx.extension.subprocess.check_output", return_value=vtool_output
        ) as check_output:
            deployment_target = _mlx_macos_deployment_target()

        self.assertEqual(deployment_target, "15.2")
        check_output.assert_called_once_with(
            [
                "xcrun",
                "vtool",
                "-show-build",
                "/tmp/mlx/lib/libmlx.dylib",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )

    def test_build_extension_uses_ninja_by_default(self):
        with patch("mlx.extension._is_ninja_available", return_value=True):
            command = BuildExtension(Distribution())
            configure_command = self._cmake_configure_command(BuildExtension)

        self.assertTrue(command.use_ninja)
        self.assertTrue(command.generate_stubs)
        self.assertIn("-DMLX_METAL_DEBUG=OFF", configure_command)
        generator_index = configure_command.index("-G")
        self.assertEqual(
            configure_command[generator_index : generator_index + 2], ["-G", "Ninja"]
        )
        binary_dir = Path(configure_command[configure_command.index("-B") + 1])
        self.assertEqual(binary_dir.name, "build-ninja")

    def test_debug_build_enables_metal_debug(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        configure_command = self._cmake_configure_command(command_type, debug=True)

        self.assertIn("-DCMAKE_BUILD_TYPE=Debug", configure_command)
        self.assertIn("-DMLX_METAL_DEBUG=ON", configure_command)

    def test_detects_current_metal_logging_capability(self):
        extension_cmake = Path(__file__).resolve().parents[2] / "cmake/extension.cmake"
        cases = [
            ("supported", "  -fmetal-enable-logging  Enable logging", 0, "TRUE"),
            ("unsupported", "Metal compiler help", 0, "FALSE"),
            ("probe failure", "-fmetal-enable-logging", 1, "FALSE"),
        ]

        for name, help_text, exit_code, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                xcrun = directory / "xcrun"
                xcrun.write_text(
                    "#!/bin/sh\n"
                    f"printf '%s\\n' '{help_text}'\n"
                    f"exit {exit_code}\n",
                    encoding="utf-8",
                )
                xcrun.chmod(0o755)

                result_file = directory / "result.txt"
                script = directory / "probe.cmake"
                script.write_text(
                    f"include([[{extension_cmake}]])\n"
                    "_mlx_metal_supports_logging(supported)\n"
                    f'file(WRITE [[{result_file}]] "${{supported}}")\n',
                    encoding="utf-8",
                )
                environment = os.environ.copy()
                environment["PATH"] = f"{directory}{os.pathsep}{environment['PATH']}"
                subprocess.run(
                    ["cmake", "-P", str(script)], env=environment, check=True
                )

                self.assertEqual(result_file.read_text(encoding="utf-8"), expected)

    def test_build_extension_uses_mlx_macos_deployment_target(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch.dict(os.environ, {}, clear=True):
            configure_command = self._cmake_configure_command(
                command_type, mlx_target="15.2"
            )

        self.assertIn("-DCMAKE_OSX_DEPLOYMENT_TARGET=15.2", configure_command)

    def test_environment_overrides_macos_deployment_target(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch.dict(os.environ, {"MACOSX_DEPLOYMENT_TARGET": "15.0"}):
            configure_command = self._cmake_configure_command(
                command_type, mlx_target="16.0"
            )

        self.assertIn("-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0", configure_command)

    def test_environment_rejects_macos_deployment_target_below_minimum(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch.dict(os.environ, {"MACOSX_DEPLOYMENT_TARGET": "13.0"}):
            with self.assertRaisesRegex(ValueError, "must be at least 14.0"):
                self._cmake_configure_command(command_type)

    def test_cmake_args_overrides_macos_deployment_target(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch.dict(os.environ, {}, clear=True):
            configure_command = self._cmake_configure_command(
                command_type,
                cmake_args="-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0",
                mlx_target="16.0",
            )

        default_index = configure_command.index("-DCMAKE_OSX_DEPLOYMENT_TARGET=16.0")
        override_index = configure_command.index("-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0")
        self.assertLess(default_index, override_index)

    def test_cmake_args_rejects_macos_deployment_target_below_minimum(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be at least 14.0"):
                self._cmake_configure_command(
                    command_type,
                    cmake_args="-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0",
                )

    def test_build_extension_can_disable_ninja(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch("mlx.extension._is_ninja_available") as is_ninja_available:
            command = command_type(Distribution())
            configure_command = self._cmake_configure_command(command_type)

        self.assertFalse(command.use_ninja)
        self.assertNotIn("-G", configure_command)
        binary_dir = Path(configure_command[configure_command.index("-B") + 1])
        self.assertEqual(binary_dir.name, "build-default")
        is_ninja_available.assert_not_called()

    def test_build_extension_falls_back_without_ninja(self):
        with patch("mlx.extension._is_ninja_available", return_value=False):
            with self.assertLogs("mlx.extension", level="WARNING") as logs:
                command = BuildExtension(Distribution())
                configure_command = self._cmake_configure_command(BuildExtension)

        self.assertFalse(command.use_ninja)
        self.assertNotIn("-G", configure_command)
        binary_dir = Path(configure_command[configure_command.index("-B") + 1])
        self.assertEqual(binary_dir.name, "build-default")
        self.assertIn("Falling back", "\n".join(logs.output))

    def test_explicit_cmake_generator_overrides_ninja_default(self):
        with patch("mlx.extension._is_ninja_available", return_value=True):
            configure_command = self._cmake_configure_command(
                BuildExtension, cmake_args="-G Xcode"
            )

        self.assertEqual(configure_command.count("-G"), 1)
        generator_index = configure_command.index("-G")
        self.assertEqual(
            configure_command[generator_index : generator_index + 2], ["-G", "Xcode"]
        )
        binary_dir = Path(configure_command[configure_command.index("-B") + 1])
        self.assertEqual(binary_dir.name, "build-xcode")

    def test_cmake_generator_environment_uses_distinct_build_directory(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch.dict(os.environ, {"CMAKE_GENERATOR": "Unix Makefiles"}):
            configure_command = self._cmake_configure_command(command_type)

        self.assertNotIn("-G", configure_command)
        binary_dir = Path(configure_command[configure_command.index("-B") + 1])
        self.assertEqual(binary_dir.name, "build-unix_makefiles")

    def test_build_extension_can_disable_stub_generation(self):
        command_type = BuildExtension.with_options(generate_stubs=False)
        command = command_type(Distribution())
        extension = MetalExtension("sample._ext", ["bindings.cpp", "kernel.metal"])

        self.assertFalse(command.generate_stubs)
        with patch(
            "mlx.extension._metal_extension_cmake", wraps=_metal_extension_cmake
        ) as generate_cmake:
            self._cmake_configure_command(command_type)
        self.assertFalse(generate_cmake.call_args.kwargs["generate_stubs"])

        cmake = _metal_extension_cmake(
            extension, Path("/tmp/output"), generate_stubs=False
        )
        self.assertNotIn("nanobind.stubgen", cmake)
        self.assertNotIn("_ext.pyi", cmake)
        self.assertIn("mlx_extension_metallib", cmake)

    def test_extension_metadata(self):
        extension = MetalExtension(
            "sample._ext",
            ["bindings.cpp", "silu.metal", "gelu.metal"],
            extra_compile_args={"cxx": ["-O2"], "metal": ["-O3"]},
        )

        self.assertEqual(
            extension.sources,
            ["bindings.cpp", "silu.metal", "gelu.metal"],
        )
        self.assertEqual(extension.language, "c++")
        self.assertFalse(extension.py_limited_api)
        self.assertEqual(extension.metal_library_name, "_ext")
        self.assertEqual(len(extension.metal_libraries), 1)
        self.assertEqual(extension.metal_libraries[0].name, "_ext")
        self.assertEqual(
            extension.metal_libraries[0].sources,
            ["silu.metal", "gelu.metal"],
        )
        self.assertEqual(
            extension.extra_compile_args,
            {"cxx": ["-O2"], "metal": ["-O3"]},
        )

    def test_requires_host_source(self):
        with self.assertRaisesRegex(ValueError, "C\\+\\+ source"):
            MetalExtension("sample._ext", ["kernel.metal"])

    def test_compile_argument_list_applies_to_cxx(self):
        extension = MetalExtension(
            "sample._ext",
            ["bindings.cpp", "silu.metal", "gelu.metal"],
            extra_compile_args=["-O2"],
        )

        self.assertEqual(
            extension.extra_compile_args,
            {"cxx": ["-O2"], "metal": []},
        )

    def test_requires_metal_source(self):
        with self.assertRaisesRegex(ValueError, "Metal source"):
            MetalExtension("sample._ext", ["bindings.cpp"])

    def test_explicit_metal_libraries(self):
        extension = MetalExtension(
            "sample._ext",
            ["bindings.cpp"],
            metal_libraries=[
                MetalLibrary("first", ["first.metal"]),
                MetalLibrary(
                    "second",
                    ["second.metal"],
                    extra_compile_args=["-O3"],
                    deployment_target="26.2",
                ),
            ],
            extra_compile_args={"metal": ["-Wno-unused"]},
        )

        self.assertEqual(
            [library.name for library in extension.metal_libraries],
            ["first", "second"],
        )
        cmake = _metal_extension_cmake(extension, Path("/tmp/output"))
        self.assertIn("TITLE [=[first]=]", cmake)
        self.assertIn("TITLE [=[second]=]", cmake)
        self.assertIn("TARGET mlx_extension_metallib_0", cmake)
        self.assertIn("TARGET mlx_extension_metallib_1", cmake)
        self.assertIn("DEPLOYMENT_TARGET [=[26.2]=]", cmake)
        self.assertIn("[=[-Wno-unused]=]\n  [=[-O3]=]", cmake)
        self.assertIn(
            "add_dependencies(mlx_extension mlx_extension_metallib_0 "
            "mlx_extension_metallib_1)",
            cmake,
        )

    def test_rejects_mixed_metal_source_forms(self):
        with self.assertRaisesRegex(ValueError, "specified in metal_libraries"):
            MetalExtension(
                "sample._ext",
                ["bindings.cpp", "kernel.metal"],
                metal_libraries=[MetalLibrary("kernels", ["other.metal"])],
            )

    def test_rejects_duplicate_metal_library_names(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            MetalExtension(
                "sample._ext",
                ["bindings.cpp"],
                metal_libraries=[
                    MetalLibrary("kernels", ["first.metal"]),
                    MetalLibrary("kernels", ["second.metal"]),
                ],
            )

    def test_rejects_unknown_compile_argument_group(self):
        with self.assertRaisesRegex(ValueError, "cxx.*metal"):
            MetalExtension(
                "sample._ext",
                ["bindings.cpp", "kernel.metal"],
                extra_compile_args={"nvcc": ["-O3"]},
            )

    def test_rejects_non_list_compile_arguments(self):
        with self.assertRaisesRegex(TypeError, "must be lists"):
            MetalExtension(
                "sample._ext",
                ["bindings.cpp", "kernel.metal"],
                extra_compile_args={"metal": "-O3"},
            )

    def test_generated_cmake_builds_module_and_metallib(self):
        extension = MetalExtension(
            "sample._ext",
            ["bindings.cpp", "silu.metal", "gelu.metal"],
            include_dirs=["include"],
            extra_compile_args={"cxx": ["-Wall"], "metal": ["-O3"]},
        )

        cmake = _metal_extension_cmake(extension, Path("/tmp/output"))

        self.assertIn("nanobind_add_module(", cmake)
        self.assertIn("cmake_minimum_required(VERSION 3.25)", cmake)
        self.assertIn(
            "find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)",
            cmake,
        )
        self.assertNotIn("Development.SABIModule", cmake)
        self.assertNotIn("  STABLE_ABI", cmake)
        self.assertIn("TITLE [=[_ext]=]", cmake)
        self.assertIn("COMPILE_OPTIONS\n  [=[-O3]=]", cmake)
        self.assertIn(str(Path("silu.metal").resolve()), cmake)
        self.assertIn(str(Path("gelu.metal").resolve()), cmake)
        self.assertIn("[=[MLX_EXTENSION_NAME=_ext]=]", cmake)
        self.assertIn('[=[MLX_METAL_LIBRARY_NAME="_ext"]=]', cmake)
        self.assertIn("target_compile_options(mlx_extension PRIVATE", cmake)
        self.assertIn("[=[-Wall]=]", cmake)
        self.assertIn("BUILD_WITH_INSTALL_RPATH TRUE", cmake)
        self.assertIn("INSTALL_RPATH [=[@loader_path;@loader_path/../mlx/lib]=]", cmake)
        self.assertNotIn("BUILD_RPATH", cmake)
        self.assertIn("from nanobind.stubgen import main", cmake)
        self.assertIn("import mlx.core", cmake)
        self.assertIn("OUTPUT [=[/tmp/output/_ext.pyi]=]", cmake)
        self.assertIn("DEPENDS mlx_extension", cmake)

    def test_generated_cmake_uses_nested_package_mlx_rpath(self):
        extension = MetalExtension("sample.ops._ext", ["bindings.cpp", "kernel.metal"])

        cmake = _metal_extension_cmake(extension, Path("/tmp/output"))

        self.assertIn("@loader_path/../../mlx/lib", cmake)

    def test_rejects_python_stable_abi(self):
        with self.assertRaisesRegex(ValueError, "does not support py_limited_api"):
            MetalExtension(
                "sample._ext",
                ["bindings.cpp", "kernel.metal"],
                py_limited_api=True,
            )


if __name__ == "__main__":
    unittest.main()
