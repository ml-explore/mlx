# Copyright © 2026 Apple Inc.

import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx.extension import BuildExtension, MetalExtension, _metal_extension_cmake
from setuptools import Distribution


@unittest.skipUnless(platform.system() == "Darwin", "MetalExtension requires macOS")
class TestMetalExtension(unittest.TestCase):
    def _cmake_configure_command(self, command_type, cmake_args="", debug=False):
        extension = MetalExtension("sample._ext", ["bindings.cpp", "kernel.metal"])
        distribution = Distribution({"ext_modules": [extension]})
        with tempfile.TemporaryDirectory() as directory:
            command = command_type(distribution)
            command.build_lib = str(Path(directory) / "lib")
            command.build_temp = str(Path(directory) / "temp")
            command.debug = debug
            command.parallel = 1
            with patch("mlx.extension.subprocess.run") as run:
                with patch.dict(os.environ, {"CMAKE_ARGS": cmake_args}):
                    command.build_extension(extension)
            return run.call_args_list[0].args[0]

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

    def test_build_extension_sets_minimum_macos_deployment_target(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "mlx.extension.sysconfig.get_config_var", return_value="11.0"
            ) as get_config_var:
                configure_command = self._cmake_configure_command(command_type)

        get_config_var.assert_called_once_with("MACOSX_DEPLOYMENT_TARGET")
        self.assertIn("-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0", configure_command)

    def test_environment_overrides_macos_deployment_target(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch.dict(os.environ, {"MACOSX_DEPLOYMENT_TARGET": "15.0"}):
            with patch("mlx.extension.sysconfig.get_config_var", return_value="11.0"):
                configure_command = self._cmake_configure_command(command_type)

        self.assertIn("-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0", configure_command)

    def test_environment_rejects_macos_deployment_target_below_minimum(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch.dict(os.environ, {"MACOSX_DEPLOYMENT_TARGET": "13.0"}):
            with self.assertRaisesRegex(ValueError, "must be at least 14.0"):
                self._cmake_configure_command(command_type)

    def test_cmake_args_overrides_macos_deployment_target(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch.dict(os.environ, {}, clear=True):
            with patch("mlx.extension.sysconfig.get_config_var", return_value="11.0"):
                configure_command = self._cmake_configure_command(
                    command_type,
                    cmake_args="-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0",
                )

        default_index = configure_command.index("-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0")
        override_index = configure_command.index("-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0")
        self.assertLess(default_index, override_index)

    def test_cmake_args_rejects_macos_deployment_target_below_minimum(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch.dict(os.environ, {}, clear=True):
            with patch("mlx.extension.sysconfig.get_config_var", return_value="14.0"):
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
        self.assertEqual(extension.metal_library_name, "_ext")
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
        self.assertIn("TITLE [=[_ext]=]", cmake)
        self.assertIn("COMPILE_OPTIONS\n  [=[-O3]=]", cmake)
        self.assertIn(str(Path("silu.metal").resolve()), cmake)
        self.assertIn(str(Path("gelu.metal").resolve()), cmake)
        self.assertIn("[=[MLX_EXTENSION_NAME=_ext]=]", cmake)
        self.assertIn('[=[MLX_METAL_LIBRARY_NAME="_ext"]=]', cmake)
        self.assertIn("target_compile_options(mlx_extension PRIVATE", cmake)
        self.assertIn("[=[-Wall]=]", cmake)
        self.assertIn("from nanobind.stubgen import main", cmake)
        self.assertIn("import mlx.core", cmake)
        self.assertIn("OUTPUT [=[/tmp/output/_ext.pyi]=]", cmake)
        self.assertIn("DEPENDS mlx_extension", cmake)


if __name__ == "__main__":
    unittest.main()
