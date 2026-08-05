# Copyright © 2026 Apple Inc.

import os
import platform
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx.extension import BuildExtension, MetalExtension, _metal_extension_cmake
from setuptools import Distribution


@unittest.skipUnless(platform.system() == "Darwin", "MetalExtension requires macOS")
class TestMetalExtension(unittest.TestCase):
    def _cmake_configure_command(self, command_type, cmake_args=""):
        extension = MetalExtension("sample._ext", ["bindings.cpp", "kernel.metal"])
        distribution = Distribution({"ext_modules": [extension]})
        with tempfile.TemporaryDirectory() as directory:
            command = command_type(distribution)
            command.build_lib = str(Path(directory) / "lib")
            command.build_temp = str(Path(directory) / "temp")
            command.debug = False
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
        generator_index = configure_command.index("-G")
        self.assertEqual(
            configure_command[generator_index : generator_index + 2], ["-G", "Ninja"]
        )

    def test_build_extension_can_disable_ninja(self):
        command_type = BuildExtension.with_options(use_ninja=False)
        with patch("mlx.extension._is_ninja_available") as is_ninja_available:
            command = command_type(Distribution())
            configure_command = self._cmake_configure_command(command_type)

        self.assertFalse(command.use_ninja)
        self.assertNotIn("-G", configure_command)
        is_ninja_available.assert_not_called()

    def test_build_extension_falls_back_without_ninja(self):
        with patch("mlx.extension._is_ninja_available", return_value=False):
            with self.assertLogs("mlx.extension", level="WARNING") as logs:
                command = BuildExtension(Distribution())
                configure_command = self._cmake_configure_command(BuildExtension)

        self.assertFalse(command.use_ninja)
        self.assertNotIn("-G", configure_command)
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
        self.assertIn("target_compile_options(mlx_extension PRIVATE", cmake)
        self.assertIn("[=[-Wall]=]", cmake)
        self.assertIn("from nanobind.stubgen import main", cmake)
        self.assertIn("import mlx.core", cmake)
        self.assertIn("OUTPUT [=[/tmp/output/_ext.pyi]=]", cmake)
        self.assertIn("DEPENDS mlx_extension", cmake)


if __name__ == "__main__":
    unittest.main()
