# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestConfig(unittest.TestCase):
    def test_get_environment(self):
        with mlx_tests.scoped_env(MLX_TEST_CONFIG_ENV="13"):
            self.assertEqual(mx.config.get("MLX_TEST_CONFIG_ENV", 0), 13)

    def test_update(self):
        name = "MLX_TEST_CONFIG_UPDATE"
        with mlx_tests.scoped_env(MLX_TEST_CONFIG_UPDATE="13"):
            self.assertIsNone(mx.config.update(name, 42))
            self.assertEqual(mx.config.get(name, 0), 42)

    def test_scoped_update(self):
        configs = {
            "MLX_TEST_1": 1,
            "MLX_TEST_2": 2,
            "MLX_TEST_3": 3,
        }
        with mx.config.scoped_update(**configs):
            for name, value in configs.items():
                self.assertEqual(mx.config.get(name, 0), value)
        for name in configs:
            self.assertEqual(mx.config.get(name, 0), 0)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
