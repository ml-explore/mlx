# Copyright © 2026 Apple Inc.

import importlib.util
import unittest
from collections import namedtuple
from pathlib import Path


def _load_utils_module():
    utils_path = Path(__file__).resolve().parents[1] / "mlx" / "utils.py"
    spec = importlib.util.spec_from_file_location("mlx_utils_pyonly", utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPyOnlyTreeUtils(unittest.TestCase):
    def test_tree_map_with_path_preserves_namedtuple_with_rest_args(self):
        utils = _load_utils_module()
        Params = namedtuple("Params", ["m", "b"])
        params = Params(m=1, b=2)
        incremented = Params(m=3, b=5)
        paths = []

        out = utils.tree_map_with_path(
            lambda path, x, y: paths.append(path) or x + y, params, incremented
        )

        self.assertIsInstance(out, Params)
        self.assertEqual(out, Params(m=4, b=7))
        self.assertEqual(paths, ["0", "1"])


if __name__ == "__main__":
    unittest.main()
