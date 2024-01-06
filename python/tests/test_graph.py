# Copyright © 2023 Apple Inc.

import io
import unittest

import mlx.core as mx
import mlx_tests


class TestGraph(mlx_tests.MLXTestCase):
    def test_to_dot(self):
        # Simply test that a few cases run.
        # Nothing too specific about the graph format
        # for now to keep it flexible
        a = mx.array(1.0)
        f = io.StringIO()
        mx.export_to_dot(f, a)
        f.seek(0)
        self.assertTrue(len(f.read()) > 0)

        b = mx.array(2.0)
        c = a + b
        f = io.StringIO()
        mx.export_to_dot(f, c)
        f.seek(0)
        self.assertTrue(len(f.read()) > 0)

        # Multi output case
        c = mx.divmod(a, b)
        f = io.StringIO()
        mx.export_to_dot(f, *c)
        f.seek(0)
        self.assertTrue(len(f.read()) > 0)


if __name__ == "__main__":
    unittest.main()
