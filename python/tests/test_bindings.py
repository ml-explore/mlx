# Copyright © 2026 Apple Inc.

import sys
import unittest

import mlx.core as mx
import mlx_tests


class TestBindings(mlx_tests.MLXTestCase):
    @unittest.skipUnless(sys.version_info >= (3, 15), "requires Python 3.15")
    def test_frozen_types(self):
        types = (
            mx.array,
            mx.Dtype,
            mx.finfo,
            mx.iinfo,
            mx.ArrayAt,
            mx.ArrayLike,
            mx.ArrayIterator,
            mx.Device,
            mx.Stream,
            mx.ThreadLocalStream,
            mx.StreamContext,
            mx.PrintOptions,
            mx.custom_function,
            mx.FunctionExporter,
            mx.distributed.Group,
        )
        for bound_type in types:
            with self.subTest(type=bound_type):
                with self.assertRaises(TypeError):
                    bound_type._test_attribute = None
                with self.assertRaises(TypeError):
                    bound_type.__doc__ = "modified"
                with self.assertRaises(TypeError):
                    del bound_type.__doc__

    def test_array_subclass(self):
        class Array(mx.array):
            def total(self):
                return self.sum().item()

        self.assertEqual(Array([1, 2, 3]).total(), 6)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
