# Copyright © 2026 Apple Inc.

import gc
import subprocess
import sys
import textwrap
import unittest
import weakref

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

    def test_transform_releases_captures(self):
        for transform in (
            mx.compile,
            mx.grad,
            mx.value_and_grad,
            mx.vmap,
            mx.checkpoint,
        ):
            for cycle in (False, True):
                with self.subTest(transform=transform, cycle=cycle):
                    captured = mx.array(2.0)
                    ref = weakref.ref(captured)

                    def fun(x, captured=captured):
                        return (x * captured).sum()

                    fn = transform(fun)
                    if cycle:
                        fun.wrapped = fn
                    result = fn(mx.ones((2, 3)))
                    mx.eval(result)
                    del result, fn, fun, captured
                    gc.collect()
                    self.assertIsNone(ref())

    def test_random_state_created_in_thread(self):
        code = textwrap.dedent("""
            import threading
            import mlx.core as mx

            results = []

            def worker():
                mx.set_default_device(mx.cpu)
                state = mx.random.state
                results.append(state is mx.random.state)
                mx.clear_streams()

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            assert results == [True]
            """)
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_vmap_rejects_keywords(self):
        fn = mx.vmap(lambda x: x)
        with self.assertRaises(TypeError):
            fn(x=mx.array([1.0]))

    @unittest.skipUnless(sys.version_info >= (3, 13), "requires Python 3.13")
    def test_import_preserves_gil_state(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "before = sys._is_gil_enabled(); "
                "import mlx.core; "
                "assert sys._is_gil_enabled() == before",
            ],
            check=True,
        )


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
