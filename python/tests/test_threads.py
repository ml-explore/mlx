# Copyright © 2026 Apple Inc.

import threading
import unittest

import mlx.core as mx
import mlx_tests


class TestReduce(mlx_tests.MLXTestCase):
    def test_threadlocal_stream(self):
        test_stream = mx.new_stream(mx.default_device())

        def test_failure():
            with self.assertRaises(RuntimeError):
                with mx.stream(test_stream):
                    x = mx.arange(10)
                    mx.eval(2 * x)

        t1 = threading.Thread(target=test_failure)
        t2 = threading.Thread(target=test_failure)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        test_stream = mx.ThreadLocalStream(mx.default_device())

        def test_success():
            with mx.stream(test_stream):
                x = mx.arange(10)
                mx.eval(2 * x)
                self.assertEqual(x.tolist(), list(range(10)))

        t1 = threading.Thread(target=test_success)
        t2 = threading.Thread(target=test_success)
        t1.start()
        t2.start()
        t1.join()
        t2.join()


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
