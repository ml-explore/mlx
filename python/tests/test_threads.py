# Copyright © 2026 Apple Inc.

import threading
import unittest

import mlx.core as mx
import mlx_tests


class TestThreads(mlx_tests.MLXTestCase):
    def test_threadlocal_stream(self):
        raised = [False, False]
        test_stream = mx.new_stream(mx.default_device())

        def test_failure(i):
            with self.assertRaises(RuntimeError):
                with mx.stream(test_stream):
                    x = mx.arange(10)
                    mx.eval(2 * x)
            mx.clear_streams()
            raised[i] = True

        t1 = threading.Thread(target=test_failure, args=(0,))
        t2 = threading.Thread(target=test_failure, args=(1,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertTrue(all(raised))

        raised = [True, True]
        test_stream = mx.new_thread_local_stream(mx.default_device())

        def test_success(i):
            with mx.stream(test_stream):
                x = mx.arange(10)
                mx.eval(2 * x)
                self.assertEqual(x.tolist(), list(range(10)))
            mx.clear_streams()
            raised[i] = False

        t1 = threading.Thread(target=test_success, args=(0,))
        t2 = threading.Thread(target=test_success, args=(1,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertFalse(any(raised))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
