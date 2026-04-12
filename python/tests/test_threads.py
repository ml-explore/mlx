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

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU is not available")
    def test_register_stream_cross_thread(self):
        """register_stream allows eval on a thread that did not create the stream."""
        s = mx.new_stream(mx.gpu)
        x = mx.ones((4, 4), stream=s)
        y = mx.abs(x, stream=s)

        errors = []

        def eval_on_thread():
            try:
                mx.register_stream(s)
                mx.eval(y)
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=eval_on_thread)
        t.start()
        t.join()

        self.assertEqual(len(errors), 0, f"eval failed on new thread: {errors}")
        self.assertTrue(mx.array_equal(y, mx.ones((4, 4))))

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU is not available")
    def test_register_stream_idempotent(self):
        """Calling register_stream multiple times does not error."""
        s = mx.new_stream(mx.gpu)
        mx.register_stream(s)
        mx.register_stream(s)
        x = mx.ones((3,), stream=s)
        mx.eval(x)
        self.assertEqual(x.tolist(), [1.0, 1.0, 1.0])


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
