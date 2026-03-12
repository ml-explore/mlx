# Copyright © 2026 Apple Inc.

import threading
import unittest

import mlx.core as mx
import mlx_tests


class TestConcurrentEval(mlx_tests.MLXTestCase):
    """Tests for thread safety of concurrent Metal evaluations.

    Validates that multiple threads can safely perform GPU operations
    on separate streams without crashes or incorrect results.

    Note: mx.random is not used here because the global PRNG state is
    not thread-safe. These tests use deterministic inputs to isolate
    the Metal stream concurrency behavior.
    """

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_concurrent_matmul_separate_streams(self):
        """Multiple threads doing matmuls on separate GPU streams."""
        errors = []
        results = [None] * 4

        def worker(idx, size):
            try:
                s = mx.new_stream(mx.gpu)
                with mx.stream(s):
                    # Deterministic but varied per-thread inputs
                    a = mx.full((size, size), float(idx + 1) * 0.01)
                    b = mx.full((size, size), float(idx + 1) * 0.02)
                    c = a @ b
                    mx.eval(c)
                    results[idx] = c.shape
            except Exception as e:
                errors.append((idx, e))

        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i, 256))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        for i in range(4):
            self.assertEqual(results[i], (256, 256))

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_concurrent_mixed_ops(self):
        """Threads performing different operation types on separate streams."""
        errors = []
        results = {}

        def matmul_worker():
            try:
                s = mx.new_stream(mx.gpu)
                with mx.stream(s):
                    a = mx.full((512, 512), 0.01)
                    b = mx.full((512, 512), 0.02)
                    c = a @ b
                    mx.eval(c)
                    results["matmul"] = c.shape
            except Exception as e:
                errors.append(("matmul", e))

        def reduction_worker():
            try:
                s = mx.new_stream(mx.gpu)
                with mx.stream(s):
                    a = mx.full((1000, 1000), 0.5)
                    r = mx.sum(a)
                    mx.eval(r)
                    results["reduction"] = r.item()
            except Exception as e:
                errors.append(("reduction", e))

        def elementwise_worker():
            try:
                s = mx.new_stream(mx.gpu)
                with mx.stream(s):
                    a = mx.full((2048, 2048), 0.1)
                    b = mx.exp(a) + mx.sin(a) * mx.cos(a)
                    mx.eval(b)
                    results["elementwise"] = b.shape
            except Exception as e:
                errors.append(("elementwise", e))

        def softmax_worker():
            try:
                s = mx.new_stream(mx.gpu)
                with mx.stream(s):
                    a = mx.broadcast_to(
                        mx.arange(1024).reshape(1, 1024), (64, 1024)
                    ).astype(mx.float32) * 0.01
                    b = mx.softmax(a, axis=-1)
                    mx.eval(b)
                    results["softmax"] = b.shape
            except Exception as e:
                errors.append(("softmax", e))

        threads = [
            threading.Thread(target=matmul_worker),
            threading.Thread(target=reduction_worker),
            threading.Thread(target=elementwise_worker),
            threading.Thread(target=softmax_worker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        self.assertEqual(results["matmul"], (512, 512))
        self.assertAlmostEqual(results["reduction"], 500000.0, places=0)
        self.assertEqual(results["elementwise"], (2048, 2048))
        self.assertEqual(results["softmax"], (64, 1024))

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_concurrent_eval_correctness(self):
        """Verify concurrent evaluations produce numerically correct results."""
        errors = []
        results = [None] * 4

        def worker(idx):
            try:
                s = mx.new_stream(mx.gpu)
                with mx.stream(s):
                    # (idx+1) * ones(100,100) @ ones(100,100) = full(100,100, (idx+1)*100)
                    scale = mx.array(float(idx + 1))
                    a = mx.ones((100, 100)) * scale
                    b = mx.ones((100, 100))
                    c = a @ b
                    mx.eval(c)
                    results[idx] = c
            except Exception as e:
                errors.append((idx, e))

        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        for i in range(4):
            expected = float((i + 1) * 100)
            self.assertTrue(
                mx.allclose(results[i], mx.full((100, 100), expected)).item(),
                f"Thread {i}: expected {expected}, got incorrect values",
            )

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_concurrent_sustained_pressure(self):
        """Sustained concurrent GPU pressure over multiple iterations."""
        num_threads = 4
        iterations = 20
        errors = []
        completions = [0] * num_threads

        def worker(idx):
            try:
                for it in range(iterations):
                    s = mx.new_stream(mx.gpu)
                    with mx.stream(s):
                        val = float(idx * iterations + it + 1) * 0.001
                        a = mx.full((256, 256), val)
                        b = mx.full((256, 256), val)
                        c = a @ b
                        c = mx.exp(c)
                        c = mx.sum(c)
                        mx.eval(c)
                        completions[idx] += 1
            except Exception as e:
                errors.append((idx, e))

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        for i in range(num_threads):
            self.assertEqual(
                completions[i],
                iterations,
                f"Thread {i} only completed {completions[i]}/{iterations}",
            )

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_concurrent_streams_with_dependencies(self):
        """Operations reading shared data from separate streams."""
        errors = []
        # Shared input computed on default stream
        shared = mx.ones((256, 256)) * 2.0
        mx.eval(shared)

        results = [None] * 4

        def worker(idx):
            try:
                s = mx.new_stream(mx.gpu)
                with mx.stream(s):
                    local = shared * (idx + 1)
                    local = local @ mx.eye(256)
                    mx.eval(local)
                    results[idx] = local
            except Exception as e:
                errors.append((idx, e))

        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        for i in range(4):
            expected = 2.0 * (i + 1)
            self.assertTrue(
                mx.allclose(results[i], mx.full((256, 256), expected)).item(),
                f"Thread {i}: expected {expected}",
            )

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_high_concurrency_separate_streams(self):
        """8 threads on separate streams with varied workloads."""
        num_threads = 8
        errors = []
        results = [None] * num_threads

        def worker(idx):
            try:
                s = mx.new_stream(mx.gpu)
                with mx.stream(s):
                    size = 128 + (idx * 64)  # 128 to 576
                    a = mx.full((size, size), float(idx + 1) * 0.01)
                    b = mx.full((size, size), 0.01)
                    c = a @ b
                    c = mx.softmax(c, axis=-1)
                    c = mx.sum(c, axis=0)
                    mx.eval(c)
                    results[idx] = c.shape
            except Exception as e:
                errors.append((idx, e))

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        for i in range(num_threads):
            expected_size = 128 + (i * 64)
            self.assertEqual(results[i], (expected_size,))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
