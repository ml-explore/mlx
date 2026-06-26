# Copyright © 2026 Apple Inc.
"""Tests for the pure-Python chunked ``mx.while_loop``."""

import math
import os
import threading
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np
from mlx._while_loop import _DEFAULT_CHUNK_SIZE, _MAX_CHUNK_SIZE, _to_bool_array


def _reference_loop(cond_fun, body_fun, init_val, max_iterations=None):
    """Ground-truth Python while loop mirroring while_loop semantics, including
    cond coercion (Python/numpy bools are wrapped like _to_bool_array)."""
    carry = init_val
    n = 0
    while bool(mx.all(_to_bool_array(cond_fun(carry)))):
        if max_iterations is not None and n >= max_iterations:
            raise RuntimeError("max_iterations exceeded")
        carry = body_fun(carry)
        n += 1
    return carry


class TestWhileLoop(mlx_tests.MLXTestCase):
    def setUp(self):
        super().setUp()
        mx.clear_cache()
        mx.reset_peak_memory()

    def tearDown(self):
        super().tearDown()

    # ---- T1-T4: correctness vs reference (scalar + pytree) ----

    def test_counter_scalar(self):
        for chunk_size in [1, 2, 8, 64]:
            with self.subTest(chunk_size=chunk_size):
                cond = lambda c: c < 100
                body = lambda c: c + 1
                out = mx.while_loop(cond, body, mx.array(0), chunk_size=chunk_size)
                ref = _reference_loop(cond, body, mx.array(0))
                self.assertEqualArray(out, ref)

    def test_factorial_tuple_carry(self):
        for chunk_size in [1, 4, 16]:
            with self.subTest(chunk_size=chunk_size):
                cond = lambda c: c[1] < 6
                body = lambda c: (c[0] * (c[1] + 1), c[1] + 1)
                init = (mx.array(1), mx.array(0))
                out = mx.while_loop(cond, body, init, chunk_size=chunk_size)
                ref = _reference_loop(cond, body, init)
                self.assertEqualArray(out[0], ref[0])
                self.assertEqualArray(out[1], ref[1])

    def test_sum_until_list_carry(self):
        cond = lambda c: c[0] < 50
        body = lambda c: [c[0] + c[1], c[1] - 1]
        init = [mx.array(0), mx.array(10)]
        out = mx.while_loop(cond, body, init, chunk_size=8)
        ref = _reference_loop(cond, body, init)
        self.assertEqualArray(out[0], ref[0])

    def test_geometric_dict_carry(self):
        for chunk_size in [1, 4, 16, 64]:
            with self.subTest(chunk_size=chunk_size):
                cond = lambda c: c["i"] < 10
                body = lambda c: {"i": c["i"] + 1, "v": c["v"] * 2}
                init = {"i": mx.array(0), "v": mx.array(1.0)}
                out = mx.while_loop(cond, body, init, chunk_size=chunk_size)
                ref = _reference_loop(cond, body, init)
                self.assertEqualArray(out["v"], ref["v"])

    # ---- T5: cond false at init ----

    def test_cond_false_at_init(self):
        cond = lambda c: mx.array(False)
        body = lambda c: c + 1
        init = mx.array(42)
        out = mx.while_loop(cond, body, init, chunk_size=8)
        self.assertEqualArray(out, init)  # unchanged

    def test_body_not_called_when_cond_false_at_init(self):
        # JAX semantics: body is never called when cond is false at entry.
        calls = [0]

        def body(c):
            calls[0] += 1
            return c + 1

        out = mx.while_loop(
            lambda c: mx.array(False),
            body,
            mx.array(0),
            chunk_size=8,
            max_iterations=20,
        )
        self.assertEqual(int(out.item()), 0)
        self.assertEqual(calls[0], 0)  # body NOT called (zero-iteration no-op)

    def test_vectorized_carry(self):
        # Multi-element carry leaf with element-wise cond (reduced by mx.all).
        init = mx.array([0, 0, 0])
        cond = (
            lambda c: c < 5
        )  # element-wise; mx.all reduces -> all lanes advance together
        body = lambda c: c + 1
        for chunk_size in [1, 2, 4, 8]:
            with self.subTest(chunk_size=chunk_size):
                out = mx.while_loop(
                    cond, body, init, max_iterations=20, chunk_size=chunk_size
                )
                ref = _reference_loop(cond, body, init)
                self.assertEqualArray(out, ref)

    def test_nested_pytree_carry(self):
        # Nested dict/list/tuple carry exercises _walk recursion.
        init = {"outer": {"a": mx.array(0)}, "list": [mx.array(1.0), (mx.array(2),)]}
        cond = lambda c: c["outer"]["a"] < 5
        body = lambda c: {
            "outer": {"a": c["outer"]["a"] + 1},
            "list": [c["list"][0] * 2, (c["list"][1][0] + 1,)],
        }
        for chunk_size in [1, 4, 16]:
            with self.subTest(chunk_size=chunk_size):
                out = mx.while_loop(
                    cond, body, init, max_iterations=20, chunk_size=chunk_size
                )
                ref = _reference_loop(cond, body, init)
                self.assertEqualArray(out["outer"]["a"], ref["outer"]["a"])
                self.assertEqualArray(out["list"][0], ref["list"][0])

    # ---- T6: max_iterations cap ----

    def test_cap_raises_when_cond_always_true(self):
        for chunk_size in [1, 2, 5, 10]:
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(RuntimeError, "max_iterations"):
                    mx.while_loop(
                        lambda c: mx.array(True),
                        lambda c: c + 1,
                        mx.array(0),
                        max_iterations=10,
                        chunk_size=chunk_size,
                    )

    def test_trip_equals_max_no_raise(self):
        # trip-count == max_iterations (multiple of chunk_size) -> RETURNS, not raise
        for chunk_size, n in [(1, 10), (2, 10), (5, 10), (10, 10), (8, 16)]:
            with self.subTest(chunk_size=chunk_size, n=n):
                out = mx.while_loop(
                    lambda c: c < n,
                    lambda c: c + 1,
                    mx.array(0),
                    max_iterations=n,
                    chunk_size=chunk_size,
                )
                self.assertEqual(int(out.item()), n)

    # ---- T7: NaN safety ----

    def test_nan_safety(self):
        # Stopped-lane NaN is masked away by the loop-level where.
        with self.subTest("stopped_lane_nan_masked"):

            def cond(c):
                return c < 3

            def body(c):
                # Active lane gets +1; stopped lane "would" get NaN but is masked.
                return mx.where(c < 3, c + 1.0, mx.array(float("nan")))

            out = mx.while_loop(
                cond, body, mx.array(0.0), max_iterations=10, chunk_size=4
            )
            self.assertFalse(math.isnan(out.item()))
            self.assertEqual(out.item(), 3.0)

        # Active-lane NaN DOES contaminate (documented limitation).
        with self.subTest("active_lane_nan_contaminates"):

            def cond2(c):
                return c < 5

            def body2(c):
                # Produces NaN on active lane (c == 3 -> sqrt(3-3)=0 ok; but force NaN)
                return mx.where(c == 3, mx.array(float("nan")), c + 1.0)

            out = mx.while_loop(
                cond2, body2, mx.array(0.0), max_iterations=20, chunk_size=4
            )
            self.assertTrue(math.isnan(out.item()))

    # ---- T8: perf (sync count, not wall-time ratio) ----

    @unittest.skipIf(
        os.getenv("MLX_SKIP_PERF_TESTS") or not mx.metal.is_available(),
        "perf test skipped on CPU or when MLX_SKIP_PERF_TESTS set",
    )
    def test_sync_count_amortized(self):
        # The driver does exactly one mx.eval per chunk. Monkeypatch mx.eval to
        # count. bool()/item() do NOT call mx.eval (they use to_scalar on
        # already-evaluated arrays), so this counts driver syncs.
        N, chunk_size = 200, 16
        orig_eval = mx.eval
        count = [0]

        def counting_eval(*args):
            count[0] += 1
            return orig_eval(*args)

        mx.eval = counting_eval
        try:
            out = mx.while_loop(
                lambda c: c < N,
                lambda c: c + 1,
                mx.array(0),
                max_iterations=N * 2,
                chunk_size=chunk_size,
            )
            mx.eval(out)
        finally:
            mx.eval = orig_eval
        # +1 for the final explicit mx.eval(out); +1 slack for the initial
        # cond(init) check. Asserts the amortization O(N/chunk_size), not a
        # wall-time ratio. Assumes bool()/item() don't call mx.eval (true on
        # this build; see impl driver).
        self.assertLessEqual(count[0], math.ceil(N / chunk_size) + 2)

    # ---- T9: chunk_size=1 ----

    def test_chunk_size_1(self):
        out = mx.while_loop(lambda c: c < 7, lambda c: c + 1, mx.array(0), chunk_size=1)
        self.assertEqual(int(out.item()), 7)

    # ---- T10: idempotency past termination ----

    def test_idempotency_past_termination(self):
        # Behavior: result is correct (carry where cond is false).
        out = mx.while_loop(
            lambda c: c < 3,
            lambda c: c + 1,
            mx.array(0),
            chunk_size=8,
            max_iterations=20,
        )
        self.assertEqual(int(out.item()), 3)

    def test_speculative_body_calls_are_eager(self):
        # White-box contract: body is evaluated eagerly (chunk_size times during
        # the first chunk's compile trace + 1 structure-validation call) even
        # past termination; output is masked by mx.where. Bodies must be pure.
        calls = [0]

        def body(c):
            calls[0] += 1
            return c + 1

        out = mx.while_loop(
            lambda c: c < 3, body, mx.array(0), chunk_size=8, max_iterations=20
        )
        self.assertEqual(int(out.item()), 3)
        # 1 (validation) + chunk_size (first-chunk trace) calls minimum.
        self.assertGreaterEqual(calls[0], 8)

    # ---- T11: structure/shape/dtype validation ----

    def test_structure_mismatch_extra_key(self):
        with self.assertRaisesRegex(ValueError, "structure"):
            mx.while_loop(
                lambda c: c["i"] < 3,
                lambda c: {"i": c["i"] + 1, "extra": mx.array(0)},
                {"i": mx.array(0)},
                max_iterations=10,
            )

    def test_structure_mismatch_missing_key(self):
        with self.assertRaisesRegex(ValueError, "structure"):
            mx.while_loop(
                lambda c: c["i"] < 3,
                lambda c: {},
                {"i": mx.array(0)},
                max_iterations=10,
            )

    def test_structure_mismatch_shape(self):
        with self.assertRaisesRegex(ValueError, "structure"):
            mx.while_loop(
                lambda c: c[0] < 3,
                lambda c: [mx.zeros((5,))],  # init is (1,)
                [mx.array(0)],
                max_iterations=10,
            )

    def test_structure_mismatch_dtype(self):
        with self.assertRaisesRegex(ValueError, "structure"):
            mx.while_loop(
                lambda c: c < 3,
                lambda c: c.astype(mx.float32),  # init int32 -> body float32
                mx.array(0),
                max_iterations=10,
            )

    def test_structure_tuple_vs_list(self):
        with self.assertRaisesRegex(ValueError, "structure"):
            mx.while_loop(
                lambda c: c[0] < 3,
                lambda c: [c[0] + 1],  # body returns list; init is tuple
                (mx.array(0),),
                max_iterations=10,
            )

    def test_structure_key_order_accepted(self):
        # Same keys, different insertion order -> accepted (order-independent).
        init = {"b": mx.array(0), "a": mx.array(0)}
        body = lambda c: {"a": c["a"] + 1, "b": c["b"] + 1}
        cond = lambda c: c["a"] < 5
        out = mx.while_loop(cond, body, init, chunk_size=2, max_iterations=20)
        self.assertEqual(int(out["a"].item()), 5)
        self.assertEqual(int(out["b"].item()), 5)

    # ---- T12: not differentiable ----

    @unittest.skipUnless(
        hasattr(mx.array(0.0), "is_tracer"),
        "requires array.is_tracer() C++ binding",
    )
    def test_not_differentiable_grad_direct(self):
        def loss(x):
            return mx.while_loop(
                lambda c: c < 5,
                lambda c: c + x,
                mx.array(0.0),
                max_iterations=10,
                chunk_size=4,
            ).sum()

        with self.assertRaisesRegex(RuntimeError, "not differentiable|tracer"):
            mx.grad(loss)(mx.array(1.0))

    @unittest.skipUnless(
        hasattr(mx.array(0.0), "is_tracer"),
        "requires array.is_tracer() C++ binding",
    )
    def test_not_differentiable_grad_closure(self):
        # The v4 key case: constant init, body closes over the traced param.
        def loss(w):
            return mx.while_loop(
                lambda c: c < 10.0,
                lambda c: c + w,
                mx.array(0.0),
                max_iterations=20,
                chunk_size=4,
            ).sum()

        with self.assertRaisesRegex(RuntimeError, "not differentiable|tracer"):
            mx.grad(loss)(mx.array(1.0))

    @unittest.skipUnless(
        hasattr(mx.array(0.0), "is_tracer"),
        "requires array.is_tracer() C++ binding",
    )
    def test_not_differentiable_vmap(self):
        with self.assertRaisesRegex(RuntimeError, "not differentiable|tracer"):
            mx.vmap(
                lambda x: mx.while_loop(
                    lambda c: c < 3,
                    lambda c: c + 1,
                    x,
                    max_iterations=10,
                    chunk_size=4,
                )
            )(mx.array([0, 1, 2]))

    # ---- T13: not compile-traceable ----

    @unittest.skipUnless(
        hasattr(mx.array(0.0), "is_tracer"),
        "requires array.is_tracer() C++ binding",
    )
    def test_not_compile_traceable(self):
        with self.assertRaisesRegex(
            RuntimeError, "not differentiable|tracer|cannot be used"
        ):
            mx.compile(
                lambda x: mx.while_loop(
                    lambda c: c < 3,
                    lambda c: c + 1,
                    x,
                    max_iterations=10,
                    chunk_size=4,
                )
            )(mx.array(0))

    # ---- T14: multi-output tuple ----

    def test_multi_output_tuple(self):
        cond = lambda c: c[1] < 5
        body = lambda c: (c[0] + c[1], c[1] + 1)
        out = mx.while_loop(cond, body, (mx.array(0), mx.array(0)), chunk_size=4)
        ref = _reference_loop(cond, body, (mx.array(0), mx.array(0)))
        self.assertEqualArray(out[0], ref[0])
        self.assertEqualArray(out[1], ref[1])

    # ---- T15: chunk_size boundaries ----

    def test_chunk_size_1024(self):
        out = mx.while_loop(
            lambda c: c < 100, lambda c: c + 1, mx.array(0), chunk_size=1024
        )
        self.assertEqual(int(out.item()), 100)

    def test_chunk_size_4096_correctness(self):
        out = mx.while_loop(
            lambda c: c < 5000, lambda c: c + 1, mx.array(0), chunk_size=4096
        )
        self.assertEqual(int(out.item()), 5000)

    def test_chunk_size_too_large_rejected(self):
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            mx.while_loop(
                lambda c: c < 5,
                lambda c: c + 1,
                mx.array(0),
                chunk_size=_MAX_CHUNK_SIZE + 1,
            )

    def test_chunk_size_zero_negative_nonint_rejected(self):
        for bad in [0, -1, 1.5, True, "8"]:
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "chunk_size"):
                    mx.while_loop(
                        lambda c: c < 5, lambda c: c + 1, mx.array(0), chunk_size=bad
                    )

    # ---- T16: kw-only ----

    def test_max_iterations_keyword_only(self):
        with self.assertRaises(TypeError):
            mx.while_loop(
                lambda c: c < 5, lambda c: c + 1, mx.array(0), 10
            )  # positional

    # ---- T17-T19: max_iterations validation ----

    def test_max_iterations_invalid(self):
        for bad in [True, 1.5, -1, 2**63, "10"]:
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "max_iterations"):
                    mx.while_loop(
                        lambda c: c < 5,
                        lambda c: c + 1,
                        mx.array(0),
                        max_iterations=bad,
                    )

    def test_max_iterations_zero(self):
        # max=0, cond true -> raise; cond false -> return init.
        with self.assertRaisesRegex(RuntimeError, "max_iterations"):
            mx.while_loop(
                lambda c: mx.array(True), lambda c: c + 1, mx.array(0), max_iterations=0
            )
        out = mx.while_loop(
            lambda c: mx.array(False), lambda c: c + 1, mx.array(7), max_iterations=0
        )
        self.assertEqual(int(out.item()), 7)

    def test_max_iterations_not_multiple_of_chunk(self):
        # max=5, chunk=2 -> raise exactly at 5 (cond always true).
        with self.assertRaisesRegex(RuntimeError, "max_iterations"):
            mx.while_loop(
                lambda c: mx.array(True),
                lambda c: c + 1,
                mx.array(0),
                max_iterations=5,
                chunk_size=2,
            )
        # trip=5, max=5, chunk=2 -> return 5.
        out = mx.while_loop(
            lambda c: c < 5,
            lambda c: c + 1,
            mx.array(0),
            max_iterations=5,
            chunk_size=2,
        )
        self.assertEqual(int(out.item()), 5)

    def test_max_iterations_none_terminating(self):
        out = mx.while_loop(
            lambda c: c < 10,
            lambda c: c + 1,
            mx.array(0),
            max_iterations=None,
            chunk_size=4,
        )
        self.assertEqual(int(out.item()), 10)

    def test_max_iterations_huge(self):
        out = mx.while_loop(
            lambda c: c < 5,
            lambda c: c + 1,
            mx.array(0),
            max_iterations=2**62,
            chunk_size=8,
        )
        self.assertEqual(int(out.item()), 5)

    # ---- T21: init_val validation ----

    def test_init_val_no_array(self):
        with self.assertRaisesRegex(ValueError, "mx.array"):
            mx.while_loop(lambda c: c < 5, lambda c: c + 1, 0, max_iterations=10)

    def test_init_val_empty(self):
        for init in [(), {}]:
            with self.subTest(init=init):
                with self.assertRaisesRegex(ValueError, "mx.array"):
                    mx.while_loop(
                        lambda c: mx.array(False), lambda c: c, init, max_iterations=10
                    )

    def test_init_val_nonarray_leaf(self):
        with self.assertRaisesRegex(ValueError, "mx.array"):
            mx.while_loop(
                lambda c: c["a"] < 5,
                lambda c: {"a": c["a"] + 1, "s": "x"},
                {"a": mx.array(0), "s": "x"},
                max_iterations=10,
            )

    def test_body_returns_nonarray_leaf(self):
        with self.assertRaisesRegex(ValueError, "mx.array"):
            mx.while_loop(
                lambda c: c < 5,
                lambda c: 1,  # Python int
                mx.array(0),
                max_iterations=10,
            )

    # ---- T22: cond coercion ----

    def test_cond_python_bool(self):
        # cond always True -> cap hit at the while_loop call (raises there).
        with self.assertRaisesRegex(RuntimeError, "max_iterations"):
            mx.while_loop(
                lambda c: True,
                lambda c: c + 1,
                mx.array(0),
                max_iterations=5,
                chunk_size=2,
            )
        # cond False -> returns init unchanged.
        out = mx.while_loop(
            lambda c: False,
            lambda c: c + 1,
            mx.array(7),
            max_iterations=5,
            chunk_size=2,
        )
        self.assertEqual(int(out.item()), 7)

    def test_cond_numpy_bool(self):
        out = mx.while_loop(
            lambda c: np.bool_(False),
            lambda c: c + 1,
            mx.array(7),
            max_iterations=5,
            chunk_size=2,
        )
        self.assertEqual(int(out.item()), 7)

    def test_cond_multielement_array(self):
        # cond returns a multi-element array -> mx.all reduces.
        out = mx.while_loop(
            lambda c: mx.array([c < 5, c < 100]),  # both must be true
            lambda c: c + 1,
            mx.array(0),
            max_iterations=20,
            chunk_size=4,
        )
        self.assertEqual(int(out.item()), 5)

    # ---- T23: cond oscillating ----

    def test_cond_oscillating_stops_at_first_false(self):
        # cond flips T->F->T; should stop at first F (carry=2).
        def cond(c):
            return c != 2

        out = mx.while_loop(
            cond, lambda c: c + 1, mx.array(0), max_iterations=20, chunk_size=4
        )
        self.assertEqual(int(out.item()), 2)

    # ---- T24: compiled body inside while_loop ----

    def test_compiled_body_inside_while_loop(self):
        @mx.compile
        def body(c):
            return c * 2 + 1

        out = mx.while_loop(
            lambda c: c < 50, body, mx.array(0), max_iterations=20, chunk_size=4
        )
        ref = _reference_loop(lambda c: c < 50, body, mx.array(0))
        self.assertEqualArray(out, ref)

    # ---- T25: determinism ----

    def test_determinism(self):
        results = []
        for _ in range(5):
            out = mx.while_loop(
                lambda c: c < 20, lambda c: c + 1, mx.array(0), chunk_size=4
            )
            results.append(int(out.item()))
        self.assertEqual(len(set(results)), 1)

    # ---- T26: random property (linear recurrence) ----

    def test_random_linear_recurrence(self):
        rng = np.random.default_rng(2024)
        for _ in range(15):
            # Positive increment guarantees termination at c == N.
            step = float(rng.uniform(0.5, 2.0))
            N = int(rng.integers(10, 100))
            chunk_size = int(rng.choice([1, 2, 4, 8, 16, 32]))
            cond = lambda c: c < N
            body = lambda c: c + step
            init = mx.array(0.0)
            out = mx.while_loop(
                cond, body, init, max_iterations=N * 2, chunk_size=chunk_size
            )
            ref = _reference_loop(cond, body, init)
            self.assertTrue(mx.allclose(out, ref).item())

    # ---- T27: random pytree carry ----

    def test_random_pytree_carry(self):
        rng = np.random.default_rng(2025)
        for _ in range(10):
            chunk_size = int(rng.choice([1, 4, 16]))
            init = {"a": mx.array(rng.integers(0, 5)), "b": mx.array(rng.uniform(0, 1))}
            cond = lambda c: c["a"] < 10
            body = lambda c: {"a": c["a"] + 1, "b": c["b"] * 2}
            out = mx.while_loop(
                cond, body, init, max_iterations=50, chunk_size=chunk_size
            )
            ref = _reference_loop(cond, body, init)
            self.assertEqualArray(out["a"], ref["a"])
            self.assertEqualArray(out["b"], ref["b"])

    # ---- T29: multi-threaded ----

    def test_multi_threaded(self):
        results = [[] for _ in range(4)]

        def runner(i):
            mx.set_default_device(mx.default_device())
            for _ in range(5):
                out = mx.while_loop(
                    lambda c: c < 10, lambda c: c + 1, mx.array(0.0), chunk_size=4
                )
                results[i].append(out.item())

        threads = [threading.Thread(target=runner, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for r in results:
            self.assertEqual(r, [10.0] * 5)

    # ---- T30: constant cond (the iter2 crash case) ----

    def test_constant_cond_no_crash(self):
        for cond in [
            lambda c: mx.array(True),
            lambda c: True,
            lambda c: np.bool_(True),
        ]:
            with self.subTest(cond=cond):
                with self.assertRaisesRegex(RuntimeError, "max_iterations"):
                    mx.while_loop(
                        cond,
                        lambda c: c + 1,
                        mx.array(0),
                        max_iterations=3,
                        chunk_size=4,
                    )
        for cond in [
            lambda c: mx.array(False),
            lambda c: False,
            lambda c: np.bool_(False),
        ]:
            with self.subTest(cond=cond):
                out = mx.while_loop(
                    cond, lambda c: c + 1, mx.array(7), max_iterations=10, chunk_size=4
                )
                self.assertEqual(int(out.item()), 7)

    # ---- T36: numpy integer args ----

    def test_numpy_integer_args(self):
        out = mx.while_loop(
            lambda c: c < 5,
            lambda c: c + 1,
            mx.array(0),
            max_iterations=np.int64(100),
            chunk_size=np.int64(4),
        )
        self.assertEqual(int(out.item()), 5)

    # ---- T33: recursive while_loop in body is NOT supported ----

    def test_recursive_while_loop_raises(self):
        # A while_loop called inside another while_loop's body receives a
        # tracer carry (the outer body is @mx.compile'd), so it must raise.
        # This is a documented limitation: while_loop cannot be nested inside
        # a compiled body because the inner loop requires a host sync.
        def body(c):
            return mx.while_loop(
                lambda d: d < 2, lambda d: d + 1, c, max_iterations=5, chunk_size=2
            )

        with self.assertRaises(RuntimeError):
            mx.while_loop(
                lambda c: c < 4, body, mx.array(0), max_iterations=20, chunk_size=2
            )

    # ---- chunk_size default ----

    def test_default_chunk_size(self):
        self.assertEqual(_DEFAULT_CHUNK_SIZE, 16)
        out = mx.while_loop(lambda c: c < 20, lambda c: c + 1, mx.array(0))
        self.assertEqual(int(out.item()), 20)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
