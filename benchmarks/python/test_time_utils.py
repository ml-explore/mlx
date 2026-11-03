# Copyright © 2026 Apple Inc.

import unittest
from unittest.mock import patch

import mlx.core as mx
from time_utils import measure_runtime


class TestMeasureRuntime(unittest.TestCase):
    def test_evaluates_before_timer_stops(self):
        events = []
        outputs = []

        def fn(x):
            events.append("call")
            out = x + 1
            outputs.append(out)
            return {"result": [out], "other": None}

        def evaluate(out):
            events.append("eval")
            original_eval(out)

        original_eval = mx.eval
        with (
            patch("time_utils.mx.eval", side_effect=evaluate),
            patch("time_utils.time.perf_counter", side_effect=[10.0, 10.006]),
        ):
            runtime = measure_runtime(fn, warmup=1, iters=2, x=mx.array(2))
        self.assertEqual(events, ["call", "eval"] * 3)
        self.assertAlmostEqual(runtime, 3.0)
        self.assertTrue(all(out.item() == 3 for out in outputs))

    def test_defaults_and_none_return(self):
        calls = []
        measure_runtime(lambda: calls.append(None))
        self.assertEqual(len(calls), 105)

    def test_zero_warmup(self):
        calls = []
        measure_runtime(lambda: calls.append(None), warmup=0, iters=2)
        self.assertEqual(len(calls), 2)

    def test_invalid_counts(self):
        for kwargs, error in [
            ({"warmup": -1}, ValueError),
            ({"iters": 0}, ValueError),
            ({"iters": -1}, ValueError),
            ({"warmup": 1.5}, TypeError),
            ({"iters": 1.5}, TypeError),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                measure_runtime(lambda: self.fail("callback ran"), **kwargs)


if __name__ == "__main__":
    unittest.main()
