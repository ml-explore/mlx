# Copyright © 2023-2024 Apple Inc.

"""
Test that nn.GRU and nn.LSTM produce the same outputs for legacy vs fast (Metal).
Matches the flow from mlx-graphs verify_fast_vs_legacy.py:
  1. Subprocess LEGACY: create model (seed 42), input (seed 43), h0/c0 zeros,
     forward, save ref (weights + input + h0, c0 + legacy outputs).
  2. Subprocess FAST: load ref, to_contiguous on all loaded arrays, update model,
     mx.eval(params), forward, compare to legacy outputs.
Same preprocessing (to_contiguous, explicit h0/c0) as the original working code.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import mlx_tests
import numpy as np

# Match original verify_fast_vs_legacy.py constants
BATCH, SEQ_LEN, INPUT_SIZE, HIDDEN_SIZE = 32, 40, 128, 200
SEED_WEIGHTS = 42
SEED_INPUT = 43


def _save_ref(ref_path: str) -> None:
    """Subprocess: MLX_RNN_IMPL=legacy. Save weights + input + h0, c0 + legacy outputs."""
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(SEED_WEIGHTS)
    gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, bias=True)
    lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, bias=True)

    mx.random.seed(SEED_INPUT)
    x = mx.random.normal((BATCH, SEQ_LEN, INPUT_SIZE)).astype(mx.float32)
    h0 = mx.zeros((BATCH, HIDDEN_SIZE), dtype=mx.float32)
    c0 = mx.zeros((BATCH, HIDDEN_SIZE), dtype=mx.float32)

    gru_out = gru(x, h0)
    lstm_h, lstm_c = lstm(x, h0, c0)
    mx.eval(gru_out, lstm_h, lstm_c)

    d = {
        "x": x,
        "h0": h0,
        "c0": c0,
        "gru_out": gru_out,
        "lstm_h": lstm_h,
        "lstm_c": lstm_c,
    }
    for k, v in gru.parameters().items():
        d[f"gru_{k}"] = v
    for k, v in lstm.parameters().items():
        d[f"lstm_{k}"] = v

    mx.savez(ref_path, **d)


def _to_contiguous(arr) -> "mx.array":
    """Force contiguous layout + float32 (loaded arrays can have bad layout)."""
    import mlx.core as mx

    return mx.array(np.ascontiguousarray(np.array(arr, dtype=np.float32)))


def _run_fast_and_compare(ref_path: str) -> bool:
    """Subprocess: MLX_RNN_IMPL=fast. Load ref, to_contiguous, update model, forward, compare. Returns True if allclose."""
    import mlx.core as mx
    import mlx.nn as nn

    data = mx.load(ref_path)

    gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, bias=True)
    lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, bias=True)

    gru.update(
        {
            "Wx": _to_contiguous(data["gru_Wx"]),
            "Wh": _to_contiguous(data["gru_Wh"]),
            "b": _to_contiguous(data["gru_b"]),
            "bhn": _to_contiguous(data["gru_bhn"]),
        }
    )
    mx.eval(gru.parameters())

    lstm.update(
        {
            "Wx": _to_contiguous(data["lstm_Wx"]),
            "Wh": _to_contiguous(data["lstm_Wh"]),
            "bias": _to_contiguous(data["lstm_bias"]),
        }
    )
    mx.eval(lstm.parameters())

    x = _to_contiguous(data["x"])
    h0 = _to_contiguous(data["h0"])
    c0 = _to_contiguous(data["c0"])

    gru_out = gru(x, h0)
    lstm_h, lstm_c = lstm(x, h0, c0)
    mx.eval(gru_out, lstm_h, lstm_c)

    def ok(a, b, rtol=1e-5, atol=1e-6):
        return bool(np.allclose(np.array(a), np.array(b), rtol=rtol, atol=atol))

    gru_ok = ok(gru_out, data["gru_out"])
    lh_ok = ok(lstm_h, data["lstm_h"])
    lc_ok = ok(lstm_c, data["lstm_c"])
    return gru_ok and lh_ok and lc_ok


class TestRecurrentFastVsLegacy(mlx_tests.MLXTestCase):
    """Compare legacy and fast RNN (same flow as mlx-graphs verify_fast_vs_legacy)."""

    def test_gru_and_lstm_fast_vs_legacy_equal(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        runner = os.path.join(test_dir, "test_recurrent_fast_vs_legacy.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "ref.npz")

            r1 = subprocess.run(
                [sys.executable, runner, "save_ref", ref_path],
                env={**os.environ, "MLX_RNN_IMPL": "legacy"},
                cwd=test_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(r1.returncode, 0, f"legacy save_ref failed: {r1.stderr}")

            r2 = subprocess.run(
                [sys.executable, runner, "run_fast_and_compare", ref_path],
                env={**os.environ, "MLX_RNN_IMPL": "fast"},
                cwd=test_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                r2.returncode,
                0,
                f"fast run_fast_and_compare failed (legacy vs fast differ): {r2.stderr}",
            )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "save_ref":
        _save_ref(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "run_fast_and_compare":
        ok = _run_fast_and_compare(sys.argv[2])
        sys.exit(0 if ok else 1)
    unittest.main()
