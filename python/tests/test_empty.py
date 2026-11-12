# Copyright © 2026 Apple Inc.


import mlx.core as mx
import mlx.nn as nn
import mlx_tests


def _quantized(shape):
    return mx.quantize(mx.zeros(shape), 64, 4)


def _matmul_into(fn):
    x = mx.zeros((0, 32)) @ mx.zeros((32, 16))
    mx.eval(x)
    return fn(x)


# Ops that give an empty output (name -> function that returns the outputs)
CASES = {
    # Compiled
    "compile_fused": lambda: mx.compile(lambda a, b: mx.exp(a) * b)(
        mx.zeros((17, 0)), mx.zeros((0,))
    ),
    "compile_fused_contig": lambda: mx.compile(lambda a: mx.exp(a) + 1)(
        mx.zeros((0, 4))
    ),
    "compile_multi_out": lambda: mx.compile(lambda a: (mx.exp(a) + 1, mx.sin(a) * 2))(
        mx.zeros((3, 0))
    ),
    "compile_shapeless": lambda: mx.compile(lambda a: mx.exp(a) * 2, shapeless=True)(
        mx.zeros((0, 3))
    ),
    "matmul_into_compile": lambda: _matmul_into(mx.compile(lambda x: x * 2 + 1)),
    "matmul_into_gelu": lambda: _matmul_into(nn.gelu),
    # Elementwise and copies
    "unary": lambda: mx.exp(mx.zeros((0, 3))),
    "binary": lambda: mx.zeros((0, 3)) * mx.zeros((3,)),
    "divmod": lambda: mx.divmod(mx.zeros((0, 3)), mx.ones((3,))),
    "where": lambda: mx.where(mx.zeros((0, 3), dtype=mx.bool_), mx.zeros((0, 3)), 1.0),
    "astype": lambda: mx.zeros((0, 3)).astype(mx.int32),
    "copy_strided": lambda: mx.contiguous(mx.zeros((3, 0)).T),
    "arange": lambda: mx.arange(0),
    "arange_float": lambda: mx.arange(1.0, 1.0, 0.5),
    "full": lambda: mx.full((0, 3), 2.0),
    "concatenate": lambda: mx.concatenate([mx.zeros((0, 3)), mx.zeros((0, 3))]),
    "pad": lambda: mx.pad(mx.zeros((0, 3)), [(0, 0), (1, 1)]),
    "slice": lambda: mx.zeros((5, 3))[2:2],
    "view": lambda: mx.zeros((0, 4)).view(mx.int32),
    "as_strided": lambda: mx.as_strided(mx.zeros((4,)), (0, 3), (1, 1)),
    # Reductions, scans and sorts
    "sum": lambda: mx.sum(mx.zeros((0, 3)), axis=1),
    "cumsum": lambda: mx.cumsum(mx.zeros((0, 3)), axis=1),
    "sort": lambda: mx.sort(mx.zeros((0, 3)), axis=1),
    "argpartition": lambda: mx.argpartition(mx.zeros((0, 3)), 1, axis=1),
    # Normalization, RoPE and attention
    "softmax": lambda: mx.softmax(mx.zeros((0, 8)), axis=-1),
    "layer_norm": lambda: mx.fast.layer_norm(
        mx.zeros((0, 8)), mx.ones((8,)), mx.zeros((8,)), 1e-5
    ),
    "layer_norm_no_affine": lambda: mx.fast.layer_norm(
        mx.zeros((0, 8)), None, None, 1e-5
    ),
    "rms_norm": lambda: mx.fast.rms_norm(mx.zeros((0, 8)), mx.ones((8,)), 1e-5),
    "rope": lambda: mx.fast.rope(
        mx.zeros((1, 0, 4, 8)), 8, traditional=False, base=10000.0, scale=1.0, offset=0
    ),
    "rope_traditional": lambda: mx.fast.rope(
        mx.zeros((0, 4, 8)), 8, traditional=True, base=10000.0, scale=1.0, offset=0
    ),
    "sdpa_empty_q": lambda: mx.fast.scaled_dot_product_attention(
        mx.zeros((1, 2, 0, 64)),
        mx.zeros((1, 2, 5, 64)),
        mx.zeros((1, 2, 5, 64)),
        scale=1.0,
    ),
    "sdpa_vector_batch0": lambda: mx.fast.scaled_dot_product_attention(
        mx.zeros((0, 2, 1, 64)),
        mx.zeros((0, 2, 5, 64)),
        mx.zeros((0, 2, 5, 64)),
        scale=1.0,
    ),
    "sdpa_batch0": lambda: mx.fast.scaled_dot_product_attention(
        mx.zeros((0, 2, 8, 64)),
        mx.zeros((0, 2, 8, 64)),
        mx.zeros((0, 2, 8, 64)),
        scale=1.0,
    ),
    # Matmul
    "matmul_m0": lambda: mx.zeros((0, 4)) @ mx.zeros((4, 5)),
    "matmul_n0": lambda: mx.zeros((3, 4)) @ mx.zeros((4, 0)),
    "matmul_batch0": lambda: mx.zeros((0, 3, 4)) @ mx.zeros((0, 4, 5)),
    "matvec_m0": lambda: mx.zeros((0, 4)) @ mx.zeros((4,)),
    "vecmat_n0": lambda: mx.zeros((4,)) @ mx.zeros((4, 0)),
    "addmm_m0": lambda: mx.addmm(mx.ones((0, 5)), mx.zeros((0, 4)), mx.zeros((4, 5))),
    "linear_m0": lambda: nn.Linear(4, 5)(mx.zeros((0, 4))),
    "block_masked_mm": lambda: mx.block_masked_mm(
        mx.zeros((0, 64)), mx.zeros((64, 64)), 32
    ),
    "gather_mm_empty_indices": lambda: mx.gather_mm(
        mx.zeros((4, 1, 8)),
        mx.zeros((4, 8, 8)),
        mx.zeros((0,), dtype=mx.uint32),
        mx.zeros((0,), dtype=mx.uint32),
    ),
    "gather_mm_sorted": lambda: mx.gather_mm(
        mx.zeros((0, 1, 8)),
        mx.zeros((4, 8, 8)),
        None,
        mx.zeros((0,), dtype=mx.uint32),
        sorted_indices=True,
    ),
    "segmented_mm_no_segments": lambda: mx.segmented_mm(
        mx.zeros((4, 8)), mx.zeros((8, 5)), mx.zeros((0, 2), dtype=mx.uint32)
    ),
    "einsum": lambda: mx.einsum("ij,jk->ik", mx.zeros((0, 4)), mx.zeros((4, 5))),
    # Quantized
    "quantize": lambda: mx.quantize(mx.zeros((0, 64)), 64, 4),
    "qmm_m0": lambda: mx.quantized_matmul(
        mx.zeros((0, 64)), *_quantized((32, 64)), transpose=True, group_size=64, bits=4
    ),
    "qmv_batch0": lambda: mx.quantized_matmul(
        mx.zeros((0, 1, 64)),
        *_quantized((32, 64)),
        transpose=True,
        group_size=64,
        bits=4,
    ),
    "qmm_n0": lambda: mx.quantized_matmul(
        mx.zeros((4, 64)),
        *[q[:0] for q in _quantized((32, 64))],
        transpose=True,
        group_size=64,
        bits=4,
    ),
    # Convolution
    "conv1d_batch0": lambda: mx.conv1d(mx.zeros((0, 8, 4)), mx.zeros((5, 3, 4))),
    "conv1d_cout0": lambda: mx.conv1d(mx.zeros((2, 8, 4)), mx.zeros((0, 3, 4))),
    "conv2d_batch0": lambda: mx.conv2d(mx.zeros((0, 8, 8, 4)), mx.zeros((5, 3, 3, 4))),
    "conv2d_cout0": lambda: mx.conv2d(mx.zeros((2, 8, 8, 4)), mx.zeros((0, 3, 3, 4))),
    "conv2d_depthwise_batch0": lambda: mx.conv2d(
        mx.zeros((0, 8, 8, 4)), mx.zeros((4, 3, 3, 1)), groups=4
    ),
    "conv3d_batch0": lambda: mx.conv3d(
        mx.zeros((0, 4, 4, 4, 2)), mx.zeros((3, 2, 2, 2, 2))
    ),
    "conv_transpose2d_batch0": lambda: mx.conv_transpose2d(
        mx.zeros((0, 4, 4, 2)), mx.zeros((3, 2, 2, 2))
    ),
    # FFT
    "fft": lambda: mx.fft.fft(mx.zeros((0, 8), dtype=mx.complex64)),
    "rfft": lambda: mx.fft.rfft(mx.zeros((0, 8))),
    "irfft": lambda: mx.fft.irfft(mx.zeros((0, 5), dtype=mx.complex64)),
    "fft_large": lambda: mx.fft.fft(mx.zeros((0, 4096), dtype=mx.complex64)),
    "fft_bluestein": lambda: mx.fft.fft(mx.zeros((0, 1031), dtype=mx.complex64)),
    "fft2": lambda: mx.fft.fft2(mx.zeros((0, 8, 8), dtype=mx.complex64)),
}

# The CPU raises a ValueError for these, so do not compare with the CPU
GPU_ONLY_CASES = {
    "dequantize": lambda: mx.dequantize(
        *[q[:0] for q in _quantized((4, 64))], group_size=64, bits=4
    ),
    "quantize_mxfp4": lambda: mx.quantize(mx.zeros((0, 32)), mode="mxfp4"),
}


class TestEmpty(mlx_tests.MLXTestCase):
    def run_case(self, fn, device):
        mx.set_default_device(device)
        outs = fn()
        outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        mx.eval(outs)
        return outs

    def check_outputs(self, outs):
        # Use each output on the CPU, so an output with no data fails
        for o in outs:
            self.assertTrue(mx.array_equal(o, o, stream=mx.cpu).item())

    def test_empty_outputs(self):
        device = mx.default_device()
        for name, fn in CASES.items():
            with self.subTest(name):
                outs = self.run_case(fn, device)
                self.check_outputs(outs)
                expected = self.run_case(fn, mx.cpu)
                self.assertEqual(len(outs), len(expected))
                for o, e in zip(outs, expected):
                    self.assertEqual(o.shape, e.shape)
                    self.assertEqual(o.dtype, e.dtype)
                    self.assertTrue(mx.array_equal(o, e, stream=mx.cpu).item())

    def test_empty_outputs_gpu_only(self):
        device = mx.default_device()
        if device == mx.cpu:
            self.skipTest("GPU only")
        for name, fn in GPU_ONLY_CASES.items():
            with self.subTest(name):
                self.check_outputs(self.run_case(fn, device))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
