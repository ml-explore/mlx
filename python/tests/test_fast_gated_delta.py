import os
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np

try:
    import torch

    has_torch = True
except ImportError as e:
    has_torch = False


def gated_delta_oracle(
    q,
    k,
    v,
    beta,
    g,
    scale=None,
    initial_state=None,
    output_final_state=False,
):
    """
    Reference PyTorch implementation of recurrent gated delta rule.
    Taken from: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]                    <--- Difference: with out kernel: this is expected as a log.
        scale: float, optional          <--- Difference: This is done by the qwen3_5 model.
        initial_state: [B, H, K, V], optional <- Difference: last two dimensions are transposed.
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


def runner(dims, stream=mx.gpu, reference=True):
    B, Hk, Hv, T, Dk, Dv = dims

    assert Hv % Hk == 0
    repeat_factor = Hv // Hk

    q = mx.random.normal(shape=(B, T, Hk, Dk))
    k = mx.random.normal(shape=(B, T, Hk, Dk))
    k = k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = mx.random.normal(shape=(B, T, Hv, Dv))
    g = mx.random.uniform(shape=(B, T, Hv))
    b = mx.sigmoid(mx.random.normal(shape=(B, T, Hv)))
    h0 = mx.random.normal((B, Hv, Dv, Dk), dtype=mx.float32)

    if reference:
        # Prepare reference inputs
        qpt = torch.from_numpy(np.array(q))
        kpt = torch.from_numpy(np.array(k))
        vpt = torch.from_numpy(np.array(v))
        bpt = torch.from_numpy(np.array(b))
        gpt = torch.from_numpy(np.array(g))
        h0pt = torch.from_numpy(np.array(h0)).transpose(-1, -2).contiguous()

        if repeat_factor > 1:
            qpt = qpt.repeat_interleave(repeat_factor, dim=2)
            kpt = kpt.repeat_interleave(repeat_factor, dim=2)

        out_on_py, hf_on_py = gated_delta_oracle(
            qpt,
            kpt,
            vpt,
            bpt,
            torch.log(gpt),
            scale=1.0,
            initial_state=h0pt,
            output_final_state=True,
        )

        out_on = mx.array(out_on_py.detach().cpu().numpy())  # [B, T, Hv, Dv]
        hf_on = mx.swapaxes(
            mx.array(hf_on_py.detach().cpu().numpy()), -1, -2
        )  # -> [B, Hv, Dv, Dk]
        out_ref = mx.array(out_on)
        hf_ref = mx.array(hf_on)
    else:
        # use fallback for tests once fallback is validated
        out_ref, hf_ref = mx.fast.gated_delta_update(
            q, k, v, g, b, initial_state=h0, stream=mx.cpu
        )

    mx.eval(out_ref, hf_ref)

    out, hf = mx.fast.gated_delta_update(q, k, v, g, b, initial_state=h0, stream=stream)

    mx.eval(out, hf)
    return (out, hf), (out_ref, hf_ref)


class TestGatedDelta(mlx_tests.MLXTestCase):
    base_dims = (1, 32, 32, 1, 128, 128)
    unaligned_dims = (1, 32, 32, 33, 128, 128)
    big_batch_dims = (128, 32, 32, 16, 128, 128)
    large_t_dims = (2, 32, 32, 1111, 128, 128)
    diff_heads = (1, 16, 32, 33, 128, 128)
    diff_heads2 = (1, 16, 48, 33, 128, 128)

    fallback_dims = [base_dims, unaligned_dims, big_batch_dims, diff_heads, diff_heads2]
    gpu_dims = fallback_dims + [large_t_dims]

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_gated_delta_fallback(self):
        for dims in self.fallback_dims:
            (out, hf), (out_ref, hf_ref) = runner(dims, mx.cpu)
            msg = f"Failed on Dimensions: {dims}"
            self.assertTrue(
                mx.allclose(out_ref, out, atol=1e-4, rtol=1e-4), msg="Out " + msg
            )
            self.assertTrue(
                mx.allclose(hf_ref, hf, atol=1e-4, rtol=1e-4), msg="State " + msg
            )

    def test_gated_delta_fallback_masked(self):
        for dims in self.fallback_dims:

            B, Hk, Hv, T, Dk, Dv = dims

            q = mx.random.normal(shape=(B, T, Hk, Dk))
            k = mx.random.normal(shape=(B, T, Hk, Dk))
            k = k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
            v = mx.random.normal(shape=(B, T, Hv, Dv))
            g = mx.random.uniform(shape=(B, T, Hv))
            b = mx.sigmoid(mx.random.normal(shape=(B, T, Hv)))
            h0 = mx.random.normal((B, Hv, Dv, Dk), dtype=mx.float32)

            # make a mask
            lengths = mx.random.randint(1, T + 1, shape=(B,))
            mask = mx.arange(T)[None, :] < lengths[:, None]
            # mask one input in python
            mask_float = mask.astype(q.dtype)
            km = k * mask_float[..., None, None]
            vm = v * mask_float[..., None, None]
            qm = q * mask_float[..., None, None]
            bm = b * mask_float[..., None]
            gm = mx.where(mask[..., None], g, 1.0)

            out_ref, hf_ref = mx.fast.gated_delta_update(
                qm, km, vm, gm, bm, initial_state=h0, stream=mx.cpu
            )

            mx.eval(out_ref, hf_ref)
            out, hf = mx.fast.gated_delta_update(
                q, k, v, g, b, initial_state=h0, mask=mask, stream=mx.cpu
            )
            mx.eval(out, hf)

            msg = f"Failed on Dimensions: {dims}"
            self.assertTrue(
                mx.allclose(out_ref, out, atol=1e-4, rtol=1e-4), msg="Out " + msg
            )
            self.assertTrue(
                mx.allclose(hf_ref, hf, atol=1e-4, rtol=1e-4), msg="State " + msg
            )

    def test_gated_delta_dtypes(self):
        dtypes = [mx.bfloat16, mx.float32]
        streams = [mx.cpu, mx.gpu] if mx.is_available(mx.gpu) else [mx.cpu]
        for stream in streams:
            for dtype in dtypes:
                for dims in [self.base_dims]:

                    B, Hk, Hv, T, Dk, Dv = dims

                    q = mx.random.normal(shape=(B, T, Hk, Dk), dtype=dtype)
                    k = mx.random.normal(shape=(B, T, Hk, Dk), dtype=dtype)
                    k = k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
                    v = mx.random.normal(shape=(B, T, Hv, Dv), dtype=dtype)
                    g = mx.random.uniform(shape=(B, T, Hv), dtype=dtype)
                    b = mx.sigmoid(mx.random.normal(shape=(B, T, Hv), dtype=dtype))
                    h0 = mx.random.normal((B, Hv, Dv, Dk), dtype=mx.float32)

                    out, hf = mx.fast.gated_delta_update(
                        q, k, v, g, b, initial_state=h0, stream=stream
                    )
                    mx.eval(out, hf)
                    msg = f"Output dtype mismatch on Dimensions: {dims}"
                    self.assertTrue(dtype == out.dtype, msg="Out " + msg)
                    self.assertTrue(hf.dtype == mx.float32, msg="State " + msg)

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_gated_delta_sequential(self):
        os.environ["GATED_DELTA_CHUNK"] = "0"
        for dims in self.gpu_dims:
            (out, hf), (out_ref, hf_ref) = runner(dims, reference=False)
            msg = f"Failed on Dimensions: {dims}"
            self.assertTrue(
                mx.allclose(out_ref, out, atol=1e-4, rtol=1e-4), msg="Out " + msg
            )
            self.assertTrue(
                mx.allclose(hf_ref, hf, atol=1e-4, rtol=1e-4), msg="State " + msg
            )

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_gated_delta_simdgroup(self):
        os.environ["GATED_DELTA_CHUNK"] = "8"
        for dims in self.gpu_dims:
            (out, hf), (out_ref, hf_ref) = runner(dims, reference=False)
            msg = f"Failed on Dimensions: {dims}"
            self.assertTrue(
                mx.allclose(out_ref, out, atol=1e-4, rtol=1e-4), msg="Out " + msg
            )
            self.assertTrue(
                mx.allclose(hf_ref, hf, atol=1e-4, rtol=1e-4), msg="State " + msg
            )

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_gated_delta_nax(self):
        os.environ["GATED_DELTA_CHUNK"] = "16"
        for dims in self.gpu_dims:
            (out, hf), (out_ref, hf_ref) = runner(dims, reference=False)
            msg = f"Failed on Dimensions: {dims}"
            self.assertTrue(
                mx.allclose(out_ref, out, atol=1e-1, rtol=1e-4), msg="Out " + msg
            )
            self.assertTrue(
                mx.allclose(hf_ref, hf, atol=1e-1, rtol=1e-4), msg="State " + msg
            )


if __name__ == "__main__":
    mlx_tests.MLXTestRunner(failfast=True)
