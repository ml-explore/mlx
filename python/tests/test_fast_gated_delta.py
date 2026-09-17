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

    Note: this reference has no notion of grouped query attention. Callers with
    Hv != Hk must repeat_interleave q and k up to Hv first, and fold the
    resulting dq/dk back down over the group.

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
        # use fallback for tests once fallback is validated by setting a mask instead of using the cpu
        mask = mx.ones((B, T))

        out_ref, hf_ref = mx.fast.gated_delta_update(
            q,
            k,
            v,
            g,
            b,
            initial_state=h0,
            mask=mask,
        )

    mx.eval(out_ref, hf_ref)

    out, hf = mx.fast.gated_delta_update(q, k, v, g, b, initial_state=h0, stream=stream)

    mx.eval(out, hf)
    return (out, hf), (out_ref, hf_ref)


def compare_grads(grads, refs, dims, dtype, g, tol, label):
    """Report every gradient against the Torch oracle and return the failures.

    Metrics rather than a bare allclose because the gradients span orders of
    magnitude: an absolute tolerance is meaningless for the large entries and a
    relative one is meaningless for the near-zero ones.
    """
    B, Hk, Hv, T, Dk, Dv = dims
    rep = Hv // Hk

    g_np = np.asarray(
        np.array(g.astype(mx.float32), dtype=np.float32), dtype=np.float64
    )
    # print("cmp: ", g_np.min(), g_np.max(), flush=True)


    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    print(f"\n=== {label} {dims} {dtype} ===")
    print(
        "  grad   status  rel_scale    rel_l2     rel_elem   "
        "max|err|    max|ref|",
        flush=True,
    )

    failures = []
    for name, gm, gr in zip("q k v g beta h0".split(), grads, refs):
        a = np.asarray(
            np.array(gm.astype(mx.float32), dtype=np.float32), dtype=np.float64
        )
        r = np.asarray(gr.contiguous().numpy(), dtype=np.float64)

        # Under GQA the oracle differentiated Hv expanded copies of q and k,
        # while the kernel accumulates over the group into Hk heads. Sum the
        # repeats back down, which is exactly what the kernel's atomic add
        # over hv does.
        if rep > 1 and name in ("q", "k"):
            r = r.reshape(B, T, Hk, rep, Dk).sum(axis=3)

        # dg = (1/g) * suffix-sum, so its entries are unbounded as g -> 0 and
        # no tolerance on dg itself is meaningful. Compare g*dg, the log-space
        # gradient, which is as well conditioned as dq/dk. Must precede the
        # metrics.
        if name == "g":
            a = a * g_np
            r = r * g_np

        e = np.abs(a - r)
        max_err = e.max()
        scale = np.abs(r).max()

        # error against the largest reference magnitude
        rel_scale = max_err / max(scale, 1e-30)
        # global energy ratio, insensitive to single outliers
        rel_l2 = np.linalg.norm(a - r) / max(np.linalg.norm(r), 1e-30)
        # worst per-element relative error over non-negligible entries; without
        # the mask this is dominated by near-zero entries where relative error
        # carries no information
        sig = np.abs(r) > 1e-3 * max(scale, 1e-30)
        rel_elem = (e[sig] / np.abs(r[sig])).max() if sig.any() else 0.0

        ok = bool(rel_scale < tol and rel_l2 < tol)

        tag = "  [log-space]" if name == "g" else ""
        status = "PASS" if ok else "FAIL"
        print(
            f"  d{name:<5} {status}   {rel_scale:.3e}  {rel_l2:.3e}  "
            f"{rel_elem:.3e}  {max_err:.3e}  {scale:.3e}{tag}",
            flush=True,
        )

        if not ok:
            en = e / max(scale, 1e-30)
            if a.ndim == 4:
                if name in ("q", "k", "v"):
                    print(
                        f"       per-t    rel: {en.max(axis=(0, 2, 3))}",
                        flush=True,
                    )
                    print(
                        f"       per-head rel: {en.max(axis=(0, 1, 3))}",
                        flush=True,
                    )
                else:  # h0
                    print(
                        f"       per-head rel: {en.max(axis=(0, 2, 3))}",
                        flush=True,
                    )
            elif a.ndim == 3:  # g, beta: [B, T, Hv]
                print(f"       per-t    rel: {en.max(axis=(0, 2))}", flush=True)
                bad = np.unravel_index(e.argmax(), e.shape)
                print(
                    f"       worst at (b,t,hv)={bad}  mlx {a[bad]:+.6f}  "
                    f"torch {r[bad]:+.6f}  g {g_np[bad]:.3e}",
                    flush=True,
                )
            failures.append(f"d{name} rel {rel_scale:.3e}")

    print("========================\n", flush=True)
    return failures


def grad_and_refs(dims, dtype):
    """Run the MLX gradient and the Torch oracle gradient on the same inputs."""
    B, Hk, Hv, T, Dk, Dv = dims
    rep = Hv // Hk

    mx.random.seed(0)
    q = mx.random.normal(shape=(B, T, Hk, Dk)).astype(dtype)
    k = mx.random.normal(shape=(B, T, Hk, Dk)).astype(dtype)
    k = k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = mx.random.normal(shape=(B, T, Hv, Dv)).astype(dtype)
    g = mx.random.uniform(shape=(B, T, Hv)).astype(dtype)
    b = mx.sigmoid(mx.random.normal(shape=(B, T, Hv))).astype(dtype)
    # state stays fp32
    h0 = mx.random.normal((B, Hv, Dv, Dk), dtype=mx.float32)

    mx.eval(q, k, v, g, b, h0)
    # print("draw:", g.min().item(), g.max().item(), g.dtype, flush=True)


    co_out = mx.random.normal(shape=(B, T, Hv, Dv))
    co_state = mx.random.normal(shape=(B, Hv, Dv, Dk))
    co_out_pt = torch.from_numpy(np.array(co_out))
    co_state_pt = torch.from_numpy(np.array(co_state)).transpose(-1, -2)

    def f(q, k, v, g, b, h0):
        out, state = mx.fast.gated_delta_update(q, k, v, g, b, h0, stream=mx.gpu)
        return (out * co_out).sum() + (state * co_state).sum()

    grads = mx.grad(f, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g, b, h0)
    mx.eval(grads)

    def to_pt(x):
        # NumPy has no bfloat16, so cast on the way out of MLX. The oracle runs
        # in fp32 regardless.
        return torch.from_numpy(np.array(x.astype(mx.float32), dtype=np.float32))

    qpt = to_pt(q).clone()
    kpt = to_pt(k).clone()
    # The oracle has no GQA, so expand the key-side heads. Interleaved, to match
    # the kernel's hk_idx = hv_idx / (Hv / Hk): heads 0,1 share key head 0.
    if rep > 1:
        qpt = qpt.repeat_interleave(rep, dim=2).contiguous()
        kpt = kpt.repeat_interleave(rep, dim=2).contiguous()
    qpt.requires_grad_()
    kpt.requires_grad_()

    vpt = to_pt(v).clone().requires_grad_()
    bpt = to_pt(b).clone().requires_grad_()
    gpt = to_pt(g).clone().requires_grad_()
    h0pt = to_pt(h0).transpose(-1, -2).contiguous()
    h0pt.requires_grad_()

    out_on_pt, hf_on_pt = gated_delta_oracle(
        qpt,
        kpt,
        vpt,
        bpt,
        torch.log(gpt),
        scale=1.0,
        initial_state=h0pt,
        output_final_state=True,
    )

    ((out_on_pt * co_out_pt).sum() + (hf_on_pt * co_state_pt).sum()).backward()

    refs = [
        qpt.grad,
        kpt.grad,
        vpt.grad,
        gpt.grad,
        bpt.grad,
        h0pt.grad.transpose(-1, -2),
    ]
    return grads, refs, g


class TestGatedDelta(mlx_tests.MLXTestCase):
    base_dims = (1, 32, 32, 1, 128, 128)
    unaligned_dims = (1, 32, 32, 33, 128, 128)
    big_batch_dims = (128, 32, 32, 16, 128, 128)
    large_t_dims = (2, 32, 32, 2048, 128, 128)
    diff_heads = (1, 16, 32, 33, 128, 128)
    diff_heads2 = (1, 16, 48, 33, 128, 128)
    unsupported_heads1 = (1, 8, 32, 33, 128, 128)
    unsupported_heads2 = (1, 24, 48, 33, 128, 128)
    gqa_large_t = (1, 16, 32, 2048, 128, 128)

    fallback_dims = [unsupported_heads1, unsupported_heads2]
    gpu_dims = [
        base_dims,
        unaligned_dims,
        big_batch_dims,
        diff_heads,
        diff_heads2,
        large_t_dims,
    ]

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
                qm,
                km,
                vm,
                gm,
                bm,
                initial_state=h0,
            )

            mx.eval(out_ref, hf_ref)
            out, hf = mx.fast.gated_delta_update(
                q,
                k,
                v,
                g,
                b,
                initial_state=h0,
                mask=mask,
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

                out, hf = mx.fast.gated_delta_update(q, k, v, g, b, initial_state=h0)

                msg = f"Output dtype mismatch on Dimensions: {dims}"
                self.assertTrue(dtype == out.dtype, msg="Out " + msg)
                self.assertTrue(hf.dtype == mx.float32, msg="State " + msg)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
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

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
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

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
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

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_gated_delta_grad(self):
        os.environ["GATED_DELTA_VJP_CHUNK"] = "0"
        for dims in self.grad_dims:
            for dtype in (mx.float32, mx.bfloat16):
                grads, refs, g = grad_and_refs(dims, dtype)

                # bf16 inputs are rounded before the kernel sees them while the
                # Torch oracle runs in fp32, so the comparison measures kernel
                # error and input rounding together. Loosen accordingly.
                tol = 2e-3 if dtype == mx.float32 else 3e-2

                failures = compare_grads(
                    grads, refs, dims, dtype, g, tol, "seq vs torch oracle"
                )
                self.assertTrue(
                    not failures,
                    msg=f"gradient mismatch on {dims} with {dtype}: "
                    + ", ".join(failures),
                )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_gated_delta_grad_nax(self):
        # Set explicitly rather than relying on the default, so the test cannot
        # silently measure the sequential kernel if the threshold logic changes.
        os.environ["GATED_DELTA_VJP_CHUNK"] = "16"
        for dims in self.grad_dims:
            for dtype in (mx.float32, mx.bfloat16):
            # for dtype in [mx.float32]:
                grads, refs, g = grad_and_refs(dims, dtype)

                # As above, plus the chunked path is intrinsically looser: it
                # forms (I + A)^-1 by a truncated Neumann series and reloads
                # forward tiles from a cache, so its floor sits near 1e-3
                # relative even at fp32.
                tol = 5e-3 if dtype == mx.float32 else 3e-2

                failures = compare_grads(
                    grads, refs, dims, dtype, g, tol, "nax vs torch oracle"
                )
                self.assertTrue(
                    not failures,
                    msg=f"nax gradient mismatch on {dims} with {dtype}: "
                    + ", ".join(failures),
                )


if __name__ == "__main__":
    mlx_tests.MLXTestRunner(failfast=True)
