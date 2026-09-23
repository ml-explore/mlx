import math
import os
import unittest
from itertools import product

import mlx.core as mx
import mlx_tests
import numpy as np

def compare_sdpa_grads(grads, refs, dims, dtype, mask_str, tol, label):
    """Report dq, dk and dv against the fp32 reference and return the failures.

    Metrics rather than a bare allclose because the three gradients have very
    different scales: dv is a convex combination of the cotangent while dq and
    dk carry the softmax Jacobian and are smaller by roughly the score spread,
    so one absolute tolerance cannot cover all three.
    """
    B, qH, kH, qL, kL, D, Dv = dims

    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    print(f"\n=== {label} {dims} {dtype} mask={mask_str} ===")
    print(
        "  grad   status  rel_scale    rel_l2     rel_elem   "
        "max|err|    max|ref|",
        flush=True,
    )

    failures = []
    for name, gm, gr in zip(("q", "k", "v"), grads, refs):
        a = np.asarray(
            np.array(gm.astype(mx.float32), dtype=np.float32), dtype=np.float64
        )
        r = np.asarray(
            np.array(gr.astype(mx.float32), dtype=np.float32), dtype=np.float64
        )

        if a.shape != r.shape:
            print(f"  d{name:<5} SHAPE  mlx {a.shape} vs ref {r.shape}", flush=True)
            failures.append(f"d{name} shape {a.shape} != {r.shape}")
            continue

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
        status = "PASS" if ok else "FAIL"
        print(
            f"  d{name:<5} {status}   {rel_scale:.3e}  {rel_l2:.3e}  "
            f"{rel_elem:.3e}  {max_err:.3e}  {scale:.3e}",
            flush=True,
        )

        if not ok:
            # Shape is [B, H, L, D]. Which axis the error concentrates on says
            # what is broken: a single sequence block points at the block loop
            # bound or a tail, a single head at the batch/head decode, a single
            # head-dim slice at the frag layout, and a uniform smear at the
            # numerics.
            en = e / max(scale, 1e-30)
            print(f"       per-seqblk  rel: {en.max(axis=(0, 1, 3))[::16]}", flush=True)
            print(f"       per-head    rel: {en.max(axis=(0, 2, 3))}", flush=True)
            print(f"       per-dimfrag rel: {en.max(axis=(0, 1, 2))[::16]}", flush=True)
            bad = np.unravel_index(e.argmax(), e.shape)
            print(
                f"       worst at (b,h,l,d)={bad}  mlx {a[bad]:+.6f}  "
                f"ref {r[bad]:+.6f}",
                flush=True,
            )
            failures.append(f"d{name} rel {rel_scale:.3e}")

    print("========================\n", flush=True)
    return failures


def sdpa_grad_and_refs(dims, dtype, mask_str):
    """Run the fused VJP and the fp32 reference VJP on the same inputs.

    The reference is mlx_ref_attn differentiated by mx.vjp with the inputs
    promoted to fp32. Differentiating it at the test dtype instead would make
    the two paths share their rounding error and hide real kernel bugs.
    """
    B, qH, kH, qL, kL, D, Dv = dims
    assert qH % kH == 0
    assert Dv == D, "different head and value dims not enabled"

    mx.random.seed(0)
    scale = 1.0 / math.sqrt(D)

    # Uniform rather than normal, matching prepare_inputs: it keeps the scores
    # in a narrow range so the softmax does not saturate, which would make the
    # gradient identically zero and the comparison vacuous.
    q = mx.random.uniform(0.0, 0.5, (B, qH, qL, D), dtype)
    k = mx.random.uniform(0.0, 0.5, (B, kH, kL, D), dtype)
    v = mx.random.uniform(0.0, scale, (B, kH, kL, Dv), dtype)

    mask = None
    if mask_str == "causal":
        mask = "causal"
    elif mask_str == "additive":
        mask = mx.random.uniform(0.0, 0.5, (B, qH, qL, kL), dtype)
    elif mask_str == "bool":
        mask = mx.random.uniform(0.0, 1.0, (B, qH, qL, kL)) < 0.5

    # A random cotangent, not ones. With an all-ones dO the rowsum term
    # <dO_i, O_i> collapses to sum(O_i) and several sign errors in the dS
    # assembly cancel, so an ones cotangent passes on kernels that are wrong.
    cot = mx.random.normal((B, qH, qL, Dv)).astype(dtype)
    mx.eval(q, k, v, cot)

    fast = lambda q_, k_, v_: mx.fast.scaled_dot_product_attention(
        q_, k_, v_, scale=scale, mask=mask
    )
    _, grads = mx.vjp(fast, [q, k, v], [cot])
    mx.eval(grads)

    # Reference in fp32. The mask must be promoted too, or an additive fp16 mask
    # reintroduces the rounding this is meant to avoid.
    mask32 = mask
    if isinstance(mask, mx.array) and mask.dtype != mx.bool_:
        mask32 = mask.astype(mx.float32)

    slow = lambda q_, k_, v_: mlx_ref_attn(q_, k_, v_, scale=scale, mask=mask32)
    _, refs = mx.vjp(
        slow,
        [q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)],
        [cot.astype(mx.float32)],
    )
    mx.eval(refs)

    return grads, refs


def mlx_ref_attn(q, k, v, scale=1.0, mask=None, sinks=None):
    q_dtype = q.dtype
    q = q * mx.array(scale, q_dtype)
    n_q_heads = q.shape[-3]
    n_kv_heads = k.shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    B = q.shape[0]
    L = q.shape[2]
    kL = k.shape[2]

    if n_repeats > 1:
        q = mx.reshape(q, [B, n_kv_heads, n_repeats, L, -1])
        k = mx.expand_dims(k, 2)
        v = mx.expand_dims(v, 2)

    scores = q @ mx.swapaxes(k, -1, -2)
    is_causal = mask == "causal"
    if mask is not None:

        if is_causal:
            offset = kL - L
            q_indices = mx.arange(L) + offset
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]

        if n_repeats > 1 and mask.ndim >= 3:
            if mask.shape[-3] == 1:
                mask = mx.expand_dims(mask, -3)
            else:
                mask = mx.unflatten(mask, -3, (n_kv_heads, n_repeats))

        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask

    if sinks is not None:
        sinks = mx.expand_dims(sinks, (0, 2, 3))
        if n_repeats > 1:
            sinks = mx.unflatten(sinks, 1, (n_kv_heads, n_repeats))
        score_shape = list(scores.shape)
        score_shape[-1] = 1
        sinks = mx.broadcast_to(sinks, score_shape)
        scores = mx.concatenate([sinks, scores], axis=-1)

    scores = mx.softmax(scores, axis=-1, precise=True)
    if sinks is not None:
        scores = scores[..., 1:]

    out = scores @ v
    if n_repeats > 1:
        out = mx.reshape(out, [B, n_q_heads, L, -1])
    return out


def do_attention(f, q, k, v, scale, mask=None, transpose=False):
    if transpose:
        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 2, 1, 3))
        v_t = mx.transpose(v, (0, 2, 1, 3))
        o_t = f(q_t, k_t, v_t, scale=scale, mask=mask)
        return mx.transpose(o_t, (0, 2, 1, 3))
    else:
        return f(q, k, v, scale=scale, mask=mask)


def prepare_inputs(B, qL, kL, D, qH, kH, mask, transpose, dtype):
    mx.random.seed(0)

    scale = 1.0 / math.sqrt(D)
    shape_q = (B, qL, qH, D) if transpose else (B, qH, qL, D)
    shape_kv = (B, kL, kH, D) if transpose else (B, kH, kL, D)

    q = mx.random.uniform(0.0, 0.5, shape_q, dtype)
    k = mx.random.uniform(0.0, 0.5, shape_kv, dtype)
    v = mx.random.uniform(0.0, scale, shape_kv, dtype)

    if mask is not None:
        if mask == "additive":
            mask = mx.random.uniform(0.0, 0.5, (B, qH, qL, kL), dtype)
        elif mask == "bool":
            mask = mx.random.uniform(0.0, 1.0, (B, qH, qL, kL)) < 0.5

    return q, k, v, scale, mask


# SDPA for MHA (n_heads == n_kv_heads)
def mlx_primitives_sdpa(q, k, v, scale, mask=None):
    p = (q * scale) @ k.transpose(0, 1, 3, 2)
    qL = q.shape[2]
    kL = k.shape[2]
    is_causal = mask == "causal"
    if mask is not None:
        if is_causal:
            offset = kL - qL
            q_indices = mx.arange(qL) + offset
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
            p = mx.where(mask, p, mx.finfo(mx.float32).min)
        elif mask.dtype == mx.bool_:
            p = mx.where(mask, p, mx.finfo(mx.float32).min)
        else:
            p += mask
    scores = mx.softmax(p.astype(mx.float32), axis=-1).astype(p.dtype)
    return scores @ v


class TestFastSDPA(mlx_tests.MLXTestCase):
    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_head_dim_72(self):
        B, D, qH, kH = (1, 72, 8, 2)
        for qL, kL, dtype, mask_str in product(
            (64, 65),
            (128, 127),
            (mx.float16, mx.bfloat16, mx.float32),
            (None, "additive", "bool", "causal"),
        ):
            with self.subTest(qL=qL, kL=kL, dtype=dtype, mask=mask_str):
                q, k, v, scale, mask = prepare_inputs(
                    B, qL, kL, D, qH, kH, mask_str, False, dtype
                )
                ref = mlx_ref_attn(q, k, v, scale, mask)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )

                if dtype == mx.float32:
                    atol = 1e-5
                elif dtype == mx.bfloat16:
                    atol = 5e-3
                else:
                    atol = 3e-4
                diff = mx.abs(out - ref) - atol * mx.abs(ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    @unittest.skipIf(not mx.metal.is_available(), "Metal kernel path only")
    def test_sdpa_pad_head_dim_opt_in(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")
        with mlx_tests.scoped_env(MLX_SDPA_PAD_HEAD_DIM="1"):
            self.test_sdpa_head_dim_72()
            self.test_sdpa_head_dim_80()
            self.test_sdpa_head_dim_72_80_sinks()

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_head_dim_80(self):
        B, D, qH, kH = (1, 80, 8, 2)
        for qL, kL, dtype, mask_str in product(
            (64, 65),
            (128, 127),
            (mx.float16, mx.bfloat16, mx.float32),
            (None, "additive", "bool", "causal"),
        ):
            with self.subTest(qL=qL, kL=kL, dtype=dtype, mask=mask_str):
                q, k, v, scale, mask = prepare_inputs(
                    B, qL, kL, D, qH, kH, mask_str, False, dtype
                )
                ref = mlx_ref_attn(q, k, v, scale, mask)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )

                if dtype == mx.float32:
                    atol = 1e-5
                elif dtype == mx.bfloat16:
                    atol = 5e-3
                else:
                    atol = 3e-4
                diff = mx.abs(out - ref) - atol * mx.abs(ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_head_dim_72_80_sinks(self):
        B, qH, kH, qL, kL = (1, 8, 2, 64, 128)
        for D, dtype in product((72, 80), (mx.float16, mx.bfloat16)):
            with self.subTest(D=D, dtype=dtype):
                q, k, v, scale, _ = prepare_inputs(
                    B, qL, kL, D, qH, kH, None, False, dtype
                )
                sinks = 10 * mx.random.normal(shape=(qH,), dtype=dtype)
                ref = mlx_ref_attn(q, k, v, scale, sinks=sinks)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, sinks=sinks
                )

                atol = 5e-3 if dtype == mx.bfloat16 else 3e-4
                diff = mx.abs(out - ref) - atol * mx.abs(ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_head_dim_96(self):
        B, D, qH, kH = (1, 96, 8, 2)
        for qL, kL, dtype, mask_str in product(
            (64, 65),
            (128, 127),
            (mx.float16, mx.bfloat16, mx.float32),
            (None, "additive", "bool", "causal"),
        ):
            with self.subTest(qL=qL, kL=kL, dtype=dtype, mask=mask_str):
                q, k, v, scale, mask = prepare_inputs(
                    B, qL, kL, D, qH, kH, mask_str, False, dtype
                )
                ref = mlx_ref_attn(q, k, v, scale, mask)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )

                if dtype == mx.float32:
                    atol = 1e-5
                elif dtype == mx.bfloat16:
                    atol = 5e-3
                else:
                    atol = 3e-4
                diff = mx.abs(out - ref) - atol * mx.abs(ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_full_head_dim_256(self):
        # On NAX devices, large nearly-square causal blocks take the fused
        # path; everything else takes the unfused fallback. Ragged lengths
        # exercise the kernel's unaligned pipelines, and K/V sliced out of a
        # longer preallocated cache (the way mlx-lm hands them over) exercise
        # the dispatch reading the slice past its end. All of it must be
        # correct.
        D = 256
        Nq, Nkv = 8, 2
        scale = D**-0.5
        mx.random.seed(0)
        cases = [
            # fused on NAX: aligned square, ragged square (unaligned Q and
            # K/V), aligned rectangle at the routing boundary, ragged
            # rectangle
            (2048, 2048, "causal", None),
            (2049, 2049, "causal", None),
            (2048, 2560, "causal", None),
            (2049, 2560, "causal", None),
            # fused on NAX with ragged K/V sliced out of a longer cache: the
            # rows behind the slice must not leak into the output
            (2049, 2049, "causal", 2304),
            (2048, 2500, "causal", 2560),
            (1031, 2049, "causal", 2304),
        ]
        for dtype in (mx.float32, mx.bfloat16):
            for qL, kL, mask, cache_len in cases:
                with self.subTest(
                    dtype=dtype, qL=qL, kL=kL, mask=mask, cache_len=cache_len
                ):
                    q = (5e-1 * mx.random.normal(shape=(1, Nq, qL, D))).astype(dtype)
                    if cache_len is None:
                        k = (5e-1 * mx.random.normal(shape=(1, Nkv, kL, D))).astype(
                            dtype
                        )
                        v = (5e-1 * mx.random.normal(shape=(1, Nkv, kL, D))).astype(
                            dtype
                        )
                    else:
                        # Large, finite stale rows behind the slice: any of
                        # them reaching the output is loud.
                        k_cache = 1e2 * mx.random.normal(shape=(1, Nkv, cache_len, D))
                        v_cache = 1e3 * mx.random.normal(shape=(1, Nkv, cache_len, D))
                        k_cache[..., :kL, :] = 5e-1 * mx.random.normal(
                            shape=(1, Nkv, kL, D)
                        )
                        v_cache[..., :kL, :] = 5e-1 * mx.random.normal(
                            shape=(1, Nkv, kL, D)
                        )
                        k = k_cache.astype(dtype)[..., :kL, :]
                        v = v_cache.astype(dtype)[..., :kL, :]
                    k_rep = mx.repeat(k, Nq // Nkv, axis=1)
                    v_rep = mx.repeat(v, Nq // Nkv, axis=1)
                    ref = mlx_primitives_sdpa(q, k_rep, v_rep, scale, mask=mask)
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, mask=mask
                    )
                    self.assertEqual(out.shape, ref.shape)
                    if dtype == mx.float32:
                        # The fused shapes run through tf32 tensor ops when
                        # MLX_ENABLE_TF32 is on (the default).
                        tol = 1e-3 if qL >= 2048 else 1e-4
                    else:
                        tol = 5e-3
                    self.assertTrue(mx.allclose(ref, out, atol=tol, rtol=tol))

    def test_sdpa_vector_kv_transposed_head_seq(self):
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))

        lengths = [43, 4096]
        for L in lengths:
            k = 5e-1 * mx.random.normal(shape=(1, L, Nkv, D))
            v = 5e-1 * mx.random.normal(shape=(1, L, Nkv, D))
            k = k.swapaxes(1, 2)
            v = v.swapaxes(1, 2)
            masks = [
                mx.array(True),
                mx.array([True] * (L - 10) + [False] * 10),
                mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
                mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            ]

            for m in masks:
                ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
                out = mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=scale,
                    mask=m,
                )
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_vector(self):
        D = 64
        L = 43
        Nq = 4
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=mx.full((Nq, 2, L), False),
            )

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            mx.random.uniform(shape=(Nq, 1, L)),
            mx.random.uniform(shape=(L, 1, Nq)).T,
            mx.log(mx.random.uniform(shape=(Nq, 1, L)) > 0.2),
            mx.log(mx.random.uniform(shape=(L, 1, Nq)).T > 0.2),
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        L = 4096
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            mx.random.uniform(shape=(Nq, 1, L)),
            mx.random.uniform(shape=(L, 1, Nq)).T,
            mx.log(mx.random.uniform(shape=(Nq, 1, L)) > 0.2),
            mx.log(mx.random.uniform(shape=(L, 1, Nq)).T > 0.2),
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_vector_gqa_long(self):
        scale = 1.0
        mx.random.seed(0)
        for Nq, Nkv, D in [(32, 4, 128), (64, 8, 64), (48, 4, 128), (64, 4, 128)]:
            for B, L in [(1, 8192), (1, 8201), (2, 8192)]:
                q = 5e-1 * mx.random.normal(shape=(B, Nq, 1, D))
                k = 5e-1 * mx.random.normal(shape=(B, Nkv, L + 32, D))[:, :, :L]
                v = 5e-1 * mx.random.normal(shape=(B, Nkv, L + 32, D))[:, :, :L]
                kr = mx.repeat(k, Nq // Nkv, axis=1)
                vr = mx.repeat(v, Nq // Nkv, axis=1)
                ref = mlx_primitives_sdpa(q, kr, vr, scale)
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    @unittest.skipIf(not mx.metal.is_available(), "Metal kernel path only")
    def test_sdpa_vector_head_dim_512(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")
        # gemma-4 global attention: 32 query heads over 4 key/value heads.
        D = 512
        Nq, Nkv = 32, 4
        scale = D**-0.5
        mx.random.seed(0)

        with mlx_tests.scoped_env(MLX_SDPA_D512_MIN_KL="0"):
            # Test 1-pass kernel.
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                with self.subTest(L=128, dtype=dtype, threshold="off"):
                    q = mx.random.normal(shape=(1, Nq, 1, D), dtype=dtype)
                    k = mx.random.normal(shape=(1, Nkv, 128, D), dtype=dtype)
                    v = mx.random.normal(shape=(1, Nkv, 128, D), dtype=dtype)
                    ref = mlx_ref_attn(q, k, v, scale)
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, force_fused=True
                    )
                    atol = 1e-5 if dtype == mx.float32 else 2e-2
                    self.assertTrue(mx.allclose(ref, out, atol=atol))

            # Test 2-pass kernel.
            for L, dtype in product(
                (8192, 8201), (mx.float32, mx.float16, mx.bfloat16)
            ):
                with self.subTest(L=L, dtype=dtype):
                    q = mx.random.normal(shape=(1, Nq, 1, D), dtype=dtype)
                    k = mx.random.normal(shape=(1, Nkv, L, D), dtype=dtype)
                    v = mx.random.normal(shape=(1, Nkv, L, D), dtype=dtype)
                    ref = mlx_ref_attn(q, k, v, scale)
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, force_fused=True
                    )
                    atol = 1e-5 if dtype == mx.float32 else 2e-2
                    self.assertTrue(mx.allclose(ref, out, atol=atol))

            # Test other heads.
            for q_heads, kv_heads in ((8, 8), (32, 8)):
                with self.subTest(q_heads=q_heads, kv_heads=kv_heads):
                    q = mx.random.normal(shape=(1, q_heads, 1, D))
                    k = mx.random.normal(shape=(1, kv_heads, L, D))
                    v = mx.random.normal(shape=(1, kv_heads, L, D))
                    ref = mlx_ref_attn(q, k, v, scale)
                    out = mx.fast.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        scale=scale,
                        force_fused=True,
                    )
                    self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

            # Test batched.
            B = 2
            sinks = 10 * mx.random.normal(shape=(Nq,))
            q = mx.random.normal(shape=(B, Nq, 1, D))
            for L in (256, 8192):
                k = mx.random.normal(shape=(B, Nkv, L, D))
                v = mx.random.normal(shape=(B, Nkv, L, D))
                for s_in in (None, sinks):
                    with self.subTest(L=L, sinks=s_in is not None):
                        ref = mlx_ref_attn(q, k, v, scale, sinks=s_in)
                        out = mx.fast.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            scale=scale,
                            sinks=s_in,
                            force_fused=True,
                        )
                        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_fully_masked(self):
        Lkv = 8
        mask = mx.array(False)
        for D in [128]:
            for Lq in [1, 8, 32]:
                q = mx.random.normal(shape=(1, 4, Lq, D))
                k = mx.random.normal(shape=(1, 4, Lkv, D))
                v = mx.random.normal(shape=(1, 4, Lkv, D))

                out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1)
                self.assertFalse(mx.any(mx.isnan(out)))

    def test_sdpa_inf_score(self):
        Lkv = 8
        for D in [4, 128]:
            for Lq in [1, 8]:
                q = mx.ones(shape=(1, 4, Lq, D))
                k = mx.ones(shape=(1, 4, Lkv, D))
                v = mx.random.normal(shape=(1, 4, Lkv, D))
                k[..., 0, :] = -float("inf")
                ref = mlx_primitives_sdpa(q, k, v, scale=1, mask=None)
                out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1)
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_few_query(self):
        D = 64
        L = 43
        Lq = 8
        Nq = 8
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Lq, Nq, D))
        q = q.swapaxes(1, 2)
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        L = 4096
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, Lq, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    @unittest.skip("Different head and value dims is not enabled")
    def test_sdpa_vector_value_dims(self):
        D = 192
        V = 128
        Nq = 4
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)

        for L in [43, 128, 237, 8192]:
            q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
            k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
            v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, V))
            ref = mlx_primitives_sdpa(q, k, v, scale)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_vector_batched(self):
        D = 64
        q = mx.random.normal(shape=(2, 1, 3, D))
        k = mx.random.normal(shape=(2, 1, 3, D))
        v = mx.random.normal(shape=(2, 1, 3, D))

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 3, 4, D)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        k = mx.random.normal(shape=(2, 3, 1, D)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        k = mx.random.normal(shape=(2, 3, 2, D)).swapaxes(1, 2)
        v = mx.random.normal(shape=(2, 2, 3, D))
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        k = mx.random.normal(shape=(2, 1, 3, D))
        v = mx.random.normal(shape=(2, 1, 3, D))
        mask = 10 * mx.random.normal(shape=(1, 2, 3, 3)).swapaxes(0, 1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        ref = mlx_ref_attn(q, k, v, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_blocks_env_override(self):
        # MLX_SDPA_BLOCKS used to be applied as-is, and values that are not
        # a multiple of 32 silently corrupted the 2-pass vector output. The
        # override is now rounded up to a multiple of 32.
        D = 128
        q = mx.random.normal(shape=(1, 32, 1, D), dtype=mx.float16)
        k = mx.random.normal(shape=(1, 8, 8192, D), dtype=mx.float16)
        v = mx.random.normal(shape=(1, 8, 8192, D), dtype=mx.float16)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
        for blocks in (16, 33, 48, 100):
            with mlx_tests.scoped_env(MLX_SDPA_BLOCKS=str(blocks)):
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    @unittest.skipIf(not mx.is_available(mx.gpu), "too slow on CPU")
    def test_sdpa(self):
        # fmt: off
        shapes_64 = [
            # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
            (  1,    20,    20,       64,    3,     3),
            (  1,    63,    63,       64,   24,    24),
            (  1,   129,   129,       64,   24,    24),
            (  1,   400,   400,       64,   24,    24),
            (  1,   128,   128,       64,   32,    32),
            (  1,    64,   128,       64,   32,    32),
            (  1,    65,   128,       64,   32,     8),
            (  1,    64,   127,       64,   32,     8),
            (  1,    65,   127,       64,   32,     8),
            (  1,   127,    65,       64,   32,     8),
        ]
        shapes_128 = [
            # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
            (  1,   128,   128,      128,   32,     8),
            (  1,    64,   128,      128,   32,     8),
            (  1,    65,   127,      128,   32,     8),
            (  1,   127,    65,      128,   32,     8),
        ]
        for ksl in [7, 9, 32, 63, 67, 129, 400, 2000]:
            shapes_128.append((1, 1, ksl, 128, 32, 32))
            shapes_128.append((1, 1, ksl, 128, 32, 8))
        # fmt: on

        shapes = shapes_64 + shapes_128
        dtypes = [mx.float16]
        if mx.metal.is_available():
            dtypes.append(mx.float32)
        masks = [None, "additive", "bool", "causal"]
        transposes = (False, True)

        for dtype, t, mask_str, (B, qL, kL, D, qH, kH) in product(
            dtypes, transposes, masks, shapes
        ):
            with self.subTest(
                B=B,
                qsl=qL,
                ksl=kL,
                head_dim=D,
                n_q_heads=qH,
                n_kv_heads=kH,
                mask=mask_str,
                transpose=t,
                dtype=dtype,
            ):
                q, k, v, scale, mask = prepare_inputs(
                    B, qL, kL, D, qH, kH, mask_str, t, dtype
                )

                out_ref = do_attention(mlx_ref_attn, q, k, v, scale, mask, t)

                out_fst = do_attention(
                    mx.fast.scaled_dot_product_attention,
                    q,
                    k,
                    v,
                    scale,
                    mask,
                    t,
                )

                # For causal mask when qL > kL, first qL-kL rows are undefined
                # Compare only the valid portion
                if mask_str == "causal" and qL > kL:
                    offset = qL - kL
                    if t:  # transpose=True: shape is (B, qL, qH, D)
                        out_ref = out_ref[:, offset:, :, :]
                        out_fst = out_fst[:, offset:, :, :]
                    else:  # transpose=False: shape is (B, qH, qL, D)
                        out_ref = out_ref[:, :, offset:, :]
                        out_fst = out_fst[:, :, offset:, :]

                atol = 2e-5 if dtype == mx.float32 else 3e-4

                self.assertListEqual(list(out_ref.shape), list(out_fst.shape))

                diff = mx.abs(out_fst - out_ref) - atol * mx.abs(out_ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    @unittest.skipIf(not mx.is_available(mx.gpu), "too slow on CPU")
    @unittest.skipIf(mx.cuda.is_available() and "CI" in os.environ, "not enough memory")
    def test_sdpa_long_masked_sequence(self):
        # Test for int16 overflow in steel_attention_nax.h mask
        # indexing (col_pos declared as short, overflows when kL > 32767).
        D = 64
        dtype = mx.float16
        atol = 1e-3  # Slightly looser than test_sdpa due to long masked sequences

        for kL, active in [
            (8192, 1024),
            (36864, 1024),
            (49152, 1024),
            (66048, 1024),
        ]:
            with self.subTest(kL=kL, active=active):
                mx.random.seed(0)
                qH, kH, qL = 32, 16, 512
                scale = 1.0 / math.sqrt(D)

                q = mx.random.normal(shape=(1, qH, qL, D)).astype(dtype)
                k = mx.random.normal(shape=(1, kH, kL, D)).astype(dtype)
                v = mx.random.normal(shape=(1, kH, kL, D)).astype(dtype)

                # Additive mask: -1e4 for inactive, 0 for last `active` positions
                mask = mx.full((1, 1, 1, kL), -1e4, dtype=dtype)
                mask[..., kL - active :] = 0.0

                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )
                ref = mlx_ref_attn(q, k, v, scale=scale, mask=mask)

                self.assertFalse(mx.isnan(out).any().item())
                self.assertListEqual(list(out.shape), list(ref.shape))

                diff = mx.abs(out - ref) - atol * mx.abs(ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    def test_sdpa_broadcast_mask(self):
        mask = mx.array(True)
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        L = 256

        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, L, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        ref = mlx_primitives_sdpa(q, k, v, scale, mask=mask)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_noncontiguous_inputs(self):
        mask = mx.ones(shape=(4, 1, 7, 7), dtype=mx.bool_)
        mx.random.seed(0)
        q = mx.random.normal(shape=(4, 7, 32, 64)).swapaxes(1, 2)

        k = mx.random.normal(shape=(4, 7, 8, 64)).swapaxes(1, 2)
        v = mx.random.normal(shape=(4, 7, 8, 64)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        ref = mlx_ref_attn(q, k, v, scale=1.0, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_promote_mask(self):
        mask = mx.array(2.0, mx.bfloat16)
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        L = 256

        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, L, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        ref = mlx_primitives_sdpa(q, k, v, scale, mask=mask)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_nan_bug(self):
        N = 128
        q_shape = (1, 1, N, 128)
        kv_shape = (1, 1, N, 128)
        q = mx.random.uniform(shape=q_shape)
        k = mx.random.uniform(shape=kv_shape)
        v = mx.random.uniform(shape=kv_shape)

        # Make boolean window causal mask
        linds = rinds = mx.arange(N)
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask = mask & (linds <= rinds + 111)

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        expected = mlx_ref_attn(q, k, v, mask=mask, scale=1.0)
        self.assertFalse(mx.isnan(out).any().item())
        self.assertLessEqual(mx.abs(out - expected).max().item(), 1e-4)

        # And an additive one
        mask = mx.log(mask)

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        expected = mlx_ref_attn(q, k, v, mask=mask, scale=1.0)
        self.assertFalse(mx.isnan(out).any().item())
        self.assertLessEqual(mx.abs(out - expected).max().item(), 1e-4)

    def test_sdpa_attention_sinks(self):
        B = 2
        N_q = N_kv = 8
        T_q = T_kv = 128
        D = 64

        q = mx.random.normal(shape=(B, N_q, T_q, D))
        k = mx.random.normal(shape=(B, N_kv, T_kv, D))
        v = mx.random.normal(shape=(B, N_kv, T_kv, D))
        scale = D**-0.5

        # sinks should promote to correct type
        sinks = mx.random.normal(shape=(N_q,))
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q.astype(mx.float16),
                k.astype(mx.float16),
                v.astype(mx.float16),
                scale=scale,
                sinks=sinks,
            )

        # Wrong shapes
        sinks = mx.random.normal(shape=(N_q + 1,))
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, sinks=sinks)

        sinks = mx.random.normal(shape=())
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, sinks=sinks)

        for T_q, T_kv, N_kv, dtype in product(
            (1, 128),
            (128, 4096),
            (2, 8),
            (mx.float16, mx.float32),
        ):
            with self.subTest(T_q=T_q, T_kv=T_kv, N_kv=N_kv, dtype=dtype):
                q = mx.random.normal(shape=(B, N_q, T_q, D), dtype=dtype)
                k = mx.random.normal(shape=(B, N_kv, T_kv, D), dtype=dtype)
                v = mx.random.normal(shape=(B, N_kv, T_kv, D), dtype=dtype)
                sinks = 10 * mx.random.normal(shape=(N_q,), dtype=dtype)

                expected = mlx_ref_attn(q, k, v, scale, sinks=sinks)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, sinks=sinks
                )
                atol = 1e-5 if dtype == mx.float32 else 1e-2
                self.assertTrue(mx.allclose(out, expected, atol=atol))

    def test_sdpa_grad(self):
        # High tolerance due to cuDNN SDPA kernel requiring tf32.
        tolerance = {"rtol": 1e-2, "atol": 1e-2}

        def test_vjp(slow, fast, primals):
            cotan = mx.ones_like(primals[0])
            o1, vjp1 = mx.vjp(slow, primals, [cotan])
            o2, vjp2 = mx.vjp(fast, primals, [cotan])

            self.assertTrue(mx.allclose(o1[0], o2[0], **tolerance))
            for i in range(3):
                self.assertTrue(mx.allclose(vjp1[i], vjp2[i], **tolerance))

        def test_grad(slow, fast, args):
            g1 = mx.grad(slow)(*args)
            g2 = mx.grad(fast)(*args)

            self.assertTrue(mx.allclose(g1, g2, **tolerance))

        B, N_kv, T, D = (2, 8, 128, 64)
        scale = D**-0.5

        for N_q in (8, 32):
            q = mx.random.normal(shape=(B, N_q, T, D), dtype=mx.float16)
            k = mx.random.normal(shape=(B, N_kv, T, D), dtype=mx.float16)
            v = mx.random.normal(shape=(B, N_kv, T, D), dtype=mx.float16)

            mask_additive = mx.random.normal((B, N_q, T, T), dtype=mx.float16)
            mask_bool = mx.random.uniform(0, 1, (B, N_q, T, T), dtype=mx.float16) < 0.5

            for mask in (None, "causal", mask_additive, mask_bool):
                sdpa_slow = lambda q, k, v: mlx_ref_attn(
                    q, k, v, scale=scale, mask=mask
                )
                sdpa_fast = lambda q, k, v: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )
                test_vjp(sdpa_slow, sdpa_fast, [q, k, v])

                loss_slow = lambda q, k, v: mlx_ref_attn(
                    q, k, v, scale=scale, mask=mask
                ).sum()
                loss_fast = lambda q, k, v: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                ).sum()
                test_grad(loss_slow, loss_fast, [q, k, v])

    @unittest.skipIf(not mx.metal.is_available(), "Metal kernel path only")
    def test_sdpa_force_fused_metal(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")

        def make_qkv(qL, kL, D, qH=8, kH=8):
            q = mx.random.normal((1, qH, qL, D), mx.float16)
            k = mx.random.normal((1, kH, kL, D), mx.float16)
            v = mx.random.normal((1, kH, kL, D), mx.float16)
            return q, k, v

        # Full attention kernel.
        for D, qL, mask in product((192, 256), (9, 16), (None, "causal")):
            with self.subTest(head_dim=D, qL=qL, mask=mask):
                q, k, v = make_qkv(qL, 512, D, 8, 4)
                scale = D**-0.5
                ref = mlx_ref_attn(q, k, v, scale=scale, mask=mask)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask, force_fused=True
                )
                self.assertTrue(mx.allclose(ref, out, atol=1e-3, rtol=1e-3))

        # Vector attention kernel.
        for D in (192, 256, 512):
            with self.subTest(head_dim=D):
                with mlx_tests.scoped_env(MLX_SDPA_D512_MIN_KL="0"):
                    q, k, v = make_qkv(4, 16385, D, 4, 2)
                    scale = D**-0.5
                    ref = mlx_ref_attn(q, k, v, scale=scale)
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, force_fused=True
                    )
                    self.assertTrue(mx.allclose(ref, out, atol=1e-3, rtol=1e-3))

        # No full attention fused kernels.
        with self.assertRaisesRegex(ValueError, "supports head dims"):
            q, k, v = make_qkv(16, 512, 512)
            mx.fast.scaled_dot_product_attention(
                q, k, v, scale=512**-0.5, force_fused=True
            )
        with self.assertRaisesRegex(
            ValueError, "query sequence to be no longer than the key sequence"
        ):
            q, k, v = make_qkv(32, 16, 64)
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=64**-0.5,
                mask="causal",
                force_fused=True,
            )

        # No vector attention fused kernels.
        with self.assertRaisesRegex(ValueError, "supports head dims"):
            q, k, v = make_qkv(1, 128, 72)
            mx.fast.scaled_dot_product_attention(
                q, k, v, scale=72**-0.5, force_fused=True
            )
        with self.assertRaisesRegex(ValueError, "GQA factor to be at most 32"):
            q, k, v = make_qkv(8, 128, 64, qH=8, kH=1)
            mx.fast.scaled_dot_product_attention(
                q, k, v, scale=64**-0.5, force_fused=True
            )
        with self.assertRaisesRegex(ValueError, r"requires at least \d+ keys"):
            with mlx_tests.scoped_env(MLX_SDPA_D512_MIN_KL=None):
                q, k, v = make_qkv(1, 512, 512, qH=32, kH=4)
                mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=512**-0.5, force_fused=True
                )

        # No CPU fused kernel.
        with mx.stream(mx.cpu):
            q, k, v = make_qkv(8, 128, 8)
            with self.assertRaisesRegex(ValueError, "require a GPU"):
                mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=64**-0.5, force_fused=True
                )

    @unittest.skipIf(not mx.cuda.is_available(), "CUDA kernel path only")
    def test_sdpa_force_fused_cuda(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")

        def make_qkv(qL, kL, D, qH=8, kH=8):
            q = mx.random.normal((1, qH, qL, D), mx.float16)
            k = mx.random.normal((1, kH, kL, D), mx.float16)
            v = mx.random.normal((1, kH, kL, D), mx.float16)
            return q, k, v

        # Vector attention kernel.
        for D in (64, 96, 128):
            with self.subTest(head_dim=D):
                q, k, v = make_qkv(3, 128, D, 4, 2)
                scale = D**-0.5
                ref = mlx_ref_attn(q, k, v, scale=scale)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, force_fused=True
                )
                self.assertTrue(mx.allclose(ref, out, atol=1e-3, rtol=1e-3))

    def test_sdpa_sliced(self):
        N = 8
        D = 64
        scale = D**-0.5

        for B, T_q, T_kv, offset, mask in product(
            (1, 2, 4),
            (1, 8),
            (256, 512),
            (8, 9, 64, 79),
            (None, "causal"),
        ):
            with self.subTest(B=B, T_q=T_q, T_kv=T_kv, offset=offset, mask=mask):
                q = mx.random.normal((B, N, T_q, D), mx.float16)
                k = mx.random.normal((B, N, T_kv, D), mx.float16)
                v = mx.random.normal((B, N, T_kv, D), mx.float16)

                k = k[..., :offset, :]
                v = v[..., :offset, :]

                ref = mlx_ref_attn(q, k, v, scale=scale, mask=mask)

                for i in range(2):
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, mask=mask
                    )
                    if B == 1:
                        tolerance = {"rtol": 1e-3, "atol": 1e-3}
                    else:
                        tolerance = {"rtol": 1e-2, "atol": 1e-2}
                    self.assertTrue(mx.allclose(ref, out, **tolerance))
    
    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_grad_per_output(self):
        # (B, qH, kH, qL, kL, D, Dv)
        grad_dims = [
            (1, 16, 16, 64, 64, 64, 64),      # smallest full tile
            (1, 16, 16, 128, 128, 128, 128),
            (2, 16, 16, 512, 512, 128, 128),
            (1, 16, 16, 4096, 4096, 128, 128),  # the size that is slow
            (1, 16, 4, 512, 512, 128, 128),     # GQA, rep = 4
            (1, 16, 16, 64, 64, 256, 256),      # widest head dim
            (1, 16, 16, 129, 127, 64, 64),      # ragged, exercises tails
        ]
        for dims in grad_dims:
            for dtype in (mx.float32, mx.bfloat16):
                # for mask_str in (None, "causal"):
                for mask_str in [None]:
                    grads, refs = sdpa_grad_and_refs(dims, dtype, mask_str)
                    tol = 5e-2 if dtype == mx.float32 else 5e-1

                    failures = compare_sdpa_grads(
                        grads, refs, dims, dtype, mask_str, tol,
                        "fused vjp vs fp32 reference",
                    )
                    self.assertTrue(
                        not failures,
                        msg=f"gradient mismatch on {dims} with {dtype}, "
                        f"mask={mask_str}: " + ", ".join(failures),
                    )

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_dq_stage(self):
        STAGE = os.environ.get("SDPA_DQ_STAGE", "ds")

        B, H, L, D = 1, 16, 64, 64  # must match an instantiated shape
        scale = 1.0 / math.sqrt(D)

        mx.random.seed(0)
        q = mx.random.uniform(0.0, 0.5, (B, H, L, D), mx.float32)
        k = mx.random.uniform(0.0, 0.5, (B, H, L, D), mx.float32)
        v = mx.random.uniform(0.0, scale, (B, H, L, D), mx.float32)
        cot = mx.random.normal((B, H, L, D)).astype(mx.float32)
        mx.eval(q, k, v, cot)

        f = lambda a, b, c: mx.fast.scaled_dot_product_attention(
            a, b, c, scale=scale, mask=None
        )
        _, grads = mx.vjp(f, [q, k, v], [cot])
        got = grads[0]
        mx.eval(got)

        # The reference intermediates, in the same order the kernel builds them.
        S = scale * (q @ mx.swapaxes(k, -1, -2))
        P = mx.softmax(S, axis=-1, precise=True)
        dP = cot @ mx.swapaxes(v, -1, -2)
        O = P @ v
        # <dO_i, O_i>, the rowsum(P * dP) term of the softmax Jacobian.
        odo = (cot * O).sum(-1, keepdims=True)
        dS = P * (dP - odo) * scale

        odo_bcast = mx.broadcast_to(odo, (B, H, L, L))

        want = {"p": P, "dp": dP, "ds": dS, "odo": odo_bcast}[STAGE] @ k
        mx.eval(want)

        a = np.array(got, dtype=np.float64)
        r = np.array(want, dtype=np.float64)
        e = np.abs(a - r)
        rel = e.max() / max(np.abs(r).max(), 1e-30)

        print(f"\n=== dq stage={STAGE} ===", flush=True)
        print(f"  rel {rel:.3e}   max|err| {e.max():.3e}   max|ref| "
              f"{np.abs(r).max():.3e}", flush=True)
        print(f"  ratio of maxima  mlx/ref = "
              f"{np.abs(a).max() / max(np.abs(r).max(), 1e-30):.4f}", flush=True)

        if rel > 1e-4:
            # A single wrong row or column localizes the index; a flat smear with
            # a clean ratio is a missing or extra scalar factor.
            en = e / max(np.abs(r).max(), 1e-30)
            print(f"  per-row  rel: {en.max(axis=(0, 1, 3))}", flush=True)
            print(f"  per-col  rel: {en.max(axis=(0, 1, 2))}", flush=True)
            bad = np.unravel_index(e.argmax(), e.shape)
            print(f"  worst (b,h,l,d)={bad}  mlx {a[bad]:+.6f}  ref {r[bad]:+.6f}",
                  flush=True)
            print(f"  elementwise ratio at worst: "
                  f"{a[bad] / r[bad] if r[bad] != 0 else float('nan'):+.4f}",
                  flush=True)

        self.assertLessEqual(rel, 1e-4, f"dq stage {STAGE} mismatch")




if __name__ == "__main__":
    mlx_tests.MLXTestRunner(failfast=True)
