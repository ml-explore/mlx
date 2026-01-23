from itertools import product

import mlx.core as mx


# In mxfp8 mode, the results do not match exactly:
# fewer than 1% of output elements differ.
# This does not appear to be a systematic error.
# The error can exceed 1 ULP for very small values,
# and is always below 1 ULP for larger values.
# For nvfp4, the results match exactly.
# therefore I suspect that the discrepancy comes from
# the mxfp8 matmul implementation in cuBLASLt..
def ulp_bf16_at(x):
    ax = mx.abs(x)
    min_normal = mx.array(2.0**-126)
    ax = mx.where(ax < min_normal, min_normal, ax)
    e = mx.floor(mx.log2(ax))
    return mx.power(2.0, e - 7.0)


def test_qqmm():
    key = mx.random.key(0)
    k1, k2 = mx.random.split(key)
    dtypes = [mx.bfloat16, mx.float32, mx.float16]

    tests = (
        (16, "nvfp4", 4),
        (32, "mxfp8", 8),
    )
    shapes = (
        [64, 65, 33, 128, 256, 1024, 1024 * 8],  # M
        [64, 128, 256, 1024, 1024 * 8],  # N
        [64, 128, 256, 1024, 1024 * 8],  # K
    )
    for group_size, mode, bits in tests:
        for M, N, K in product(*shapes):
            for dtype in dtypes:
                x = mx.random.normal(shape=(M, K), key=k1, dtype=dtype)
                w = mx.random.normal(shape=(N, K), key=k2, dtype=dtype)
                w_q, scales_w = mx.quantize(w, group_size, bits, mode=mode)
                w_dq = mx.dequantize(
                    w_q,
                    scales_w,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                    dtype=dtype,
                )
                y_q = mx.qqmm(
                    x,
                    w_q,
                    scales_w,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                )
                x_q, scales_x = mx.quantize(
                    x, group_size=group_size, bits=bits, mode=mode
                )
                x_dq = mx.dequantize(
                    x_q,
                    scales_x,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                    dtype=dtype,
                )
                y_hat = mx.matmul(x_dq, mx.transpose(w_dq))
                ulp = ulp_bf16_at(y_hat)
                error = (y_q - y_hat).abs()
                if not (mx.logical_or(error < 1e-3, error <= ulp).all()):
                    raise AssertionError(
                        f"qqmm test failed for shape {(M, N, K)}, "
                        f"group_size={group_size}, bits={bits}, "
                        f"mode={mode}, dtype={dtype}"
                    )


def test_qqmm_vjp():
    key = mx.random.key(0)
    k1, k2 = mx.random.split(key)
    M = 64
    N = 1024
    K = 512
    tests = (
        (16, "nvfp4", 4),
        (32, "mxfp8", 8),
    )
    x = mx.random.normal(shape=(M, K), key=k1)
    c = mx.ones(shape=(M, N))

    for group_size, mode, bits in tests:
        w = mx.random.normal(shape=(N, K), key=k2)

        def fn(x):
            return mx.qqmm(x, w, group_size=group_size, bits=bits, mode=mode)

        _, vjp_out = mx.vjp(fn, primals=(x,), cotangents=(c,))
        w_tq, scales_wt = mx.quantize(
            mx.transpose(w), group_size=group_size, bits=bits, mode=mode
        )
        expected_out = mx.qqmm(
            c, w_tq, scales_wt, group_size=group_size, bits=bits, mode=mode
        )
        ulp = ulp_bf16_at(expected_out)
        error = (vjp_out[0] - expected_out).abs()
        if not (mx.logical_or(error < 1e-3, error <= ulp).all()):
            raise AssertionError(
                f"qqmm vjp test failed for shape {(M, N, K)}, "
                f"group_size={group_size}, bits={bits}, mode={mode}"
            )


if __name__ == "__main__":
    test_qqmm()
    test_qqmm_vjp()
