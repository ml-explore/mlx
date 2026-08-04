import mlx.core as mx
from mlx_sample_metal_extension import fatrelu_and_mul, gelu, silu_and_mul


def run_cases():
    gated_base = mx.arange(2 * 74 * 3, dtype=mx.float32).reshape(2, 74, 3)
    gated_x = mx.sin(gated_base * 0.17).transpose(0, 2, 1).astype(mx.float16)
    elementwise_base = mx.arange(2 * 37 * 3, dtype=mx.float32).reshape(2, 37, 3)
    elementwise_x = (
        mx.cos(elementwise_base * 0.11).transpose(0, 2, 1).astype(mx.float16)
    )

    d = gated_x.shape[-1] // 2
    left = gated_x[..., :d]
    right = gated_x[..., d:]
    threshold = 0.2
    cases = {
        "silu_and_mul": (
            silu_and_mul(gated_x, stream=mx.gpu),
            (left * mx.sigmoid(left)) * right,
        ),
        "gelu": (
            gelu(elementwise_x, stream=mx.gpu),
            0.5 * elementwise_x * (1.0 + mx.erf(elementwise_x * 0.7071067811865475)),
        ),
        "fatrelu_and_mul": (
            fatrelu_and_mul(gated_x, threshold, stream=mx.gpu),
            mx.where(left > threshold, left, mx.zeros_like(left)) * right,
        ),
    }

    mx.eval(*(array for pair in cases.values() for array in pair))
    for name, (output, expected) in cases.items():
        correct = mx.allclose(output, expected, rtol=2e-3, atol=2e-3).item()
        print(f"{name}: shape={output.shape}, dtype={output.dtype}, correct={correct}")
        assert correct


run_cases()
