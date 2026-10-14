import mlx.core as mx
from mlx_sample_metal_extension import silu_and_mul

base = mx.arange(2 * 74 * 3, dtype=mx.float32).reshape(2, 74, 3)
x = mx.sin(base * 0.17).transpose(0, 2, 1).astype(mx.float16)

d = x.shape[-1] // 2
left = x[..., :d]
right = x[..., d:]

output = silu_and_mul(x, stream=mx.gpu)
expected = (left * mx.sigmoid(left)) * right
mx.eval(output, expected)

correct = mx.allclose(output, expected, rtol=2e-3, atol=2e-3).item()
print(f"silu_and_mul: shape={output.shape}, dtype={output.dtype}, correct={correct}")
assert correct
