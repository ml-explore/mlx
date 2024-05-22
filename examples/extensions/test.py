import mlx.core as mx
from mlx_sample_extensions import axpby

a = mx.ones((3, 4))
b = mx.ones((3, 4))
c = axpby(a, b, 4.0, 2.0, stream=mx.cpu)

print(f"c shape: {c.shape}")
print(f"c dtype: {c.dtype}")
print(f"c correct: {mx.all(c == 6.0).item()}")
