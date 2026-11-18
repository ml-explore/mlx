import mlx.core as mx
from mlx_sample_extensions import axpby

a = mx.ones((3, 4))
b = mx.ones((3, 4))
c_cpu = axpby(a, b, 4.0, 2.0, stream=mx.cpu)
c_gpu = axpby(a, b, 4.0, 2.0, stream=mx.gpu)

print(f"c shape: {c_cpu.shape}")
print(f"c dtype: {c_cpu.dtype}")
print(f"c_cpu correct: {mx.all(c_cpu == 6.0).item()}")
print(f"c_gpu correct: {mx.all(c_gpu == 6.0).item()}")
