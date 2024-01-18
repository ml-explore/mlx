import mlx.core as mx
from mlx.onnx import MlxBackend
from onnx import hub

model = hub.load("mnist")
backend = MlxBackend(model)
res = backend.run(mx.ones((1, 1, 28, 28)))
print(res)
