import math
import mlx.core as mx
from mlx.nn.layers.base import Module

class LazyLinear(Module):
    def __init__(self, output_dims: int, bias: bool = True) -> None:
        super().__init__()
        self.output_dims = output_dims
        self.bias = bias
        self.weight_initialized = False

    def _initialize_parameters(self, input_dims: int) -> None:
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(self.output_dims, input_dims)
        )
        if self.bias:
            self.bias = mx.random.uniform(
                low=-scale, high=scale, shape=(self.output_dims,)
            )
        self.weight_initialized = True

    def __call__(self, x: mx.array) -> mx.array:
        if not self.weight_initialized:
            self._initialize_parameters(x.shape[-1])

        if self.bias is not None:
            return mx.addmm(self.bias, x, self.weight.T)
        else:
            return x @ self.weight.T

    def _extra_repr(self) -> str:
        return f"output_dims={self.output_dims}, bias={self.bias is not None}"
