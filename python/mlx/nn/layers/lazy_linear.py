import math

import mlx.core as mx
from mlx.nn.layers.base import Module


class LazyLinear(Module):
    r"""A linear layer with lazy weight and bias initialization.

    This module represents a linear transformation which initializes its
    parameters the first time it is called with input data. This allows
    for dynamic determination of input feature dimensions.

    The transformation is defined as:

    .. math::

        y = x W^\top + b

    where :math:`W` has shape ``[output_dims, input_dims]`` (initialized upon the
    first forward pass) and :math:`b` has shape ``[output_dims]``.

    The values of :math:`W` and :math:`b` are initialized from the uniform distribution
    :math:`\mathcal{U}(-k, k)`, where :math:`k = \frac{1}{\sqrt{D_i}}` and :math:`D_i` is the
    input dimension determined at runtime.

    Args:
        output_dims (int): The dimensionality of the output features.
        bias (bool, optional): If set to ``False``, the layer will not use a bias.
                               Default is ``True``.
    """

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
