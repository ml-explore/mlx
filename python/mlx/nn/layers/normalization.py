# Copyright © 2023 Apple Inc.

import mlx.core as mx
from mlx.nn.layers.base import Module


class LayerNorm(Module):
    r"""Applies layer normalization [1] on the inputs.

    Computes

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively.

    [1]: https://arxiv.org/abs/1607.06450

    Args:
        dims (int): The feature dimension of the input to normalize over
        eps (float): A small additive constant for numerical stability
        affine (bool): If True learn an affine transform to apply after the
            normalization
    """

    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if affine:
            self.bias = mx.zeros((dims,))
            self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}"

    def __call__(self, x):
        means = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - means) * mx.rsqrt(var + self.eps)
        return (self.weight * x + self.bias) if "weight" in self else x


class RMSNorm(Module):
    r"""Applies Root Mean Square normalization [1] to the inputs.

    Computes

    ..  math::

        y = \frac{x}{\sqrt{E[x^2] + \epsilon}} \gamma

    where :math:`\gamma` is a learned per feature dimension parameter initialized at
    1.

    [1]: https://arxiv.org/abs/1910.07467

    Args:
        dims (int): The feature dimension of the input to normalize over
        eps (float): A small additive constant for numerical stability
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.eps}"

    def __call__(self, x):
        # S is 1/sqrt(N) where N is the size of the features of x and is used
        # to compute a numerically more stable RMS of x by multiplying with S
        # first and summing.
        #
        # This way we prefer underflow over overflow which is controlled with
        # the parameter epsilon anyway.
        S = 1 / x.shape[-1] ** 0.5

        n = (x * S).square().sum(axis=-1, keepdims=True)
        n = mx.rsqrt(n + self.eps)

        return self.weight * x * n


class GroupNorm(Module):
    r"""Applies Group Normalization [1] to the inputs.

    Computes the same normalization as layer norm, namely

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively. However, the mean and
    variance are computed over the spatial dimensions and each group of
    features. In particular, the input is split into num_groups across the
    feature dimension.

    The feature dimension is assumed to be the last dimension and the dimensions
    that precede it (except the first) are considered the spatial dimensions.

    [1]: https://arxiv.org/abs/1803.08494

    Args:
        num_groups (int): Number of groups to separate the features into
        dims (int): The feature dimensions of the input to normalize over
        eps (float): A small additive constant for numerical stability
        affine (bool): If True learn an affine transform to apply after the
            normalization.
        pytorch_compatible (bool): If True perform the group normalization in
            the same order/grouping as PyTorch.
    """

    def __init__(
        self,
        num_groups: int,
        dims: int,
        eps: float = 1e-5,
        affine: bool = True,
        pytorch_compatible: bool = False,
    ):
        super().__init__()
        if affine:
            self.bias = mx.zeros((dims,))
            self.weight = mx.ones((dims,))
        self.num_groups = num_groups
        self.dims = dims
        self.eps = eps
        self.pytorch_compatible = pytorch_compatible

    def _extra_repr(self):
        return (
            f"{self.num_groups}, {self.dims}, eps={self.eps}, "
            f"affine={'weight' in self}, pytorch_compatible={self.pytorch_compatible}"
        )

    def _pytorch_compatible_group_norm(self, x):
        num_groups = self.num_groups
        batch, *rest, dims = x.shape

        # Split into groups
        x = x.reshape(batch, -1, num_groups, dims // num_groups)
        x = x.transpose(0, 1, 3, 2).reshape(batch, -1, num_groups)

        # Normalize
        means = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)
        x = (x - means) * mx.rsqrt(var + self.eps)
        x = x.reshape(batch, -1, dims // num_groups, num_groups)
        x = x.transpose(0, 1, 3, 2).reshape(batch, *rest, dims)

        return x

    def _group_norm(self, x):
        num_groups = self.num_groups
        batch, *rest, dims = x.shape

        # Split into groups
        x = x.reshape(batch, -1, num_groups)

        # Normalize
        means = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)
        x = (x - means) * mx.rsqrt(var + self.eps)
        x = x.reshape(batch, *rest, dims)

        return x

    def __call__(self, x):
        group_norm = (
            self._pytorch_compatible_group_norm
            if self.pytorch_compatible
            else self._group_norm
        )
        x = group_norm(x)
        return (self.weight * x + self.bias) if "weight" in self else x
