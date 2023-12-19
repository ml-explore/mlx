# Copyright © 2023 Apple Inc.

from typing import Tuple

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


class BatchNorm1d(Module):
    r"""Applies Batch Normalization over a 2D or 3D input.

    Computes

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively.

    [1]: https://arxiv.org/abs/1502.03167

    Args:
        num_features (int): The feature dimension of the input to normalize over.
        eps (float, optional): A small additive constant for numerical stability. Default is 1e-5.
        momentum (float, optional): The momentum for updating the running mean and variance. Default is 0.1.
        affine (bool, optional): If True, learn an affine transform to apply after the normalization. Default is True.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> mx.random.seed(42)
        >>> input = mx.random.normal((5, 4), dtype=mx.float32)
        >>> # Batch norm
        >>> bn = nn.BatchNorm1d(num_features=4, affine=True)
        >>> output = bn(x)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))

        if self.track_running_stats:
            self.running_mean = mx.zeros((num_features,))
            self.running_var = mx.ones((num_features,))

    def _extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={'weight' in self}, track_running_stats={self.track_running_stats}"

    def _calc_stats(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Calculate the mean and variance of the input tensor.

        Args:
            x (mx.array): Input tensor.

        Returns:
            tuple: Tuple containing mean and variance.
        """

        if len(x.shape) == 2:
            means = mx.mean(x, axis=0, keepdims=True)
            var = mx.var(x, axis=0, keepdims=True)
        else:
            means = mx.mean(x, axis=(0, 2), keepdims=True)
            var = mx.var(x, axis=(0, 2), keepdims=True)

        if self.track_running_stats and self.training:
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * means
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
        return means, var

    def _check_and_expand_dims(self, x: mx.array):
        """
        Check if the input is a 2D or 3D tensor and expand the weight, bias, running mean, and running variance accordingly.

        Args:
            x (mx.array): Input tensor.
        """

        if x.ndim != 2 and x.ndim != 3:
            raise ValueError(f"expected 2D or 3D input (got {x.ndim}D input)")

        if x.ndim == 3 and self.weight.ndim != x.ndim:
            self.weight = mx.expand_dims(self.weight, [0, 2])
            self.bias = mx.expand_dims(self.bias, [0, 2])

        if self.track_running_stats:
            if x.ndim == 3 and self.running_mean.ndim != x.ndim:
                self.running_mean = mx.expand_dims(self.running_mean, [0, 2])
                self.running_var = mx.expand_dims(self.running_var, [0, 2])

    def __call__(self, x: mx.array):
        """
        Forward pass of BatchNorm1d.

        Args:
            x (mx.array): Input tensor.

        Returns:
            mx.array: Output tensor.
        """

        self._check_and_expand_dims(x)

        if self.training or not self.track_running_stats:
            means, var = self._calc_stats(x)
        else:
            means, var = self.running_mean, self.running_var
        x = (x - means) * mx.rsqrt(var + self.eps)
        return (self.weight * x + self.bias) if "weight" in self else x
