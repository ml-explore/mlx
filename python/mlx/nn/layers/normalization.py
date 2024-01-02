# Copyright © 2023 Apple Inc.

from typing import Tuple

import mlx.core as mx
from mlx.nn.layers.base import Module


class InstanceNorm(Module):
    r"""Applies instance normalization [1] on the inputs.

    Computes

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively. Both are of size num_features,
    if :attr:`affine` is ``True``.

    [1]: https://arxiv.org/abs/1607.08022

    Args:
        num_features: number of features of the input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
    Shape:
        - Input: :math:`(N, C, L)` or :math:`(N, C, H, W)` or :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, L)` or :math:`(N, C, H, W)` or :math:`(N, C, D, H, W)` (same shape as input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
    ):
        super().__init__()
        if affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))
        self.eps = eps
        self.affine = affine
        self.param_shapes = {
            3: (1, num_features, 1),  # input shape: (B, C, L)
            4: (1, num_features, 1, 1),  # input shape: (B, C, H, W)
            5: (1, num_features, 1, 1, 1),  # input shape: (B, C, D, H, W)
        }
        self.reduction_axes = {
            3: [2],  # input shape: (B, C, L)
            4: [2, 3],  # input shape: (B, C, H, W)
            5: [2, 3, 4],  # input shape: (B, C, D, H, W)
        }

    def extra_repr(self):
        return "{num_features}, eps={eps}, momentum={momentum}, affine={affine}".format(
            **self.__dict__
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.affine and self.weight.ndim != x.ndim:
            if x.ndim not in self.reduction_axes:
                raise ValueError("Unsupported number shape")
            # Ensure parameters are reshaped for correct broadcasting
            self.weight = mx.reshape(self.weight, self.param_shapes[x.ndim])
            self.bias = mx.reshape(self.bias, self.param_shapes[x.ndim])
        # Compute stats
        mean = mx.mean(x, axis=self.reduction_axes[x.ndim], keepdims=True)
        var = mx.var(x, axis=self.reduction_axes[x.ndim], keepdims=True)
        # Normalize
        normalized = (x - mean) * mx.rsqrt(var + self.eps)
        # Scale and shift if necessary
        if self.affine:
            return self.weight * normalized + self.bias
        else:
            return normalized


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


class BatchNorm(Module):
    r"""Applies Batch Normalization over a 2D or 3D input.

    Computes

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively.

    The input shape is specified as ``NC`` or ``NLC``, where ``N`` is the
    batch, ``C`` is the number of features or channels, and ``L`` is the
    sequence length. The output has the same shape as the input. For
    four-dimensional arrays, the shape is ``NHWC``, where ``H`` and ``W`` are
    the height and width respecitvely.

    For more information on Batch Normalization, see the original paper `Batch
    Normalization: Accelerating Deep Network Training by Reducing Internal
    Covariate Shift <https://arxiv.org/abs/1502.03167>`_.

    Args:
        num_features (int): The feature dimension to normalize over.
        eps (float, optional): A small additive constant for numerical
            stability. Default: ``1e-5``.
        momentum (float, optional): The momentum for updating the running
            mean and variance. Default: ``0.1``.
        affine (bool, optional): If ``True``, apply a learned affine
            transformation after the normalization. Default: ``True``.
        track_running_stats (bool, optional): If ``True``, track the
            running mean and variance. Default: ``True``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> x = mx.random.normal((5, 4))
        >>> bn = nn.BatchNorm(num_features=4, affine=True)
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
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))

        if self.track_running_stats:
            self._running_mean = mx.zeros((num_features,))
            self._running_var = mx.ones((num_features,))

    def _extra_repr(self):
        return (
            f"{self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={'weight' in self}, "
            f"track_running_stats={self.track_running_stats}"
        )

    def _calc_stats(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Calculate the mean and variance of the input tensor.

        Args:
            x (mx.array): Input tensor.

        Returns:
            tuple: Tuple containing mean and variance.
        """
        reduction_axes = tuple(range(0, x.ndim - 1))
        means = mx.mean(x, axis=reduction_axes, keepdims=True)
        var = mx.var(x, axis=reduction_axes, keepdims=True)

        if self.track_running_stats and self.training:
            self._running_mean = (
                1 - self.momentum
            ) * self._running_mean + self.momentum * means
            self._running_var = (
                1 - self.momentum
            ) * self._running_var + self.momentum * var
        return means, var

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of BatchNorm.

        Args:
            x (mx.array): Input tensor.

        Returns:
            mx.array: Output tensor.
        """

        if x.ndim < 2 or x.ndim > 4:
            raise ValueError(
                f"Expected input tensor to have 2, 3 or 4 dimensions, but got {x.ndim}"
            )

        if self.training or not self.track_running_stats:
            means, var = self._calc_stats(x)
        else:
            means, var = self._running_mean, self._running_var
        x = (x - means) * mx.rsqrt(var + self.eps)
        return (self.weight * x + self.bias) if "weight" in self else x
