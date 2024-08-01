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
    parameters initialized at 1 and 0 respectively. Both are of size :attr:`dims`,
    if :attr:`affine` is ``True``.

    Args:
        dims (int): The number of features of the input.
        eps (float): A value added to the denominator for numerical stability. Default: ``1e-5``.
        affine (bool): Default: ``False``.

    Shape:
      - Input: :math:`(..., C)` where :math:`C` is equal to :attr:`dims`.
      - Output: Same shape as the input.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> x = mx.random.normal((8, 4, 4, 16))
        >>> inorm = nn.InstanceNorm(dims=16)
        >>> output = inorm(x)

    References:
        [1]: https://arxiv.org/abs/1607.08022
    """

    def __init__(
        self,
        dims: int,
        eps: float = 1e-5,
        affine: bool = False,
    ):
        super().__init__()
        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))
        self.dims = dims
        self.eps = eps

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}"

    def __call__(self, x: mx.array) -> mx.array:
        reduction_axes = tuple(range(1, x.ndim - 1))
        # Compute stats
        mean = mx.mean(x, axis=reduction_axes, keepdims=True)
        var = mx.var(x, axis=reduction_axes, keepdims=True)
        # Normalize
        x = (x - mean) * mx.rsqrt(var + self.eps)
        # Scale and shift if necessary
        return (self.weight * x + self.bias) if "weight" in self else x


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
        bias (bool): If True include a translation to the affine
            transformation. If set to False the transformation is not really affine
            just scaling.
    """

    def __init__(
        self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True
    ):
        super().__init__()
        if affine:
            self.weight = mx.ones((dims,))
            if bias:
                self.bias = mx.zeros((dims,))
        self.eps = eps
        self.dims = dims

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}"

    def __call__(self, x):
        weight = self.weight if "weight" in self else None
        bias = self.bias if "bias" in self else None
        return mx.fast.layer_norm(x, weight, bias, self.eps)


class RMSNorm(Module):
    r"""Applies Root Mean Square normalization [1] to the inputs.

    Computes

    ..  math::

        y = \frac{x}{\sqrt{E[x^2] + \epsilon}} \gamma

    where :math:`\gamma` is a learned per feature dimension parameter initialized at
    1.

    Note the accumulation for the mean is done in 32-bit precision.

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
        return mx.fast.rms_norm(x, self["weight"], self.eps)


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
        group_size = dims // num_groups

        # Split into groups
        x = x.reshape(batch, -1, num_groups, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, num_groups, -1)

        # Normalize
        x = mx.fast.layer_norm(x, eps=self.eps, weight=None, bias=None)

        x = x.reshape(batch, num_groups, -1, group_size)
        x = x.transpose(0, 2, 1, 3).reshape(batch, *rest, dims)
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
    the height and width respectively.

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
            self.running_mean = mx.zeros((num_features,))
            self.running_var = mx.ones((num_features,))
            self.freeze(keys=["running_mean", "running_var"], recurse=False)

    def unfreeze(self, *args, **kwargs):
        """Wrap unfreeze to make sure that running_mean and var are always
        frozen parameters."""
        super().unfreeze(*args, **kwargs)
        self.freeze(keys=["running_mean", "running_var"], recurse=False)

    def _extra_repr(self):
        return (
            f"{self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={'weight' in self}, "
            f"track_running_stats={self.track_running_stats}"
        )

    def _calc_stats(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Calculate the mean and variance of the input tensor across the batch
        and spatial dimensions.

        Args:
            x (array): Input tensor.

        Returns:
            tuple: Tuple containing mean and variance.
        """
        reduction_axes = tuple(range(0, x.ndim - 1))

        mean = mx.mean(x, axis=reduction_axes)
        var = mx.var(x, axis=reduction_axes)

        return mean, var

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of BatchNorm.

        Args:
            x (array): Input tensor.

        Returns:
            array: Normalized output tensor.
        """
        if x.ndim < 2 or x.ndim > 4:
            raise ValueError(
                f"Expected input tensor to have 2, 3 or 4 dimensions, but got {x.ndim}"
            )

        # Calculate the mean and variance used to normalize the input x. If we
        # are in training mode update the running stats if needed.
        mean, var = self._calc_stats(x)
        if self.training and self.track_running_stats:
            mu = self.momentum
            self.running_mean = (1 - mu) * self.running_mean + mu * mean
            self.running_var = (1 - mu) * self.running_var + mu * var
        elif self.track_running_stats:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) * mx.rsqrt(var + self.eps)
        return (self.weight * x + self.bias) if "weight" in self else x
