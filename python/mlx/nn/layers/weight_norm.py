import mlx.core as mx
from mlx.nn.layers.base import Module


class WeightNormWrapper(Module):
    r"""Applies weight normalization [1] to a module's parameter.

    This module wraps another module and applies weight normalization to one of its
    parameters. Weight normalization reparameterizes the weight vectors in terms of
    a magnitude (scale) and a direction as:

    .. math::

        w = g \frac{v}{||v||}

    where :math:`g` is a scalar and :math:`v` is a vector. The module computes the
    normalized weight and updates the wrapped module's parameter with this value.

    Args:
        module (Module): The module to wrap.
        name (str): The name of the parameter to normalize. Default: ``"weight"``.
        dim (int or None): The dimension along which to normalize. If None, the
            weight is normalized by its L2 norm. Default: ``0``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> linear = nn.Linear(20, 30)
        >>> weight_norm_linear = nn.weight_norm(linear)
        >>> x = mx.random.normal((8, 20))
        >>> output = weight_norm_linear(x)

    References:
        [1]: https://arxiv.org/abs/1602.07868
    """

    def __init__(self, module, name="weight", dim=0):
        super().__init__()
        self.module = module
        self.wn_name = name
        params = module.parameters()
        if name not in params:
            raise ValueError(f"Parameter '{name}' not found in module")
        mx.eval(params)
        weight = params[name]
        self.weight_v = mx.array(weight)
        self.wn_module_type = type(module).__name__

        if dim is None:
            self.weight_g = mx.linalg.norm(weight)
            self.wn_axes = []
        else:
            dim = dim if dim >= 0 else weight.ndim + dim
            if dim < 0 or dim >= weight.ndim:
                raise ValueError(
                    f"dim {dim} out of bounds for {weight.ndim} dimensions"
                )
            axes = [i for i in range(weight.ndim) if i != dim]
            if len(axes) > 2:
                reshape_dims = [weight.shape[dim], -1]
                weight_reshaped = mx.reshape(weight, reshape_dims)
                self.weight_g = mx.linalg.norm(weight_reshaped, axis=1, keepdims=True)
                g_shape = [1] * weight.ndim
                g_shape[dim] = weight.shape[dim]
                self.weight_g = mx.reshape(self.weight_g, g_shape)
            elif "Conv" in self.wn_module_type and dim == 0:
                weight_flat = mx.reshape(weight, (weight.shape[0], -1))
                self.weight_g = mx.linalg.norm(weight_flat, axis=1, keepdims=True)
                g_shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
                self.weight_g = mx.reshape(self.weight_g, g_shape)
            else:
                self.weight_g = mx.linalg.norm(weight, axis=tuple(axes), keepdims=True)
            self.wn_axes = axes
        self.wn_dim = dim

    def __call__(self, *args, **kwargs):
        """Apply weight normalization to the wrapped module and then call it."""
        normalized_weight = mx.weight_norm(
            self.weight_v,
            self.weight_g,
            axes=None if self.wn_axes == [] else self.wn_axes,
            eps=1e-5,
        )
        setattr(self.module, self.wn_name, normalized_weight)
        return self.module(*args, **kwargs)


def weight_norm(module, name="weight", dim=0):
    """Apply weight normalization to a module.

    This is a convenience function that wraps the module with WeightNormWrapper.

    Args:
        module (Module): The module to apply weight normalization to.
        name (str): The name of the parameter to normalize. Default: ``"weight"``.
        dim (int): The dimension along which to normalize. Default: ``0``.

    Returns:
        A WeightNormWrapper instance which wraps the module.
    """
    return WeightNormWrapper(module, name, dim)


class WeightNormConv1d:
    r"""Applies a 1D convolution with weight normalization over an input signal.

    This module is a convenience wrapper that combines Conv1d with weight normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: ``1``.
        padding (int): Zero-padding added to both sides of the input. Default: ``0``.
        dilation (int): Spacing between kernel elements. Default: ``1``.
        groups (int): Number of blocked connections from input to output channels. Default: ``1``.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        dim (int): Dimension along which to normalize weights. Default: ``0``.

    Returns:
        A Conv1d module with weight normalization applied.
    """

    def __new__(
        cls,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        dim=0,
    ):
        from mlx.nn import Conv1d

        conv = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        return weight_norm(conv, "weight", dim)


class WeightNormConv2d:
    r"""Applies a 2D convolution with weight normalization over an input signal.

    This module is a convenience wrapper that combines Conv2d with weight normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple): Stride of the convolution. Default: ``1``.
        padding (int or tuple): Zero-padding added to both sides of the input. Default: ``0``.
        dilation (int or tuple): Spacing between kernel elements. Default: ``1``.
        groups (int): Number of blocked connections from input to output channels. Default: ``1``.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        dim (int): Dimension along which to normalize weights. Default: ``0``.

    Returns:
        A Conv2d module with weight normalization applied.
    """

    def __new__(
        cls,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        dim=0,
    ):
        from mlx.nn import Conv2d

        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        return weight_norm(conv, "weight", dim)


class WeightNormLinear:
    r"""Applies a linear transformation with weight normalization to the incoming data.

    This module is a convenience wrapper that combines Linear with weight normalization.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        dim (int): Dimension along which to normalize weights. Default: ``0``.

    Returns:
        A Linear module with weight normalization applied.
    """

    def __new__(cls, in_features, out_features, bias=True, dim=0):
        from mlx.nn import Linear

        linear = Linear(in_features, out_features, bias)
        return weight_norm(linear, "weight", dim)
