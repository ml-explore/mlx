# weight_norm.py
# Copyright © 2025 Apple Inc.
import mlx.core as mx


def weight_norm(module, name="weight", dim=0):
    """Apply weight normalization to a module's parameter.

    Weight normalization is a reparameterization that decouples the magnitude of a weight tensor
    from its direction:

        w = g * v / ||v||

    This is achieved by computing normalized weights on-the-fly. In this implementation,
    we store both the unnormalized parameter 'v' and the magnitude parameter 'g'.

    Args:
        module: The module to modify (e.g., Conv1d, Linear).
        name (str, optional): Parameter name to normalize. Default: 'weight'.
        dim (int, optional): Dimension to keep; norm over others. Default: 0.
                            Use None for norm over entire tensor.

    Returns:
        The modified module.

    Example:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> from mlx.nn.layers.weight_norm import weight_norm
        >>>
        >>> # Apply to Linear layer
        >>> linear = nn.Linear(10, 20)
        >>> linear_wn = weight_norm(linear)
        >>>
        >>> # Apply to Conv1d layer
        >>> conv1d = nn.Conv1d(16, 32, kernel_size=3)
        >>> conv1d_wn = weight_norm(conv1d)
    """
    params = module.parameters()
    if name not in params:
        raise ValueError(f"Parameter '{name}' not found in module")

    # Ensure parameters are materialized
    mx.eval(params)

    weight = params[name]
    v = mx.array(weight)

    # Store original module type for dimension handling
    module_type = type(module).__name__

    # Compute initial g
    if dim is None:
        g = mx.linalg.norm(weight)  # Scalar
    else:
        # Handle dimension ordering differences for Conv layers
        if "Conv" in module_type:
            # For Conv layers, special handling needed
            # MLX Conv1d weight shape: [out_channels, kernel_size, in_channels]
            # MLX Conv2d weight shape: [out_channels, kernel_height, kernel_width, in_channels]
            if dim == 0:
                # For Conv1d and Conv2d, flatten all dimensions except out_channels
                weight_flat = mx.reshape(weight, (weight.shape[0], -1))
                g = mx.linalg.norm(weight_flat, axis=1, keepdims=True)
                # Reshape g to match the output channels dimension with singleton dimensions for the rest
                g_shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
                g = mx.reshape(g, g_shape)
            else:
                dim = dim if dim >= 0 else weight.ndim + dim
                if dim < 0 or dim >= weight.ndim:
                    raise ValueError(
                        f"dim {dim} out of bounds for {weight.ndim} dimensions"
                    )
                # For dimensions other than 0, use a single axis for normalization
                axes = tuple(i for i in range(weight.ndim) if i != dim)
                # Handle multiple axes if needed
                if len(axes) > 2:
                    # Use reshape workaround for >2 axes
                    weight_flat = mx.reshape(weight, (weight.shape[0], -1))
                    g = mx.linalg.norm(weight_flat, axis=1, keepdims=True)
                    g_shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
                    g = mx.reshape(g, g_shape)
                else:
                    g = mx.linalg.norm(weight, axis=axes, keepdims=True)
        else:
            # Standard handling for other layer types
            dim = dim if dim >= 0 else weight.ndim + dim
            if dim < 0 or dim >= weight.ndim:
                raise ValueError(
                    f"dim {dim} out of bounds for {weight.ndim} dimensions"
                )
            axes = tuple(i for i in range(weight.ndim) if i != dim)
            g = mx.linalg.norm(weight, axis=axes, keepdims=True)

    # Store parameters on module
    module.v = v
    module.g = g
    module.wn_dim = dim
    module.wn_name = name
    module.wn_module_type = module_type

    # Override __call__ method to apply weight normalization
    original_call = module.__call__

    def weight_norm_call(*args, **kwargs):
        # Update weight before calling the original function
        params = module.parameters()

        if module.wn_dim is None:
            v_norm = mx.linalg.norm(module.v)
        else:
            # Use dimension handling logic based on module type
            if "Conv" in module.wn_module_type:
                # Special handling for Conv layers based on their dimension structure
                v_flat = mx.reshape(module.v, (module.v.shape[0], -1))
                v_norm = mx.linalg.norm(v_flat, axis=1, keepdims=True)
                # Reshape back to match the original shape with singleton dimensions
                v_norm_shape = [module.v.shape[0]] + [1] * (module.v.ndim - 1)
                v_norm = mx.reshape(v_norm, v_norm_shape)
            else:
                axes = tuple(i for i in range(module.v.ndim) if i != module.wn_dim)
                if len(axes) > 2:
                    # Handle multiple axes with reshape approach
                    v_flat = mx.reshape(module.v, (module.v.shape[0], -1))
                    v_norm = mx.linalg.norm(v_flat, axis=1, keepdims=True)
                    v_norm_shape = [module.v.shape[0]] + [1] * (module.v.ndim - 1)
                    v_norm = mx.reshape(v_norm, v_norm_shape)
                else:
                    v_norm = mx.linalg.norm(module.v, axis=axes, keepdims=True)

        # Compute normalized weight: g * v / ||v||
        normalized_weight = module.g * module.v / mx.maximum(v_norm, 1e-5)

        # Update the weight parameter
        params[name] = normalized_weight

        # Now call the original method
        return original_call(*args, **kwargs)

    # Replace the __call__ method
    module.__call__ = weight_norm_call

    return module


class WeightNormConv1d:
    """1D convolution with weight normalization.

    Weight normalization is a reparameterization technique that decouples the magnitude
    of a weight tensor from its direction. This class applies weight normalization
    to a Conv1d layer.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input to output channels. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        dim (int, optional): Dimension to keep; norm over others. Default: 0

    Note:
        Due to dimension ordering differences between PyTorch and MLX,
        this implementation properly handles normalization for MLX's Conv1d
        weight shape: [out_channels, kernel_size, in_channels]
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
    """2D convolution with weight normalization.

    Weight normalization is a reparameterization technique that decouples the magnitude
    of a weight tensor from its direction. This class applies weight normalization
    to a Conv2d layer.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input to output channels. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        dim (int, optional): Dimension to keep; norm over others. Default: 0

    Note:
        Due to dimension ordering differences between PyTorch and MLX,
        this implementation properly handles normalization for MLX's Conv2d
        weight shape: [out_channels, kernel_height, kernel_width, in_channels]
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
    """Linear layer with weight normalization.

    Weight normalization is a reparameterization technique that decouples the magnitude
    of a weight tensor from its direction. This class applies weight normalization
    to a Linear layer.

    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        dim (int, optional): Dimension to keep; norm over others. Default: 0
    """

    def __new__(cls, in_features, out_features, bias=True, dim=0):
        from mlx.nn import Linear

        linear = Linear(in_features, out_features, bias)
        return weight_norm(linear, "weight", dim)
