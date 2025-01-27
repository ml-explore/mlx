# Copyright © 2023-2024 Apple Inc.

import math
from typing import Callable, Literal

import mlx.core as mx


def constant(
    value: float, dtype: mx.Dtype = mx.float32
) -> Callable[[mx.array], mx.array]:
    r"""An initializer that returns an array filled with ``value``.

    Args:
        value (float): The value to fill the array with.
        dtype (Dtype, optional): The data type of the array. Default:
          ``float32``.

    Returns:
        Callable[[array], array]: An initializer that returns an array with the
        same shape as the input, filled with ``value``.

    Example:

        >>> init_fn = nn.init.constant(0.5)
        >>> init_fn(mx.zeros((2, 2)))
        array([[0.5, 0.5],
               [0.5, 0.5]], dtype=float32)
    """

    def initializer(a: mx.array) -> mx.array:
        return mx.full(a.shape, value, dtype=dtype)

    return initializer


def normal(
    mean: float = 0.0, std: float = 1.0, dtype: mx.Dtype = mx.float32
) -> Callable[[mx.array], mx.array]:
    r"""An initializer that returns samples from a normal distribution.

    Args:
        mean (float, optional): Mean of the normal distribution. Default:
          ``0.0``.
        std (float, optional): Standard deviation of the normal distribution.
          Default: ``1.0``.
        dtype (Dtype, optional): The data type of the array. Default:
          ``float32``.

    Returns:
        Callable[[array], array]: An initializer that returns an array with the
        same shape as the input, filled with samples from a normal distribution.

    Example:

        >>> init_fn = nn.init.normal()
        >>> init_fn(mx.zeros((2, 2)))
        array([[-0.982273, -0.534422],
               [0.380709, 0.0645099]], dtype=float32)
    """

    def initializer(a: mx.array) -> mx.array:
        return mx.random.normal(shape=a.shape, scale=std, loc=mean, dtype=dtype)

    return initializer


def uniform(
    low: float = 0.0, high: float = 1.0, dtype: mx.Dtype = mx.float32
) -> Callable[[mx.array], mx.array]:
    r"""An initializer that returns samples from a uniform distribution.

    Args:
        low (float, optional): The lower bound of the uniform distribution.
          Default: ``0.0``.
        high (float, optional): The upper bound of the uniform distribution.
          Default: ``1.0``
        dtype (Dtype, optional): The data type of the array. Default: ``float32``.

    Returns:
        Callable[[array], array]: An initializer that returns an array
        with the same shape as the input, filled with samples from a uniform
        distribution

    Example:

        >>> init_fn = nn.init.uniform(low=0, high=1)
        >>> init_fn(mx.zeros((2, 2)))
        array([[0.883935, 0.863726],
               [0.617261, 0.417497]], dtype=float32)
    """

    def initializer(a: mx.array) -> mx.array:
        return mx.random.uniform(low, high, a.shape, dtype=dtype)

    return initializer


def identity(dtype: mx.Dtype = mx.float32) -> Callable[[mx.array], mx.array]:
    r"""An initializer that returns an identity matrix.

    Args:
        dtype (Dtype, optional): The data type of the array. Defaults:
          ``float32``.

    Returns:
        Callable[[array], array]: An initializer that returns an identity
        matrix with the same shape as the input.

    Example:

        >>> init_fn = nn.init.identity()
        >>> init_fn(mx.zeros((2, 2)))
        array([[1, 0],
               [0, 1]], dtype=float32)
    """

    def initializer(arr: mx.array) -> mx.array:
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"The input array must be a square matrix but got shape {arr.shape}."
            )
        return mx.eye(n=arr.shape[0], dtype=dtype)

    return initializer


def _calculate_fan_in_fan_out(x):
    if x.ndim < 2:
        raise ValueError(
            "Glorot / He initialization requires at least 2 dimensional input"
            f" but input with {x.ndim} dimensions."
        )

    fan_in = x.shape[-1]
    fan_out = x.shape[0]

    if x.ndim > 2:
        receptive_field = 1
        for d in x.shape[1:-1]:
            receptive_field *= d

        fan_in = fan_in * receptive_field
        fan_out = fan_out * receptive_field

    return fan_in, fan_out


def glorot_normal(
    dtype: mx.Dtype = mx.float32,
) -> Callable[[mx.array, float], mx.array]:
    r"""A Glorot normal initializer.

    This initializer samples from a normal distribution with a standard
    deviation computed from the number of input (``fan_in``) and output
    (``fan_out``) units according to:

    .. math::
        \sigma = \gamma \sqrt{\frac{2.0}{\text{fan\_in} + \text{fan\_out}}}

    For more details see the original reference: `Understanding the difficulty
    of training deep feedforward neural networks
    <https://proceedings.mlr.press/v9/glorot10a.html>`_

    Args:
        dtype (Dtype, optional): The data type of the array. Default: ``float32``.

    Returns:
        Callable[[array, float], array]: An initializer that returns an array
        with the same shape as the input, filled with samples from the Glorot
        normal distribution.

    Example:

        >>> init_fn = nn.init.glorot_normal()
        >>> init_fn(mx.zeros((2, 2)))
        array([[0.191107, 1.61278],
               [-0.150594, -0.363207]], dtype=float32)
        >>> init_fn(mx.zeros((2, 2)), gain=4.0)
        array([[1.89613, -4.53947],
               [4.48095, 0.995016]], dtype=float32)
    """

    def initializer(a: mx.array, gain: float = 1.0) -> mx.array:
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        return mx.random.normal(shape=a.shape, scale=std, dtype=dtype)

    return initializer


def glorot_uniform(
    dtype: mx.Dtype = mx.float32,
) -> Callable[[mx.array, float], mx.array]:
    r"""A Glorot uniform initializer.

    This initializer samples from a uniform distribution with a range
    computed from the number of input (``fan_in``) and output (``fan_out``)
    units according to:

    .. math::
        \sigma = \gamma \sqrt{\frac{6.0}{\text{fan\_in} + \text{fan\_out}}}

    For more details see the original reference: `Understanding the difficulty
    of training deep feedforward neural networks
    <https://proceedings.mlr.press/v9/glorot10a.html>`_

    Args:
        dtype (Dtype, optional): The data type of the array. Default: ``float32``.

    Returns:
        Callable[[array, float], array]: An initializer that returns an array
        with the same shape as the input, filled with samples from the Glorot
        uniform distribution.

    Example:

        >>> init_fn = nn.init.glorot_uniform()
        >>> init_fn(mx.zeros((2, 2)))
        array([[0.223404, -0.890597],
               [-0.379159, -0.776856]], dtype=float32)
        >>> init_fn(mx.zeros((2, 2)), gain=4.0)
        array([[-1.90041, 3.02264],
               [-0.912766, 4.12451]], dtype=float32)
    """

    def initializer(a: mx.array, gain: float = 1.0) -> mx.array:
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
        return mx.random.uniform(-limit, limit, a.shape, dtype=dtype)

    return initializer


def he_normal(
    dtype: mx.Dtype = mx.float32,
) -> Callable[[mx.array, Literal["fan_in", "fan_out"], float], mx.array]:
    r"""Build a He normal initializer.

    This initializer samples from a normal distribution with a standard
    deviation computed from the number of input (``fan_in``) or output
    (``fan_out``) units according to:

    .. math::
        \sigma = \gamma \frac{1}{\sqrt{\text{fan}}}

    where :math:`\text{fan}` is either the number of input units when the
    ``mode`` is ``"fan_in"`` or output units when the ``mode`` is
    ``"fan_out"``.

    For more details see the original reference: `Delving Deep into Rectifiers:
    Surpassing Human-Level Performance on ImageNet Classification
    <https://arxiv.org/abs/1502.01852>`_

    Args:
        dtype (Dtype, optional): The data type of the array. Defaults to mx.float32.

    Returns:
        Callable[[array, str, float], array]: An initializer that returns an
        array with the same shape as the input, filled with samples from the He
        normal distribution.

    Example:

        >>> init_fn = nn.init.he_normal()
        >>> init_fn(mx.zeros((2, 2)))  # uses fan_in
        array([[-1.25211, 0.458835],
               [-0.177208, -0.0137595]], dtype=float32)
        >>> init_fn(mx.zeros((2, 2)), mode="fan_out", gain=5)
        array([[5.6967, 4.02765],
               [-4.15268, -2.75787]], dtype=float32)
    """

    def initializer(
        a: mx.array,
        mode: Literal["fan_in", "fan_out"] = "fan_in",
        gain: float = 1.0,
    ) -> mx.array:
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        if mode == "fan_in":
            fan = fan_in
        elif mode == "fan_out":
            fan = fan_out
        else:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: fan_in, fan_out")

        std = gain / math.sqrt(fan)
        return mx.random.normal(shape=a.shape, scale=std, dtype=dtype)

    return initializer


def he_uniform(
    dtype: mx.Dtype = mx.float32,
) -> Callable[[mx.array, Literal["fan_in", "fan_out"], float], mx.array]:
    r"""A He uniform (Kaiming uniform) initializer.

    This initializer samples from a uniform distribution with a range
    computed from the number of input (``fan_in``) or output (``fan_out``)
    units according to:

    .. math::

        \sigma = \gamma \sqrt{\frac{3.0}{\text{fan}}}

    where :math:`\text{fan}` is either the number of input units when the
    ``mode`` is ``"fan_in"`` or output units when the ``mode`` is
    ``"fan_out"``.

    For more details see the original reference: `Delving Deep into Rectifiers:
    Surpassing Human-Level Performance on ImageNet Classification
    <https://arxiv.org/abs/1502.01852>`_


    Args:
        dtype (Dtype, optional): The data type of the array. Default: ``float32``.

    Returns:
        Callable[[array, str, float], array]: An initializer that returns an
        array with the same shape as the input, filled with samples from  the
        He uniform distribution.

    Example:

        >>> init_fn = nn.init.he_uniform()
        >>> init_fn(mx.zeros((2, 2)))  # uses fan_in
        array([[0.0300242, -0.0184009],
               [0.793615, 0.666329]], dtype=float32)
        >>> init_fn(mx.zeros((2, 2)), mode="fan_out", gain=5)
        array([[-1.64331, -2.16506],
               [1.08619, 5.79854]], dtype=float32)
    """

    def initializer(
        a: mx.array,
        mode: Literal["fan_in", "fan_out"] = "fan_in",
        gain: float = 1.0,
    ) -> mx.array:
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        if mode == "fan_in":
            fan = fan_in
        elif mode == "fan_out":
            fan = fan_out
        else:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: fan_in, fan_out")

        limit = gain * math.sqrt(3.0 / fan)
        return mx.random.uniform(-limit, limit, a.shape, dtype=dtype)

    return initializer


def sparse(
    sparsity: float,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: mx.Dtype = mx.float32,
) -> Callable[[mx.array], mx.array]:
    r"""An initializer that returns a sparse matrix.

    Args:
        sparsity (float): The fraction of elements in each column to be set to
        zero.
        mean (float, optional): Mean of the normal distribution. Default:
          ``0.0``.
        std (float, optional): Standard deviation of the normal distribution.
          Default: ``1.0``.
        dtype (Dtype, optional): The data type of the array. Default:
          ``float32``.

    Returns:
        Callable[[array], array]: An initializer that returns an array with the
        same shape as the input, filled with samples from a normal distribution.

    Example:

        >>> init_fn = nn.init.sparse(sparsity=0.5)
        >>> init_fn(mx.zeros((2, 2)))
        array([[-1.91187, -0.117483],
       [0, 0]], dtype=float32)
    """

    def initializer(a: mx.array) -> mx.array:
        if a.ndim != 2:
            raise ValueError("Only tensors with 2 dimensions are supported")

        rows, cols = a.shape
        num_zeros = int(math.ceil(sparsity * cols))

        order = mx.argsort(mx.random.uniform(shape=a.shape), axis=1)
        a = mx.random.normal(shape=a.shape, scale=std, loc=mean, dtype=dtype)

        a[mx.arange(rows).reshape(rows, 1), order[:, :num_zeros]] = 0

        return a

    return initializer


def orthogonal(
    gain: float = 1.0, dtype: mx.Dtype = mx.float32
) -> Callable[[mx.array], mx.array]:
    r"""An initializer that returns an orthogonal matrix.

    Args:
        gain (float, optional): Scaling factor for the orthogonal matrix.
            Default: ``1.0``.
        dtype (Dtype, optional): Data type of the array. Default: ``float32``.

    Returns:
        Callable[[array], array]: An initializer that returns
        an orthogonal matrix with the same shape as the input.
    """

    def initializer(a: mx.array) -> mx.array:
        if a.ndim != 2:
            raise ValueError(
                f"Orthogonal initialization requires a 2D array but got"
                " a {a.ndim}D array."
            )

        rows, cols = a.shape
        n = max(rows, cols)

        rmat = mx.random.normal(shape=(n, n))

        # Perform QR decomposition on CPU
        q, r = mx.linalg.qr(rmat, stream=mx.cpu)

        # Adjust the sign of Q using the diagonal of R
        d = mx.diag(r)
        q = q * mx.sign(d)

        # Slice Q to the desired shape
        q = q[:rows, :cols]

        # Scale Q by gain
        q = q * gain
        return q.astype(dtype)

    return initializer
