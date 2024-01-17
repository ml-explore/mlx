# Copyright © 2023 Apple Inc.

from typing import Callable, Literal

import mlx.core as mx


def constant(
    value: float, dtype: mx.Dtype = mx.float32
) -> Callable[[mx.array], mx.array]:
    r"""Build an initializer that returns a array filled with 'value'.

    Args:
        value (float): The value to fill the array with.
        dtype (mx.Dtype, optional): The data type of the array. Defaults to mx.float32.

    Returns:
        Callable[[mx.array], mx.array]: An initializer that returns an array, of the same
                                        shape as the input array, filled with 'value'.
    """

    def initializer(a: mx.array) -> mx.array:
        return mx.full(a.shape, value, dtype=dtype)

    return initializer


def normal(
    mean: float = 0.0, std: float = 1.0, dtype: mx.Dtype = mx.float32
) -> Callable[[mx.array], mx.array]:
    r"""Build an initializer that returns random values from a normal distribution.

    Args:
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        dtype (Dtype): The data type of the array.

    Returns:
        Callable[[mx.array], mx.array]: An initializer that returns an array, of the same
                                        shape as the input array, filled with values drawn
                                        from the normal distribution.
    """

    def initializer(a: mx.array) -> mx.array:
        standard_normal = mx.random.normal(shape=a.shape, dtype=dtype)
        return standard_normal * std + mean

    return initializer


def uniform(
    low: float = 0.0, high: float = 1.0, dtype: mx.Dtype = mx.float32
) -> Callable[[mx.array], mx.array]:
    r"""Build an initializer that returns random values from a uniform distribution.

    Args:
        low (float): The lower bound of the uniform distribution.
        high (float): The upper bound of the uniform distribution.
        dtype (Dtype): The data type of the array.

    Returns:
        Callable[[mx.array], mx.array]: An initializer that returns an array, of the same
                                        shape as the input array, filled with values drawn
                                        from the uniform distribution
    """

    def initializer(a: mx.array) -> mx.array:
        return mx.random.uniform(low, high, a.shape, dtype=dtype)

    return initializer


def identity(dtype: mx.Dtype = mx.float32) -> Callable[[mx.array], mx.array]:
    r"""Build an initializer that returns an identity matrix.

    Args:
        dtype (Dtype, optional): The data type of the array. Defaults to mx.float32.

    Returns:
        Callable[[mx.array], mx.array]: An initializer that returns an identity array, of
                                    the same shape as the input array.
    """

    def initializer(arr: mx.array) -> mx.array:
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(
                "The input array must be square (same number of rows and columns)."
            )
        return mx.core.eye(n=arr.shape[0], dtype=dtype)

    return initializer


def _calculate_fan_in_fan_out(x):
    if x.ndim < 2:
        raise ValueError("Glorot initialization requires at least 2 dimensional input")

    fan_in = x.shape[0]
    fan_out = x.shape[1]
    receptive_field = 1

    if x.ndim > 2:
        for d in x.shape[2:]:
            receptive_field *= d

        fan_in = fan_in * receptive_field
        fan_out = fan_out * receptive_field

    return fan_in, fan_out


def glorot_normal(
    dtype: mx.Dtype = mx.float32,
) -> Callable[[mx.array, mx.float32], mx.array]:
    r"""Build a Xavier Glorot normal initializer.

    This initializer generates values from a normal distribution centered around 0.
    The standard deviation is calculated based on the number of input (`fan_in`) and
    output (`fan_out`) units in the weight tensor. The method is described in
    `Understanding the difficulty of training deep feedforward neural networks`
    - Glorot, X. & Bengio, Y. (2010).

    Args:
        dtype (mx.Dtype, optional): The data type of the array. Defaults to mx.float32.

    Returns:
        Callable[[mx.array, mx.float32], mx.array]: An initializer that returns an array, of the same
                                    shape as the input array, filled with random values from
                                    the Glorot normal distribution.
    """

    def initializer(a: mx.array, gain: mx.float32 = 1.0) -> mx.array:
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        std = gain * mx.sqrt(mx.array(2.0 / (fan_in + fan_out)))
        return mx.random.normal(shape=a.shape, dtype=dtype) * std

    return initializer


def glorot_uniform(
    dtype: mx.Dtype = mx.float32,
) -> Callable[[mx.array, mx.float32], mx.array]:
    r"""Build a Xavier Glorot uniform initializer.

    This initializer generates values from a uniform distribution within a range
    determined by the number of input (`fan_in`) and output (`fan_out`) units in the
    weight tensor. The method is described in  `Understanding the difficulty of training
    deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010).

    Args:
        dtype (mx.Dtype, optional): The data type of the array. Defaults to mx.float32.

    Returns:
        Callable[[mx.array, mx.array], mx.array]: An initializer that returns an array, of the same
                                    shape as the input array, filled with random values from
                                    the Glorot uniform distribution.
    """

    def initializer(a: mx.array, gain: mx.float32 = 1.0) -> mx.array:
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        limit = gain * mx.sqrt(mx.array(6.0 / (fan_in + fan_out)))
        return mx.random.uniform(-limit, limit, a.shape, dtype=dtype)

    return initializer


def he_normal(
    dtype: mx.Dtype = mx.float32,
) -> Callable[[mx.array, str, mx.float32], mx.array]:
    r"""Build a Kaming He normal initializer.

    This initializer generates values from a normal distribution centered around 0,
    with a standard deviation calculated based on the number of input (`fan_in`) or
    output (`fan_out`) units in the weight tensor. The method is described in `Delving
    deep into rectifiers: Surpassing human-level performance on ImageNet classification`
    - He, K. et al. (2015).

    Args:
        dtype (mx.Dtype, optional): The data type of the array. Defaults to mx.float32.

    Returns:
        Callable[[mx.array, str, mx.float32], mx.array]: An initializer that returns an array, of the same
                                        shape as the input array, filled with random values from
                                        the He normal distribution.
    """

    def initializer(
        a: mx.array,
        mode: Literal["fan_in", "fan_out"] = "fan_in",
        gain: mx.float32 = 1.0,
    ) -> mx.array:
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        if mode == "fan_in":
            fan = fan_in
        elif mode == "fan_out":
            fan = fan_out
        else:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: fan_in, fan_out")

        std = gain / mx.sqrt(mx.array(fan))
        return mx.random.normal(shape=a.shape, dtype=dtype) * std

    return initializer


def he_uniform(
    dtype: mx.Dtype = mx.float32,
) -> Callable[[mx.array, str, mx.float32], mx.array]:
    r"""Create a He uniform (Kaiming uniform) initializer.

    This initializer generates values from a uniform distribution within a range
    determined by the number of input (`fan_in`) or output (`fan_out`) units in the
    weight tensor. The method is described in `Delving deep into rectifiers: Surpassing
    human-level performance on ImageNet classification` - He, K. et al. (2015).

    Args:
        dtype (mx.Dtype, optional): The data type of the array. Defaults to mx.float32.

    Returns:
        Callable[[mx.array, str, mx.float32], mx.array]: An initializer that returns an array, of the same
                                        shape as the input array, filled with random values from
                                        the He uniform distribution.
    """

    def initializer(
        a: mx.array,
        mode: Literal["fan_in", "fan_out"] = "fan_in",
        gain: mx.float32 = 1.0,
    ) -> mx.array:
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        if mode == "fan_in":
            fan = fan_in
        elif mode == "fan_out":
            fan = fan_out
        else:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: fan_in, fan_out")

        limit = gain * mx.sqrt(mx.array(3.0 / fan))
        return mx.random.uniform(-limit, limit, a.shape, dtype=dtype)

    return initializer