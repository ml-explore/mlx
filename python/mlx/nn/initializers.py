# Copyright © 2023 Apple Inc.

import mlx.core as mx
from typing import Union, List

def constant(value: float, dtype: mx.Dtype = mx.float32) -> mx.array:
    r"""Build an initializer that returns an array filled with 'value'.

    Args:
        value (float): The value to fill the array with.
    """
    def initializer(shape: Union[int, List[int]]) -> mx.array:
        return mx.full(shape, value, dtype=dtype)
    return initializer

def normal(mean: float = 0.0, std: float = 1.0, dtype: mx.Dtype = mx.float32) -> mx.array:
    r"""Build an initializer that returns an array with random values from a normal distribution.

    Args:
        dtype (Dtype): The data type of the array.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
    """
    def initializer(shape: List[int]) -> mx.array:
        standard_normal = mx.random.normal(shape=shape, dtype=dtype)
        return standard_normal * std + mean
    return initializer

def uniform(low: float = 0.0, high: float = 1.0, dtype: mx.Dtype = mx.float32) -> mx.array:
    r"""Build an initializer that returns an array with random values from a uniform distribution.

    Args:
        low (float): The lower bound of the uniform distribution.
        high (float): The upper bound of the uniform distribution.
        dtype (Dtype): The data type of the array.
    """
    def initializer(shape: List[int]) -> mx.array:
        return mx.random.uniform(low, high, shape, dtype=dtype)
    return initializer




