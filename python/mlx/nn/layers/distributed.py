# Copyright © 2024 Apple Inc.

from functools import lru_cache
from typing import Optional

import mlx.core as mx
from mlx.nn.layers.base import Module


@lru_cache
def sum_gradients(group):
    if group.size() == 1:
        return lambda x: x

    @mx.custom_function
    def f(x):
        return x

    @f.vjp
    def f(x, dx, _):
        return mx.distributed.all_sum(dx, group=group)

    return f


class AllToShardedLinear(Module):
    """Each member of the group applies part of the affine transformation such
    that the result is sharded across the group.

    The gradients are automatically aggregated from each member of the group.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` the the layer will not use a
            bias. Default is ``True``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Optional[mx.distributed.Group] = None,
    ):
        super().__init__()

        # Initialize the parameters
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        if (output_dims % N) != 0:
            raise ValueError(
                f"Cannot shard the output of size {output_dims} across {N} devices."
            )

        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims // N, input_dims),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims // N,),
            )

    def _extra_repr(self) -> str:
        N = self.group.size()
        return f"input_dims={self.weight.shape[1]}, output_dims={N * self.weight.shape[0]}, bias={'bias' in self}"

    def __call__(self, x: mx.array) -> mx.array:
        # Aggregate the gradients coming from each shard
        if self.group.size() > 1:
            x = sum_gradients(self.group)(x)

        # Compute the affine projection
        if "bias" in self:
            x = mx.addmm(self["bias"], x, self["weight"].T)
        else:
            x = x @ self["weight"].T
        return x


class ShardedToAllLinear(Module):
    """Each member of the group applies part of the affine transformation and
    then aggregates the results.

    All nodes will have the same exact result after this layer.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` the the layer will not use a
            bias. Default is ``True``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Optional[mx.distributed.Group] = None,
    ):
        super().__init__()

        # Initialize the parameters
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        if (input_dims % N) != 0:
            raise ValueError(
                f"The input of size {input_dims} cannot be sharded across {N} devices."
            )

        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims // N),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims,),
            )

    def _extra_repr(self) -> str:
        N = self.group.size()
        return f"input_dims={N * self.weight.shape[1]}, output_dims={self.weight.shape[0]}, bias={'bias' in self}"

    def __call__(self, x: mx.array) -> mx.array:
        if self.group.size() > 1:
            # Perform the local projection and aggregate the results
            x = x @ self["weight"].T
            x = mx.distributed.all_sum(x, group=group)

            # Add the bias if we have one
            if "bias" in self:
                x = x + self["bias"]
        else:
            # Normal linear layer as we are not in a distributed setting.
            if "bias" in self:
                x = mx.addmm(self["bias"], x, self["weight"].T)
            else:
                x = x @ self["weight"].T
        return x
