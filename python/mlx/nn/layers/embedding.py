# Copyright © 2023 Apple Inc.

import math
from typing import Optional

import mlx.core as mx
from mlx.nn.layers.base import Module


class Embedding(Module):
    """Implements a simple lookup table that maps each input integer to a
    high-dimensional vector.

    Typically used to embed discrete tokens for processing by neural networks.

    Args:
        num_embeddings (int): How many possible discrete tokens can we embed.
                              Usually called the vocabulary size.
        dims (int): The dimensionality of the embeddings.
    """

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        _weight: Optional[mx.array],
        padding_idx: Optional[int],
        freeze: Optional[bool],
    ):
        super().__init__()
        scale = math.sqrt(1 / dims)
        if _weight is None:
            self.weight = mx.random.normal((num_embeddings, dims)) * scale
        else:
            assert _weight.shape == [
                num_embeddings,
                dims,
            ], "Shape of weight does not match num_embeddings and dims"
            self.weight = _weight

        if freeze:
            self.weight.freeze()

        if padding_idx is not None:
            self.weight[padding_idx] = 0

    def _extra_repr(self):
        return f"{self.weight.shape[0]}, {self.weight.shape[1]}"

    def __call__(self, x):
        return self.weight[x]
