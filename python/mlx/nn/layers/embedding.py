# Copyright © 2023-2024 Apple Inc.

import math

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.quantized import QuantizedEmbedding


class Embedding(Module):
    """Implements a simple lookup table that maps each input integer to a
    high-dimensional vector.

    Typically used to embed discrete tokens for processing by neural networks.

    Args:
        num_embeddings (int): How many possible discrete tokens can we embed.
           Usually called the vocabulary size.
        dims (int): The dimensionality of the embeddings.
    """

    def __init__(self, num_embeddings: int, dims: int):
        super().__init__()
        scale = math.sqrt(1 / dims)
        self.weight = mx.random.normal(shape=(num_embeddings, dims), scale=scale)

    def _extra_repr(self):
        return f"{self.weight.shape[0]}, {self.weight.shape[1]}"

    def __call__(self, x):
        return self.weight[x]

    def as_linear(self, x):
        """
        Call the embedding layer as a linear layer.

        Use this for example when input embedding and output projection
        weights are tied.
        """
        return x @ self.weight.T

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        """Return a :obj:`QuantizedEmbedding` layer that approximates this embedding layer."""
        return QuantizedEmbedding.from_embedding(self, group_size, bits, mode)
