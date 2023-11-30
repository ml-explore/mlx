# Copyright © 2023 Apple Inc.

from mlx.nn.layers.base import Module
from mlx.nn.layers.activations import (
    GELU,
    ReLU,
    SiLU,
    gelu,
    gelu_approx,
    gelu_fast_approx,
    relu,
    silu,
)
from mlx.nn.layers.containers import Sequential
from mlx.nn.layers.convolution import Conv1d, Conv2d
from mlx.nn.layers.dropout import Dropout
from mlx.nn.layers.embedding import Embedding
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.normalization import GroupNorm, LayerNorm, RMSNorm
from mlx.nn.layers.positional_encoding import RoPE, SinusoidalPositionalEncoding
from mlx.nn.layers.transformer import (
    MultiHeadAttention,
    TransformerEncoder,
    TransformerEncoderLayer,
)
