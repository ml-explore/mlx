# Copyright © 2023 Apple Inc.

from mlx.nn.layers.activations import (
    CELU,
    ELU,
    GELU,
    GLU,
    SELU,
    Hardswish,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    PReLU,
    ReLU,
    ReLU6,
    Sigmoid,
    SiLU,
    Softmax,
    Softplus,
    Softsign,
    Step,
    Tanh,
    celu,
    elu,
    gelu,
    gelu_approx,
    gelu_fast_approx,
    glu,
    hardswish,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    mish,
    prelu,
    relu,
    relu6,
    selu,
    silu,
    softmax,
    softplus,
    softsign,
    step,
    tanh,
)
from mlx.nn.layers.base import Module
from mlx.nn.layers.containers import Sequential
from mlx.nn.layers.convolution import Conv1d, Conv2d
from mlx.nn.layers.dropout import Dropout, Dropout2d, Dropout3d
from mlx.nn.layers.embedding import Embedding
from mlx.nn.layers.linear import Bilinear, Identity, Linear
from mlx.nn.layers.normalization import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    RMSNorm,
)
from mlx.nn.layers.positional_encoding import ALiBi, RoPE, SinusoidalPositionalEncoding
from mlx.nn.layers.quantized import QuantizedLinear
from mlx.nn.layers.transformer import (
    MultiHeadAttention,
    Transformer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
