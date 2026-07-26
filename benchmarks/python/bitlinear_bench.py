# Copyright © 2026 8b-is

import argparse

import mlx.core as mx
import mlx.nn as nn
from time_utils import time_fn

from mlx.nn.layers.bitlinear import BitLinear

B = 32
T = 256


def time_bitlinear(dim):
    mx.random.seed(3)
    x = mx.random.normal((B, T, dim))
    linear = nn.Linear(dim, dim)
    ternary = BitLinear(dim, dim)
    binary = BitLinear(dim, dim, binary=True)
    mx.eval(x, linear.parameters(), ternary.parameters(), binary.parameters())

    time_fn(linear, x, msg=f"nn.Linear(dim={dim})")
    time_fn(ternary, x, msg=f"BitLinear(dim={dim}, binary=False)")
    time_fn(binary, x, msg=f"BitLinear(dim={dim}, binary=True)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLX benchmarks.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if args.gpu:
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

    # BitLinear does strictly more work per forward pass than nn.Linear (an
    # RMSNorm plus two quantization passes, all still in full precision --
    # this is a quantization-*aware-training* layer, not a packed/memory-
    # efficient format). Expect it to be slower, not faster; that is the
    # honest baseline a future packed ternary format should be measured
    # against, not a claim this benchmark is trying to win.
    for dim in (768, 4096):
        time_bitlinear(dim)
