# Copyright © 2026 Apple Inc.

import argparse

import mlx.core as mx
from time_utils import time_fn

SHAPES = {
    "qwen3.6-35b-a3b-up": (512, 2048),
    "qwen3.6-35b-a3b-down": (2048, 512),
}


def qqmm(x, weight, scales, global_scale_x, global_scale_w):
    return mx.qqmm(
        x,
        weight,
        scales,
        mode="nvfp4",
        global_scale_x=global_scale_x,
        global_scale_w=global_scale_w,
    )


def benchmark(shape, tokens):
    output_dims, input_dims = shape
    mx.random.seed(0)
    global_scale_x = mx.array(1.0, dtype=mx.float32)
    global_scale_w = mx.array(1.0, dtype=mx.float32)
    x = mx.random.uniform(low=-0.5, high=0.5, shape=(tokens, input_dims)).astype(
        mx.bfloat16
    )
    weight = mx.random.uniform(
        low=-0.5, high=0.5, shape=(output_dims, input_dims)
    ).astype(mx.bfloat16)
    weight, scales = mx.quantize(
        weight,
        mode="nvfp4",
        global_scale=global_scale_w,
    )
    mx.eval(x, weight, scales)

    time_fn(
        qqmm,
        x,
        weight,
        scales,
        global_scale_x,
        global_scale_w,
        msg=(
            f"mode=nvfp4 dtype=bfloat16 tokens={tokens} "
            f"N={output_dims} K={input_dims}"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=SHAPES, default="qwen3.6-35b-a3b-up")
    parser.add_argument("--tokens", type=int, default=2048)
    args = parser.parse_args()
    benchmark(SHAPES[args.shape], args.tokens)
