# Copyright © 2026 Apple Inc.

import argparse

import mlx.core as mx
from time_utils import time_fn

SHAPES = {
    "qwen3.6-35b-a3b-up": {
        "experts": 256,
        "top_k": 8,
        "input": 2048,
        "output": 512,
    },
    "qwen3.6-35b-a3b-down": {
        "experts": 256,
        "top_k": 8,
        "input": 512,
        "output": 2048,
    },
}


def make_weights(shape):
    weight = mx.random.uniform(
        low=-0.5,
        high=0.5,
        shape=(shape["experts"], shape["output"], shape["input"]),
    ).astype(mx.bfloat16)
    quantized, scales = mx.quantize(
        weight,
        mode="nvfp4",
        global_scale=mx.array(1.0, dtype=mx.float32),
    )
    global_scale = mx.ones((shape["experts"],), dtype=mx.float32)
    mx.eval(quantized, scales, global_scale)
    return quantized, scales, global_scale


def gather_qqmm(x, weight, scales, global_scale_x, global_scale_w, rhs, sorted_):
    return mx.gather_qqmm(
        x,
        weight,
        scales,
        rhs_indices=rhs,
        mode="nvfp4",
        global_scale_x=global_scale_x,
        global_scale_w=global_scale_w,
        sorted_indices=sorted_,
    )


def benchmark(shape, workload, tokens):
    weight, scales, global_scale_w = make_weights(shape)
    global_scale_x = mx.array(1.0, dtype=mx.float32)

    if workload == "prompt":
        routes = tokens * shape["top_k"]
        x = mx.random.uniform(
            low=-0.5, high=0.5, shape=(routes, 1, shape["input"])
        ).astype(mx.bfloat16)
        rhs = mx.repeat(
            mx.arange(shape["experts"], dtype=mx.uint32),
            (routes + shape["experts"] - 1) // shape["experts"],
        )[:routes]
        sorted_ = True
    elif workload == "matrix":
        x = mx.random.uniform(
            low=-0.5,
            high=0.5,
            shape=(shape["top_k"], tokens, shape["input"]),
        ).astype(mx.bfloat16)
        rhs = (mx.arange(shape["top_k"], dtype=mx.uint32) * 17) % shape["experts"]
        sorted_ = False
    else:
        x = mx.random.uniform(
            low=-0.5, high=0.5, shape=(1, 1, 1, shape["input"])
        ).astype(mx.bfloat16)
        rhs = (
            (mx.arange(shape["top_k"], dtype=mx.uint32) * 17) % shape["experts"]
        ).reshape(1, shape["top_k"])
        sorted_ = False

    mx.eval(x, rhs)
    time_fn(
        gather_qqmm,
        x,
        weight,
        scales,
        global_scale_x,
        global_scale_w,
        rhs,
        sorted_,
        msg=(
            f"{workload} routes={rhs.size} " f"N={shape['output']} K={shape['input']}"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=SHAPES, default="qwen3.6-35b-a3b-up")
    parser.add_argument(
        "--workload", choices=("prompt", "matrix", "decode"), default="prompt"
    )
    parser.add_argument("--tokens", type=int, default=2048)
    args = parser.parse_args()
    benchmark(SHAPES[args.shape], args.workload, args.tokens)
