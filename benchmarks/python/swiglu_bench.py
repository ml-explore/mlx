# Copyright © 2025 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
from time_utils import time_fn

SEQ_LENS = [512, 713, 1024, 2048, 4123, 8192, 8192 * 2]
MAX_UNSORTED_N = 1024

# https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/blob/main/config.json
# https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json
# https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json
# https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/config.json
# https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json
CONFIGS = {
    "glm-5.3-flash": (4096, 2048, 288, 8),
    "glm-5.3": (6144, 2048, 256, 8),
    "qwen3.5-35b-a3b": (2048, 512, 256, 8),
    "qwen3.5-122b-a10b": (3072, 1024, 256, 8),
    "qwen3.5-397b-a17b": (4096, 1024, 512, 10),
}


def gather_sort(x, indices):
    N, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


def time_gather_mm(name, D, M, E, I):
    w1 = mx.random.normal((E, M, D), dtype=mx.bfloat16, scale=D**-0.5)
    w2 = mx.random.normal((E, M, D), dtype=mx.bfloat16, scale=D**-0.5)
    w3 = mx.random.normal((E, D, M), dtype=mx.bfloat16, scale=M**-0.5)
    mx.eval(w1, w2, w3)

    def gather_mm(x, w1, w2, w3, indices, sort):
        idx = indices
        inv_order = None
        if sort:
            x, idx, inv_order = gather_sort(x, indices)
        gate = mx.gather_mm(
            x, w1.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=sort
        )
        up = mx.gather_mm(x, w2.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=sort)
        x = mx.gather_mm(
            nn.silu(gate) * up,
            w3.swapaxes(-1, -2),
            rhs_indices=idx,
            sorted_indices=sort,
        )
        if sort:
            x = scatter_unsort(x, inv_order, indices.shape)
        return x

    for N in SEQ_LENS:
        x = mx.random.normal((N, 1, 1, D), dtype=mx.bfloat16)
        scores = mx.random.uniform(shape=(N, E))
        indices = mx.argpartition(scores, E - I, axis=-1)[:, -I:].astype(mx.uint32)
        sorted_indices = mx.sort(indices.flatten()).reshape(N, I)
        mx.eval(x, indices, sorted_indices)

        label = f"{name} N={N}"
        time_fn(
            gather_mm, x, w1, w2, w3, indices, True, msg=f"{label} swiglu"
        )

if __name__ == "__main__":
    for name, config in CONFIGS.items():
        time_gather_mm(name, *config)
