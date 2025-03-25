# Copyright © 2023-2024 Apple Inc.

import mlx.core as mx
from time_utils import time_fn

N = 1024
K = 1024
M = 1024
E = 32
I = 4


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


def time_gather_mm():
    x = mx.random.normal((N, 1, 1, K)) / 1024**0.5
    w = mx.random.normal((E, M, K)) / 1024**0.5
    indices = (mx.random.uniform(shape=(N, I)) * E).astype(mx.uint32)
    sorted_indices = mx.sort(indices.flatten()).reshape(N, I)
    mx.eval(x, w, indices, sorted_indices)

    def gather_mm(x, w, indices, sort):
        idx = indices
        inv_order = None
        if sort:
            x, idx, inv_order = gather_sort(x, indices)
        for _ in range(2):
            x = mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=idx)
        if sort:
            x = scatter_unsort(x, inv_order, indices.shape)
        return x

    time_fn(gather_mm, x, w, indices, False)
    time_fn(gather_mm, x, w, indices, True)
    time_fn(gather_mm, x, w, sorted_indices, False)

    x = mx.random.normal((N * I, K)) / 1024**0.5
    w = mx.random.normal((M, K)) / 1024**0.5
    mx.eval(x, w)

    def equivalent_matmul(x, w):
        for _ in range(2):
            x = x @ w.T
        return x

    time_fn(equivalent_matmul, x, w)


if __name__ == "__main__":
    time_gather_mm()
