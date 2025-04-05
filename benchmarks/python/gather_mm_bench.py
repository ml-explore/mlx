# Copyright © 2023-2024 Apple Inc.

import mlx.core as mx
from time_utils import time_fn

N = 1024
D = 1024
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


def gather_mm_simulate(x, w, indices):
    x, idx, inv_order = gather_sort(x, indices)
    for i in range(2):
        y = mx.concatenate([x[i] @ w[j].T for i, j in enumerate(idx.tolist())], axis=0)
        x = y[:, None]
    x = scatter_unsort(x, inv_order, indices.shape)
    return x


def time_gather_mm():
    x = mx.random.normal((N, 1, 1, D)) / 1024**0.5
    w1 = mx.random.normal((E, M, D)) / 1024**0.5
    w2 = mx.random.normal((E, D, M)) / 1024**0.5
    indices = (mx.random.uniform(shape=(N, I)) * E).astype(mx.uint32)
    sorted_indices = mx.sort(indices.flatten()).reshape(N, I)
    mx.eval(x, w1, w2, indices, sorted_indices)

    def gather_mm(x, w1, w2, indices, sort):
        idx = indices
        inv_order = None
        if sort:
            x, idx, inv_order = gather_sort(x, indices)
        x = mx.gather_mm(x, w1.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=sort)
        x = mx.gather_mm(x, w2.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=sort)
        if sort:
            x = scatter_unsort(x, inv_order, indices.shape)
        return x

    time_fn(gather_mm, x, w1, w2, indices, False)
    time_fn(gather_mm, x, w1, w2, sorted_indices, False)
    time_fn(gather_mm, x, w1, w2, indices, True)

    x = mx.random.normal((N * I, D)) / 1024**0.5
    w1 = mx.random.normal((M, D)) / 1024**0.5
    w2 = mx.random.normal((D, M)) / 1024**0.5
    mx.eval(x, w1, w2)

    def equivalent_matmul(x, w1, w2):
        x = x @ w1.T
        x = x @ w2.T
        return x

    time_fn(equivalent_matmul, x, w1, w2)


if __name__ == "__main__":
    time_gather_mm()
