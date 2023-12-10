# Copyright © 2023 Apple Inc.

import numpy as np
from time_utils import time_fn


def time_add():
    a = np.ones((100, 100, 10), dtype=np.float32)
    b = np.ones((100, 100, 10), dtype=np.float32)
    time_fn(np.add, a, b)


def time_matmul():
    a = np.random.rand(1000, 500).astype(np.float32)
    b = np.random.rand(500, 1000).astype(np.float32)
    time_fn(np.matmul, a, b)


def time_exp():
    a = np.random.randn(1000, 100).astype(np.float32)
    time_fn(np.exp, a)


def time_take():
    a = np.random.rand(10000, 500)
    ids = np.random.randint(0, 10000, (20, 10))
    ids = [idx.reshape(-1) for idx in np.split(ids, 20)]

    def random_take():
        return [np.take(a, idx, 0) for idx in ids]

    time_fn(random_take)


if __name__ == "__main__":
    time_add()
    time_matmul()
    time_exp()
    time_take()
