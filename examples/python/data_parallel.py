# Copyright © 2026 Apple Inc.

"""train linear regression with data parallel workers.

run locally with:

    mlx.launch -n 2 python examples/python/data_parallel.py
"""

import time

import mlx.core as mx
import mlx.nn as nn

num_features = 100
num_examples = 1_000
num_iters = 100
lr = 0.1

world = mx.distributed.init()

# every rank learns the same parameters from a different synthetic data shard.
w_star = mx.random.normal((num_features,), key=mx.random.key(0))
X = mx.random.normal(
    (num_examples, num_features), key=mx.random.key(world.rank() + 1)
)
eps = 1e-2 * mx.random.normal(
    (num_examples,), key=mx.random.key(world.rank() + 1_000)
)
y = X @ w_star + eps

# parameters must start with the same values on every rank.
w = 1e-2 * mx.random.normal((num_features,), key=mx.random.key(2_000))


def loss_fn(w):
    return 0.5 * mx.mean(mx.square(X @ w - y))


grad_fn = mx.grad(loss_fn)

tic = time.perf_counter()
for _ in range(num_iters):
    grad = grad_fn(w)
    grad = nn.average_gradients(grad, group=world)
    w = w - lr * grad
    mx.eval(w)
toc = time.perf_counter()

loss = mx.distributed.all_sum(loss_fn(w), group=world) / world.size()
loss_value = loss.item()
error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5
throughput = num_iters / (toc - tic)

if world.rank() == 0:
    print(
        f"Loss {loss_value:.5f}, L2 distance: |w-w*| = {error_norm:.5f}, "
        f"Throughput {throughput:.5f} (it/s), Ranks {world.size()}"
    )
