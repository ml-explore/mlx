# Copyright © 2026 Apple Inc.

"""
Data parallel training of a small MLP.

Every rank holds a different slice of the same dataset and averages the
gradients each step, so the training is equivalent to running on one process
and the final loss is the same at any number of ranks, up to the order the
floating point additions happen in:

    python examples/python/distributed_data_parallel.py
    mlx.launch -n 2 python examples/python/distributed_data_parallel.py
    mlx.launch -n 4 python examples/python/distributed_data_parallel.py

The model is replicated and the batch is split, which is what you reach for
when the model fits on one machine but the data is large.
"""

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

num_features = 100
num_examples = 1_000
hidden = 64
num_iters = 200
lr = 0.05

world = mx.distributed.init()

if num_examples % world.size() != 0:
    raise ValueError(
        f"Cannot split {num_examples} examples evenly over {world.size()} ranks."
    )

# Fixed keys, so every rank draws the same dataset rather than one of its own.
w_star = mx.random.normal((num_features,), key=mx.random.key(0))
X = mx.random.normal((num_examples, num_features), key=mx.random.key(1))
y = X @ w_star + 1e-2 * mx.random.normal((num_examples,), key=mx.random.key(2))

# Keep this rank's slice and drop the rest.
examples_per_rank = num_examples // world.size()
start = world.rank() * examples_per_rank
X = X[start : start + examples_per_rank]
y = y[start : start + examples_per_rank]


class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        self.layers = [nn.Linear(dims, hidden), nn.Linear(hidden, 1)]

    def __call__(self, x):
        return self.layers[1](nn.relu(self.layers[0](x))).squeeze(-1)


# Seeding the global rng starts every rank from the same weights, which the
# averaged gradient then keeps in step.
mx.random.seed(0)
model = MLP(num_features, hidden)
mx.eval(model.parameters())

optimizer = optim.SGD(learning_rate=lr)


def loss_fn(model, X, y):
    return 0.5 * mx.mean(mx.square(model(X) - y))


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

tic = time.perf_counter()
for _ in range(num_iters):
    loss, grads = loss_and_grad_fn(model, X, y)

    # Each rank has gradients for its own slice. Averaging them gives the
    # gradients of the whole dataset, which is what keeps this equivalent to
    # single process training. One call handles the whole parameter tree.
    grads = nn.average_gradients(grads, group=world)

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
toc = time.perf_counter()

# Every slice is the same size, so averaging the per rank losses gives the loss
# over the whole dataset.
loss = mx.distributed.all_sum(loss_fn(model, X, y), group=world) / world.size()

# Only rank 0 prints the loss, but every rank has to evaluate it. Arrays are
# lazy, so leaving this to the print below would mean the other ranks never
# join the all_sum and everyone waits forever.
mx.eval(loss)

throughput = num_iters / (toc - tic)

if world.rank() == 0:
    print(
        f"Loss {loss.item():.6f}, Throughput {throughput:.2f} (it/s) "
        f"over {world.size()} rank(s), {examples_per_rank} examples per rank"
    )
