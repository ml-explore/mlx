# Copyright © 2025 Apple Inc.

"""A minimal data-parallel training example using mlx.distributed.

Each rank trains the same model on a different shard of a synthetic
classification dataset. After every step, gradients are averaged across
ranks with ``mx.distributed.all_sum`` (wrapped in ``nn.average_gradients``
for batched communication). When run with a single process the
distributed calls are no-ops, so the same script works for local
debugging.

Run with::

    mlx.launch -n 2 examples/distributed/data_parallel/main.py
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

num_features = 64
num_classes = 10
batch_size = 32
num_steps = 200
lr = 0.1
seed = 0

world = mx.distributed.init()
rank = world.rank()
size = world.size()

# Same true labelling function on every rank.
mx.random.seed(seed)
W_true = mx.random.normal((num_features, num_classes))

# Different shard of data per rank.
mx.random.seed(seed + 1 + rank)
X = mx.random.normal((batch_size * num_steps, num_features))
y = mx.argmax(X @ W_true, axis=1)

# Identical model init across ranks.
mx.random.seed(seed)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


model = MLP()
mx.eval(model.parameters())
optimizer = optim.SGD(learning_rate=lr)


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)


def step(x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    # All-reduce: sum gradients across ranks then divide by world size.
    # On a single rank this is a no-op.
    grads = nn.average_gradients(grads)
    optimizer.update(model, grads)
    return loss


for i in range(num_steps):
    xb = X[i * batch_size : (i + 1) * batch_size]
    yb = y[i * batch_size : (i + 1) * batch_size]
    loss = step(xb, yb)
    mx.eval(loss, model.parameters())
    if rank == 0 and (i % 20 == 0 or i == num_steps - 1):
        print(f"step {i:4d}  loss {loss.item():.4f}")

if rank == 0:
    preds = mx.argmax(model(X[:batch_size]), axis=1)
    acc = mx.mean(preds == y[:batch_size]).item()
    print(f"rank 0 train-batch accuracy: {acc:.3f} (world size {size})")
