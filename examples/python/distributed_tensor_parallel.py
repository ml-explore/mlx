# Copyright © 2026 Apple Inc.

"""
Tensor parallel inference for a small MLP.

The two linear layers are sharded across ranks: the first splits its output
features so each rank computes part of the hidden state, the second splits its
input features and sums the partial results. That costs one all reduce per
block, and the sharded model returns what the full model returns:

    python examples/python/distributed_tensor_parallel.py
    mlx.launch -n 2 python examples/python/distributed_tensor_parallel.py
    mlx.launch -n 4 python examples/python/distributed_tensor_parallel.py

Unlike data parallelism this splits the model rather than the batch, so it is
what you reach for when the weights are too big for one machine.
"""

import mlx.core as mx
import mlx.nn as nn

dims = 256
hidden = 1024
num_tokens = 8

world = mx.distributed.init()

if hidden % world.size() != 0:
    raise ValueError(
        f"Cannot split {hidden} hidden features evenly over {world.size()} ranks."
    )


class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dims, hidden)
        self.down = nn.Linear(hidden, dims)

    def __call__(self, x):
        return self.down(nn.silu(self.up(x)))


# Seeding the global rng gives every rank the same weights to shard.
mx.random.seed(0)
model = MLP(dims, hidden)
mx.eval(model.parameters())

x = mx.random.normal((num_tokens, dims), key=mx.random.key(1))
expected = model(x)
mx.eval(expected)

# Each rank keeps a slice of each weight. from_linear takes the slice belonging
# to this rank, so the layers never hold the full weight afterwards.
model.up = nn.AllToShardedLinear.from_linear(model.up, group=world)
model.down = nn.ShardedToAllLinear.from_linear(model.down, group=world)
mx.eval(model.parameters())

y = model(x)

# Every rank evaluates: the down projection ends in an all reduce, so a rank
# that skipped this would leave the others waiting.
mx.eval(y)

difference = mx.abs(y - expected).max().item()
hidden_per_rank = hidden // world.size()

if world.rank() == 0:
    print(
        f"Max |sharded - full| = {difference:.3e} over {world.size()} rank(s), "
        f"{hidden_per_rank} of {hidden} hidden features per rank"
    )
