# Copyright © 2024 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)


if __name__ == "__main__":

    batch_size = 8
    input_dim = 32
    output_dim = 10

    # Seed for the parameter initialization
    mx.random.seed(0)

    def init():
        model = MLP(
            num_layers=3, input_dim=input_dim, hidden_dim=64, output_dim=output_dim
        )
        optimizer = optim.SGD(learning_rate=1e-1)
        optimizer.init(model.parameters())
        state = [model.parameters(), optimizer.state]
        tree_structure, state = zip(*mlx.utils.tree_flatten(state))
        return model, optimizer, tree_structure, state

    model, optimizer, tree_structure, state = init()
    mx.eval(state)
    print(state[0])
    mx.export_function("init_mlp.mlxfn", lambda: init()[-1])

    def loss_fn(params, X, y):
        model.update(params)
        return nn.losses.cross_entropy(model(X), y, reduction="mean")

    def step(*inputs):
        *state, X, y = inputs
        params, opt_state = mlx.utils.tree_unflatten(list(zip(tree_structure, state)))
        optimizer.state = opt_state
        loss, grads = mx.value_and_grad(loss_fn)(params, X, y)
        params = optimizer.apply_gradients(grads, params)
        _, state = zip(*mlx.utils.tree_flatten([params, optimizer.state]))
        return *state, loss

    # Make some random data
    mx.random.seed(42)
    example_X = mx.random.normal(shape=(batch_size, input_dim))
    example_y = mx.random.randint(low=0, high=output_dim, shape=(batch_size,))
    mx.export_function("train_mlp.mlxfn", step, *state, example_X, example_y)

    imported_step = mx.import_function("train_mlp.mlxfn")
    for it in range(100):
        *state, loss = imported_step(*state, example_X, example_y)
        if it % 10 == 0:
            print(f"Loss {loss.item():.6}")
