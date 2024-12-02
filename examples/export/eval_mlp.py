# Copyright © 2024 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
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

    # Load the model
    mx.random.seed(0)  # Seed for params
    model = MLP(num_layers=5, input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
    mx.eval(model)

    # Note, the model parameters are saved in the export function
    def forward(x):
        return model(x)

    mx.random.seed(42)  # Seed for input
    example_x = mx.random.uniform(shape=(batch_size, input_dim))

    mx.export_function("eval_mlp.mlxfn", forward, example_x)

    # Import in Python
    imported_forward = mx.import_function("eval_mlp.mlxfn")
    expected = forward(example_x)
    (out,) = imported_forward(example_x)
    assert mx.allclose(expected, out)
    print(out)
