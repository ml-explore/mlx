import math
from typing import Optional

import mlx.core as mx
from mlx.nn.layers.base import Module


class LSTMCell(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias

        # These weights are the learnables parameters of the cell's gates and state vectors:
        # the input gate, forget gate, output gate, cell state candidate and biases.
        k = 1 / math.sqrt(hidden_size)
        self.weight_ih = mx.random.uniform(-k, k, shape=(4 * hidden_size, input_size))
        self.weight_hh = mx.random.uniform(-k, k, shape=(4 * hidden_size, hidden_size))
        if bias:
            self.bias_h = mx.random.uniform(-k, k, shape=(4 * hidden_size,))

    def __call__(
        self,
        input: mx.array,
        h_0: mx.array,
        c_0: mx.array,
    ) -> mx.array:
        x_hidden = mx.matmul(input, self.weight_ih.T)  # (B, 4 * hidden_size)
        h_hidden = mx.matmul(h_0, self.weight_hh.T)  # (B, 4 * hidden_size)
        if self.bias:
            x_hidden += self.bias_h

        x_hidden = x_hidden.reshape(-1, 4, self.hidden_size)  # (B, 4, hidden_size)
        h_hidden = h_hidden.reshape(-1, 4, self.hidden_size)  # (B, 4, hidden_size)

        i = mx.sigmoid(x_hidden[:, 0] + h_hidden[:, 0])
        f = mx.sigmoid(x_hidden[:, 1] + h_hidden[:, 1])
        g = mx.tanh(x_hidden[:, 2] + h_hidden[:, 2])
        o = mx.sigmoid(x_hidden[:, 3] + h_hidden[:, 3])

        c_1 = f * c_0 + i * g  # (B, hidden_size)
        h_1 = o * mx.tanh(c_1)  # (B, hidden_size)

        if input.ndim == 1:
            h_1 = h_1.squeeze(0)
            c_1 = c_1.squeeze(0)

        return h_1, c_1
