from typing import Any

import mlx.core as mx
import mlx.nn as nn
from message_passing import MessagePassing


class GCNLayer(MessagePassing):
    r"""Applies a GCN convolution over input node features.

    Args:
        x_dim (int): size of input node features
        h_dim (int): size of hidden node embeddings
        bias (bool): whether to use bias in the node projection
    """

    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        bias: bool = True,
    ):
        super(GCNLayer, self).__init__(aggr="add")
        self.linear = nn.Linear(x_dim, h_dim, bias)

    def __call__(
        self, x: mx.array, edge_index: mx.array, normalize: bool = True, **kwargs: Any
    ) -> mx.array:
        x = self.linear(x)

        row, col = edge_index

        # Compute node degree normalization for the mean aggregation.
        norm: mx.array = None
        if normalize:
            deg = self._degree(col, x.shape[0])
            deg_inv_sqrt = deg ** (-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        else:
            norm = mx.ones_like(row)

        # Compute messages and aggregate them with sum and norm.
        x = self.propagate(x=x, edge_index=edge_index, norm=norm)

        return x

    def message(
        self, x_i: mx.array, x_j: mx.array, norm: mx.array, **kwargs: Any
    ) -> mx.array:
        return norm.reshape(-1, 1) * x_i

    def _degree(self, index: mx.array, num_edges: int) -> mx.array:
        out = mx.zeros((num_edges,))
        one = mx.ones((index.shape[0],), dtype=out.dtype)
        return mx.scatter_add(out, index, one.reshape(-1, 1), 0)


class GCN(nn.Module):
    r"""Graph Convolutional Network implementation [1]

    Args:
        x_dim (int): size of input node features
        h_dim (int): size of hidden node embeddings
        bias (bool): whether to use bias in the node projection

    References:
        [1] Kipf et al. Semi-Supervised Classification with Graph Convolutional Networks.
        https://arxiv.org/abs/1609.02907
    """

    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        out_dim: int,
        nb_layers: int = 2,
        dropout: float = 0.5,
        bias: bool = True,
    ):
        super(GCN, self).__init__()

        layer_sizes = [x_dim] + [h_dim] * nb_layers + [out_dim]
        self.gcn_layers = [
            GCNLayer(in_dim, out_dim, bias)
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x: mx.array, edge_index: mx.array) -> mx.array:
        for layer in self.gcn_layers[:-1]:
            x = nn.relu(layer(x, edge_index))
            x = self.dropout(x)

        x = self.gcn_layers[-1](x, edge_index)
        return x