from typing import Any

import mlx.core as mx
import mlx.nn as nn


class MessagePassing(nn.Module):
    r"""Base class for creating Message Passing Neural Networks (MPNNs) [1].

    Inherit this class to build arbitrary GNN models based on the message
    passing paradigm. This implementation is inspired from PyTorch Geometric [2].

    Args:
        aggr (str): the aggregation strategy used to aggregate messages

    References:
        [1] Gilmer et al. Neural Message Passing for Quantum Chemistry.
        https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

        [2] Fey et al. PyG
        https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html
    """

    def __init__(self, aggr="add"):
        super().__init__()

        self.aggr = aggr

    def __call__(self, x: mx.array, edge_index: mx.array, **kwargs: Any):
        raise NotImplementedError

    def propagate(self, x: mx.array, edge_index: mx.array, **kwargs: Any) -> mx.array:
        r"""Computes messages from neighbors, aggregates them and updates
        the final node embeddings.

        Args:
            x (mx.array): input node features/embeddings
            edge_index (mx.array): graph representation of shape (2, |E|) in COO format
            **kwargs (Any): arguments to pass to message, aggregate and update
        """
        src_idx, dst_idx = edge_index
        x_i = x[src_idx]
        x_j = x[dst_idx]

        messages = self.message(x_i, x_j, **kwargs)

        aggregated = self.aggregate(messages, dst_idx, **kwargs)

        output = self.update_(aggregated, **kwargs)

        return output

    def message(self, x_i: mx.array, x_j: mx.array, **kwargs: Any) -> mx.array:
        r"""Computes messages between connected nodes.

        Args:
            x_i (mx.array): source node embeddings
            x_j (mx.array): destination node embeddings
            **kwargs (Any): optional args to compute messages
        """
        return x_i

    def aggregate(
        self, messages: mx.array, indices: mx.array, **kwargs: Any
    ) -> mx.array:
        r"""Aggregates the messages using the `self.aggr` strategy.

        Args:
            messages (mx.array): computed messages
            indices: (mx.array): indices representing the nodes that receive messages
            **kwargs (Any): optional args to aggregate messages
        """
        if self.aggr == "add":
            nb_unique_indices = _unique(indices)
            empty_tensor = mx.zeros((nb_unique_indices, messages.shape[-1]))
            update_dim = (messages.shape[0], 1, messages.shape[1])

            return mx.scatter_add(
                empty_tensor, indices, messages.reshape(update_dim), 0
            )

        raise NotImplementedError(f"Aggregation {self.aggr} not implemented yet.")

    # NOTE: this method can't be named `update()`, or the grads will be always set to 0.
    def update_(self, aggregated: mx.array, **kwargs: Any) -> mx.array:
        r"""Updates the final embeddings given the aggregated messages.

        Args:
            aggregated (mx.array): aggregated messages
            **kwargs (Any): optional args to update messages
        """
        return aggregated


def _unique(array: mx.array):
    return len(set(array.tolist()))
