from typing import Optional

import mlx.core as mx
from mlx.nn.layers.base import Module


class GraphNetworkBlock(Module):
    """Implements a generic Graph Network block as defined in [1].

    A Graph Network block takes as input a graph with N nodes and E edges and
    returns a graph with the same topology.

    The input graph can have:
    - `node_features`: features associated with each node in the graph, provided as an
        array of size [N, F_N]
    - `edge_features`: features associated with each edge in the graph, provided as an
        array of size [E, F_E]
    - `global_features`: features associated to the graph itself, of size [F_U]
    The topology of the graph is specified as an `edge_index`, an array of size [2, E],
    containing the source and destination nodes of each edge as column vectors.

    A Graph Network block is initialized by providing node, edge and global models (all
    optional), that are used to update node, edge and global features (if present).
    Depending on which models are provided and how they are implemented, the Graph
    Network block acts as a flexible ``meta-layer`` that can be used to implement other
    architectures, like message-passing networks, deep sets, relation networks and more
    (see [1]).


    Args:
        node_model (mlx.nn.layers.base.Module, optional): a callable Module which updates
            a graph's node features
        edge_model (mlx.nn.layers.base.Module, optional): a callable Module which updates
            a graph's edge features
        global_model (mlx.nn.layers.base.Module, optional): a callable Module which updates
            a graph's global features

    References:
        [1] Battaglia et al. Relational Inductive Biases, Deep Learning, and Graph Networks. https://arxiv.org/pdf/1806.01261.pdf


    .. code-block:: python
        import mlx.core as mx

        from mlx.nn.layers.linear import Linear
        from mlx.nn.layers.gnn import GraphNetworkBlock


        class NodeModel(Module):
            def __init__(
                self,
                node_features_dim: int,
                edge_features_dim: int,
                global_features_dim: int,
                output_dim: int,
            ):
                super().__init__()
                self.model = Linear(
                    input_dims=node_features_dim + edge_features_dim + global_features_dim,
                    output_dims=output_dim,
                )

            def __call__(
                self,
                edge_index: mx.array,
                node_features: mx.array,
                edge_features: mx.array,
                global_features: mx.array,
            ):
                destination_nodes = edge_index[1]
                aggregated_edges = mx.zeros([node_features.shape[0], edge_features.shape[1]])
                for i in range(node_features.shape[0]):
                    aggregated_edges[i] = mx.where(
                        (destination_nodes == i).reshape(edge_features.shape[0], 1),
                        edge_features,
                        0,
                    ).mean()
                model_input = mx.concatenate(
                    [
                        node_features,
                        aggregated_edges,
                        mx.ones([node_features.shape[0], global_features.shape[0]])
                        * global_features,
                    ],
                    1,
                )
                return self.model(model_input)


        class EdgeModel(Module):
            def __init__(
                self,
                edge_features_dim: int,
                node_features_dim: int,
                global_features_dim: int,
                output_dim: int,
            ):
                super().__init__()
                self.model = Linear(
                    input_dims=2 * node_features_dim + edge_features_dim + global_features_dim,
                    output_dims=output_dim,
                )

            def __call__(
                self,
                edge_index: mx.array,
                node_features: mx.array,
                edge_features: mx.array,
                global_features: mx.array,
            ):
                source_nodes = edge_index[0]
                destination_nodes = edge_index[1]
                model_input = mx.concatenate(
                    [
                        node_features[destination_nodes],
                        node_features[source_nodes],
                        edge_features,
                        mx.ones([edge_features.shape[0], global_features.shape[0]])
                        * global_features,
                    ],
                    1,
                )
                return self.model(model_input)


        class GlobalModel(Module):
            def __init__(
                self,
                edge_features_dim: int,
                node_features_dim: int,
                global_features_dim: int,
                output_dim: int,
            ):
                super().__init__()
                self.model = Linear(
                    input_dims=node_features_dim + edge_features_dim + global_features_dim,
                    output_dims=output_dim,
                )

            def __call__(
                self,
                edge_index: mx.array,
                node_features: mx.array,
                edge_features: mx.array,
                global_features: mx.array,
            ):
                aggregated_edges = edge_features.mean(axis=0)
                aggregated_nodes = node_features.mean(axis=0)
                model_input = mx.concatenate(
                    [aggregated_nodes, aggregated_edges, global_features], 0
                )
                return self.model(model_input)

        N = 4 # number of nodes
        F_N = 2 # number of node features
        F_E = 1 # number of edge features
        F_U = 2 # number of global features

        edge_index = mx.array([[0, 0, 1, 2, 3], [1, 2, 3, 3, 0]])
        node_features = mx.random.normal([N, F_N])
        edge_features = mx.random.normal([edge_index.shape[1], F_E])
        global_features = mx.random.normal([F_U])

        # edge model
        output_edge_feature_dim = F_E
        edge_model = EdgeModel(
            edge_features_dim=F_E,
            node_features_dim=F_N,
            global_features_dim=F_U,
            output_dim=output_edge_feature_dim,
        )

        # node model
        output_node_features_dim = F_N
        node_model = NodeModel(
            node_features_dim=F_N,
            edge_features_dim=output_edge_feature_dim,
            global_features_dim=F_U,
            output_dim=output_node_features_dim,
        )

        # global_model
        output_global_features_dim = F_U
        global_model = GlobalModel(
            node_features_dim=output_node_features_dim,
            edge_features_dim=output_edge_feature_dim,
            global_features_dim=F_U,
            output_dim=output_global_features_dim,
        )


        # Graph Network block
        gnn = GraphNetworkBlock(
            node_model=node_model, edge_model=edge_model, global_model=global_model
        )
        node_features, edge_features, global_features = gnn(
            edge_index=edge_index,
            node_features=node_features,
            edge_features=edge_features,
            global_features=global_features,
        )

    """

    def __init__(
        self,
        node_model: Optional[Module] = None,
        edge_model: Optional[Module] = None,
        global_model: Optional[Module] = None,
    ):
        super().__init__()
        self.node_model = node_model
        self.edge_model = edge_model
        self.global_model = global_model

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Optional[mx.array] = None,
        edge_features: Optional[mx.array] = None,
        global_features: Optional[mx.array] = None,
    ) -> tuple[Optional[mx.array], Optional[mx.array], Optional[mx.array]]:
        """Forward pass of the Graph Network block

        Args:
            edge_index (array): array of size [2, E], where each columns contains the source
                and destination nodes of an edge.
            node_features (array, optional): features associated with each node in the
                graph, provided as an array of size [N, F_N]
            edge_features (array, optional): features associated with each edge in the
                graph, provided as an array of size [E, F_E]
            global_features (array, optional): features associated to the graph itself,
                of size [F_U]

        Returns:
            The tuple of updated node, edge and global attributes.
        """
        if self.edge_model:
            edge_features = self.edge_model(
                edge_index, node_features, edge_features, global_features
            )

        if self.node_model:
            node_features = self.node_model(
                edge_index, node_features, edge_features, global_features
            )

        if self.global_model:
            global_features = self.global_model(
                edge_index, node_features, edge_features, global_features
            )

        return node_features, edge_features, global_features
