// Copyright © 2023 Apple Inc.

#pragma once

namespace mlx::core {

// Forward declaration
class array;
class Primitive;

class GraphNode {
 public:
  explicit GraphNode(){};

  GraphNode(
      std::unique_ptr<Primitive> primitive,
      const std::vector<array>& inputs,
      const std::vector<array>& outputs);

  Primitive& primitive() const {
    return *(graph_node_->primitive);
  }

  bool has_primitive() const {
    return graph_node_ != nullptr && graph_node_->primitive != nullptr;
  }

  const std::vector<array>& inputs() const;

  std::vector<array>& inputs() {
    return graph_node_->inputs;
  }
  std::vector<array>& outputs() {
    return graph_node_->outputs;
  }
  const std::vector<array>& outputs() const;

  bool is_evaled() const;

  std::uintptr_t id() const {
    return reinterpret_cast<std::uintptr_t>(graph_node_.get());
  }

  void detach();

 private:
  struct GraphNodeImpl {
    std::unique_ptr<Primitive> primitive{nullptr};
    std::vector<array> inputs;
    std::vector<array> outputs;
    GraphNodeImpl(
        std::unique_ptr<Primitive> primitive,
        const std::vector<array>& inputs,
        const std::vector<array>& outputs);

    ~GraphNodeImpl();
  };

  std::shared_ptr<GraphNodeImpl> graph_node_{nullptr};
};
} // namespace mlx::core
