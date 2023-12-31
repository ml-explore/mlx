// Copyright © 2023 Apple Inc.

#include "mlx/graph.h"
#include "mlx/array.h"
#include "mlx/primitives.h"

namespace mlx::core {

const std::vector<array>& GraphNode::inputs() const {
  static std::vector<array> empty{};
  if (graph_node_ != nullptr) {
    return graph_node_->inputs;
  } else {
    return empty;
  }
}

const std::vector<array>& GraphNode::outputs() const {
  static std::vector<array> empty{};
  if (graph_node_ != nullptr) {
    return graph_node_->outputs;
  } else {
    return empty;
  }
}

bool GraphNode::is_evaled() const {
  // Evaluated if its a null graph or if it has an output that is not
  // evaluated
  return graph_node_ == nullptr || outputs()[0].is_evaled();
}

void GraphNode::detach() {
  inputs().clear();
  for (auto& o : outputs()) {
    o.detach();
  }
  outputs().clear();
}

GraphNode::GraphNode(
    std::unique_ptr<Primitive> primitive,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs)
    : graph_node_(std::make_shared<GraphNodeImpl>(
          std::move(primitive),
          inputs,
          outputs)){};

GraphNode::GraphNodeImpl::GraphNodeImpl(
    std::unique_ptr<Primitive> primitive,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs)
    : primitive(std::move(primitive)), inputs(inputs), outputs(outputs){};

GraphNode::GraphNodeImpl::~GraphNodeImpl() = default;

} // namespace mlx::core
