// Copyright © 2023-2024 Apple Inc.

#pragma once

namespace mlx::core::detail {

std::pair<std::vector<array>, std::vector<array>> vmap_trace(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& inputs,
    const std::vector<int>& in_axes);

std::vector<array> vmap_replace(
    const std::vector<array>& inputs,
    const std::vector<array>& s_inputs,
    const std::vector<array>& s_outputs,
    const std::vector<int>& in_axes,
    const std::vector<int>& out_axes);

// Create an InTracing object during tracing operations to signify to the rest
// of the codebase that we are during tracing so evals should not throw away
// the graph.
struct InTracing {
  explicit InTracing(bool dynamic = false, bool grad = false) {
    grad_counter += grad;
    trace_stack().push_back({dynamic, grad});
  }
  ~InTracing() {
    grad_counter -= trace_stack().back().second;
    trace_stack().pop_back();
  }

  static bool in_tracing() {
    return !trace_stack().empty();
  }
  static bool in_dynamic_tracing() {
    // compile is always and only the outer-most transform
    return in_tracing() && trace_stack().front().first;
  }

  static bool in_grad_tracing() {
    return grad_counter > 0;
  }

 private:
  static int grad_counter;
  static std::vector<std::pair<char, char>>& trace_stack();
};

struct RetainGraph {
  RetainGraph() {
    tracing_counter++;
  }
  ~RetainGraph() {
    tracing_counter--;
  }

  static bool retain_graph() {
    return tracing_counter > 0;
  }

 private:
  static int tracing_counter;
};

/** Return true if we are currently performing a function transformation in
 * order to keep the graph when evaluating tracer arrays. */
inline bool in_tracing() {
  return detail::InTracing::in_tracing();
}

/** Return true if we are in a dynamic (shapeless) trace used for compiling or
 * exporting graphs with dynamic shapes.  */
inline bool in_dynamic_tracing() {
  return detail::InTracing::in_dynamic_tracing();
}

/** Return true if we are in a gradient trace (vjp, jvp, etc).  */
inline bool in_grad_tracing() {
  return detail::InTracing::in_grad_tracing();
}

inline bool retain_graph() {
  return detail::RetainGraph::retain_graph();
}

} // namespace mlx::core::detail
