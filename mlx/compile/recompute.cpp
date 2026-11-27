// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "mlx/compile_impl.h"
#include "mlx/primitives.h"

namespace mlx::core::detail {

// Max-flow solver (Dinic) for compile_recompute.
class MaxFlow {
 public:
  static constexpr int64_t inf = std::numeric_limits<int64_t>::max() / 4;

  explicit MaxFlow(int n) : adj_(n), level_(n), next_(n) {}

  void add_edge(int u, int v, int64_t cap) {
    adj_[u].push_back(edges_.size());
    edges_.push_back({v, cap});
    adj_[v].push_back(edges_.size());
    edges_.push_back({u, 0});
  }

  // Returns the max flow, or inf if there is no finite cut.
  int64_t solve(int s, int t) {
    int64_t flow = 0;
    while (make_levels(s, t)) {
      std::fill(next_.begin(), next_.end(), 0);
      while (int64_t f = augment(s, t)) {
        flow += f;
        if (flow >= inf) {
          return inf;
        }
      }
    }
    return flow;
  }

  // Vertices that s still reaches in the residual graph.
  std::vector<bool> source_side(int s) const {
    std::vector<bool> seen(adj_.size(), false);
    std::vector<int> stack = {s};
    seen[s] = true;
    while (!stack.empty()) {
      int u = stack.back();
      stack.pop_back();
      for (int e : adj_[u]) {
        int v = edges_[e].to;
        if (edges_[e].cap > 0 && !seen[v]) {
          seen[v] = true;
          stack.push_back(v);
        }
      }
    }
    return seen;
  }

 private:
  struct Edge {
    int to;
    int64_t cap;
  };

  bool make_levels(int s, int t) {
    std::fill(level_.begin(), level_.end(), -1);
    std::queue<int> queue;
    level_[s] = 0;
    queue.push(s);
    while (!queue.empty()) {
      int u = queue.front();
      queue.pop();
      for (int e : adj_[u]) {
        int v = edges_[e].to;
        if (edges_[e].cap > 0 && level_[v] < 0) {
          level_[v] = level_[u] + 1;
          queue.push(v);
        }
      }
    }
    return level_[t] >= 0;
  }

  // Push flow along one path of the level graph. A stack is used instead of
  // recursion because compiled graphs can be deep.
  int64_t augment(int s, int t) {
    std::vector<int> path;
    int u = s;
    while (u != t) {
      bool moved = false;
      for (; next_[u] < adj_[u].size(); ++next_[u]) {
        int e = adj_[u][next_[u]];
        int v = edges_[e].to;
        if (edges_[e].cap > 0 && level_[v] == level_[u] + 1) {
          path.push_back(e);
          u = v;
          moved = true;
          break;
        }
      }
      if (!moved) {
        if (path.empty()) {
          return 0;
        }
        // Dead end: remove u from the level graph and step back.
        level_[u] = -1;
        u = edges_[path.back() ^ 1].to;
        path.pop_back();
      }
    }
    int64_t f = inf;
    for (int e : path) {
      f = std::min(f, edges_[e].cap);
    }
    for (int e : path) {
      edges_[e].cap -= f;
      edges_[e ^ 1].cap += f;
    }
    return f;
  }

  std::vector<Edge> edges_;
  std::vector<std::vector<int>> adj_;
  std::vector<int> level_;
  std::vector<size_t> next_;
};

void compile_recompute(std::vector<array>& tape, ParentsMap& parents_map) {
  int n = tape.size();
  std::unordered_map<std::uintptr_t, int> vid;
  for (int i = 0; i < n; ++i) {
    vid.insert({tape[i].id(), i});
    for (auto& s : tape[i].siblings()) {
      vid.insert({s.id(), i});
    }
  }

  // Backward nodes are made by autodiff, or computed from such nodes, like an
  // optimizer update.
  std::vector<bool> backward(n, false);
  bool has_backward = false;
  for (int i = 0; i < n; ++i) {
    auto& a = tape[i];
    if (!a.has_primitive()) {
      continue;
    }
    bool b = a.is_backward();
    for (auto& in : a.inputs()) {
      b = b || backward[vid.at(in.id())];
    }
    backward[i] = b;
    has_backward |= b;
  }
  if (!has_backward) {
    return;
  }

  auto readers = [&](int i) {
    std::vector<int> rs;
    for (auto& o : tape[i].outputs()) {
      if (auto it = parents_map.find(o.id()); it != parents_map.end()) {
        for (auto& [p, _] : it->second) {
          if (auto r = vid.find(p.id()); r != vid.end()) {
            rs.push_back(r->second);
          }
        }
      }
    }
    return rs;
  };
  auto fusable = [&](int i) {
    return tape[i].has_primitive() && is_fusable(tape[i].primitive());
  };
  auto is_view = [&](int i) {
    if (!tape[i].has_primitive()) {
      return false;
    }
    auto& p = tape[i].primitive();
    return typeid(p) == typeid(Broadcast) || typeid(p) == typeid(Reshape) ||
        typeid(p) == typeid(Transpose) || typeid(p) == typeid(ExpandDims) ||
        typeid(p) == typeid(Squeeze);
  };

  // True if the backward reads node i, directly or through views
  std::function<bool(int)> backward_reads;
  backward_reads = [&](int i) -> bool {
    for (int r : readers(i)) {
      if (backward[r] || (is_view(r) && backward_reads(r))) {
        return true;
      }
    }
    return false;
  };

  // True if the backward reads node i only with fusable ops and not through a
  // view. After a transpose or a reshape the copy does not fuse, for example a
  // weight cast read by a matmul. After a broadcast the copy runs once for each
  // element, for example a scalar learning rate in each optimizer update.
  auto fuses_into_backward = [&](int i) {
    for (int r : readers(i)) {
      if (is_view(r) ? (backward[r] || backward_reads(r))
                     : (backward[r] && !fusable(r))) {
        return false;
      }
    }
    return true;
  };

  MaxFlow graph(2 * n + 2);
  int source = 2 * n;
  int sink = 2 * n + 1;
  for (int i = 0; i < n; ++i) {
    auto& a = tape[i];
    int in = 2 * i;
    int out = 2 * i + 1;
    if (backward[i]) {
      graph.add_edge(in, sink, MaxFlow::inf);
      continue;
    }
    auto rs = readers(i);
    for (int r : rs) {
      graph.add_edge(out, 2 * r, MaxFlow::inf);
    }
    if (!a.has_primitive()) {
      // Inputs and constants stay alive anyway, so keeping them is free.
      graph.add_edge(source, in, MaxFlow::inf);
      graph.add_edge(in, out, 0);
      continue;
    }
    if (is_view(i)) {
      // Views are free to recompute, so they are never kept.
      graph.add_edge(in, out, MaxFlow::inf);
      continue;
    }

    // Keeping a value costs its size. The cost doubles when a fused kernel
    // would otherwise keep the value in registers.
    bool in_registers =
        fusable(i) && !rs.empty() && std::all_of(rs.begin(), rs.end(), fusable);
    int64_t bytes = 0;
    for (auto& o : a.outputs()) {
      bytes += o.nbytes();
    }
    graph.add_edge(in, out, in_registers ? 2 * bytes : bytes);

    // Only cheap elementwise ops are recomputed, and only when the backward
    // can fuse them. Other ops must run in the forward.
    bool recompute =
        fusable(i) && a.siblings().empty() && fuses_into_backward(i);
    if (!recompute) {
      graph.add_edge(source, in, MaxFlow::inf);
    }
  }

  auto kept_bytes = graph.solve(source, sink);
  if (kept_bytes >= MaxFlow::inf) {
    return;
  }
  auto source_side = graph.source_side(source);

  // The value the backward reads for `x`: x itself if it is kept, otherwise a
  // copy computed from kept values.
  std::unordered_map<int, array> copies;
  std::vector<array> new_nodes;
  std::function<array(const array&)> backward_value;
  backward_value = [&](const array& x) -> array {
    int j = vid.at(x.id());
    if (backward[j] || source_side[2 * j]) {
      return x;
    }
    if (auto it = copies.find(j); it != copies.end()) {
      return it->second;
    }
    auto& a = tape[j];
    std::vector<array> inputs;
    for (auto& in : a.inputs()) {
      inputs.push_back(backward_value(in));
    }
    array c(a.shape(), a.dtype(), a.primitive_ptr(), std::move(inputs));
    c.set_backward(true);
    for (int k = 0; k < c.inputs().size(); ++k) {
      parents_map[c.inputs()[k].id()].push_back({c, k});
    }
    copies.insert({j, c});
    new_nodes.push_back(c);
    return c;
  };

  // Point the backward readers at the copies.
  for (int i = 0; i < n; ++i) {
    if (!backward[i]) {
      continue;
    }
    // Give each backward node its own copies, so each kernel can fuse them.
    copies.clear();
    for (int k = 0; k < tape[i].inputs().size(); ++k) {
      auto x = tape[i].inputs()[k];
      auto c = backward_value(x);
      if (c.id() == x.id()) {
        continue;
      }
      auto& from = parents_map[x.id()];
      auto& to = parents_map[c.id()];
      for (auto it = from.begin(); it != from.end();) {
        if (it->second == k && vid.at(it->first.id()) == i) {
          it->first.inputs()[k] = c;
          to.push_back(std::move(*it));
          it = from.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  if (new_nodes.empty()) {
    return;
  }

  // Put each copy in the tape before the first node that reads it.
  std::unordered_map<std::uintptr_t, array> copy_of_id;
  for (auto& c : new_nodes) {
    copy_of_id.insert({c.id(), c});
  }
  std::vector<array> new_tape;
  new_tape.reserve(tape.size() + new_nodes.size());
  std::unordered_set<std::uintptr_t> placed;
  std::function<void(const array&)> place_inputs;
  place_inputs = [&](const array& a) {
    for (auto& in : a.inputs()) {
      auto it = copy_of_id.find(in.id());
      if (it != copy_of_id.end() && placed.insert(in.id()).second) {
        place_inputs(it->second);
        new_tape.push_back(it->second);
      }
    }
  };
  for (auto& a : tape) {
    place_inputs(a);
    new_tape.push_back(a);
  }
  tape = std::move(new_tape);
}

} // namespace mlx::core::detail
