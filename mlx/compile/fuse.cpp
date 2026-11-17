// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "mlx/compile_impl.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr int max_compile_depth = 11;
constexpr int max_compile_arrays = 24;
constexpr int max_fuse_distance = 0;

namespace detail {

bool is_unary(const Primitive& p) {
  return (
      typeid(p) == typeid(Abs) || typeid(p) == typeid(ArcCos) ||
      typeid(p) == typeid(ArcCosh) || typeid(p) == typeid(ArcSin) ||
      typeid(p) == typeid(ArcSinh) || typeid(p) == typeid(ArcTan) ||
      typeid(p) == typeid(ArcTanh) || typeid(p) == typeid(AsType) ||
      typeid(p) == typeid(Ceil) || typeid(p) == typeid(Cos) ||
      typeid(p) == typeid(Conjugate) || typeid(p) == typeid(Cosh) ||
      typeid(p) == typeid(Remainder) || typeid(p) == typeid(Erf) ||
      typeid(p) == typeid(ErfInv) || typeid(p) == typeid(Exp) ||
      typeid(p) == typeid(Floor) || typeid(p) == typeid(Log) ||
      typeid(p) == typeid(Log1p) || typeid(p) == typeid(LogicalNot) ||
      typeid(p) == typeid(Negative) || typeid(p) == typeid(Round) ||
      typeid(p) == typeid(Sigmoid) || typeid(p) == typeid(Sign) ||
      typeid(p) == typeid(Sin) || typeid(p) == typeid(Sinh) ||
      typeid(p) == typeid(Square) || typeid(p) == typeid(Sqrt) ||
      typeid(p) == typeid(Tan) || typeid(p) == typeid(Tanh) ||
      typeid(p) == typeid(Expm1) || typeid(p) == typeid(Real) ||
      typeid(p) == typeid(Imag) || typeid(p) == typeid(BitwiseInvert));
}

bool is_binary(const Primitive& p) {
  return typeid(p) == typeid(Add) || typeid(p) == typeid(ArcTan2) ||
      typeid(p) == typeid(BitwiseBinary) || typeid(p) == typeid(Divide) ||
      typeid(p) == typeid(Equal) || typeid(p) == typeid(FloorDivide) ||
      typeid(p) == typeid(Greater) || typeid(p) == typeid(GreaterEqual) ||
      typeid(p) == typeid(Less) || typeid(p) == typeid(LessEqual) ||
      typeid(p) == typeid(LogicalNot) || typeid(p) == typeid(LogicalAnd) ||
      typeid(p) == typeid(LogicalOr) || typeid(p) == typeid(LogAddExp) ||
      typeid(p) == typeid(Maximum) || typeid(p) == typeid(Minimum) ||
      typeid(p) == typeid(Multiply) || typeid(p) == typeid(NotEqual) ||
      typeid(p) == typeid(Power) || typeid(p) == typeid(Subtract);
}

bool is_ternary(const Primitive& p) {
  return typeid(p) == typeid(Select);
}

bool is_broadcast(const Primitive& p) {
  return typeid(p) == typeid(Broadcast);
}

bool is_noop(const Primitive& p) {
  return typeid(p) == typeid(Copy) || typeid(p) == typeid(StopGradient);
}

bool is_reduction(const Primitive& p) {
  return typeid(p) == typeid(Reduce) || typeid(p) == typeid(ArgReduce);
}

bool is_fusable(const Primitive& p) {
  return is_unary(p) || is_binary(p) || is_ternary(p) || is_broadcast(p);
}

// Any parent in the divider will continue to refer to `x` but any parent not
// in the divider will refer to a copy of the operation.
array split_one(
    const array& x,
    ParentsMap& parents_map,
    const std::unordered_set<uintptr_t>& divider) {
  array y(x.shape(), x.dtype(), x.primitive_ptr(), x.inputs());

  auto& x_parents = parents_map[x.id()];
  auto& y_parents = parents_map[y.id()];

  for (auto it = x_parents.begin(); it != x_parents.end();) {
    if (divider.find(it->first.id()) != divider.end()) {
      it->first.inputs()[it->second] = y;
      y_parents.emplace_back(std::move(*it));
      it = x_parents.erase(it);
    } else {
      it++;
    }
  }

  return y;
}

// Extract sub-graphs of the graph that can be compiled
// and replace them with a Compiled Primitive.
// Replace the arrays in `fused_tape` with one Compiled primitive. The arrays in
// `old_outputs` become the outputs of the primitive.
std::vector<array> make_compiled(
    std::vector<array> fused_tape,
    const std::vector<array>& inputs,
    std::vector<array> old_outputs,
    const std::unordered_set<uintptr_t>& input_ids,
    std::unordered_map<uintptr_t, array>& output_map,
    ParentsMap& parents_map) {
  std::unordered_set<uintptr_t> fused_ids;
  for (auto& f : fused_tape) {
    fused_ids.insert(f.id());
  }
  std::unordered_set<uintptr_t> output_ids;
  for (auto& o : old_outputs) {
    output_ids.insert(o.id());
  }

  for (auto& f : fused_tape) {
    if (output_ids.find(f.id()) != output_ids.end()) {
      // Parents are now siblings, update the parent map
      auto& pairs = parents_map[f.id()];
      pairs.erase(
          std::remove_if(
              pairs.begin(),
              pairs.end(),
              [&](auto& p) {
                return fused_ids.find(p.first.id()) != fused_ids.end();
              }),
          pairs.end());
    } else {
      // Remove inner fused arrays parents from the parents map
      // to keep the parents map in a valid state
      parents_map.erase(f.id());
    }
  }

  std::vector<Shape> shapes;
  std::vector<Dtype> types;
  for (auto& o : old_outputs) {
    if (o.shape() != old_outputs.back().shape()) {
      throw std::runtime_error(
          "[compile] Compilation failed. Tried to fuse operations with different output shapes");
    }
    shapes.push_back(o.shape());
    types.push_back(o.dtype());
  }
  std::unordered_set<uintptr_t> constant_ids;
  for (auto& in : inputs) {
    // Scalar constant
    if (in.size() == 1 && !in.has_primitive() &&
        input_ids.find(in.id()) == input_ids.end()) {
      constant_ids.insert(in.id());
    }
  }
  auto compiled_outputs = array::make_arrays(
      std::move(shapes),
      types,
      std::make_shared<Compiled>(
          old_outputs.back().primitive().stream(),
          inputs,
          old_outputs,
          std::move(fused_tape),
          std::move(constant_ids)),
      inputs);

  // Replace inputs old parents with compiled_outputs
  for (int i = 0; i < inputs.size(); ++i) {
    auto& pairs = parents_map[inputs[i].id()];
    pairs.erase(
        std::remove_if(
            pairs.begin(),
            pairs.end(),
            [&](auto& p) {
              return fused_ids.find(p.first.id()) != fused_ids.end();
            }),
        pairs.end());
    for (auto& o : compiled_outputs) {
      pairs.push_back({o, i});
    }
  }

  // - Update outputs parents to point to compiled outputs
  // - Update any overall graph outputs to be compiled outputs
  for (int o = 0; o < old_outputs.size(); ++o) {
    merge_one(compiled_outputs[o], old_outputs[o], parents_map);
    if (auto it = output_map.find(old_outputs[o].id());
        it != output_map.end()) {
      it->second = compiled_outputs[o];
    }
  }
  return compiled_outputs;
}

// Groups of arrays to fuse. Each group with more than one array becomes one
// Compiled primitive.
struct FuseGroups {
  std::vector<std::vector<array>> members;
  std::unordered_map<uintptr_t, int> group_of;

  // The group of `a`, or -1 if `a` is not fusable
  int find(const array& a) const {
    auto it = group_of.find(a.id());
    return (it == group_of.end()) ? -1 : it->second;
  }

  void add(const array& a, int g) {
    group_of[a.id()] = g;
    members[g].push_back(a);
  }

  void make_group(const array& a) {
    members.emplace_back();
    add(a, members.size() - 1);
  }

  // Move the arrays of group `src` to group `dst`
  void merge(int dst, int src) {
    for (auto& a : members[src]) {
      add(a, dst);
    }
    members[src].clear();
  }

  // Undo merge(dst, src), where `n` is the size of `dst` before the merge
  void unmerge(int dst, int src, int n) {
    for (int i = n; i < members[dst].size(); ++i) {
      add(members[dst][i], src);
    }
    members[dst].erase(members[dst].begin() + n, members[dst].end());
  }
};

// True if `a` is read outside of its group or is an output of the graph
bool is_group_output(
    const array& a,
    const FuseGroups& groups,
    const ParentsMap& parents_map,
    const std::unordered_map<uintptr_t, array>& output_map) {
  if (output_map.find(a.id()) != output_map.end()) {
    return true;
  }
  int g = groups.find(a);
  if (auto it = parents_map.find(a.id()); it != parents_map.end()) {
    for (auto& [p, idx] : it->second) {
      if (groups.find(p) != g) {
        return true;
      }
    }
  }
  return false;
}

// True if group `g` can become one Compiled primitive
bool is_valid_group(
    int g,
    const FuseGroups& groups,
    const ParentsMap& parents_map,
    const std::unordered_map<uintptr_t, array>& output_map) {
  // All outputs have the same shape. Do not write out a broadcast.
  std::vector<array> outputs;
  for (auto& a : groups.members[g]) {
    if (!is_group_output(a, groups, parents_map, output_map)) {
      continue;
    }
    if (is_broadcast(a.primitive()) &&
        output_map.find(a.id()) == output_map.end()) {
      return false;
    }
    if (!outputs.empty() && a.shape() != outputs[0].shape()) {
      return false;
    }
    outputs.push_back(a);
  }

  // Count the inputs and find the longest chain of arrays in the group
  std::unordered_set<uintptr_t> inputs;
  std::unordered_map<uintptr_t, int> depths;
  std::function<int(const array&)> recurse;
  recurse = [&](const array& a) -> int {
    if (groups.find(a) != g) {
      inputs.insert(a.id());
      return 0;
    }
    if (auto it = depths.find(a.id()); it != depths.end()) {
      return it->second;
    }
    int depth = 0;
    for (auto& in : a.inputs()) {
      depth = std::max(depth, recurse(in));
    }
    depths.insert({a.id(), depth + 1});
    return depth + 1;
  };
  int depth = 0;
  for (auto& a : groups.members[g]) {
    depth = std::max(depth, recurse(a));
  }
  return depth <= max_compile_depth &&
      inputs.size() + outputs.size() <= max_compile_arrays;
}

// Give group `g` its own copy of the broadcast `b` with split_one. Return
// true if the copy is fused into the group.
bool split_broadcast(
    array b,
    int g,
    FuseGroups& groups,
    ParentsMap& parents_map,
    const std::unordered_map<uintptr_t, array>& output_map) {
  std::unordered_set<uintptr_t> divider;
  for (auto& a : groups.members[g]) {
    divider.insert(a.id());
  }
  array c = split_one(b, parents_map, divider);
  for (int i = 0; i < c.inputs().size(); ++i) {
    parents_map[c.inputs()[i].id()].push_back({c, i});
  }
  groups.add(c, g);
  if (is_valid_group(g, groups, parents_map, output_map)) {
    return true;
  }

  // Undo the copy
  groups.members[g].pop_back();
  groups.group_of.erase(c.id());
  for (auto& in : c.inputs()) {
    auto& pairs = parents_map[in.id()];
    pairs.erase(
        std::remove_if(
            pairs.begin(),
            pairs.end(),
            [&](auto& p) { return p.first.id() == c.id(); }),
        pairs.end());
  }
  merge_one(b, c, parents_map);
  return false;
}

// Fuse a group into the group that reads it, if all the readers of the group
// are in that group. Return true if a group changed.
bool compile_fuse_vertical(
    FuseGroups& groups,
    const std::vector<array>& tape,
    ParentsMap& parents_map,
    const std::unordered_map<uintptr_t, array>& output_map) {
  bool changed = false;

  // Go through the tape in reverse order so readers are fused first
  for (int i = tape.size() - 1; i >= 0; --i) {
    auto& a = tape[i];
    int src = groups.find(a);
    if (src < 0) {
      continue;
    }

    // Find the groups that read the group of `a`
    std::vector<int> readers;
    bool all_fusable = true;
    for (auto& m : groups.members[src]) {
      auto it = parents_map.find(m.id());
      if (it == parents_map.end()) {
        continue;
      }
      for (auto& [p, idx] : it->second) {
        int dst = groups.find(p);
        if (dst < 0) {
          all_fusable = false;
        } else if (
            dst != src &&
            std::find(readers.begin(), readers.end(), dst) == readers.end()) {
          readers.push_back(dst);
        }
      }
    }

    // Split a broadcast with readers in more than one place, so that each
    // group can fuse its own copy
    if (is_broadcast(a.primitive()) && groups.members[src].size() == 1 &&
        (readers.size() > 1 || (readers.size() == 1 && !all_fusable))) {
      for (int dst : readers) {
        changed |= split_broadcast(a, dst, groups, parents_map, output_map);
      }
      continue;
    }

    if (!all_fusable || readers.size() != 1) {
      continue;
    }
    int dst = readers[0];
    if (a.primitive().stream() != groups.members[dst][0].primitive().stream()) {
      continue;
    }
    int n = groups.members[dst].size();
    groups.merge(dst, src);
    if (is_valid_group(dst, groups, parents_map, output_map)) {
      changed = true;
    } else {
      groups.unmerge(dst, src, n);
    }
  }
  return changed;
}

// True if a path through arrays outside of groups `g1` and `g2` connects them.
// The two groups cannot be one kernel then.
bool is_connected(
    int g1,
    int g2,
    const FuseGroups& groups,
    const std::unordered_map<uintptr_t, int>& position,
    const ParentsMap& parents_map) {
  auto in_groups = [&](const array& a) {
    int g = groups.find(a);
    return g == g1 || g == g2;
  };

  // Readers come after their inputs in the tape, so stop the search after the
  // last array of the two groups
  int last = 0;
  for (int g : {g1, g2}) {
    for (auto& a : groups.members[g]) {
      if (auto it = position.find(a.id()); it != position.end()) {
        last = std::max(last, it->second);
      }
    }
  }

  // Start from the readers outside of the two groups
  std::vector<array> stack;
  for (int g : {g1, g2}) {
    for (auto& a : groups.members[g]) {
      if (auto it = parents_map.find(a.id()); it != parents_map.end()) {
        for (auto& [p, idx] : it->second) {
          if (!in_groups(p)) {
            stack.push_back(p);
          }
        }
      }
    }
  }

  std::unordered_set<uintptr_t> visited;
  while (!stack.empty()) {
    auto a = std::move(stack.back());
    stack.pop_back();
    if (in_groups(a)) {
      return true;
    }
    if (!visited.insert(a.id()).second) {
      continue;
    }
    if (auto it = position.find(a.id());
        it != position.end() && it->second > last) {
      continue;
    }
    if (auto it = parents_map.find(a.id()); it != parents_map.end()) {
      for (auto& [p, idx] : it->second) {
        stack.push_back(p);
      }
    }
  }
  return false;
}

// Fuse two groups which read the same arrays, if no path through other arrays
// connects them and they run at about the same time in the eval order. Return
// true if a group changed.
bool compile_fuse_horizontal(
    FuseGroups& groups,
    const std::vector<array>& tape,
    const std::unordered_map<uintptr_t, int>& position,
    ParentsMap& parents_map,
    const std::unordered_map<uintptr_t, array>& output_map) {
  // Score each pair of groups with the bytes of the arrays they both read
  std::map<std::pair<int, int>, size_t> scores;
  for (auto& x : tape) {
    // Scalars become constants in the kernel, so they are not read
    if (x.size() == 1) {
      continue;
    }
    auto it = parents_map.find(x.id());
    if (it == parents_map.end()) {
      continue;
    }
    std::vector<int> readers;
    for (auto& [p, idx] : it->second) {
      int g = groups.find(p);
      if (g >= 0 && g != groups.find(x) &&
          std::find(readers.begin(), readers.end(), g) == readers.end()) {
        readers.push_back(g);
      }
    }
    for (int i = 0; i < readers.size(); ++i) {
      for (int j = i + 1; j < readers.size(); ++j) {
        auto key = std::minmax(readers[i], readers[j]);
        scores[{key.first, key.second}] += x.nbytes();
      }
    }
  }

  // Eval runs an array as late as possible, just before its readers. So the
  // level of an array, its longest path to an output, sets when it runs.
  std::unordered_map<uintptr_t, int> levels;
  std::function<int(const array&)> level;
  level = [&](const array& a) -> int {
    if (auto it = levels.find(a.id()); it != levels.end()) {
      return it->second;
    }
    int l = 0;
    if (auto it = parents_map.find(a.id()); it != parents_map.end()) {
      for (auto& [p, idx] : it->second) {
        l = std::max(l, level(p) + 1);
      }
    }
    levels.insert({a.id(), l});
    return l;
  };

  // A group runs at the level of its deepest output
  auto group_level = [&](int g) {
    int l = 0;
    for (auto& a : groups.members[g]) {
      if (is_group_output(a, groups, parents_map, output_map)) {
        l = std::max(l, level(a));
      }
    }
    return l;
  };
  auto distance = [&](int g1, int g2) {
    return std::abs(group_level(g1) - group_level(g2));
  };
  auto is_used = [&](int g) {
    for (auto& a : groups.members[g]) {
      if (is_group_output(a, groups, parents_map, output_map)) {
        return true;
      }
    }
    return false;
  };

  // Try the pairs with the most shared bytes first, then the closest pairs.
  // Keep one array of each group, since groups change when they merge.
  struct Candidate {
    size_t score;
    int distance;
    array a;
    array b;
  };
  std::vector<Candidate> candidates;
  for (auto& [key, score] : scores) {
    auto [g1, g2] = key;
    candidates.push_back(
        {score,
         distance(g1, g2),
         groups.members[g1][0],
         groups.members[g2][0]});
  }
  std::sort(candidates.begin(), candidates.end(), [](auto& x, auto& y) {
    return (x.score != y.score) ? x.score > y.score : x.distance < y.distance;
  });

  // Merging groups at different levels moves a kernel in the eval order, and
  // this can keep large arrays longer in memory
  bool changed = false;
  for (auto& c : candidates) {
    int g1 = groups.find(c.a);
    int g2 = groups.find(c.b);
    if (g1 == g2 || c.a.primitive().stream() != c.b.primitive().stream() ||
        distance(g1, g2) > max_fuse_distance || !is_used(g1) || !is_used(g2) ||
        is_connected(g1, g2, groups, position, parents_map)) {
      continue;
    }
    int size = groups.members[g1].size();
    groups.merge(g1, g2);
    if (is_valid_group(g1, groups, parents_map, output_map)) {
      changed = true;
    } else {
      groups.unmerge(g1, g2, size);
    }
  }
  return changed;
}

// Fuse arrays in groups with the fusion rules, then replace each group with
// one Compiled primitive.
void compile_fuse(
    std::vector<array>& tape,
    ParentsMap& parents_map,
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Track outputs to replace with new compiled outputs
  std::unordered_map<uintptr_t, array> output_map;
  for (auto& o : outputs) {
    output_map.insert({o.id(), o});
  }

  // Set of inputs to distinguish constants
  std::unordered_set<uintptr_t> input_ids;
  for (auto& in : inputs) {
    input_ids.insert(in.id());
  }

  // Start with one group for each fusable array
  FuseGroups groups;
  std::unordered_map<uintptr_t, int> position;
  for (int i = 0; i < tape.size(); ++i) {
    auto& a = tape[i];
    position.insert({a.id(), i});
    if (a.has_primitive() && is_fusable(a.primitive()) &&
        input_ids.find(a.id()) == input_ids.end()) {
      groups.make_group(a);
    }
  }

  // Merge groups until no rule changes them. A merge can make a new merge
  // possible, so repeat the rules.
  while (compile_fuse_vertical(groups, tape, parents_map, output_map) ||
         compile_fuse_horizontal(
             groups, tape, position, parents_map, output_map)) {
  }

  // Replace each group of more than one array with a Compiled primitive
  for (int g = 0; g < groups.members.size(); ++g) {
    auto& group = groups.members[g];
    if (group.size() <= 1) {
      continue;
    }

    // The roots are the arrays which are not read in the group
    std::vector<array> roots;
    for (auto& a : group) {
      bool is_root = true;
      if (auto it = parents_map.find(a.id()); it != parents_map.end()) {
        for (auto& [p, idx] : it->second) {
          if (groups.find(p) == g) {
            is_root = false;
            break;
          }
        }
      }
      if (is_root) {
        roots.push_back(a);
      }
    }
    std::sort(roots.begin(), roots.end(), [&](auto& a, auto& b) {
      return position.at(a.id()) < position.at(b.id());
    });

    // Build the tape of the group in the right order and collect the inputs
    std::vector<array> inputs;
    std::vector<array> fused_tape;
    std::unordered_set<uintptr_t> input_set;
    std::unordered_set<uintptr_t> tape_set;
    std::function<void(const array&)> recurse_tape;
    recurse_tape = [&](const array& a) {
      if (groups.find(a) != g) {
        if (input_set.find(a.id()) == input_set.end()) {
          input_set.insert(a.id());
          inputs.push_back(a);
        }
        return;
      }
      if (tape_set.find(a.id()) != tape_set.end()) {
        return;
      }
      tape_set.insert(a.id());
      for (auto& in : a.inputs()) {
        recurse_tape(in);
      }
      fused_tape.push_back(a);
    };
    for (auto& r : roots) {
      recurse_tape(r);
    }

    std::vector<array> old_outputs;
    for (auto& f : fused_tape) {
      if (is_group_output(f, groups, parents_map, output_map)) {
        old_outputs.push_back(f);
      }
    }

    // Nothing reads the group, so it is not used
    if (old_outputs.empty()) {
      continue;
    }
    make_compiled(
        std::move(fused_tape),
        inputs,
        std::move(old_outputs),
        input_ids,
        output_map,
        parents_map);
  }

  // Build the new tape in execution order
  std::vector<array> new_tape;
  std::unordered_set<uintptr_t> cache;
  std::function<void(const array&)> recurse;
  recurse = [&](const array& a) {
    if (cache.find(a.id()) != cache.end()) {
      return;
    }
    // Don't recurse on inputs
    if (input_ids.find(a.id()) == input_ids.end()) {
      for (auto& in : a.inputs()) {
        recurse(in);
      }
    }
    cache.insert(a.id());
    for (auto& s : a.siblings()) {
      cache.insert(s.id());
    }
    new_tape.push_back(a);
  };
  for (auto& o : outputs) {
    o = output_map.at(o.id());
    recurse(o);
  }
  tape = std::move(new_tape);
}

} // namespace detail

} // namespace mlx::core
