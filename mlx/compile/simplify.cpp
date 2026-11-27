// Copyright © 2023-2024 Apple Inc.

#include <map>
#include <unordered_map>
#include <unordered_set>

#include "mlx/compile_impl.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::detail {

// Helper like below but only merges the two provided arrays. If the src has
// siblings then these won't be merged to the dst.
void merge_one(array& dst, array& src, ParentsMap& parents_map) {
  auto src_parents = parents_map.find(src.id());
  if (src_parents == parents_map.end()) {
    return;
  }
  auto& pairs = parents_map[dst.id()];
  for (auto& parent : src_parents->second) {
    parent.first.inputs()[parent.second] = dst;
    pairs.push_back(parent);
  }

  // If src is a parent of dst, remove it from dst's parents
  for (auto it = pairs.begin(); it != pairs.end();) {
    if (it->first.id() == src.id()) {
      it = pairs.erase(it);
    } else {
      it++;
    }
  }
  // Remove the source from the map to avoid fusing with it again
  parents_map.erase(src_parents);
}

// Helper that merges two arrays in the graph by setting the parents of the
// source to point to the destination. The arrays are assumed to be coming from
// equivalent primitives so their siblings are merged as well.
void merge(array& dst, array& src, ParentsMap& parents_map) {
  // Canonicalize the order of the primitives outputs
  auto sources = src.outputs();
  auto dests = dst.outputs();
  // For each src parent, point it to the corresponding dst
  for (int i = 0; i < sources.size(); ++i) {
    merge_one(dests[i], sources[i], parents_map);
  }
}

inline uint64_t splitmix64(uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

struct VecU64Hash {
  size_t operator()(const std::vector<uint64_t>& s) const noexcept {
    uint64_t h =
        0x243f6a8885a308d3ull ^ (uint64_t)s.size() * 0x9e3779b97f4a7c15ull;
    for (uint64_t x : s) {
      h = splitmix64(x ^ splitmix64(h + 0x9e3779b97f4a7c15ull));
    }
    return (size_t)h;
  }
};

// Simplify the tape. Note, this function modifies in-place both the tape,
// the parents map to remove orphaned arrays, and potentially the outputs
void compile_simplify(
    std::vector<array>& tape,
    ParentsMap& parents_map,
    std::vector<array>& outputs,
    int passes) {
  // Helpers to identify identical scalars
  std::map<std::pair<uint64_t, Dtype::Val>, array> scalars;
  auto is_scalar = [](const array& a) {
    // Condition for when it's safe to read an array
    return a.is_available() && a.ndim() == 0;
  };
  auto get_scalar_rep = [](const array& a) {
    uint64_t v = 0;
    switch (a.dtype().size()) {
      case 1:
        v = *a.data<uint8_t>();
        break;
      case 2:
        v = *a.data<uint16_t>();
        break;
      case 4:
        v = *a.data<uint32_t>();
        break;
      case 8:
        v = *a.data<uint64_t>();
        break;
    }
    return std::make_pair(v, a.dtype().val());
  };

  for (auto& a : tape) {
    if (is_scalar(a)) {
      scalars.insert({get_scalar_rep(a), a});
    }
  }

  // Depth-1 array equivalence check.
  auto array_equivalent = [](const array& a, const array& b) {
    if (!a.has_primitive() || !b.has_primitive()) {
      return false;
    }
    if (a.primitive_id() == b.primitive_id()) {
      return false;
    }
    const auto& pa = a.primitive();
    const auto& pb = b.primitive();
    if (typeid(pa) != typeid(pb)) {
      return false;
    }

    if (a.inputs().size() != b.inputs().size()) {
      return false;
    }

    for (int i = 0; i < a.inputs().size(); i++) {
      if (a.inputs()[i].id() != b.inputs()[i].id()) {
        return false;
      }
    }

    return pa.is_equivalent(pb);
  };

  std::unordered_map<uintptr_t, array> output_map;
  for (auto& o : outputs) {
    output_map.insert({o.id(), o});
  }

  // Merge scalars, except outputs
  std::vector<array> new_tape;
  for (auto& arr : tape) {
    // Check if we can merge scalars
    if (is_scalar(arr) && output_map.find(arr.id()) == output_map.end()) {
      auto scalar = scalars.find(get_scalar_rep(arr));
      if (scalar->second.id() != arr.id()) {
        merge(scalar->second, arr, parents_map);
        // Don't keep orphaned scalars in the tape
        continue;
      }
    }
    new_tape.push_back(std::move(arr));
  }
  tape = std::move(new_tape);

  // Remove no-ops
  {
    for (auto& arr : tape) {
      if (!arr.has_primitive() || !is_noop(arr.primitive())) {
        new_tape.push_back(std::move(arr));
        continue;
      }
      merge_one(arr.inputs()[0], arr, parents_map);
      if (auto it = output_map.find(arr.id()); it != output_map.end()) {
        it->second = arr.inputs()[0];
      }
    }
    tape = std::move(new_tape);
    for (auto& o : outputs) {
      o = output_map.at(o.id());
    }
  }

  std::unordered_map<std::uintptr_t, uint32_t> tape_order;
  for (uint32_t i = 0; i < tape.size(); ++i) {
    tape_order.insert({tape[i].id(), i});
  }

  // Outputs were rewritten above, so output_map keys are no longer their ids.
  std::unordered_set<uintptr_t> output_set;
  for (auto& o : outputs) {
    output_set.insert(o.id());
  }

  // Multi-pass merge only keeping non-orphaned arrays in the tape
  for (int pass = 0; pass < passes; ++pass) {
    for (auto& arr : tape) {
      // Helper to check if we can merge the parents of the
      // given array
      auto maybe_merge_parents = [&](auto& a) {
        auto parents = parents_map.find(a.id());
        if (parents != parents_map.end()) {
          auto N = parents->second.size();
          std::vector<bool> mask(N, false);

          auto try_merge = [&](int dst_idx, int src_idx) {
            if (tape_order[parents->second[src_idx].first.id()] <
                tape_order[parents->second[dst_idx].first.id()]) {
              std::swap(src_idx, dst_idx);
            }
            auto& src = parents->second[src_idx].first;
            auto& dst = parents->second[dst_idx].first;
            if (src.id() != dst.id() && array_equivalent(src, dst) &&
                output_set.find(src.id()) == output_set.end()) {
              merge(dst, src, parents_map);
              mask[src_idx] = true;
            }
          };

          if (N > 100) {
            std::unordered_map<
                std::vector<uint64_t>,
                std::vector<int>,
                VecU64Hash>
                dst_map;
            // Find possibly mergeable groups
            for (int i = 0; i < N; i++) {
              // Make the hash key
              std::vector<uint64_t> key;
              auto& curr = parents->second[i].first;
              key.reserve(curr.inputs().size() + 2);
              for (auto& in : curr.inputs()) {
                key.push_back(in.id());
              }
              auto& p = curr.primitive();
              key.push_back(curr.inputs().size());
              key.push_back(typeid(p).hash_code());
              auto it = dst_map.find(key);
              if (it == dst_map.end()) {
                bool _;
                std::tie(it, _) = dst_map.insert({key, std::vector<int>{}});
              }
              it->second.push_back(i);
            }
            for (auto& [_, group] : dst_map) {
              for (int i = 0; i < group.size(); ++i) {
                if (mask[group[i]]) {
                  continue;
                }
                for (int j = i + 1; j < group.size(); ++j) {
                  if (mask[group[j]]) {
                    continue;
                  }
                  try_merge(group[i], group[j]);
                }
              }
            }
          } else {
            for (int i = 0; i < N; ++i) {
              if (mask[i]) {
                continue;
              }
              for (int j = i + 1; j < N; ++j) {
                if (mask[j]) {
                  continue;
                }
                try_merge(i, j);
              }
            }
          }

          // Erase orphaned parents so we don't keep fusing with them
          for (int i = N - 1; i >= 0; --i) {
            if (mask[i]) {
              parents->second.erase(parents->second.begin() + i);
            }
          }
          return false;
        } else {
          return output_set.find(a.id()) == output_set.end();
        }
      };
      bool discard = maybe_merge_parents(arr);
      for (auto& s : arr.siblings()) {
        discard &= maybe_merge_parents(s);
      }
      // If an array and its siblings have no parents, and none of them are
      // outputs, it is safe to remove it from the tape
      if (!discard) {
        new_tape.push_back(std::move(arr));
      }
    }
    tape = std::move(new_tape);
  }
}

} // namespace mlx::core::detail
