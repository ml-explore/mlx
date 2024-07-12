// Copyright © 2024 Apple Inc.
#include <iostream> // TODO
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <numeric>
#include "mlx/einsum.h"
#include "mlx/ops.h"

namespace mlx::core {

using CharSet = std::unordered_set<char>;

// A little helper struct to hold the string and set
// representation of a subscript to avoid needing
// to keep recomputing the set
struct Subscript {
  Subscript(std::string str, CharSet set)
      : str(std::move(str)), set(std::move(set)) {};
  std::string str;
  CharSet set;
};

struct PathNode {
  PathNode(
      std::vector<Subscript> inputs,
      Subscript output,
      std::vector<int> positions)
      : inputs(std::move(inputs)),
        output(std::move(output)),
        positions(std::move(positions)) {};

  std::vector<Subscript> inputs;
  Subscript output;

  std::vector<int> positions;
};

struct Contraction {
  Contraction(size_t size, size_t cost, CharSet output, int x, int y)
      : size(size), cost(cost), output(std::move(output)), x(x), y(y) {};

  size_t size;
  size_t cost;
  CharSet output;
  int x;
  int y;
};

namespace {

// The MLX einsum implementation is based on NumPy (which is based on
// opt_einsum):
// https://github.com/numpy/numpy/blob/1d49c7f7ff527c696fc26ab2278ad51632a66660/numpy/_core/einsumfunc.py#L743
// https://github.com/dgasmith/opt_einsum

// Parse the comma separated subscripts into a vector of strings. If the
// output subscripts are missing they are inferred.
//
// For example:
//  "ij,jk -> ik" becomes {{"ij", "jk"}, "ik"}
//  "ij,jk" becomes {{"ij", "jk"}, "ik"}
std::pair<std::vector<std::string>, std::string> parse(std::string subscripts) {
  std::string lhs, rhs;

  // Start by removing all white space
  subscripts.erase(
      std::remove(subscripts.begin(), subscripts.end(), ' '), subscripts.end());

  if (auto pos = subscripts.find("->"); pos != std::string::npos) {
    lhs = subscripts.substr(0, pos);
    rhs = subscripts.substr(pos + 2);
  } else {
    lhs = subscripts;
    std::unordered_map<char, int> temp;
    for (auto& c : subscripts) {
      if (c == ',') {
        continue;
      }
      if (auto it = temp.find(c); it != temp.end()) {
        it->second += 1;
      } else {
        temp.insert({c, 1});
      }
    }
    for (auto& k : temp) {
      if (k.second == 1) {
        rhs += k.first;
      }
    }
  }
  std::vector<std::string> input_list;
  std::stringstream ss(lhs);
  std::string token;
  while (getline(ss, token, ',')) {
    input_list.push_back(token);
  }
  return {input_list, rhs};
}

// Check if two sets are disjoint
bool disjoint(const CharSet& x, const CharSet& y) {
  for (auto& c : x) {
    if (y.find(c) != y.end()) {
      return false;
    }
  }
  return true;
}

// Intersect two sets
CharSet intersect(const CharSet& x, const CharSet& y) {
  CharSet intersection;
  for (auto& a : x) {
    if (y.find(a) != y.end()) {
      intersection.insert(a);
    }
  }
  return intersection;
}

template <typename T>
size_t term_size(const T& term, std::unordered_map<char, int> dict) {
  size_t size = 1;
  for (auto c : term) {
    size *= dict[c];
  }
  return size;
}

size_t flop_count(
    const CharSet& term,
    bool inner,
    int num_terms,
    std::unordered_map<char, int> dict) {
  size_t size = term_size(term, dict);
  auto op_factor = 1;
  if ((num_terms - 1) > op_factor) {
    op_factor = num_terms - 1;
  }
  if (inner) {
    op_factor += 1;
  }
  return size * op_factor;
}

// Look for a contraction using the given positions.
//
// Returns:
// - The subscripts of the contracted result
// - The subscripts participating in the contraction
std::pair<CharSet, CharSet> contract_two(
    int p1,
    int p2,
    const std::vector<Subscript>& inputs,
    const Subscript& out) {
  CharSet idx_contracted;
  CharSet idx_remaining(out.set);
  for (int i = 0; i < inputs.size(); i++) {
    auto& in = inputs[i].set;
    if (i == p1 || i == p2) {
      idx_contracted.insert(in.begin(), in.end());
    } else {
      idx_remaining.insert(in.begin(), in.end());
    }
  }

  // The subscripts of the contracted result
  auto new_result = intersect(idx_remaining, idx_contracted);
  return {new_result, idx_contracted};
}

// Contract all the inputs (e.g. naive einsum)
std::pair<CharSet, CharSet> contract_all(
    const std::vector<Subscript>& inputs,
    const Subscript& out) {
  CharSet idx_contracted;
  for (auto& in : inputs) {
    idx_contracted.insert(in.set.begin(), in.set.end());
  }

  // The subscripts of the contracted result
  auto new_result = intersect(out.set, idx_contracted);

  return {new_result, idx_contracted};
}

std::vector<PathNode> greedy_path(
    std::vector<Subscript> inputs,
    const Subscript& output,
    std::unordered_map<char, int> dim_dict,
    size_t memory_limit) {
  // Get the full naive cost
  size_t naive_cost;
  {
    auto [new_term, contractions] = contract_all(inputs, output);
    naive_cost = flop_count(
        contractions,
        contractions.size() > new_term.size(),
        inputs.size(),
        dim_dict);
  }

  // Start by iterating over all possible combinations
  std::vector<std::pair<int, int>> pos_pairs;
  for (int i = 0; i < inputs.size(); ++i) {
    for (int j = i + 1; j < inputs.size(); ++j) {
      pos_pairs.emplace_back(i, j);
    }
  }

  std::vector<PathNode> path;
  std::vector<Contraction> possible_contractions;
  size_t path_cost = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    for (auto& [p1, p2] : pos_pairs) {
      // Ignore outer products
      if (disjoint(inputs[p1].set, inputs[p2].set)) {
        continue;
      }

      // Find possible contraction
      auto [new_term, contractions] = contract_two(p1, p2, inputs, output);

      // Ignore if:
      // - The size of the new result is greater than the memory limit
      // - The cost is larger than the naive cost
      auto new_size = term_size(new_term, dim_dict);
      if (new_size > memory_limit) {
        continue;
      }
      auto removed_size = term_size(inputs[p1].set, dim_dict) +
          term_size(inputs[p2].set, dim_dict) - new_size;

      bool inner = contractions.size() > new_term.size();
      auto cost = flop_count(contractions, inner, 2, dim_dict);
      if (path_cost + cost > naive_cost) {
        continue;
      }
      possible_contractions.emplace_back(
          removed_size, cost, std::move(new_term), p1, p2);
    }

    // If there's nothing in the contraction list,
    // go over the pairs again without ignoring outer products
    if (possible_contractions.empty()) {
    }

    if (possible_contractions.empty()) {
      break;
    }

    // Find the best contraction
    auto& best = *std::min_element(
        possible_contractions.begin(),
        possible_contractions.end(),
        [](const auto& x, const auto& y) {
          return x.size > y.size || (x.size == y.size && x.cost < y.cost);
        });

    // Construct the output subscripts
    std::string out_str(best.output.begin(), best.output.end());
    // TODO, sorting by dimension size seems suboptimal
    std::sort(out_str.begin(), out_str.end(), [&dim_dict](auto x, auto y) {
      return dim_dict[x] < dim_dict[y];
    });
    Subscript new_output(std::move(out_str), std::move(best.output));

    // Add the chosen contraction to the path
    {
      std::vector<Subscript> in_terms;
      in_terms.push_back(std::move(inputs[best.y]));
      in_terms.push_back(std::move(inputs[best.x]));
      path.emplace_back(
          std::move(in_terms), new_output, std::vector<int>{best.x, best.y});
    }
    // Remove used terms
    inputs.erase(inputs.begin() + best.y);
    inputs.erase(inputs.begin() + best.x);

    // Add the new result
    inputs.push_back(std::move(new_output));

    // Update the existing contractions based on the selected one
    std::vector<Contraction> updated_contractions;
    for (auto& contraction : possible_contractions) {
      // Drop contractions which contain either selected term
      if (contraction.x == best.x || contraction.x == best.y ||
          contraction.y == best.x || contraction.y == best.y) {
        continue;
      }

      // Update the positions of other contractions
      int x =
          contraction.x - (contraction.x > best.x) - (contraction.x > best.y);
      int y =
          contraction.y - (contraction.y > best.x) - (contraction.y > best.y);
      contraction.x = x;
      contraction.y = y;
      updated_contractions.push_back(std::move(contraction));
    }

    pos_pairs.clear();
    for (int i = 0; i < inputs.size() - 1; ++i) {
      pos_pairs.emplace_back(i, inputs.size() - 1);
    }
    path_cost += best.cost;

    possible_contractions = std::move(updated_contractions);
  }
  return path;
}

template <typename Map>
bool comp_map(Map const& lhs, Map const& rhs) {
  return lhs.size() == rhs.size() &&
      std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename Map>
bool comp_keys(Map const& lhs, Map const& rhs) {
  return lhs.size() == rhs.size() &&
      std::equal(
             lhs.begin(),
             lhs.end(),
             rhs.begin(),
             [](auto const& lhs_pair, auto const& rhs_pair) {
               return lhs_pair.first == rhs_pair.first;
             });
}

bool can_dot(
    const std::vector<CharSet>& inputs,
    const CharSet& result,
    const CharSet& removed) {
  if (removed.size() == 0) {
    return false;
  }
  if (inputs.size() != 2) {
    return false;
  }

  /*  auto& in_left = inputs[0];
    auto& in_left = inputs[0];
    for (auto c : in_left) {

    }*/
}

// std::map<char, int> str_idx_map(const std::string inp) {
//   std::map<char, int> counts;
//   int i = 0;
//   for (auto c : inp) {
//     if (c != ' ' && counts.find(c) == counts.end()) {
//       counts[c] = i;
//       i += 1;
//     }
//   }
//   return counts;
// }

// Collapse repeated subscripts and return the resulting array. The subscript
// is also updated in place. For example:
// - Given an input with shape (4, 4) and subscript "ii", returns
//   the diagonal of shape (4,) and updates the subscript to "i".
// - Given an input with shape (4, 2, 4, 2) and subscript "ijij",
//   returns an output with shape (4, 2) and updates the subscript
//   to "ij".
array collapse_repeats(array in, Subscript& subscript, StreamOrDevice s) {
  // Build a list of (repeat chars, num repeats, first axis the char appears in)
  auto& str = subscript.str;
  std::string new_str;
  std::vector<std::tuple<char, int, int>> repeats;
  {
    std::unordered_map<char, std::pair<int, int>> counts;
    for (int i = 0; i < str.size(); ++i) {
      auto [it, _] = counts.insert({str[i], {0, i}});
      if (it->second.first == 0) {
        new_str.push_back(str[i]);
      }
      it->second.first++;
    }

    for (auto& v : counts) {
      if (v.second.first >= 2) {
        repeats.emplace_back(v.first, v.second.first, v.second.second);
      }
    }
  }

  // Sort by the first axis of each repeat
  std::sort(repeats.begin(), repeats.end(), [](const auto& x, const auto& y) {
    return std::get<2>(x) < std::get<2>(y);
  });

  // Build the inputs for gather
  auto slice_sizes = in.shape();
  std::vector<int> axes;
  std::vector<array> indices;
  int n_expand = repeats.size();
  for (auto [c, v, _] : repeats) {
    for (int i = 0; i < str.size(); ++i) {
      if (str[i] == c) {
        slice_sizes[i] = 1;
        axes.push_back(i);
      }
    }
    std::vector<int> idx_shape(n_expand--, 1);
    idx_shape[0] = in.shape(axes.back());
    auto idx = reshape(arange(in.shape(axes.back()), s), idx_shape, s);
    for (int i = 0; i < v; ++i) {
      indices.push_back(idx);
    }
  }

  in = gather(in, indices, axes, slice_sizes, s);

  // Update subscript string with removed dups
  str = new_str;

  // Squeeze singleton dimensions left over from the gather
  for (auto& ax : axes) {
    ax += indices[0].ndim();
  }
  return squeeze(in, axes, s);
}

array einsum_naive(
    std::vector<Subscript> inputs,
    const Subscript& output,
    std::vector<array> operands,
    StreamOrDevice s) {
  // Collapse repeat indices
  for (int i = 0; i < inputs.size(); ++i) {
    auto& in = inputs[i];
    // TODO the inputs should have the same shape and the same char
    if (in.set.size() < in.str.size()) {
      operands[i] = collapse_repeats(operands[i], in, s);
    }
  }

  // Map each character to an axis
  std::unordered_map<char, int> char_to_ax;
  for (auto& in : inputs) {
    for (auto c : in.str) {
      char_to_ax.insert({c, char_to_ax.size()});
    }
  }

  // Expand and transpose inputs as needed
  for (int i = 0; i < inputs.size(); ++i) {
    auto& op = operands[i];

    // Add missing dimensions at the end
    std::cout << op.ndim() << " " << char_to_ax.size() << std::endl;
    if (op.ndim() != char_to_ax.size()) {
      auto shape = operands[i].shape();
      shape.insert(shape.end(), char_to_ax.size() - shape.size(), 1);
      op = reshape(op, std::move(shape), s);
    }

    // Transpose:
    // - Build a vector of (char, ax) pairs for the current input
    // - Sort the vector by the canonical axis in char_to_ax
    // - Extract the sorted axis to get transpose order
    std::vector<std::pair<char, int>> str_ax;
    for (auto c : inputs[i].str) {
      str_ax.emplace_back(c, str_ax.size());
    }
    for (auto [c, ax] : char_to_ax) {
      if (inputs[i].set.find(c) == inputs[i].set.end()) {
        str_ax.emplace_back(c, str_ax.size());
      }
    }
    std::sort(
        str_ax.begin(),
        str_ax.end(),
        [&char_to_ax](const auto& x, const auto& y) {
          return char_to_ax[x.first] < char_to_ax[y.first];
        });

    // Skip the transpose if not needed
    if (std::is_sorted(
            str_ax.begin(), str_ax.end(), [](const auto& x, const auto& y) {
              return x.second < y.second;
            })) {
      std::cout << "SKIP? " << std::endl;
      break;
    }

    std::vector<int> reorder;
    for (auto [c, ax] : str_ax) {
      reorder.push_back(ax);
    }
    op = transpose(op, reorder, s);
  }

  // Multiply and sum
  auto out = operands[0];
  for (int i = 1; i < operands.size(); ++i) {
    out = multiply(out, operands[i], s);
  }
  std::vector<int> sum_axes;
  for (auto [c, ax] : char_to_ax) {
    if (output.set.find(c) == output.set.end()) {
      sum_axes.push_back(ax);
    }
  }
  std::cout << "SUM AXES " << sum_axes << std::endl;
  if (!sum_axes.empty()) {
    out = sum(out, sum_axes, true, s);
  }

  // Transpose output if needed
  std::vector<int> reorder;
  for (auto c : output.str) {
    reorder.push_back(char_to_ax[c]);
  }
  reorder.insert(reorder.end(), sum_axes.begin(), sum_axes.end());
  out = transpose(out, reorder, s);

  // Remove reduced axes
  if (!sum_axes.empty()) {
    out = squeeze(out, sum_axes, s);
  }

  return out;
}

std::pair<std::vector<PathNode>, std::unordered_map<char, int>>
einsum_path_helper(
    const std::string& subscripts,
    const std::vector<array>& operands,
    const std::string& fn_name) {
  if (operands.size() == 0) {
    std::ostringstream msg;
    msg << "[" << fn_name << "] At least one operand is required.";
    throw std::invalid_argument(msg.str());
  }

  auto [in_subscripts, out_subscript] = parse(subscripts);

  if (operands.size() != in_subscripts.size()) {
    std::ostringstream msg;
    msg << "[" << fn_name << "] Number of operands, " << operands.size()
        << ", does not match number of input subscripts, "
        << in_subscripts.size();
    throw std::invalid_argument(msg.str());
  }

  auto check_letters = [&](const auto& subscript) {
    for (auto c : subscript) {
      if (!isalpha(c)) {
        std::ostringstream msg;
        msg << "[" << fn_name << "] Subscripts must be letters, but got '" << c
            << "'.";
        throw std::invalid_argument(msg.str());
      }
    }
  };
  for (auto& in : in_subscripts) {
    check_letters(in);
  }
  check_letters(out_subscript);

  CharSet out_set(out_subscript.begin(), out_subscript.end());
  if (out_set.size() != out_subscript.size()) {
    std::ostringstream msg;
    msg << "[" << fn_name << "] Repeat indices not allowed in output.";
    throw std::invalid_argument(msg.str());
  }
  Subscript output(out_subscript, std::move(out_set));

  std::unordered_map<char, int> dim_map;
  //  std::vector<CharSet> broadcast_indicies;
  std::vector<Subscript> inputs;
  for (int i = 0; i < in_subscripts.size(); ++i) {
    auto& in = in_subscripts[i];
    CharSet in_set(in.begin(), in.end());
    inputs.emplace_back(in, in_set);

    if (in.size() != operands[i].ndim()) {
      std::ostringstream msg;
      msg << "[" << fn_name << "] Invalid number of subscripts " << in.size()
          << " for input " << i << " with " << operands[i].ndim()
          << " dimensions.";
      throw std::invalid_argument(msg.str());
    }

    // Check repeat subscripts are valid
    if (in_set.size() < in.size()) {
      std::unordered_map<char, int> local_dims;
      for (int j = 0; j < in.size(); ++j) {
        auto dim = operands[i].shape(j);
        auto inserted = local_dims.insert({in[j], dim});
        if (!inserted.second) {
          if (inserted.first->second != dim) {
            std::ostringstream msg;
            msg << "[" << fn_name << "] Dimensions of repeated subscripts "
                << "do not have the same size (" << inserted.first->second
                << " != " << dim << ").";
            throw std::invalid_argument(msg.str());
          }
        }
      }
    }

    //    broadcast_indicies.push_back({});
    for (int j = 0; j < in.size(); j++) {
      auto c = in[j];
      auto dim = operands[i].shape(j);
      //      if (dim == 1) {
      //        broadcast_indices.at(i).insert(c);
      //      }
      if (auto it = dim_map.find(c); it != dim_map.end()) {
        if (dim != 1 && it->second != dim) {
          std::ostringstream msg;
          msg << "[" << fn_name << "] Cannot broadcast dimension " << j
              << " of input " << i << " with shape " << operands[i].shape()
              << " to size " << it->second << ".";
          throw std::invalid_argument(msg.str());
        }
        // Ensure the broadcasted size is used
        it->second = std::max(it->second, dim);
      } else {
        dim_map[c] = dim;
      }
    }
  }

  size_t max_size = term_size(out_subscript, dim_map);
  for (auto& in : in_subscripts) {
    max_size = std::max(max_size, term_size(in, dim_map));
  }

  // Calculate the path
  std::vector<PathNode> path;
  if (inputs.size() <= 2) {
    std::vector<int> positions(in_subscripts.size());
    std::iota(positions.begin(), positions.end(), 0);
    path.emplace_back(inputs, output, std::move(positions));
  } else {
    path = greedy_path(inputs, output, dim_map, max_size);
  }
  return {path, dim_map};
}

} // namespace

/** Computes an einsum_path */
std::pair<std::vector<std::vector<int>>, std::string> einsum_path(
    const std::string& subscripts,
    const std::vector<array>& operands) {
  auto [path, dim_map] =
      einsum_path_helper(subscripts, operands, "einsum_path");

  // At the end
  std::vector<std::vector<int>> pos_path;
  for (auto& p : path) {
    pos_path.push_back(p.positions);
  }
  return {pos_path, ""};

  // Now we have the path
  // Go through and construct path nodes
  // - Get the input subscripts
  // - Recompute (?) output subscripts
  // - Do the updates, and continue until path is processed
  /*  std::vector<EinsumPath> result;
    // Go through the generated path and construct einsum path
    for (int i = 0; i < path.size(); i++) {
      auto curr = path[i];
      // sort by greater idx so that pop later does not mess up the order
      std::sort(curr.begin(), curr.end(), std::greater<int>());

      auto cont = find_contraction(curr, in_sets, out_set);
      in_sets = std::get<1>(cont);

      bool do_blas = false;
      CharSet bcast;
      std::vector<std::string> tmp_inputs;

      for (auto j : curr) {
        tmp_inputs.push_back(extract.first.at(j));
        extract.first.erase(extract.first.begin() + j);
        bcast.insert(
            broadcast_indicies.at(j).begin(), broadcast_indicies.at(j).end());
        broadcast_indicies.erase(broadcast_indicies.begin() + j);
      }
      // check if tensordot can be used
      if (!has_intersection(std::get<2>(cont), bcast)) {
        do_blas = can_dot(tmp_inputs, std::get<0>(cont), std::get<2>(cont));
      }

      // Construct the sub-einsum subscripts
      std::string ein_res = extract.second;
      if ((i - path.size()) != -1) {
        std::string tmp(std::get<0>(cont).begin(), std::get<0>(cont).end());
        std::sort(tmp.begin(), tmp.end(), [dim_map](char a, char b) {
          return dim_map.find(a)->second < dim_map.find(b)->second;
        });
        ein_res = tmp;
      }
      extract.first.emplace_back(ein_res);
      std::string new_ein_res;
      for (auto ti : tmp_inputs) {
        new_ein_res += ti;
        new_ein_res += ",";
      }
      new_ein_res.pop_back();
      new_ein_res += "->";
      new_ein_res += ein_res;
      // finish constructing the sub-einsum subscripts

      CharSet new_bcast;
      std::set_difference(
          bcast.begin(),
          bcast.end(),
          std::get<2>(cont).begin(),
          std::get<2>(cont).end(),
          std::inserter(new_bcast, new_bcast.begin()));
      broadcast_indicies.emplace_back(new_bcast);
      result.push_back({curr, std::get<2>(cont), new_ein_res, do_blas});
    }

    return result;*/
}

array einsum(
    const std::string& subscripts,
    const std::vector<array>& operands,
    StreamOrDevice s /* = {} */) {
  std::vector<PathNode> path;
  std::unordered_map<char, int> dim_map;
  std::tie(path, dim_map) = einsum_path_helper(subscripts, operands, "einsum");
  auto inputs = operands;
  for (auto node : path) {
    // Needs to know "can dot"
    // Inputs and output (contraction has input pos, output pos
    // C
    // remove items from inputs
    if (false) { // TODO check if can dot
      auto extract_axes = [&node, &dim_map](const auto& input) {
        std::vector<int> axes;
        for (auto c : input.str) {
          //          if (node.output.set.find(c) != node.output.set.end()) {
          //            axes.push_back(dim_map[c]);
          //          }
        }
        return axes;
      };
      auto a_axes = extract_axes(node.inputs[0]);
      auto b_axes = extract_axes(node.inputs[1]);
      auto& a = inputs[node.positions[0]];
      auto& b = inputs[node.positions[1]];
      inputs.emplace_back(tensordot(a, b, a_axes, b_axes, s));
    } else {
      inputs.emplace_back(einsum_naive(node.inputs, node.output, operands, s));
    }

    // Positions are always sorted increasing, so start from the back
    for (auto it = node.positions.rbegin(); it != node.positions.rend(); ++it) {
      inputs.erase(inputs.begin() + *it);
    }
  }
  return inputs.front();
}

} // namespace mlx::core
