// Copyright © 2024 Apple Inc.
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "mlx/einsum.h"
#include "mlx/ops.h"

namespace mlx::core {

namespace {

// The MLX einsum implementation is based on NumPy (which is based on
// opt_einsum):
// https://github.com/numpy/numpy/blob/1d49c7f7ff527c696fc26ab2278ad51632a66660/numpy/_core/einsumfunc.py#L743
// https://github.com/dgasmith/opt_einsum

using CharSet = std::unordered_set<char>;

// A helper struct to hold the string and set
// representation of a subscript to avoid needing
// to recompute the set
struct Subscript {
  Subscript(std::string str, CharSet set)
      : str(std::move(str)), set(std::move(set)) {};
  std::string str;
  CharSet set;
};

struct PathInfo {
  size_t naive_cost;
  size_t naive_scaling;
  size_t optimized_cost;
  size_t optimized_scaling;
  size_t largest_term;
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
    // Explicit mode
    lhs = subscripts.substr(0, pos);
    rhs = subscripts.substr(pos + 2);
  } else {
    // Implicit mode:
    // - repeats are summed
    // - remaining output axes are ordered alphabetically
    lhs = subscripts;
    std::unordered_map<char, int> temp;
    for (auto& c : subscripts) {
      if (c == ',') {
        continue;
      }
      auto inserted = temp.insert({c, 0});
      inserted.first->second++;
    }
    for (auto& k : temp) {
      if (k.second == 1) {
        rhs += k.first;
      }
    }
    std::sort(rhs.begin(), rhs.end());
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

std::tuple<std::vector<PathNode>, size_t, int> greedy_path(
    std::vector<Subscript> inputs,
    const Subscript& output,
    std::unordered_map<char, int> dim_dict,
    size_t cost_limit,
    size_t memory_limit) {
  // Helper struct for building the greedy path
  struct Contraction {
    Contraction(
        size_t size,
        size_t cost,
        CharSet output,
        int dims,
        int x,
        int y)
        : size(size),
          cost(cost),
          output(std::move(output)),
          dims(dims),
          x(x),
          y(y) {};

    int64_t size; // Size difference, can be negative
    size_t cost;
    CharSet output;
    int dims; // Number of dimensions in the contraction
    int x;
    int y;
  };

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
  int path_scaling = 0;
  auto num_in = inputs.size();
  for (int i = 0; i < num_in; ++i) {
    auto add_contraction = [&](int p1, int p2) {
      CharSet new_term;
      CharSet contractions(inputs[p1].set.begin(), inputs[p1].set.end());
      contractions.insert(inputs[p2].set.begin(), inputs[p2].set.end());
      for (int i = 0; i < inputs.size(); i++) {
        if (i == p1 || i == p2) {
          continue;
        }
        auto& in = inputs[i].set;
        for (auto c : in) {
          if (contractions.find(c) != contractions.end()) {
            new_term.insert(c);
          }
        }
      }
      for (auto c : output.set) {
        if (contractions.find(c) != contractions.end()) {
          new_term.insert(c);
        }
      }

      // Ignore if:
      // - The size of the new result is greater than the memory limit
      // - The cost is larger than the naive cost
      auto new_size = term_size(new_term, dim_dict);
      if (new_size > memory_limit) {
        return;
      }
      int64_t removed_size = term_size(inputs[p1].set, dim_dict) +
          term_size(inputs[p2].set, dim_dict) - new_size;

      bool inner = contractions.size() > new_term.size();
      auto cost = flop_count(contractions, inner, 2, dim_dict);
      if (path_cost + cost > cost_limit) {
        return;
      }
      possible_contractions.emplace_back(
          removed_size, cost, std::move(new_term), contractions.size(), p1, p2);
    };

    for (auto& [p1, p2] : pos_pairs) {
      // Ignore outer products
      if (!disjoint(inputs[p1].set, inputs[p2].set)) {
        add_contraction(p1, p2);
      }
    }

    // If there's nothing in the contraction list,
    // go over the pairs again without ignoring outer products
    if (possible_contractions.empty()) {
      for (auto& [p1, p2] : pos_pairs) {
        add_contraction(p1, p2);
      }
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
    path_scaling = std::max(best.dims, path_scaling);

    // Construct the output subscripts
    std::string out_str(best.output.begin(), best.output.end());
    // TODO, sorting by dimension size seems suboptimal?
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
  return {path, path_cost, path_scaling};
}

// Assumes inputs have already have had repeats and single axis sums collapsed
bool can_dot(const std::vector<Subscript>& inputs, const Subscript& output) {
  if (inputs.size() != 2) {
    return false;
  }

  for (auto c : inputs[0].set) {
    // Use batched tensordot if anything is being contracted
    if (output.set.find(c) == output.set.end()) {
      return true;
    }
  }
  return false;
}

array batch_tensordot(
    array a,
    array b,
    std::vector<int> a_contract,
    std::vector<int> a_batch,
    std::vector<int> a_concat,
    std::vector<int> b_contract,
    std::vector<int> b_batch,
    std::vector<int> b_concat,
    StreamOrDevice s) {
  // Broadcast contracting dimensions
  {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    for (int i = 0; i < a_contract.size(); ++i) {
      auto d = std::max(a.shape(a_contract[i]), b.shape(b_contract[i]));
      a_shape[a_contract[i]] = d;
      b_shape[b_contract[i]] = d;
    }
    a = broadcast_to(a, a_shape, s);
    b = broadcast_to(b, b_shape, s);
  }

  auto transpose_reshape = [&s](
                               const array& x,
                               const std::vector<int>& i,
                               const std::vector<int>& j,
                               const std::vector<int>& k) {
    std::vector<int> reorder(i.begin(), i.end());
    reorder.insert(reorder.end(), j.begin(), j.end());
    reorder.insert(reorder.end(), k.begin(), k.end());

    int size1 = 1;
    for (auto s : j) {
      size1 *= x.shape(s);
    }

    int size2 = 1;
    for (auto s : k) {
      size2 *= x.shape(s);
    }

    std::vector<int> shape;
    for (auto ax : i) {
      shape.push_back(x.shape(ax));
    }
    shape.push_back(size1);
    shape.push_back(size2);

    return reshape(transpose(x, reorder, s), std::move(shape), s);
  };

  std::vector<int> out_shape;
  for (auto ax : a_batch) {
    out_shape.push_back(a.shape(ax));
  }
  for (auto ax : a_concat) {
    out_shape.push_back(a.shape(ax));
  }
  for (auto ax : b_concat) {
    out_shape.push_back(b.shape(ax));
  }

  a = transpose_reshape(a, a_batch, a_concat, a_contract);
  b = transpose_reshape(b, b_batch, b_contract, b_concat);

  return reshape(matmul(a, b, s), std::move(out_shape), s);
}

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

// Collapse repeat indices and sum single dimensions.
// For example:
// - "aa" becomes "a"
// - "ij,jk->k" becoms "j,jk->k"
void preprocess_einsum_inputs(
    std::vector<Subscript>& inputs,
    const Subscript& output,
    std::vector<array>& operands,
    StreamOrDevice s) {
  // Collapse repeat indices
  for (int i = 0; i < inputs.size(); ++i) {
    auto& in = inputs[i];
    if (in.set.size() < in.str.size()) {
      operands[i] = collapse_repeats(operands[i], in, s);
    }
  }

  // Sum indices that are only in a single input
  {
    std::unordered_map<char, int> counts;
    for (auto& in : inputs) {
      for (auto c : in.set) {
        auto inserted = counts.insert({c, 0});
        inserted.first->second++;
      }
    }
    for (auto c : output.set) {
      auto inserted = counts.insert({c, 0});
      inserted.first->second++;
    }
    for (int i = 0; i < inputs.size(); ++i) {
      auto& in = inputs[i];
      std::vector<int> sum_axes;
      for (int ax = 0; ax < in.str.size(); ++ax) {
        if (counts[in.str[ax]] == 1) {
          sum_axes.push_back(ax);
        }
      }
      if (!sum_axes.empty()) {
        operands[i] = sum(operands[i], sum_axes, false, s);
      }
      for (auto it = sum_axes.rbegin(); it != sum_axes.rend(); ++it) {
        in.set.erase(in.str[*it]);
        in.str.erase(in.str.begin() + *it);
      }
    }
  }
}

array einsum_naive(
    std::vector<Subscript> inputs,
    const Subscript& output,
    std::vector<array> operands,
    StreamOrDevice s) {
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
      continue;
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
  if (!sum_axes.empty()) {
    out = sum(out, sum_axes, false, s);
  }

  // Transpose output if needed
  std::vector<int> reorder;
  for (auto c : output.str) {
    reorder.push_back(char_to_ax[c]);
  }
  for (auto& r : reorder) {
    int offset = 0;
    for (auto s : sum_axes) {
      if (r > s) {
        offset++;
      }
    }
    r -= offset;
  }
  return transpose(out, reorder, s);
}

std::pair<std::vector<PathNode>, PathInfo> einsum_path_helper(
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

    for (int j = 0; j < in.size(); j++) {
      auto c = in[j];
      auto dim = operands[i].shape(j);
      auto inserted = dim_map.insert({c, dim});
      auto& in_dim = inserted.first->second;
      if (dim != 1 && in_dim != 1 && in_dim != dim) {
        std::ostringstream msg;
        msg << "[" << fn_name << "] Cannot broadcast dimension " << j
            << " of input " << i << " with shape " << operands[i].shape()
            << " to size " << in_dim << ".";
        throw std::invalid_argument(msg.str());
      }
      // Ensure the broadcasted size is used
      in_dim = std::max(in_dim, dim);
    }
  }

  size_t max_size = term_size(out_subscript, dim_map);
  for (auto& in : in_subscripts) {
    max_size = std::max(max_size, term_size(in, dim_map));
  }

  PathInfo path_info;

  // Get the full naive cost
  size_t naive_cost;
  {
    CharSet contractions;
    for (auto& in : inputs) {
      contractions.insert(in.set.begin(), in.set.end());
    }

    bool inner = false;
    for (auto c : contractions) {
      if (output.set.find(c) == output.set.end()) {
        inner = true;
        break;
      }
    }
    naive_cost = flop_count(contractions, inner, inputs.size(), dim_map);
    path_info.naive_cost = naive_cost;
  }
  path_info.naive_scaling = dim_map.size();

  // Calculate the path
  std::vector<PathNode> path;
  if (inputs.size() <= 2) {
    std::vector<int> positions(in_subscripts.size());
    std::iota(positions.begin(), positions.end(), 0);
    path.emplace_back(
        std::move(inputs), std::move(output), std::move(positions));
  } else {
    std::tie(path, path_info.optimized_cost, path_info.optimized_scaling) =
        greedy_path(inputs, output, dim_map, naive_cost, max_size);
  }
  return {path, path_info};
}

} // namespace

std::pair<std::vector<std::vector<int>>, std::string> einsum_path(
    const std::string& subscripts,
    const std::vector<array>& operands) {
  auto [path, path_info] =
      einsum_path_helper(subscripts, operands, "einsum_path");

  std::vector<std::vector<int>> pos_path;
  for (auto& p : path) {
    pos_path.push_back(p.positions);
  }

  std::ostringstream path_print;
  path_print << "  Complete contraction:  " << subscripts << "\n"
             << "         Naive scaling:  " << path_info.naive_scaling << "\n"
             << "     Optimized scaling:  " << path_info.optimized_scaling
             << "\n"
             << "      Naive FLOP count:  " << path_info.naive_cost << "\n"
             << "  Optimized FLOP count:  " << path_info.optimized_cost << "\n";
  // TODO add more info here
  return {pos_path, path_print.str()};
}

array einsum(
    const std::string& subscripts,
    const std::vector<array>& operands,
    StreamOrDevice s /* = {} */) {
  auto [path, path_info] = einsum_path_helper(subscripts, operands, "einsum");
  auto inputs = operands;
  for (auto node : path) {
    preprocess_einsum_inputs(node.inputs, node.output, inputs, s);

    if (can_dot(node.inputs, node.output)) {
      auto extract_axes =
          [](const auto& input, const auto& other, const auto& output) {
            std::vector<int> contract;
            std::vector<int> batch;
            std::vector<int> concat;
            for (int i = 0; i < input.str.size(); ++i) {
              auto c = input.str[i];
              if (output.set.find(c) == output.set.end()) {
                // Not in the output, contraction
                contract.push_back(i);
              } else if (other.set.find(c) != other.set.end()) {
                // Not a contraction but in both inputs, batch dim
                batch.push_back(i);
              } else {
                // Not a batch dim or contract dim, so concat dim
                concat.push_back(i);
              }
            }
            return std::make_tuple(contract, batch, concat);
          };
      auto [a_contract, a_batch, a_concat] =
          extract_axes(node.inputs[0], node.inputs[1], node.output);
      auto [b_contract, b_batch, b_concat] =
          extract_axes(node.inputs[1], node.inputs[0], node.output);
      auto& a = inputs[node.positions[0]];
      auto& b = inputs[node.positions[1]];

      std::unordered_map<char, int> char_map;
      for (auto i : a_batch) {
        char_map.insert({node.inputs[0].str[i], char_map.size()});
      }
      for (auto i : a_concat) {
        char_map.insert({node.inputs[0].str[i], char_map.size()});
      }
      for (auto i : b_concat) {
        char_map.insert({node.inputs[1].str[i], char_map.size()});
      }

      inputs.emplace_back(batch_tensordot(
          a,
          b,
          std::move(a_contract),
          std::move(a_batch),
          std::move(a_concat),
          std::move(b_contract),
          std::move(b_batch),
          std::move(b_concat),
          s));

      std::vector<int> reorder;
      for (auto c : node.output.str) {
        reorder.push_back(char_map[c]);
      }
      inputs.back() = transpose(inputs.back(), reorder, s);

    } else {
      inputs.emplace_back(einsum_naive(node.inputs, node.output, inputs, s));
    }

    // Positions are always sorted increasing, so start from the back
    for (auto it = node.positions.rbegin(); it != node.positions.rend(); ++it) {
      inputs.erase(inputs.begin() + *it);
    }
  }
  return inputs.front();
}

} // namespace mlx::core
