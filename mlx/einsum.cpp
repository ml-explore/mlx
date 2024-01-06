// Copyright © 2023 Apple Inc.

#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>

#include "ops.h"

namespace mlx::core {

std::pair<std::vector<std::string>, std::string> einsum_parse(
    const std::string& equation) {
  std::string lhs, rhs;
  if (equation.find("->") != std::string::npos) {
    auto pos = equation.find("->");
    lhs = equation.substr(0, pos);
    rhs = equation.substr(pos + 2);
  } else {
    lhs = equation;
    std::map<char, int> temp;
    for (int i = 0; i < equation.size(); i++) {
      if (equation[i] == ',') {
        continue;
      }
      if (temp.find(equation[i]) != temp.end()) {
        temp[equation[i]] += 1;
      } else {
        temp[equation[i]] = 1;
      }
    }
    for (auto k : temp) {
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

/**
 * Calculates the size of a term
 */
template <typename Iter>
size_t term_size(Iter begin, Iter end, std::unordered_map<char, int> dict) {
  size_t size = 1;
  while (begin != end) {
    auto c = *begin;
    ++begin;
    size *= dict[c];
  }
  return size;
}

template <typename Iter>
size_t flop_count(
    Iter begin,
    Iter end,
    bool inner,
    int num_terms,
    std::unordered_map<char, int> dict) {
  size_t size = term_size(begin, end, dict);
  auto op_factor = 1;
  if ((num_terms - 1) > op_factor) {
    op_factor = num_terms - 1;
  }
  if (inner) {
    op_factor += 1;
  }
  return size * op_factor;
}

std::tuple<
    std::set<char>,
    std::vector<std::set<char>>,
    std::set<char>,
    std::set<char>>
find_contraction(
    std::vector<int> positions,
    std::vector<std::set<char>> in_sets,
    std::set<char> out_set) {
  std::set<char> idx_contracted;
  std::set<char> idx_remaining(out_set);
  std::vector<std::set<char>> remaining_sets;
  for (int i = 0; i < in_sets.size(); i++) {
    if (auto it = std::find(positions.begin(), positions.end(), i);
        it != positions.end()) {
      idx_contracted.insert(in_sets[i].begin(), in_sets[i].end());
    } else {
      remaining_sets.push_back(in_sets[i]);
      idx_remaining.insert(in_sets[i].begin(), in_sets[i].end());
    }
  }
  std::set<char> idx_contracted_remaining;
  std::set_intersection(
      idx_remaining.begin(),
      idx_remaining.end(),
      idx_contracted.begin(),
      idx_contracted.end(),
      std::inserter(
          idx_contracted_remaining, idx_contracted_remaining.begin()));
  remaining_sets.push_back(idx_contracted_remaining);
  std::set<char> idx_removed;
  std::set_difference(
      idx_contracted.begin(),
      idx_contracted.end(),
      idx_contracted_remaining.begin(),
      idx_contracted_remaining.end(),
      std::inserter(idx_removed, idx_removed.begin()));
  return {
      idx_contracted_remaining, remaining_sets, idx_removed, idx_contracted};
}

/** helper functions */
std::vector<int> rangeHelper(int r) {
  std::vector<int> result(r);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

std::vector<std::pair<int, int>> combinations(std::vector<int> r) {
  std::vector<std::pair<int, int>> result;
  for (int i = 0; i < r.size(); i++) {
    for (int j = i + 1; j < r.size(); j++) {
      std::pair<int, int> p = {r[i], r[j]};
      result.emplace_back(p);
    }
  }
  return result;
}

std::map<char, int> str_idx_map(const std::string inp) {
  std::map<char, int> counts;
  int i = 0;
  for (auto c : inp) {
    if (c != ' ' && counts.find(c) == counts.end()) {
      counts[c] = i;
      i += 1;
    }
  }
  return counts;
}

std::vector<std::vector<int>> optimal_path(
    std::vector<std::set<char>> in_sets,
    std::set<char> out_set,
    std::unordered_map<char, int> dim_dict,
    size_t max_size) {
  std::vector<std::tuple<
      int,
      std::vector<std::vector<int>>,
      std::vector<std::set<char>>>>
      results;
  results.push_back({0, {}, in_sets});
  for (int i = 0; i < in_sets.size() - 1; i++) {
    std::vector<std::tuple<
        int,
        std::vector<std::vector<int>>,
        std::vector<std::set<char>>>>
        temp;
    for (auto curr : results) {
      for (auto j : combinations(rangeHelper(in_sets.size() - i))) {
        auto cont =
            find_contraction({j.first, j.second}, std::get<2>(curr), out_set);
        auto size = term_size(
            std::get<0>(cont).begin(), std::get<0>(cont).end(), dim_dict);
        if (size > max_size) {
          continue;
        }
        auto total_cost = std::get<0>(curr) +
            flop_count(std::get<3>(cont).begin(),
                       std::get<3>(cont).end(),
                       std::get<2>(cont).size() > 0,
                       2,
                       dim_dict);
        std::vector<std::vector<int>> positions(
            std::get<1>(curr).begin(), std::get<1>(curr).end());
        std::vector<int> jv = {j.first, j.second};
        positions.emplace_back(jv);
        temp.emplace_back(total_cost, positions, std::get<1>(cont));
      }
    }
    if (temp.size() > 0) {
      results = temp;
    } else {
      auto path_parent = results.at(0);
      for (int i = 1; i < results.size(); i++) {
        if (std::get<0>(results.at(i)) < std::get<0>(path_parent)) {
          path_parent = results.at(i);
        }
      }
      auto positions = std::get<1>(path_parent);
      positions.emplace_back(rangeHelper(in_sets.size() - i));
      return positions;
    }
  }
  if (results.size() == 0) {
    return {rangeHelper(in_sets.size())};
  }
  auto path_parent = results.at(0);
  for (int i = 1; i < results.size(); i++) {
    if (std::get<0>(results.at(i)) < std::get<0>(path_parent)) {
      path_parent = results.at(i);
    }
  }
  return std::get<1>(path_parent);
}

bool has_intersection(std::set<char> a, std::set<char> b) {
  std::set<char> intersection;
  std::set_intersection(
      a.begin(),
      a.end(),
      b.begin(),
      b.end(),
      std::inserter(intersection, intersection.begin()));
  return intersection.size() > 0;
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
    std::vector<std::string> inputs,
    std::set<char> result,
    std::set<char> idx_removed) {
  if (idx_removed.size() == 0) {
    return false;
  }
  if (inputs.size() != 2) {
    return false;
  }
  std::unordered_map<char, int> lhs_count;
  for (auto c : inputs.at(0)) {
    if (lhs_count.find(c) != lhs_count.end()) {
      lhs_count[c] += 1;
    } else {
      lhs_count[c] = 1;
    }
  }
  std::unordered_map<char, int> rhs_count;
  for (auto c : inputs.at(1)) {
    if (rhs_count.find(c) != rhs_count.end()) {
      rhs_count[c] += 1;
    } else {
      rhs_count[c] = 1;
    }
  }
  if (comp_map(lhs_count, rhs_count)) {
    return true;
  }
  for (auto k : lhs_count) {
    auto rc = 0;
    if (rhs_count.find(k.first) != rhs_count.end()) {
      rc = rhs_count[k.first];
    }
    if (k.second > 1 || rc > 1 || (k.second + rc > 2)) {
      return false;
    }
    auto fc = 0;
    if (result.find(k.first) != result.end()) {
      fc = 1;
    }
    if ((k.second + rc - 1) == fc) {
      return false;
    }
  }
  for (auto k : rhs_count) {
    auto lc = 0;
    if (lhs_count.find(k.first) != lhs_count.end()) {
      lc = lhs_count[k.first];
    }
    if (k.second > 1 || lc > 1 || (k.second + lc > 2)) {
      return false;
    }
    auto fc = 0;
    if (result.find(k.first) != result.end()) {
      fc = 1;
    }
    if ((k.second + lc - 1) == fc) {
      return false;
    }
  }
  if (comp_keys(lhs_count, rhs_count)) {
    return false;
  }
  auto rs = idx_removed.size();
  if (inputs.at(0).substr(0, rs) == inputs.at(1).substr(0, rs)) {
    return true;
  }
  if (inputs.at(0).substr(0, rs) ==
      inputs.at(1).substr(inputs.at(1).size() - rs)) {
    return true;
  }
  if (inputs.at(0).substr(inputs.at(0).size() - rs) ==
      inputs.at(1).substr(0, rs)) {
    return true;
  }
  if (inputs.at(0).substr(inputs.at(0).size() - rs) ==
      inputs.at(1).substr(inputs.at(1).size() - rs)) {
    return true;
  }
  std::set<char> kleft;
  std::set<char> kright;
  for (auto k : lhs_count) {
    if (idx_removed.find(k.first) == idx_removed.end()) {
      kleft.insert(k.first);
    }
  }
  for (auto k : rhs_count) {
    if (idx_removed.find(k.first) == idx_removed.end()) {
      kright.insert(k.first);
    }
  }
  if (kleft.size() == 0 || kright.size() == 0) {
    return false;
  }
  return true;
}

/** Computes the optimal einsum_path */
std::vector<std::tuple<
    std::vector<int>,
    std::set<char>,
    std::string,
    std::vector<std::string>,
    bool>>
einsum_path(const std::string& equation, const std::vector<array>& operands) {
  auto extract = einsum_parse(equation);
  if (operands.size() != extract.first.size()) {
    throw std::invalid_argument("[einsum_path] operands size mismatch");
  }
  std::vector<std::set<char>> in_sets;
  std::set<char> out_set(extract.second.begin(), extract.second.end());
  for (auto& input : extract.first) {
    std::set<char> temp(input.begin(), input.end());
    in_sets.push_back(temp);
  }
  std::unordered_map<char, int> dim_map;
  std::vector<std::set<char>> broadcast_indicies;

  for (int i = 0; i < extract.first.size(); i++) {
    auto input = extract.first[i];
    broadcast_indicies.push_back(std::set<char>());
    for (int j = 0; j < input.size(); j++) {
      auto c = input[j];
      auto dim = operands[i].shape(j);
      if (dim == 1) {
        broadcast_indicies.at(i).insert(c);
      }
      if (dim_map.find(c) != dim_map.end()) {
        if (dim != 1 && dim_map[c] != dim) {
          throw new std::runtime_error("[einsum_path] dimension mismatch");
        }
        dim_map[c] = std::max(dim_map[c], dim);
      } else {
        dim_map[c] = dim;
      }
    }
  }

  size_t max_size =
      term_size(extract.second.begin(), extract.second.end(), dim_map);
  for (auto input : extract.first) {
    max_size =
        std::max(max_size, term_size(input.begin(), input.end(), dim_map));
  }
  // calculate the optimal path
  auto path = optimal_path(in_sets, out_set, dim_map, max_size);
  std::vector<std::tuple<
      std::vector<int>,
      std::set<char>,
      std::string,
      std::vector<std::string>,
      bool>>
      result;

  for (int i = 0; i < path.size(); i++) {
    auto curr = path[i];
    std::sort(curr.begin(), curr.end(), std::greater<int>());
    auto cont = find_contraction(curr, in_sets, out_set);
    in_sets = std::get<1>(cont);
    bool do_blas = false;
    std::set<char> bcast;
    std::vector<std::string> tmp_inputs;
    for (auto j : curr) {
      tmp_inputs.push_back(extract.first.at(j));
      extract.first.erase(extract.first.begin() + j);
      bcast.insert(
          broadcast_indicies.at(j).begin(), broadcast_indicies.at(j).end());
      broadcast_indicies.erase(broadcast_indicies.begin() + j);
    }
    if (!has_intersection(std::get<2>(cont), bcast)) {
      do_blas = can_dot(tmp_inputs, std::get<0>(cont), std::get<2>(cont));
    }
    std::string ein_res = extract.second;
    if ((i - path.size()) != -1) {
      std::string tmp(std::get<0>(cont).begin(), std::get<0>(cont).end());
      std::sort(tmp.begin(), tmp.end(), [dim_map](char a, char b) {
        auto pa = dim_map.find(a)->second;
        auto pb = dim_map.find(b)->second;
        return pa < pb;
      });
      ein_res = tmp;
    }
    extract.first.emplace_back(ein_res);
    std::set<char> new_bcast;
    std::set_difference(
        bcast.begin(),
        bcast.end(),
        std::get<2>(cont).begin(),
        std::get<2>(cont).end(),
        std::inserter(new_bcast, new_bcast.begin()));
    std::string new_ein_res;
    for (auto ti : tmp_inputs) {
      new_ein_res += ti;
      new_ein_res += ",";
    }
    new_ein_res.pop_back();
    new_ein_res += "->";
    new_ein_res += ein_res;
    broadcast_indicies.emplace_back(new_bcast);
    auto in_list_cp = extract.first;
    result.emplace_back(
        curr, std::get<2>(cont), new_ein_res, in_list_cp, do_blas);
  }

  return result;
}

array einsum_naive(
    std::string equation,
    const std::vector<array>& operands,
    StreamOrDevice s /* = {} */) {
  if (operands.empty()) {
    throw std::runtime_error("[einsum] Must provide at least one operand");
  }
  auto extract = einsum_parse(equation);

  if (operands.size() != extract.first.size()) {
    throw std::runtime_error(
        "[einsum] Number of operands (" + std::to_string(operands.size()) +
        ") must match the number of input characters(" +
        std::to_string(extract.first.size()) + ")");
  }

  std::map<char, int> input_map;
  for (int i = 0; i < extract.first.size(); i++) {
    auto arr = operands[i];
    auto inp = extract.first[i];
    for (int j = 0; j < std::min(arr.shape().size(), inp.size()); j++) {
      input_map[inp[j]] = arr.shape(j);
    }
  }
  std::vector<int> broad;
  for (auto key : input_map) {
    broad.push_back(key.second);
  }
  std::vector<array> inputs_arr;
  for (int i = 0; i < operands.size(); i++) {
    auto arr = operands[i];
    auto ord_map = str_idx_map(extract.first[i]);
    std::vector<int> new_shape;
    for (auto key : input_map) {
      if (ord_map.find(key.first) != ord_map.end()) {
        new_shape.push_back(key.second);
      } else {
        new_shape.push_back(1);
      }
    }
    std::vector<int> axis;
    for (auto key : ord_map) {
      axis.push_back(key.second);
    }
    inputs_arr.push_back(
        broadcast_to(reshape(transpose(arr, axis, s), new_shape, s), broad, s));
  }

  auto ord_output = str_idx_map(extract.second);
  std::vector<int> rhs_order;
  for (auto key : ord_output) {
    rhs_order.push_back(key.second);
  }

  std::vector<int> sum_axis;
  int i = 0;
  for (auto key : input_map) {
    if (ord_output.find(key.first) == ord_output.end()) {
      sum_axis.push_back(i);
    }
    i += 1;
  }
  // TODO: this should just start with the first and then accumulate
  auto acc = ones_like(inputs_arr.at(0), s);
  for (int i = 0; i < inputs_arr.size(); i++) {
    acc = multiply(acc, inputs_arr[i], s);
  }
  return transpose(sum(acc, sum_axis, false, s), rhs_order, s);
}
} // namespace mlx::core