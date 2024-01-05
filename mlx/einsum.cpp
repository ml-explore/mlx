// Copyright © 2023 Apple Inc.

#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>

#include "ops.h"

namespace mlx::core {

std::pair<std::string, std::string> parse_path(const std::string& equation) {
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
  return {lhs, rhs};
}
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
  size_t size = 0;
  while (begin != end) {
    auto c = *begin;
    ++begin;
    size *= dict[c];
  }
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
        auto cont = find_contraction({j.first, j.second}, in_sets, out_set);
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

std::vector<std::tuple<
    std::vector<int>,
    std::set<char>,
    std::string,
    std::vector<std::string>,
    bool>>
einsum_path(
    const std::string& equation,
    const std::vector<array>& operands,
    StreamOrDevice s /** = {} */
) {
  auto extract = parse_path(equation);
  std::vector<std::string> input_list;
  std::stringstream ss(extract.first);
  std::string token;
  while (getline(ss, token, ',')) {
    input_list.push_back(token);
  }
  std::vector<std::set<char>> in_sets;
  std::set<char> out_set(extract.second.begin(), extract.second.end());
  for (auto& input : input_list) {
    std::set<char> temp(input.begin(), input.end());
    in_sets.push_back(temp);
  }
  std::unordered_map<char, int> dim_dict;
  std::vector<std::set<char>> broadcast_indicies;

  for (int i = 0; i < input_list.size(); i++) {
    auto input = input_list[i];
    broadcast_indicies.push_back(std::set<char>());
    for (int j = 0; j < input.size(); j++) {
      auto c = input[j];
      auto dim = operands[i].shape(j);
      if (dim == 1) {
        broadcast_indicies.at(i).insert(c);
      }
      if (dim_dict.find(c) != dim_dict.end()) {
        if (dim != 1 && dim_dict[c] != dim) {
          throw new std::runtime_error("[einsum_path] dimension mismatch");
        }
        dim_dict[c] = std::max(dim_dict[c], dim);
      } else {
        dim_dict[c] = dim;
      }
    }
  }

  size_t max_size =
      term_size(extract.second.begin(), extract.second.end(), dim_dict);
  for (auto input : input_list) {
    max_size =
        std::max(max_size, term_size(input.begin(), input.end(), dim_dict));
  }

  auto path = optimal_path(in_sets, out_set, dim_dict, max_size);
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
      tmp_inputs.push_back(input_list.at(j));
      input_list.erase(input_list.begin() + j);
      bcast.insert(
          broadcast_indicies.at(j).begin(), broadcast_indicies.at(j).end());
      broadcast_indicies.erase(broadcast_indicies.begin() + j);
    }
    if (has_intersection(std::get<2>(cont), bcast)) {
      do_blas = true;
      // can_dot(tmp_inputs, std::get<0>(cont), std::get<2>(cont));
    }
    std::string ein_res = extract.second;
    if ((i - path.size()) != -1) {
      // TODO: do this....
    }
    input_list.emplace_back(ein_res);
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
    new_ein_res += "->";
    new_ein_res += ein_res;
    broadcast_indicies.emplace_back(new_bcast);
    auto in_list_cp = input_list;
    result.emplace_back(
        curr, std::get<2>(cont), new_ein_res, in_list_cp, do_blas);
  }

  return result;
}
} // namespace mlx::core