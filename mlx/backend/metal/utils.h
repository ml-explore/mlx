// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core {

namespace {

void set_array_buffer(
    MTL::ComputeCommandEncoder* enc,
    const array& a,
    int idx) {
  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  auto offset = a.data<char>() -
      static_cast<char*>(const_cast<MTL::Buffer*>(a_buf)->contents());
  enc->setBuffer(a_buf, offset, idx);
}

std::string type_to_name(const array& a) {
  std::string tname;
  switch (a.dtype()) {
    case bool_:
      tname = "bool_";
      break;
    case uint8:
      tname = "uint8";
      break;
    case uint16:
      tname = "uint16";
      break;
    case uint32:
      tname = "uint32";
      break;
    case uint64:
      tname = "uint64";
      break;
    case int8:
      tname = "int8";
      break;
    case int16:
      tname = "int16";
      break;
    case int32:
      tname = "int32";
      break;
    case int64:
      tname = "int64";
      break;
    case float16:
      tname = "float16";
      break;
    case float32:
      tname = "float32";
      break;
    case bfloat16:
      tname = "bfloat16";
      break;
    case complex64:
      tname = "complex64";
      break;
  }
  return tname;
}

MTL::Size get_block_dims(int dim0, int dim1, int dim2) {
  int pows[3] = {0, 0, 0};
  int sum = 0;
  while (true) {
    int presum = sum;
    // Check all the pows
    if (dim0 >= (1 << (pows[0] + 1))) {
      pows[0]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (dim1 >= (1 << (pows[1] + 1))) {
      pows[1]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (dim2 >= (1 << (pows[2] + 1))) {
      pows[2]++;
      sum++;
    }
    if (sum == presum || sum == 10) {
      break;
    }
  }
  return MTL::Size{1ul << pows[0], 1ul << pows[1], 1ul << pows[2]};
}

// Collapse dims that are contiguous to possibly route to a better kernel
// e.g. for x = transpose(array({0, 1, 2, 3, 4, 5, 6, 7}, {2, 2, 2}), {2, 0, 1})
// should return {{2, 4}, {{1, 2}}}.
//
// When multiple arrays are passed they should all have the same shape. The
// collapsed axes are also the same so one shape is returned.
std::tuple<std::vector<int>, std::vector<std::vector<size_t>>>
collapse_contiguous_dims(
    const std::vector<int>& shape,
    const std::vector<std::vector<size_t>> strides) {
  // Make a vector that has axes separated with -1. Collapse all axes between
  // -1.
  std::vector<int> to_collapse;
  if (shape.size() > 0) {
    to_collapse.push_back(0);
    for (int i = 1; i < shape.size(); i++) {
      bool contiguous = true;
      for (const std::vector<size_t>& st : strides) {
        if (st[i] * shape[i] != st[i - 1]) {
          contiguous = false;
        }
        if (!contiguous) {
          break;
        }
      }
      if (!contiguous) {
        to_collapse.push_back(-1);
      }
      to_collapse.push_back(i);
    }
    to_collapse.push_back(-1);
  }

  std::vector<int> out_shape;
  std::vector<std::vector<size_t>> out_strides(strides.size());
  for (int i = 0; i < to_collapse.size(); i++) {
    int current_shape = shape[to_collapse[i]];
    while (to_collapse[++i] != -1) {
      current_shape *= shape[to_collapse[i]];
    }
    out_shape.push_back(current_shape);
    for (int j = 0; j < strides.size(); j++) {
      const std::vector<size_t>& st = strides[j];
      out_strides[j].push_back(st[to_collapse[i - 1]]);
    }
  }

  return std::make_tuple(out_shape, out_strides);
}

std::tuple<std::vector<int>, std::vector<std::vector<size_t>>>
collapse_contiguous_dims(const std::vector<array>& xs) {
  std::vector<std::vector<size_t>> strides;
  for (auto& x : xs) {
    strides.emplace_back(x.strides());
  }
  return collapse_contiguous_dims(xs[0].shape(), strides);
}

template <typename... Arrays>
std::tuple<std::vector<int>, std::vector<std::vector<size_t>>>
collapse_contiguous_dims(Arrays... xs) {
  return collapse_contiguous_dims(
      std::vector<array>{std::forward<Arrays>(xs)...});
}

} // namespace

} // namespace mlx::core
