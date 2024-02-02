// Copyright © 2023 Apple Inc.

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/select.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T>
void select_binary_op(
    const array& a,
    const array& b,
    array& out,
    BinaryOpType bopt,
    bool invert_predicate) {
  SelectScalarScalar<T> opss(invert_predicate);
  SelectScalarVector<T> opsv(invert_predicate);
  SelectVectorScalar<T> opvs(invert_predicate);
  SelectVectorVector<T> opvv(invert_predicate);
  binary_op<bool, T, T>(a, b, out, bopt, opss, opsv, opvs, opvv);
}

void select_with_predicate(
    const array& a,
    const array& b,
    array& out,
    BinaryOpType bopt,
    bool invert_predicate = false) {
  switch (out.dtype()) {
    case bool_:
      select_binary_op<bool>(a, b, out, bopt, invert_predicate);
      break;
    case uint8:
      select_binary_op<uint8_t>(a, b, out, bopt, invert_predicate);
      break;
    case uint16:
      select_binary_op<uint16_t>(a, b, out, bopt, invert_predicate);
      break;
    case uint32:
      select_binary_op<uint32_t>(a, b, out, bopt, invert_predicate);
      break;
    case uint64:
      select_binary_op<uint64_t>(a, b, out, bopt, invert_predicate);
      break;
    case int8:
      select_binary_op<int8_t>(a, b, out, bopt, invert_predicate);
      break;
    case int16:
      select_binary_op<int16_t>(a, b, out, bopt, invert_predicate);
      break;
    case int32:
      select_binary_op<int32_t>(a, b, out, bopt, invert_predicate);
      break;
    case int64:
      select_binary_op<int64_t>(a, b, out, bopt, invert_predicate);
      break;
    case float16:
      select_binary_op<float16_t>(a, b, out, bopt, invert_predicate);
      break;
    case float32:
      select_binary_op<float>(a, b, out, bopt, invert_predicate);
      break;
    case bfloat16:
      select_binary_op<bfloat16_t>(a, b, out, bopt, invert_predicate);
      break;
    case complex64:
      select_binary_op<complex64_t>(a, b, out, bopt, invert_predicate);
      break;
  }
}

} // namespace

void Select::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);
  const auto& condition = inputs[0];
  const auto& a = inputs[1];
  const auto& b = inputs[2];

  BinaryOpType bopt = get_select_binary_op_type(condition, a, b);
  set_select_binary_op_output_data(condition, a, b, out, bopt);

  select_with_predicate(condition, a, out, bopt);
  select_with_predicate(condition, b, out, bopt, true /* invert_predicate */);
}

} // namespace mlx::core
