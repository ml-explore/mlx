// Copyright © 2023 Apple Inc.

#include <cassert>
#include <cmath>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/binary_two.h"
#include "mlx/backend/common/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

template <typename T1, typename T2, typename U, typename Op>
void comparison_op(const array& a, const array& b, array& out, Op op) {
  DefaultScalarScalar<T1, T2, U, Op> opss(op);
  DefaultScalarVector<T1, T2, U, Op> opsv(op);
  DefaultVectorScalar<T1, T2, U, Op> opvs(op);
  DefaultVectorVector<T1, T2, U, Op> opvv(op);
  binary_op<T1, T2, U>(a, b, out, opss, opsv, opvs, opvv);
}

template <typename Op>
void comparison_op(const array& a, const array& b, array& out, Op op) {
  switch (a.dtype()) {
    case bool_:
      comparison_op<bool, bool, bool>(a, b, out, op);
      break;
    case uint8:
      comparison_op<uint8_t, uint8_t, bool>(a, b, out, op);
      break;
    case uint16:
      comparison_op<uint16_t, uint16_t, bool>(a, b, out, op);
      break;
    case uint32:
      comparison_op<uint32_t, uint32_t, bool>(a, b, out, op);
      break;
    case uint64:
      comparison_op<uint64_t, uint64_t, bool>(a, b, out, op);
      break;
    case int8:
      comparison_op<int8_t, int8_t, bool>(a, b, out, op);
      break;
    case int16:
      comparison_op<int16_t, int16_t, bool>(a, b, out, op);
      break;
    case int32:
      comparison_op<int32_t, int32_t, bool>(a, b, out, op);
      break;
    case int64:
      comparison_op<int64_t, int64_t, bool>(a, b, out, op);
      break;
    case float16:
      comparison_op<float16_t, float16_t, bool>(a, b, out, op);
      break;
    case float32:
      comparison_op<float, float, bool>(a, b, out, op);
      break;
    case bfloat16:
      comparison_op<bfloat16_t, bfloat16_t, bool>(a, b, out, op);
      break;
    case complex64:
      comparison_op<complex64_t, complex64_t, bool>(a, b, out, op);
      break;
  }
}

} // namespace

void Add::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Add());
}

void DivMod::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto integral_op = [](auto x, auto y) {
    return std::make_pair(x / y, x % y);
  };
  auto float_op = [](auto x, auto y) {
    return std::make_pair(std::trunc(x / y), std::fmod(x, y));
  };
  switch (outputs[0].dtype()) {
    case bool_:
      binary_op<bool>(a, b, outputs, integral_op);
    case uint8:
      binary_op<uint8_t>(a, b, outputs, integral_op);
      break;
    case uint16:
      binary_op<uint16_t>(a, b, outputs, integral_op);
      break;
    case uint32:
      binary_op<uint32_t>(a, b, outputs, integral_op);
      break;
    case uint64:
      binary_op<uint64_t>(a, b, outputs, integral_op);
      break;
    case int8:
      binary_op<int8_t>(a, b, outputs, integral_op);
      break;
    case int16:
      binary_op<int16_t>(a, b, outputs, integral_op);
      break;
    case int32:
      binary_op<int32_t>(a, b, outputs, integral_op);
      break;
    case int64:
      binary_op<int64_t>(a, b, outputs, integral_op);
      break;
    case float16:
      binary_op<float16_t>(a, b, outputs, float_op);
      break;
    case float32:
      binary_op<float>(a, b, outputs, float_op);
      break;
    case bfloat16:
      binary_op<bfloat16_t>(a, b, outputs, float_op);
      break;
    case complex64:
      // Should never get here
      throw std::runtime_error("[DivMod] Complex type not supported");
      break;
  }
}

void Divide::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Divide());
}

void Remainder::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Remainder());
}

void Equal::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (equal_nan_) {
    comparison_op(inputs[0], inputs[1], out, detail::NaNEqual());
  } else {
    comparison_op(inputs[0], inputs[1], out, detail::Equal());
  }
}

void Greater::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op(inputs[0], inputs[1], out, detail::Greater());
}

void GreaterEqual::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op(inputs[0], inputs[1], out, detail::GreaterEqual());
}

void Less::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op(inputs[0], inputs[1], out, detail::Less());
}

void LessEqual::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op(inputs[0], inputs[1], out, detail::LessEqual());
}

void LogAddExp::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  if (is_floating_point(out.dtype())) {
    if (out.dtype() == float32) {
      binary_op<float>(a, b, out, detail::LogAddExp());
    } else if (out.dtype() == float16) {
      binary_op<float16_t>(a, b, out, detail::LogAddExp());
    } else if (out.dtype() == bfloat16) {
      binary_op<bfloat16_t>(a, b, out, detail::LogAddExp());
    } else {
      std::ostringstream err;
      err << "[logaddexp] Does not support " << out.dtype();
      throw std::invalid_argument(err.str());
    }
  } else {
    throw std::invalid_argument(
        "[logaddexp] Cannot compute logaddexp for arrays with"
        " non floating point type.");
  }
}

void Maximum::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Maximum());
}

void Minimum::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Minimum());
}

void Multiply::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Multiply());
}

void NotEqual::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op(inputs[0], inputs[1], out, detail::NotEqual());
}

void Power::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Power());
}

void Subtract::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Subtract());
}

} // namespace mlx::core
