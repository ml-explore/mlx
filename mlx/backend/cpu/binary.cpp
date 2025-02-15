// Copyright © 2023 Apple Inc.

#include <cassert>
#include <cmath>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/binary.h"
#include "mlx/backend/cpu/binary_ops.h"
#include "mlx/backend/cpu/binary_two.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

template <typename Op>
void comparison_op(const array& a, const array& b, array& out) {
  switch (a.dtype()) {
    case bool_:
      binary_op<bool, bool, Op>(a, b, out);
      break;
    case uint8:
      binary_op<uint8_t, bool, Op>(a, b, out);
      break;
    case uint16:
      binary_op<uint16_t, bool, Op>(a, b, out);
      break;
    case uint32:
      binary_op<uint32_t, bool, Op>(a, b, out);
      break;
    case uint64:
      binary_op<uint64_t, bool, Op>(a, b, out);
      break;
    case int8:
      binary_op<int8_t, bool, Op>(a, b, out);
      break;
    case int16:
      binary_op<int16_t, bool, Op>(a, b, out);
      break;
    case int32:
      binary_op<int32_t, bool, Op>(a, b, out);
      break;
    case int64:
      binary_op<int64_t, bool, Op>(a, b, out);
      break;
    case float16:
      binary_op<float16_t, bool, Op>(a, b, out);
      break;
    case float32:
      binary_op<float, bool, Op>(a, b, out);
      break;
    case float64:
      binary_op<double, bool, Op>(a, b, out);
      break;
    case bfloat16:
      binary_op<bfloat16_t, bool, Op>(a, b, out);
      break;
    case complex64:
      binary_op<complex64_t, bool, Op>(a, b, out);
      break;
  }
}

} // namespace

void Add::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Add());
}

void DivMod::eval_cpu(
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
    case float64:
      binary_op<double>(a, b, outputs, float_op);
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

void Divide::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Divide());
}

void Remainder::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Remainder());
}

void Equal::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  if (equal_nan_) {
    switch (a.dtype()) {
      case float16:
        binary_op<float16_t, bool, detail::NaNEqual>(a, b, out);
        break;
      case float32:
        binary_op<float, bool, detail::NaNEqual>(a, b, out);
        break;
      case float64:
        binary_op<double, bool, detail::NaNEqual>(a, b, out);
        break;
      case bfloat16:
        binary_op<bfloat16_t, bool, detail::NaNEqual>(a, b, out);
        break;
      case complex64:
        binary_op<complex64_t, bool, detail::NaNEqual>(a, b, out);
        break;
      default:
        throw std::runtime_error(
            "[NanEqual::eval_cpu] Only for floating point types.");
    }
  } else {
    comparison_op<detail::Equal>(a, b, out);
  }
}

void Greater::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op<detail::Greater>(inputs[0], inputs[1], out);
}

void GreaterEqual::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op<detail::GreaterEqual>(inputs[0], inputs[1], out);
}

void Less::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op<detail::Less>(inputs[0], inputs[1], out);
}

void LessEqual::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op<detail::LessEqual>(inputs[0], inputs[1], out);
}

void LogAddExp::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  switch (out.dtype()) {
    case float16:
      binary_op<float16_t, detail::LogAddExp>(a, b, out);
      break;
    case float32:
      binary_op<float, detail::LogAddExp>(a, b, out);
      break;
    case float64:
      binary_op<double, detail::LogAddExp>(a, b, out);
      break;
    case bfloat16:
      binary_op<bfloat16_t, detail::LogAddExp>(a, b, out);
      break;
    default:
      throw std::runtime_error(
          "[LogAddExp::eval_cpu] Only supports non-complex floating point types.");
  }
}

void LogicalAnd::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2); // LogicalAnd requires two input arrays
  auto& in1 = inputs[0];
  auto& in2 = inputs[1];
  binary(in1, in2, out, detail::LogicalAnd());
}

void LogicalOr::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2); // LogicalOr requires two input arrays
  auto& in1 = inputs[0];
  auto& in2 = inputs[1];
  binary(in1, in2, out, detail::LogicalOr());
}

void Maximum::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Maximum());
}

void Minimum::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Minimum());
}

void Multiply::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Multiply());
}

void NotEqual::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op<detail::NotEqual>(inputs[0], inputs[1], out);
}

void Power::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Power());
}

void Subtract::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Subtract());
}

void BitwiseBinary::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto dispatch_type = [&a, &b, &out](auto op) {
    switch (out.dtype()) {
      case bool_:
        binary_op<bool>(a, b, out, op);
      case uint8:
        binary_op<uint8_t>(a, b, out, op);
        break;
      case uint16:
        binary_op<uint16_t>(a, b, out, op);
        break;
      case uint32:
        binary_op<uint32_t>(a, b, out, op);
        break;
      case uint64:
        binary_op<uint64_t>(a, b, out, op);
        break;
      case int8:
        binary_op<int8_t>(a, b, out, op);
        break;
      case int16:
        binary_op<int16_t>(a, b, out, op);
        break;
      case int32:
        binary_op<int32_t>(a, b, out, op);
        break;
      case int64:
        binary_op<int64_t>(a, b, out, op);
        break;
      default:
        throw std::runtime_error(
            "[BitwiseBinary::eval_cpu] Type not supported");
        break;
    }
  };
  switch (op_) {
    case BitwiseBinary::And:
      dispatch_type(detail::BitwiseAnd());
      break;
    case BitwiseBinary::Or:
      dispatch_type(detail::BitwiseOr());
      break;
    case BitwiseBinary::Xor:
      dispatch_type(detail::BitwiseXor());
      break;
    case BitwiseBinary::LeftShift:
      dispatch_type(detail::LeftShift());
      break;
    case BitwiseBinary::RightShift:
      dispatch_type(detail::RightShift());
      break;
  }
}

void ArcTan2::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  switch (out.dtype()) {
    case float16:
      binary_op<float16_t>(a, b, out, detail::ArcTan2());
      break;
    case float32:
      binary_op<float>(a, b, out, detail::ArcTan2());
      break;
    case float64:
      binary_op<double>(a, b, out, detail::ArcTan2());
      break;
    case bfloat16:
      binary_op<bfloat16_t>(a, b, out, detail::ArcTan2());
      break;
    default:
      throw std::runtime_error(
          "[ArcTan2::eval_cpu] Only supports non-complex floating point types.");
  }
}

} // namespace mlx::core
