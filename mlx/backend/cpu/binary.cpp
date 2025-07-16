// Copyright © 2023 Apple Inc.

#include <cassert>
#include <cmath>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/binary.h"
#include "mlx/backend/cpu/binary_ops.h"
#include "mlx/backend/cpu/binary_two.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

template <typename Op>
void binary(const array& a, const array& b, array& out, Op op, Stream stream) {
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    b = array::unsafe_weak_copy(b),
                    out = array::unsafe_weak_copy(out),
                    bopt]() mutable {
    switch (out.dtype()) {
      case bool_:
        binary_op<bool, Op>(a, b, out, bopt);
        break;
      case uint8:
        binary_op<uint8_t, Op>(a, b, out, bopt);
        break;
      case uint16:
        binary_op<uint16_t, Op>(a, b, out, bopt);
        break;
      case uint32:
        binary_op<uint32_t, Op>(a, b, out, bopt);
        break;
      case uint64:
        binary_op<uint64_t, Op>(a, b, out, bopt);
        break;
      case int8:
        binary_op<int8_t, Op>(a, b, out, bopt);
        break;
      case int16:
        binary_op<int16_t, Op>(a, b, out, bopt);
        break;
      case int32:
        binary_op<int32_t, Op>(a, b, out, bopt);
        break;
      case int64:
        binary_op<int64_t, Op>(a, b, out, bopt);
        break;
      case float16:
        binary_op<float16_t, Op>(a, b, out, bopt);
        break;
      case float32:
        binary_op<float, Op>(a, b, out, bopt);
        break;
      case float64:
        binary_op<double, Op>(a, b, out, bopt);
        break;
      case bfloat16:
        binary_op<bfloat16_t, Op>(a, b, out, bopt);
        break;
      case complex64:
        binary_op<complex64_t, Op>(a, b, out, bopt);
        break;
    }
  });
}

template <typename Op>
void comparison_op(
    const array& a,
    const array& b,
    array& out,
    Op op,
    Stream stream) {
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    b = array::unsafe_weak_copy(b),
                    out = array::unsafe_weak_copy(out),
                    bopt]() mutable {
    switch (a.dtype()) {
      case bool_:
        binary_op<bool, bool, Op>(a, b, out, bopt);
        break;
      case uint8:
        binary_op<uint8_t, bool, Op>(a, b, out, bopt);
        break;
      case uint16:
        binary_op<uint16_t, bool, Op>(a, b, out, bopt);
        break;
      case uint32:
        binary_op<uint32_t, bool, Op>(a, b, out, bopt);
        break;
      case uint64:
        binary_op<uint64_t, bool, Op>(a, b, out, bopt);
        break;
      case int8:
        binary_op<int8_t, bool, Op>(a, b, out, bopt);
        break;
      case int16:
        binary_op<int16_t, bool, Op>(a, b, out, bopt);
        break;
      case int32:
        binary_op<int32_t, bool, Op>(a, b, out, bopt);
        break;
      case int64:
        binary_op<int64_t, bool, Op>(a, b, out, bopt);
        break;
      case float16:
        binary_op<float16_t, bool, Op>(a, b, out, bopt);
        break;
      case float32:
        binary_op<float, bool, Op>(a, b, out, bopt);
        break;
      case float64:
        binary_op<double, bool, Op>(a, b, out, bopt);
        break;
      case bfloat16:
        binary_op<bfloat16_t, bool, Op>(a, b, out, bopt);
        break;
      case complex64:
        binary_op<complex64_t, bool, Op>(a, b, out, bopt);
        break;
    }
  });
}

template <typename Op>
void binary_float(
    const array& a,
    const array& b,
    array& out,
    Op op,
    Stream stream) {
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    b = array::unsafe_weak_copy(b),
                    out = array::unsafe_weak_copy(out),
                    bopt]() mutable {
    switch (out.dtype()) {
      case float16:
        binary_op<float16_t, Op>(a, b, out, bopt);
        break;
      case float32:
        binary_op<float, Op>(a, b, out, bopt);
        break;
      case float64:
        binary_op<double, Op>(a, b, out, bopt);
        break;
      case bfloat16:
        binary_op<bfloat16_t, Op>(a, b, out, bopt);
        break;
      case complex64:
        binary_op<complex64_t, Op>(a, b, out, bopt);
        break;
      default:
        throw std::runtime_error(
            "[binary_float] Only supports floating point types.");
    }
  });
}

template <typename Op>
void binary_int(
    const array& a,
    const array& b,
    array& out,
    Op op,
    Stream stream) {
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    b = array::unsafe_weak_copy(b),
                    out = array::unsafe_weak_copy(out),
                    bopt]() mutable {
    switch (out.dtype()) {
      case bool_:
        binary_op<bool, Op>(a, b, out, bopt);
      case uint8:
        binary_op<uint8_t, Op>(a, b, out, bopt);
        break;
      case uint16:
        binary_op<uint16_t, Op>(a, b, out, bopt);
        break;
      case uint32:
        binary_op<uint32_t, Op>(a, b, out, bopt);
        break;
      case uint64:
        binary_op<uint64_t, Op>(a, b, out, bopt);
        break;
      case int8:
        binary_op<int8_t, Op>(a, b, out, bopt);
        break;
      case int16:
        binary_op<int16_t, Op>(a, b, out, bopt);
        break;
      case int32:
        binary_op<int32_t, Op>(a, b, out, bopt);
        break;
      case int64:
        binary_op<int64_t, Op>(a, b, out, bopt);
        break;
      default:
        throw std::runtime_error("[binary_int] Type not supported");
        break;
    }
  });
}

} // namespace

void Add::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Add(), stream());
}

void DivMod::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  auto& out_a = outputs[0];
  auto& out_b = outputs[1];
  set_binary_op_output_data(a, b, out_a, bopt);
  set_binary_op_output_data(a, b, out_b, bopt);

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out_a);
  encoder.set_output_array(out_b);

  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    b = array::unsafe_weak_copy(b),
                    out_a = array::unsafe_weak_copy(out_a),
                    out_b = array::unsafe_weak_copy(out_b),
                    bopt]() mutable {
    auto integral_op = [](auto x, auto y) {
      return std::make_pair(x / y, x % y);
    };
    auto float_op = [](auto x, auto y) {
      return std::make_pair(std::trunc(x / y), std::fmod(x, y));
    };

    switch (out_a.dtype()) {
      case bool_:
        binary_op<bool>(a, b, out_a, out_b, integral_op, bopt);
      case uint8:
        binary_op<uint8_t>(a, b, out_a, out_b, integral_op, bopt);
        break;
      case uint16:
        binary_op<uint16_t>(a, b, out_a, out_b, integral_op, bopt);
        break;
      case uint32:
        binary_op<uint32_t>(a, b, out_a, out_b, integral_op, bopt);
        break;
      case uint64:
        binary_op<uint64_t>(a, b, out_a, out_b, integral_op, bopt);
        break;
      case int8:
        binary_op<int8_t>(a, b, out_a, out_b, integral_op, bopt);
        break;
      case int16:
        binary_op<int16_t>(a, b, out_a, out_b, integral_op, bopt);
        break;
      case int32:
        binary_op<int32_t>(a, b, out_a, out_b, integral_op, bopt);
        break;
      case int64:
        binary_op<int64_t>(a, b, out_a, out_b, integral_op, bopt);
        break;
      case float16:
        binary_op<float16_t>(a, b, out_a, out_b, float_op, bopt);
        break;
      case float32:
        binary_op<float>(a, b, out_a, out_b, float_op, bopt);
        break;
      case float64:
        binary_op<double>(a, b, out_a, out_b, float_op, bopt);
        break;
      case bfloat16:
        binary_op<bfloat16_t>(a, b, out_a, out_b, float_op, bopt);
        break;
      case complex64:
        // Should never get here
        throw std::runtime_error("[DivMod] Complex type not supported");
        break;
    }
  });
}

void Divide::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Divide(), stream());
}

void Remainder::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Remainder(), stream());
}

void Equal::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  if (equal_nan_) {
    auto bopt = get_binary_op_type(a, b);
    set_binary_op_output_data(a, b, out, bopt);

    auto& encoder = cpu::get_command_encoder(stream());
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);
    encoder.dispatch([a = array::unsafe_weak_copy(a),
                      b = array::unsafe_weak_copy(b),
                      out = array::unsafe_weak_copy(out),
                      bopt]() mutable {
      switch (a.dtype()) {
        case float16:
          binary_op<float16_t, bool, detail::NaNEqual>(a, b, out, bopt);
          break;
        case float32:
          binary_op<float, bool, detail::NaNEqual>(a, b, out, bopt);
          break;
        case float64:
          binary_op<double, bool, detail::NaNEqual>(a, b, out, bopt);
          break;
        case bfloat16:
          binary_op<bfloat16_t, bool, detail::NaNEqual>(a, b, out, bopt);
          break;
        case complex64:
          binary_op<complex64_t, bool, detail::NaNEqual>(a, b, out, bopt);
          break;
        default:
          throw std::runtime_error(
              "[NanEqual::eval_cpu] Only for floating point types.");
      }
    });
  } else {
    comparison_op(a, b, out, detail::Equal(), stream());
  }
}

void Greater::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op(inputs[0], inputs[1], out, detail::Greater(), stream());
}

void GreaterEqual::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op(inputs[0], inputs[1], out, detail::GreaterEqual(), stream());
}

void Less::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op(inputs[0], inputs[1], out, detail::Less(), stream());
}

void LessEqual::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op(inputs[0], inputs[1], out, detail::LessEqual(), stream());
}

void LogAddExp::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary_float(a, b, out, detail::LogAddExp(), stream());
}

void LogicalAnd::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2); // LogicalAnd requires two input arrays
  auto& in1 = inputs[0];
  auto& in2 = inputs[1];
  binary(in1, in2, out, detail::LogicalAnd(), stream());
}

void LogicalOr::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2); // LogicalOr requires two input arrays
  auto& in1 = inputs[0];
  auto& in2 = inputs[1];
  binary(in1, in2, out, detail::LogicalOr(), stream());
}

void Maximum::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Maximum(), stream());
}

void Minimum::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Minimum(), stream());
}

void Multiply::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Multiply(), stream());
}

void NotEqual::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op(inputs[0], inputs[1], out, detail::NotEqual(), stream());
}

void Power::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Power(), stream());
}

void Subtract::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary(a, b, out, detail::Subtract(), stream());
}

void BitwiseBinary::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  switch (op_) {
    case BitwiseBinary::And:
      binary_int(a, b, out, detail::BitwiseAnd(), stream());
      break;
    case BitwiseBinary::Or:
      binary_int(a, b, out, detail::BitwiseOr(), stream());
      break;
    case BitwiseBinary::Xor:
      binary_int(a, b, out, detail::BitwiseXor(), stream());
      break;
    case BitwiseBinary::LeftShift:
      binary_int(a, b, out, detail::LeftShift(), stream());
      break;
    case BitwiseBinary::RightShift:
      binary_int(a, b, out, detail::RightShift(), stream());
      break;
  }
}

void ArcTan2::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  binary_float(a, b, out, detail::ArcTan2(), stream());
}

} // namespace mlx::core
