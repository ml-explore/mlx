// Copyright © 2023 Apple Inc.

#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/binary.h"
#include "mlx/backend/cpu/binary_ops.h"
#include "mlx/backend/cpu/binary_two.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

void Add::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary_op_cpu(a, b, out, detail::Add(), stream());
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
    // Match numpy's semantics: the quotient is floor(a / b) and the remainder
    // carries the divisor's sign, so q * b + r == a holds for every sign
    // combination.
    auto integral_op = [](auto x, auto y) {
      // Start from the truncating quotient/remainder and shift both down by
      // one when the signs differ. This avoids (a - r), which can overflow
      // signed integers at the extremes (e.g. INT_MAX / -2).
      auto q = x / y;
      auto r = x % y;
      if (r != 0 && ((r < 0) != (y < 0))) {
        q -= 1;
        r += y;
      }
      return std::make_pair(q, r);
    };
    auto float_op = [](auto x, auto y) {
      auto r = std::fmod(x, y);
      decltype(r) q;
      if (y == 0) {
        // numpy treats b == 0 specially: the quotient is a / b (which yields
        // +/-inf for a nonzero) and the remainder is fmod(a, b) (nan).
        q = static_cast<decltype(r)>(x / y);
      } else if (std::isnan(x) || std::isnan(y) || std::isinf(x)) {
        // numpy's floor_divide returns nan for nan inputs and for an infinite
        // dividend (but keeps +/-inf when a finite dividend overflows).
        q = std::numeric_limits<decltype(r)>::quiet_NaN();
      } else {
        // floor(a / b) matches numpy bit for bit; deriving the quotient from
        // the remainder instead can differ by one ulp.
        q = std::floor(x / y);
      }
      if (r != 0 && ((r < 0) != (y < 0))) {
        r += y;
      }
      return std::make_pair(q, r);
    };

    dispatch_all_types(out_a.dtype(), [&](auto type_tag) {
      using T = MLX_GET_TYPE(type_tag);
      if constexpr (std::is_same_v<T, complex64_t>) {
        throw std::runtime_error("[DivMod] Complex type not supported");
      } else if constexpr (std::is_integral_v<T>) {
        binary_op<T>(a, b, out_a, out_b, integral_op, bopt);
      } else {
        binary_op<T>(a, b, out_a, out_b, float_op, bopt);
      }
    });
  });
}

void Divide::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary_op_cpu(a, b, out, detail::Divide(), stream());
}

void Remainder::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary_op_cpu(a, b, out, detail::Remainder(), stream());
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
      dispatch_inexact_types(
          a.dtype(), "[NanEqual::eval_cpu]", [&](auto type_tag) {
            using T = MLX_GET_TYPE(type_tag);
            binary_op<T, bool, detail::NaNEqual>(a, b, out, bopt);
          });
    });
  } else {
    comparison_op_cpu(a, b, out, detail::Equal(), stream());
  }
}

void Greater::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op_cpu(inputs[0], inputs[1], out, detail::Greater(), stream());
}

void GreaterEqual::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op_cpu(
      inputs[0], inputs[1], out, detail::GreaterEqual(), stream());
}

void Less::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op_cpu(inputs[0], inputs[1], out, detail::Less(), stream());
}

void LessEqual::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op_cpu(inputs[0], inputs[1], out, detail::LessEqual(), stream());
}

void LogAddExp::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary_float_op_cpu(a, b, out, detail::LogAddExp(), stream());
}

void LogicalAnd::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2); // LogicalAnd requires two input arrays
  auto& in1 = inputs[0];
  auto& in2 = inputs[1];
  binary_op_cpu(in1, in2, out, detail::LogicalAnd(), stream());
}

void LogicalOr::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2); // LogicalOr requires two input arrays
  auto& in1 = inputs[0];
  auto& in2 = inputs[1];
  binary_op_cpu(in1, in2, out, detail::LogicalOr(), stream());
}

void Maximum::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary_op_cpu(a, b, out, detail::Maximum(), stream());
}

void Minimum::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary_op_cpu(a, b, out, detail::Minimum(), stream());
}

void Multiply::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary_op_cpu(a, b, out, detail::Multiply(), stream());
}

void NotEqual::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  comparison_op_cpu(inputs[0], inputs[1], out, detail::NotEqual(), stream());
}

void Power::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary_op_cpu(a, b, out, detail::Power(), stream());
}

void Subtract::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary_op_cpu(a, b, out, detail::Subtract(), stream());
}

void BitwiseBinary::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  switch (op_) {
    case BitwiseBinary::And:
      binary_int_op_cpu(a, b, out, detail::BitwiseAnd(), stream());
      break;
    case BitwiseBinary::Or:
      binary_int_op_cpu(a, b, out, detail::BitwiseOr(), stream());
      break;
    case BitwiseBinary::Xor:
      binary_int_op_cpu(a, b, out, detail::BitwiseXor(), stream());
      break;
    case BitwiseBinary::LeftShift:
      binary_int_op_cpu(a, b, out, detail::LeftShift(), stream());
      break;
    case BitwiseBinary::RightShift:
      binary_int_op_cpu(a, b, out, detail::RightShift(), stream());
      break;
  }
}

void ArcTan2::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  binary_float_op_cpu(a, b, out, detail::ArcTan2(), stream());
}

} // namespace mlx::core
