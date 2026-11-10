// Copyright © 2023 Apple Inc.

#include <cassert>
#include <cmath>
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

namespace {

struct LogAddExpGradOp {
  template <typename T>
  using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;

  template <int N, typename T>
  std::pair<simd::Simd<T, N>, simd::Simd<T, N>> operator()(
      simd::Simd<T, N> a,
      simd::Simd<T, N> b) const {
    using A = Acc<T>;
    auto diff = simd::Simd<A, N>(a) - simd::Simd<A, N>(b);
    auto e = simd::exp(-simd::abs(diff));
    auto large = simd::Simd<A, N>(1) / (simd::Simd<A, N>(1) + e);
    auto small = e * large;
    auto positive = diff >= simd::Simd<A, N>(0);
    return {
        simd::Simd<T, N>(simd::select(positive, large, small)),
        simd::Simd<T, N>(simd::select(positive, small, large))};
  }

  template <typename T>
  std::pair<T, T> operator()(T a, T b) const {
    auto [first, second] = (*this)(simd::Simd<T, 1>(a), simd::Simd<T, 1>(b));
    return {first.value, second.value};
  }
};

} // namespace

void LogAddExpGrad::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, outputs[0], bopt);
  set_binary_op_output_data(a, b, outputs[1], bopt);
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(outputs[0]);
  encoder.set_output_array(outputs[1]);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    b = array::unsafe_weak_copy(b),
                    first = array::unsafe_weak_copy(outputs[0]),
                    second = array::unsafe_weak_copy(outputs[1]),
                    bopt]() mutable {
    dispatch_float_types(first.dtype(), "LogAddExpGrad", [&](auto tag) {
      using T = MLX_GET_TYPE(tag);
      LogAddExpGradOp op;
      if (bopt == BinaryOpType::General) {
        binary_op<T>(a, b, first, second, op, bopt);
        return;
      }
      constexpr int N = simd::max_size<LogAddExpGradOp::Acc<T>>;
      const bool scalar_a = bopt == BinaryOpType::ScalarScalar ||
          bopt == BinaryOpType::ScalarVector;
      const bool scalar_b = bopt == BinaryOpType::ScalarScalar ||
          bopt == BinaryOpType::VectorScalar;
      const T* ap = a.data<T>();
      const T* bp = b.data<T>();
      T* firstp = first.data<T>();
      T* secondp = second.data<T>();
      size_t count = first.data_size();
      while (count >= N) {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
          simd::Simd<float, N> av, bv;
          for (int i = 0; i < N; ++i) {
            av[i] = static_cast<float>(ap[scalar_a ? 0 : i]);
            bv[i] = static_cast<float>(bp[scalar_b ? 0 : i]);
          }
          auto [x, y] = op(av, bv);
          for (int i = 0; i < N; ++i) {
            firstp[i] = T(x[i]);
            secondp[i] = T(y[i]);
          }
        } else {
          auto av = scalar_a ? simd::Simd<T, N>(*ap) : simd::load<T, N>(ap);
          auto bv = scalar_b ? simd::Simd<T, N>(*bp) : simd::load<T, N>(bp);
          auto [x, y] = op(av, bv);
          simd::store(firstp, x);
          simd::store(secondp, y);
        }
        ap += scalar_a ? 0 : N;
        bp += scalar_b ? 0 : N;
        firstp += N;
        secondp += N;
        count -= N;
      }
      while (count-- > 0) {
        std::tie(*firstp++, *secondp++) = op(*ap, *bp);
        ap += !scalar_a;
        bp += !scalar_b;
      }
    });
  });
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
      auto q = x / y;
      auto r = x % y;
      if constexpr (std::is_signed_v<decltype(x)>) {
        if (r != 0 && (r < 0) != (y < 0)) {
          q -= 1;
          r += y;
        }
      }
      return std::make_pair(q, r);
    };
    auto float_op = [](auto x, auto y) {
      auto r = std::fmod(x, y);
      if (r != 0 && (r < 0) != (y < 0)) {
        r += y;
      }
      return std::make_pair(std::floor(x / y), r);
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

void FloorDivide::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  binary_op_cpu(a, b, out, detail::FloorDivide(), stream());
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
